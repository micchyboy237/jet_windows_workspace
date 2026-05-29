"""
audio_tagger_base.py
====================
Abstract base class and all shared utilities for sherpa-onnx audio tagging.
Design patterns used
--------------------
Template Method  — BaseAudioTagger.tag_file() defines the invariant pipeline;
                   _get_model_paths() and _build_sherpa_config() are the variant
                   hooks that each backend overrides.
Strategy         — ChunkProcessor and ResultsReporter are injected via composition
                   so they can be swapped or tested independently.
DRY              — Every line that was duplicated across ced / zipformer lives here
                   exactly once.

Aligned with FireRed VAD:
    - Uses FRAME_SHIFT_SAMPLE (160 samples) as the fundamental unit
    - Windows are multiples of FireRed frames (100 frames = 1s window)
    - Supports per-segment tagging with absolute UTC timestamps
"""
from __future__ import annotations
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import sherpa_onnx
import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# FireRed alignment imports
from fireredvad.core.constants import (
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE as FIRERED_SAMPLE_RATE,
)

console = Console()
log = logging.getLogger(__name__)

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()

# Aligned with FireRed VAD constants
SAMPLE_RATE = FIRERED_SAMPLE_RATE  # 16000 Hz - must match FireRed
HOP_LENGTH = FRAME_SHIFT_SAMPLE    # 160 samples (10ms) - must match FireRed frame shift


@dataclass
class TaggingEvent:
    """Normalised event produced by either model backend.
    
    Now supports absolute UTC timestamps when processing speech segments.
    """
    name: Optional[str]
    class_index: Optional[int]
    prob: float
    max_prob: float = 0.0
    occurrences: int = 1
    
    # Relative timing within audio file/segment
    chunk_start: float = 0.0
    chunk_end: float = 0.0
    chunk_index: int = 0
    
    # Absolute UTC timestamps (only set when processing from FireRed stream)
    time_utc_start: Optional[datetime] = None
    time_utc_end: Optional[datetime] = None


@dataclass
class TaggingResult:
    """Full result from one audio file or speech segment tagging run.
    
    Now supports both file-based and segment-based processing.
    """
    audio_path: str
    sample_rate: int
    duration: float
    elapsed_time: float
    events: List[TaggingEvent]
    chunk_count: int
    backend_name: str
    model_variant: str
    top_k: int
    
    # Set to True when this result is from a live speech segment
    is_speech_segment: bool = False
    
    # Absolute UTC timestamps for speech segments
    segment_start_utc: Optional[datetime] = None
    segment_end_utc: Optional[datetime] = None
    
    @property
    def real_time_factor(self) -> float:
        return self.elapsed_time / self.duration if self.duration > 0 else 0.0


def read_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Read any soundfile-supported file.
    Returns (mono float32 samples, original sample rate).
    """
    p = Path(audio_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\n"
            "Please check the path and try again."
        )
    data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    samples = np.ascontiguousarray(data[:, 0])
    log.debug(
        f"Read [cyan]{len(samples):,}[/cyan] samples at [cyan]{sr} Hz[/cyan] "
        f"from [cyan]{audio_path}[/cyan]"
    )
    return samples, sr


def resample_if_needed(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Linear-interpolation resample; no-op when rates already match."""
    if orig_sr == target_sr:
        return samples
    log.info(f"Resampling [cyan]{orig_sr} Hz → {target_sr} Hz[/cyan]")
    target_len = int(len(samples) / orig_sr * target_sr)
    idx = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


def find_model_file(model_path: Path, model_int8_path: Path) -> str:
    """
    Return the path to whichever model file exists, preferring int8.
    Raises FileNotFoundError with a clear message when neither exists.
    """
    if model_int8_path.is_file():
        log.info("Using int8 quantised model")
        return str(model_int8_path)
    if model_path.is_file():
        log.info("Using standard (fp32) model")
        return str(model_path)
    raise FileNotFoundError(
        "No model file found. Checked:\n"
        f"  • {model_int8_path}\n"
        f"  • {model_path}\n"
        "Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
    )


def _validate_firered_alignment(expected_frames: int, hop_length: int) -> None:
    """
    Validate that tagger parameters align with FireRed VAD frame boundaries.
    
    FireRed operates on 10ms frames (160 samples at 16kHz).
    The tagger's window and hop must be multiples of this frame size
    to ensure perfect alignment when processing speech segments.
    
    Args:
        expected_frames: Number of frames per tagger window
        hop_length: Hop length in samples (should equal FRAME_SHIFT_SAMPLE)
    
    Raises:
        ValueError: If alignment check fails
    """
    window_samples = expected_frames * hop_length
    hop_samples = window_samples // 2
    
    # Window must be multiple of FireRed frame size
    if window_samples % FRAME_SHIFT_SAMPLE != 0:
        raise ValueError(
            f"Tagger window ({window_samples} samples) is not aligned to FireRed "
            f"frame size ({FRAME_SHIFT_SAMPLE} samples). "
            f"Window must be a multiple of {FRAME_SHIFT_SAMPLE}."
        )
    
    # Hop must be multiple of FireRed frame size
    if hop_samples % FRAME_SHIFT_SAMPLE != 0:
        raise ValueError(
            f"Tagger hop ({hop_samples} samples) is not aligned to FireRed "
            f"frame size ({FRAME_SHIFT_SAMPLE} samples). "
            f"Hop must be a multiple of {FRAME_SHIFT_SAMPLE}."
        )
    
    log.debug(
        f"✓ FireRed alignment verified: window={window_samples}samples "
        f"({window_samples//FRAME_SHIFT_SAMPLE} frames), "
        f"hop={hop_samples}samples ({hop_samples//FRAME_SHIFT_SAMPLE} frames)"
    )


def process_audio_chunks(
    audio_tagger: sherpa_onnx.AudioTagging,
    samples: np.ndarray,
    sample_rate: int,
    expected_frames: int = 100,
    hop_length: int = HOP_LENGTH,
    segment_start_utc: Optional[datetime] = None,
) -> List[dict]:
    """
    Slide a 50%-overlapping window over the audio and collect raw events.
    
    Window  = expected_frames × hop_length samples  (0.8 s at 16 kHz)
    Step    = window // 2  (50 % overlap)
    Padding — short clips are zero-padded to at least one full window.
    
    Now aligned with FireRed VAD:
        - Validates that window/hop are multiples of FRAME_SHIFT_SAMPLE
        - Supports optional UTC timestamps for speech segments
    
    Args:
        audio_tagger: Configured sherpa-onnx AudioTagging instance
        samples: Audio samples (mono, float32)
        sample_rate: Sample rate in Hz (must be 16000 for FireRed alignment)
        expected_frames: Frames per window (default 100 = 1s)
        hop_length: Hop length in samples (default FRAME_SHIFT_SAMPLE = 160)
        segment_start_utc: Optional UTC timestamp for segment start
    
    Returns:
        List of event dicts with timing information
    """
    # Validate alignment with FireRed
    _validate_firered_alignment(expected_frames, hop_length)
    
    window_samples = expected_frames * hop_length
    hop_samples = window_samples // 2
    total_samples = len(samples)
    
    log.info(
        f"Audio: [cyan]{total_samples:,}[/cyan] samples "
        f"([cyan]{total_samples / sample_rate:.2f}s[/cyan]) | "
        f"Window: [cyan]{window_samples / sample_rate:.2f}s[/cyan] "
        f"({expected_frames} frames × {hop_length} samples)"
    )
    
    # Handle short audio segments
    if total_samples < window_samples:
        log.info(f"Padding short audio ({total_samples} → {window_samples} samples)")
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:total_samples] = samples
        samples = padded
        total_samples = window_samples
    
    num_chunks = max(1, (total_samples - window_samples) // hop_samples + 1)
    all_events: List[dict] = []
    
    log.info(f"Processing [cyan]{num_chunks}[/cyan] overlapping chunk(s)…")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Tagging chunks…", total=num_chunks)
        
        for i in range(num_chunks):
            start = i * hop_samples
            end = min(start + window_samples, total_samples)
            
            # Extract window (zero-pad if at edge)
            chunk = np.zeros(window_samples, dtype=np.float32)
            available = min(window_samples, total_samples - start)
            chunk[:available] = samples[start:start + available]
            
            # Run inference
            stream = audio_tagger.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
            result = audio_tagger.compute(stream)
            
            # Calculate timestamps
            chunk_start_sec = start / sample_rate
            chunk_end_sec = end / sample_rate
            
            # Compute absolute UTC timestamps if segment start is provided
            time_utc_start = None
            time_utc_end = None
            if segment_start_utc is not None:
                from datetime import timedelta
                time_utc_start = segment_start_utc + timedelta(seconds=chunk_start_sec)
                time_utc_end = segment_start_utc + timedelta(seconds=chunk_end_sec)
            
            log.debug(
                f"Chunk {i}: [{chunk_start_sec:.2f}s – {chunk_end_sec:.2f}s] "
                f"events={len(result)}"
            )
            
            for event in result:
                all_events.append({
                    "name": getattr(event, "name", None),
                    "index": getattr(event, "index", None),
                    "prob": getattr(event, "prob", None),
                    "chunk_start": chunk_start_sec,
                    "chunk_end": chunk_end_sec,
                    "chunk_index": i,
                    "time_utc_start": time_utc_start,
                    "time_utc_end": time_utc_end,
                })
            
            progress.update(task, advance=1)
    
    return all_events


def aggregate_chunk_results(
    chunk_events: List[dict],
    top_k: int = 5,
) -> List[TaggingEvent]:
    """
    Average probabilities for the same event across all chunks.
    Returns top-k TaggingEvent objects sorted by mean probability (desc).
    """
    if not chunk_events:
        return []
    
    groups: dict[tuple, dict] = {}
    for ev in chunk_events:
        key = (ev["name"], ev["index"])
        if key not in groups:
            groups[key] = {
                "name": ev["name"],
                "class_index": ev["index"],
                "probs": [],
                "occurrences": 0,
                "chunk_start": ev["chunk_start"],
                "chunk_end": ev["chunk_end"],
                "time_utc_start": ev.get("time_utc_start"),
                "time_utc_end": ev.get("time_utc_end"),
            }
        groups[key]["probs"].append(ev["prob"])
        groups[key]["occurrences"] += 1
        # Track the earliest start and latest end
        groups[key]["chunk_start"] = min(groups[key]["chunk_start"], ev["chunk_start"])
        groups[key]["chunk_end"] = max(groups[key]["chunk_end"], ev["chunk_end"])
    
    aggregated = [
        TaggingEvent(
            name=g["name"],
            class_index=g["class_index"],
            prob=float(np.mean(g["probs"])),
            max_prob=float(np.max(g["probs"])),
            occurrences=g["occurrences"],
            chunk_start=g["chunk_start"],
            chunk_end=g["chunk_end"],
            time_utc_start=g["time_utc_start"],
            time_utc_end=g["time_utc_end"],
        )
        for g in groups.values()
    ]
    
    aggregated.sort(key=lambda e: e.prob, reverse=True)
    return aggregated[:top_k]


def print_results_table(
    events: List[TaggingEvent],
    chunk_count: int,
    backend_name: str
) -> None:
    """Display aggregated results in a Rich table."""
    tbl = Table(
        title=f"🎵 Audio Tagging Results — {backend_name} (aggregated)",
        header_style="bold magenta",
    )
    tbl.add_column("Rank", style="cyan", width=6)
    tbl.add_column("Label", style="green")
    tbl.add_column("Index", style="yellow", width=8)
    tbl.add_column("Avg prob", style="bold white", justify="right")
    tbl.add_column("Max prob", style="dim white", justify="right")
    tbl.add_column("Occurrences", style="blue", justify="right")
    tbl.add_column("Bar", style="magenta")
    
    for i, ev in enumerate(events):
        bar_len = int(ev.prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # Add timing info if available
        label = ev.name or "N/A"
        if ev.time_utc_start:
            label += f"\n[dim]{ev.time_utc_start.strftime('%H:%M:%S.%f')[:-3]}[/dim]"
        
        tbl.add_row(
            str(i + 1),
            label,
            str(ev.class_index) if ev.class_index is not None else "N/A",
            f"{ev.prob * 100:.2f}%",
            f"{ev.max_prob * 100:.2f}%",
            f"{ev.occurrences}/{chunk_count}",
            bar,
        )
    console.print(tbl)


def print_perf_table(result: TaggingResult) -> None:
    """Display performance metrics in a Rich table."""
    rtf = result.real_time_factor
    rtf_style = "green" if rtf < 1.0 else "yellow"
    rtf_icon = "✓" if rtf < 1.0 else "⚠"
    
    perf = Table(title="⚡ Performance metrics")
    perf.add_column("Metric", style="cyan")
    perf.add_column("Value", style="green")
    perf.add_column("Status", style=rtf_style)
    perf.add_row("Processing time", f"{result.elapsed_time:.3f}s", "")
    perf.add_row("Audio duration", f"{result.duration:.3f}s", "")
    perf.add_row(
        "Real-time factor",
        f"{rtf:.3f}x",
        f"{rtf_icon} {'Real-time' if rtf < 1.0 else 'Slower than real-time'}",
    )
    perf.add_row(
        "Processing speed",
        f"{result.duration / result.elapsed_time:.1f}x",
        ""
    )
    perf.add_row("Chunks processed", str(result.chunk_count), "")
    
    if result.is_speech_segment:
        perf.add_row("Type", "[cyan]Speech Segment[/cyan]", "")
        if result.segment_start_utc:
            perf.add_row(
                "UTC Start",
                result.segment_start_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                ""
            )
    
    console.print(perf)


def save_results(result: TaggingResult, output_dir: Path) -> None:
    """Persist results.json + metadata.json and print all summary tables."""
    print_results_table(result.events, result.chunk_count, result.backend_name)
    print_perf_table(result)
    
    events_as_dicts = [
        {
            "rank": i + 1,
            "name": ev.name,
            "class_index": ev.class_index,
            "prob": ev.prob,
            "max_prob": ev.max_prob,
            "occurrences": ev.occurrences,
            "chunk_start": ev.chunk_start,
            "chunk_end": ev.chunk_end,
            "time_utc_start": ev.time_utc_start.isoformat() if ev.time_utc_start else None,
            "time_utc_end": ev.time_utc_end.isoformat() if ev.time_utc_end else None,
        }
        for i, ev in enumerate(result.events)
    ]
    
    results_json = output_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(events_as_dicts, f, indent=2, ensure_ascii=False)
    
    metadata = {
        "audio_file": result.audio_path,
        "sample_rate": result.sample_rate,
        "audio_duration_seconds": round(result.duration, 3),
        "processing_time_seconds": round(result.elapsed_time, 3),
        "real_time_factor": round(result.real_time_factor, 3),
        "backend": result.backend_name,
        "model_variant": result.model_variant,
        "top_k": result.top_k,
        "chunks_processed": result.chunk_count,
        "is_speech_segment": result.is_speech_segment,
        "segment_start_utc": result.segment_start_utc.isoformat() if result.segment_start_utc else None,
        "segment_end_utc": result.segment_end_utc.isoformat() if result.segment_end_utc else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregation_method": "average_probability_with_max",
    }
    
    metadata_json = output_dir / "metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    console.print(Panel(
        f"[cyan]Results:[/cyan]  {results_json}\n"
        f"[cyan]Metadata:[/cyan] {metadata_json}",
        title="💾 Saved files",
        border_style="green",
    ))


class BaseAudioTagger(ABC):
    """
    Template Method base for all sherpa-onnx audio taggers.
    
    Now aligned with FireRed VAD:
        - Uses FRAME_SHIFT_SAMPLE (160 samples) as fundamental unit
        - Windows are multiples of FireRed frames
        - Supports tag_speech_segment() for live audio processing
    
    Subclasses must implement:
      _get_model_paths(variant) → dict with keys: model, model_int8, labels,
                                   test_wavs_dir, model_info
      _build_sherpa_config(model_file, label_file, top_k) → sherpa_onnx.AudioTaggingConfig
    
    The public interface is:
      tagger.build() → self
      tagger.tag_file(audio_path, output_dir) → TaggingResult
      tagger.tag_speech_segment(audio, start_utc, end_utc) → TaggingResult
      tagger.default_test_wav → Path
    """
    
    BACKEND_NAME: str = "base"
    DEFAULT_VARIANT: str = ""
    VALID_VARIANTS: tuple = ()
    
    # Aligned with FireRed: 100 frames × 160 samples = 12,800 samples = 0.8s
    EXPECTED_FRAMES: int = 100
    
    def __init__(self, variant: str = "", top_k: int = 5):
        if not variant:
            variant = self.DEFAULT_VARIANT
        if self.VALID_VARIANTS and variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Valid: {', '.join(self.VALID_VARIANTS)}"
            )
        self.variant = variant
        self.top_k = top_k
        self._tagger: Optional[sherpa_onnx.AudioTagging] = None
        
        # Validate FireRed alignment on init
        _validate_firered_alignment(self.EXPECTED_FRAMES, HOP_LENGTH)
    
    @abstractmethod
    def _get_model_paths(self) -> dict:
        """
        Return a dict describing where the model files live.
        Required keys: model, model_int8, labels, test_wavs_dir, model_info
        """
    
    @abstractmethod
    def _build_sherpa_config(
        self,
        model_file: str,
        label_file: str,
        top_k: int,
    ) -> sherpa_onnx.AudioTaggingConfig:
        """Build the backend-specific AudioTaggingConfig."""
    
    def build(self) -> "BaseAudioTagger":
        """Validate paths, build config, instantiate the sherpa tagger. Returns self."""
        paths = self._get_model_paths()
        model_file = find_model_file(paths["model"], paths["model_int8"])
        label_file = paths["labels"]
        if not Path(label_file).is_file():
            raise FileNotFoundError(
                f"Labels file not found: {label_file}\n"
                "Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
            )
        
        config = self._build_sherpa_config(model_file, str(label_file), self.top_k)
        if not config.validate():
            raise ValueError(f"Invalid AudioTaggingConfig: {config}")
        
        self._print_config_table(paths["model_info"], model_file, str(label_file))
        self._tagger = sherpa_onnx.AudioTagging(config)
        return self
    
    def tag_file(
        self,
        audio_path: str,
        output_dir: Path,
    ) -> TaggingResult:
        """
        Full pipeline: load audio → chunk → infer → aggregate → save.
        Returns TaggingResult for programmatic use.
        """
        if self._tagger is None:
            raise RuntimeError("Call .build() before .tag_file()")
        
        samples, orig_sr = read_audio(audio_path)
        samples = resample_if_needed(samples, orig_sr, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
        audio_duration = len(samples) / sample_rate
        
        log.info(
            f"Audio loaded: [cyan]{len(samples):,}[/cyan] samples | "
            f"[cyan]{sample_rate} Hz[/cyan] | [cyan]{audio_duration:.2f}s[/cyan]"
        )
        
        start_time = time.time()
        chunk_events = process_audio_chunks(
            audio_tagger=self._tagger,
            samples=samples,
            sample_rate=sample_rate,
            expected_frames=self.EXPECTED_FRAMES,
        )
        aggregated = aggregate_chunk_results(chunk_events, self.top_k)
        elapsed = time.time() - start_time
        
        chunk_count = len({e["chunk_index"] for e in chunk_events})
        result = TaggingResult(
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration=audio_duration,
            elapsed_time=elapsed,
            events=aggregated,
            chunk_count=chunk_count,
            backend_name=self.BACKEND_NAME,
            model_variant=self.variant,
            top_k=self.top_k,
            is_speech_segment=False,
        )
        
        save_results(result, output_dir)
        return result
    
    def tag_speech_segment(
        self,
        segment_audio: np.ndarray,
        segment_start_utc: datetime,
        segment_end_utc: datetime,
        segment_id: Optional[int] = None,
    ) -> TaggingResult:
        """
        Tag a single speech segment from FireRed VAD pipeline.
        
        This method is designed to be called directly from the recording loop
        in speech_detector.py. It processes the segment with proper alignment
        to FireRed's frame boundaries and preserves absolute UTC timestamps.
        
        Args:
            segment_audio: Audio samples for the speech segment (mono, float32, 16kHz)
            segment_start_utc: UTC timestamp when speech started
            segment_end_utc: UTC timestamp when speech ended
            segment_id: Optional identifier for this segment
            
        Returns:
            TaggingResult with absolute UTC timestamps preserved
            
        Example usage in speech_detector.py:
            tagger = CEDAudioTagger(variant="base", top_k=5).build()
            for segment, audio in record_from_mic():
                result = tagger.tag_speech_segment(
                    segment_audio=audio,
                    segment_start_utc=segment["start_time_utc"],
                    segment_end_utc=segment["end_time_utc"],
                    segment_id=segment_index
                )
                # Process result.events with time_utc_start/time_utc_end
        """
        if self._tagger is None:
            raise RuntimeError("Call .build() before .tag_speech_segment()")
        
        # Ensure correct shape
        if segment_audio.ndim > 1:
            segment_audio = segment_audio.squeeze()
        segment_audio = segment_audio.astype(np.float32)
        
        segment_duration = len(segment_audio) / SAMPLE_RATE
        
        log.debug(
            f"Tagging speech segment {segment_id}: "
            f"{segment_duration:.2f}s audio, "
            f"UTC [{segment_start_utc.isoformat()} → {segment_end_utc.isoformat()}]"
        )
        
        start_time = time.time()
        
        # Process with segment-level UTC timestamps
        chunk_events = process_audio_chunks(
            audio_tagger=self._tagger,
            samples=segment_audio,
            sample_rate=SAMPLE_RATE,
            expected_frames=self.EXPECTED_FRAMES,
            segment_start_utc=segment_start_utc,
        )
        
        aggregated = aggregate_chunk_results(chunk_events, self.top_k)
        elapsed = time.time() - start_time
        
        chunk_count = len({e["chunk_index"] for e in chunk_events})
        
        result = TaggingResult(
            audio_path=f"speech_segment_{segment_id or 'unknown'}",
            sample_rate=SAMPLE_RATE,
            duration=segment_duration,
            elapsed_time=elapsed,
            events=aggregated,
            chunk_count=chunk_count,
            backend_name=self.BACKEND_NAME,
            model_variant=self.variant,
            top_k=self.top_k,
            is_speech_segment=True,
            segment_start_utc=segment_start_utc,
            segment_end_utc=segment_end_utc,
        )
        
        log.info(
            f"Segment {segment_id}: tagged in {elapsed:.3f}s "
            f"(RTF: {result.real_time_factor:.3f}) | "
            f"Top event: {aggregated[0].name if aggregated else 'none'}"
        )
        
        return result
    
    @property
    def default_test_wav(self) -> Path:
        """Return the path to the bundled test wav for this backend/variant."""
        return self._get_model_paths()["test_wavs_dir"] / "6.wav"
    
    def _print_config_table(
        self,
        model_info: dict,
        model_file: str,
        label_file: str
    ) -> None:
        """Display model configuration in a Rich table."""
        window_samples = self.EXPECTED_FRAMES * HOP_LENGTH
        window_sec = window_samples / SAMPLE_RATE
        
        tbl = Table(
            title=f"🎯 {self.BACKEND_NAME.upper()} audio tagger configuration "
                  f"[dim](Aligned with FireRed VAD)[/dim]"
        )
        tbl.add_column("Parameter", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_row("Backend", self.BACKEND_NAME)
        tbl.add_row("Variant", model_info.get("description", self.variant))
        tbl.add_row("Model size", model_info.get("size", "—"))
        tbl.add_row("Model path", model_file)
        tbl.add_row("Labels path", label_file)
        tbl.add_row("Top K", str(self.top_k))
        tbl.add_row("Provider", "cpu")
        tbl.add_row("Threads", "1")
        tbl.add_row("FireRed Alignment", "✓ Verified")
        tbl.add_row(
            "Window",
            f"{self.EXPECTED_FRAMES} frames × {HOP_LENGTH} samples = "
            f"{window_samples} samples ({window_sec:.1f}s)"
        )
        tbl.add_row(
            "Frame Shift",
            f"{FRAME_SHIFT_SAMPLE} samples (10ms)"
        )
        console.print(tbl)
