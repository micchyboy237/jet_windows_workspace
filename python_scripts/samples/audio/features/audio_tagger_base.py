"""
audio_tagger_base.py
====================
Abstract base class for sherpa-onnx audio tagging.
Dataclasses and constants have been extracted to audio_tagger_core.py.
Standalone utilities have been extracted to audio_tagger_utils.py.
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
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Generator, Tuple, Union
import numpy as np
import sherpa_onnx
from audio_tagger_core import (
    FRAME_SHIFT_SAMPLE,
    HOP_LENGTH,
    SAMPLE_RATE,
    TaggingEvent,
    TaggingResult,
    console,
    log,
)
from audio_tagger_utils import (
    _validate_firered_alignment,
    aggregate_chunk_results,
    find_model_file,
    process_audio_chunks,
    read_audio,
    resample_if_needed,
    save_results,
)
from audio_utils import (
    AudioInput,
    load_audio,
    split_audio,
)
from rich.table import Table


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
      tagger.tag_audio(audio_input, output_dir) → TaggingResult
      tagger.tag_audio_chunks(audio_input, chunk_duration_s, overlap_s) → TaggingResult
      tagger.tag_file(audio_path, output_dir) → TaggingResult  [deprecated, delegates to tag_audio]
      tagger.tag_speech_segment(audio, start_utc, end_utc) → TaggingResult
      tagger.default_test_wav → Path
    Auto-build: __init__ automatically calls build() so the tagger is ready immediately.
    """

    BACKEND_NAME: str = "base"
    DEFAULT_VARIANT: str = ""
    VALID_VARIANTS: tuple = ()
    EXPECTED_FRAMES: int = 100

    def __init__(self, variant: str = "", top_k: int = 5):
        if not variant:
            variant = self.DEFAULT_VARIANT
        if self.VALID_VARIANTS and variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Valid: {', '.join(self.VALID_VARIANTS)}"
            )
        self.variant = variant
        self.top_k = top_k
        self._tagger: Optional[sherpa_onnx.AudioTagging] = None
        _validate_firered_alignment(self.EXPECTED_FRAMES, HOP_LENGTH)
        # Auto-build so the tagger is ready immediately
        self.build()

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
        if self._tagger is not None:
            log.debug("Tagger already built; skipping rebuild.")
            return self
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

    def process_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
        expected_frames: Optional[int] = None,
        hop_length: Optional[int] = None,
        segment_start_utc: Optional[datetime] = None,
        show_progress: bool = True,
    ) -> List[TaggingEvent]:
        """
        Process audio samples through the tagger and return aggregated results.
        This is the core processing method that handles the full pipeline:
        chunk → infer → aggregate. It's backend-agnostic and works with any
        sherpa-onnx model (Zipformer, CED, etc.).
        Args:
            samples: Audio samples as numpy array (mono, float32)
            sample_rate: Sample rate in Hz (should be 16000 for FireRed alignment)
            expected_frames: Frames per window (default: self.EXPECTED_FRAMES)
            hop_length: Hop length in samples (default: HOP_LENGTH)
            segment_start_utc: Optional UTC timestamp for speech segments
            show_progress: Whether to show Rich progress bars
        Returns:
            List of aggregated TaggingEvent objects, sorted by probability
        Example:
            tagger = ZipformerAudioTagger(variant="standard", top_k=5)
            tagger.build()
            samples, sr = read_audio("speech.wav")
            events = tagger.process_audio(samples, sr)
            for event in events:
                print(f"{event.name}: {event.prob:.2%}")
        """
        if self._tagger is None:
            raise RuntimeError("Call .build() before .process_audio()")
        frames = (
            expected_frames if expected_frames is not None else self.EXPECTED_FRAMES
        )
        hop = hop_length if hop_length is not None else HOP_LENGTH
        log.debug(
            f"process_audio called with {len(samples)} samples, "
            f"backend={self.BACKEND_NAME}, variant={self.variant}"
        )
        chunk_events = process_audio_chunks(
            audio_tagger=self._tagger,
            samples=samples,
            sample_rate=sample_rate,
            expected_frames=frames,
            hop_length=hop,
            segment_start_utc=segment_start_utc,
        )
        aggregated = aggregate_chunk_results(chunk_events, self.top_k)
        log.info(
            f"process_audio complete: {len(aggregated)} events from "
            f"{len(chunk_events)} raw events across "
            f"{len({e['chunk_index'] for e in chunk_events})} chunks"
        )
        return aggregated

    def tag_audio(
        self,
        audio_input: AudioInput,
        output_dir: Optional[Path] = None,
    ) -> TaggingResult:
        """
        Full pipeline: load audio (from path, bytes, array, or tensor) →
        chunk → infer → aggregate → optionally save.
        
        Accepts AudioInput types for maximum flexibility:
        - File paths (str, Path)
        - In-memory WAV bytes
        - NumPy arrays (any shape/layout/dtype/sr)
        - Torch tensors
        
        Args:
            audio_input: Audio input to process
            output_dir: Optional directory for saving results. If None, prints results to console.
        
        Returns:
            TaggingResult with comprehensive analysis including:
            - Aggregated top-K events
            - Performance metrics (RTF, processing speed)
            - Speech label statistics (if speech detected)
            - Per-chunk raw events for detailed analysis
        
        Example:
            # Tag from file
            result = tagger.tag_audio("speech.wav", Path("output"))
            
            # Tag from bytes
            with open("audio.wav", "rb") as f:
                result = tagger.tag_audio(f.read())
            
            # Tag from numpy array
            audio_array, sr = librosa.load("audio.wav", sr=16000)
            result = tagger.tag_audio(audio_array)
            
            # Tag from torch tensor
            audio_tensor = torch.randn(16000)  # 1 second
            result = tagger.tag_audio(audio_tensor)
        """
        # Step 1: Load audio using AudioInput-compatible loader
        try:
            samples = load_audio(audio_input, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            raise ValueError(
                f"Failed to load audio from {type(audio_input).__name__}: {e}\n"
                f"Supported types: str, Path, bytes, np.ndarray, torch.Tensor"
            )
        
        sample_rate = SAMPLE_RATE
        audio_duration = len(samples) / sample_rate
        
        # Step 2: Extract display path for logging
        if isinstance(audio_input, (str, os.PathLike)):
            audio_path_str = str(audio_input)
            log.info(f"Processing audio file: [cyan]{Path(audio_path_str).name}[/cyan]")
        elif isinstance(audio_input, bytes):
            audio_path_str = "audio_bytes"
            log.info(f"Processing audio bytes: [cyan]{len(audio_input):,} bytes[/cyan]")
        elif isinstance(audio_input, np.ndarray):
            audio_path_str = f"numpy_array_{audio_input.shape}"
            log.info(f"Processing numpy array: [cyan]shape={audio_input.shape}[/cyan]")
        elif HAS_TORCH and isinstance(audio_input, torch.Tensor):
            audio_path_str = f"torch_tensor_{list(audio_input.shape)}"
            log.info(f"Processing torch tensor: [cyan]shape={list(audio_input.shape)}[/cyan]")
        else:
            audio_path_str = "unknown_audio_input"
        
        log.debug(
            f"Audio loaded: [cyan]{len(samples):,}[/cyan] samples | "
            f"[cyan]{audio_duration:.2f}s[/cyan] | "
            f"[cyan]{sample_rate} Hz[/cyan]"
        )
        
        # Step 3: Process audio chunks
        start_time = time.time()
        
        chunk_events = process_audio_chunks(
            audio_tagger=self._tagger,
            samples=samples,
            sample_rate=sample_rate,
            expected_frames=self.EXPECTED_FRAMES,
        )
        
        # Step 4: Aggregate results
        aggregated = aggregate_chunk_results(chunk_events, self.top_k)
        
        elapsed = time.time() - start_time
        
        # Step 5: Calculate chunk count
        window_samples = self.EXPECTED_FRAMES * HOP_LENGTH
        hop_samples = window_samples // 2
        total_samples = max(len(samples), window_samples)
        chunk_count = max(1, (total_samples - window_samples) // hop_samples + 1)
        
        # Step 6: Build result
        result = TaggingResult(
            audio_path=audio_path_str,
            sample_rate=sample_rate,
            duration=audio_duration,
            elapsed_time=elapsed,
            events=aggregated,
            chunk_count=chunk_count,
            backend_name=self.BACKEND_NAME,
            model_variant=self.variant,
            top_k=self.top_k,
            is_speech_segment=False,
            _chunk_events=chunk_events,
        )
        
        # Step 7: Display or save results
        if output_dir is not None:
            save_results(result, output_dir, chunk_events=chunk_events)
        else:
            from audio_tagger_utils import print_results_table, print_perf_table
            print_results_table(result.events, result.chunk_count, result.backend_name)
            print_perf_table(result)
        
        return result

    def tag_audio_chunks(
        self,
        audio_input: AudioInput,
        chunk_duration_s: float = 15.0,
        overlap_s: float = 3.0,
        output_dir: Optional[Path] = None,
    ) -> List[TaggingResult]:
        """
        Process long audio by splitting into overlapping chunks, tagging each
        independently, and returning a list of per-chunk TaggingResult objects.
        
        This is ideal for very long recordings (hours) that would exceed memory
        or model context windows. Each chunk is processed independently with
        its own set of events.
        
        Args:
            audio_input: Audio input (path, bytes, numpy array, or torch tensor)
            chunk_duration_s: Duration of each chunk in seconds (default: 15.0s)
            overlap_s: Overlap between chunks in seconds (default: 3.0s)
            output_dir: Optional directory for saving results. If provided,
                       saves per-chunk results in subdirectories (chunk_0000/, chunk_0001/, etc.)
        
        Returns:
            List of TaggingResult objects, one per chunk.
        
        Example:
            # Process a 2-hour recording in 30-second chunks with 5s overlap
            results = tagger.tag_audio_chunks("long_recording.wav", 
                                              chunk_duration_s=30.0, 
                                              overlap_s=5.0)
            for i, result in enumerate(results):
                print(f"Chunk {i}: {result.events[0].name if result.events else 'none'}")
        """
        start_time = time.time()
        
        # Load audio using the unified loader
        log.debug(f"Loading audio from {type(audio_input).__name__} for chunking...")
        samples = load_audio(audio_input, sr=SAMPLE_RATE, mono=True)
        total_duration = len(samples) / SAMPLE_RATE
        
        log.info(
            f"Audio loaded for chunking: [cyan]{len(samples):,}[/cyan] samples | "
            f"[cyan]{total_duration:.1f}s[/cyan] | "
            f"chunk_size={chunk_duration_s}s | overlap={overlap_s}s"
        )
        
        # Extract audio path for metadata
        if isinstance(audio_input, (str, os.PathLike)):
            base_path = Path(audio_input).stem
        else:
            base_path = "audio_chunks"
        
        results: List[TaggingResult] = []
        chunk_index = 0
        
        for chunk_samples, chunk_start_time_s in split_audio(
            samples, sr=SAMPLE_RATE, 
            chunk_duration_s=chunk_duration_s, 
            overlap_s=overlap_s
        ):
            chunk_duration = len(chunk_samples) / SAMPLE_RATE
            chunk_start = time.time()
            
            log.info(
                f"Processing chunk {chunk_index}: "
                f"[{chunk_start_time_s:.1f}s – {chunk_start_time_s + chunk_duration:.1f}s] "
                f"({len(chunk_samples)} samples)"
            )
            
            chunk_events_list = process_audio_chunks(
                audio_tagger=self._tagger,
                samples=chunk_samples,
                sample_rate=SAMPLE_RATE,
                expected_frames=self.EXPECTED_FRAMES,
            )
            aggregated = aggregate_chunk_results(chunk_events_list, self.top_k)
            chunk_elapsed = time.time() - chunk_start
            
            window_samples = self.EXPECTED_FRAMES * HOP_LENGTH
            hop_samples = window_samples // 2
            chunk_total_samples = max(len(chunk_samples), window_samples)
            sub_chunk_count = max(1, (chunk_total_samples - window_samples) // hop_samples + 1)
            
            result = TaggingResult(
                audio_path=f"{base_path}_chunk_{chunk_index:04d}",
                sample_rate=SAMPLE_RATE,
                duration=chunk_duration,
                elapsed_time=chunk_elapsed,
                events=aggregated,
                chunk_count=sub_chunk_count,
                backend_name=self.BACKEND_NAME,
                model_variant=self.variant,
                top_k=self.top_k,
                is_speech_segment=False,
            )
            results.append(result)
            
            if output_dir is not None:
                chunk_output_dir = output_dir / f"chunk_{chunk_index:04d}"
                chunk_output_dir.mkdir(parents=True, exist_ok=True)
                save_results(result, chunk_output_dir, chunk_events=chunk_events_list)
            
            chunk_index += 1
        
        total_elapsed = time.time() - start_time
        log.info(
            f"tag_audio_chunks complete: {len(results)} chunks in "
            f"{total_elapsed:.1f}s (RTF: {total_elapsed / total_duration:.3f})"
        )
        
        return results

    def tag_file(
        self,
        audio_path: str,
        output_dir: Path,
    ) -> TaggingResult:
        """
        [DEPRECATED] Full pipeline: load audio → chunk → infer → aggregate → save.
        
        This method is kept for backward compatibility but delegates to tag_audio().
        New code should use tag_audio() directly, which supports more input types.
        
        Saves all 5 output files:
          1. results.json        — Aggregated top-K predictions
          2. metadata.json       — Processing metadata
          3. chunk_results.json  — Per-chunk raw probabilities
          4. chunk_timeline.png  — Probability timeline plot
          5. results_bar.png     — Aggregated results bar chart
        
        Returns TaggingResult for programmatic use.
        """
        log.warning(
            "tag_file() is deprecated. Use tag_audio() instead, which supports "
            "AudioInput types (paths, bytes, arrays, tensors)."
        )
        return self.tag_audio(audio_input=audio_path, output_dir=output_dir)

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
        aggregated = self.process_audio(
            samples=segment_audio,
            sample_rate=SAMPLE_RATE,
            expected_frames=self.EXPECTED_FRAMES,
            segment_start_utc=segment_start_utc,
            show_progress=False,
        )
        elapsed = time.time() - start_time
        window_samples = self.EXPECTED_FRAMES * HOP_LENGTH
        hop_samples = window_samples // 2
        total_samples = max(len(segment_audio), window_samples)
        chunk_count = max(1, (total_samples - window_samples) // hop_samples + 1)
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
        self, model_info: dict, model_file: str, label_file: str
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
            f"{window_samples} samples ({window_sec:.1f}s)",
        )
        tbl.add_row("Frame Shift", f"{FRAME_SHIFT_SAMPLE} samples (10ms)")
        console.print(tbl)
