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

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import sherpa_onnx

# ── Shared types & constants from the zero-dependency core ─────────────────
from audio_tagger_core import (
    FRAME_SHIFT_SAMPLE,
    HOP_LENGTH,
    SAMPLE_RATE,
    TaggingEvent,
    TaggingResult,
    console,
    log,
)

# ── Extracted utility functions ────────────────────────────────────────────
# Safe: audio_tagger_utils only imports from audio_tagger_core, never from here.
from audio_tagger_utils import (
    _validate_firered_alignment,
    aggregate_chunk_results,
    find_model_file,
    process_audio_chunks,
    read_audio,
    resample_if_needed,
    save_results,
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
      tagger.build() → self
      tagger.process_audio(samples, sample_rate, **kwargs) → List[TaggingEvent]
      tagger.tag_file(audio_path, output_dir) → TaggingResult
      tagger.tag_speech_segment(audio, start_utc, end_utc) → TaggingResult
      tagger.default_test_wav → Path
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

    def tag_file(
        self,
        audio_path: str,
        output_dir: Path,
    ) -> TaggingResult:
        """
        Full pipeline: load audio → chunk → infer → aggregate → save.
        Saves all 5 output files:
          1. results.json        — Aggregated top-K predictions
          2. metadata.json       — Processing metadata
          3. chunk_results.json  — Per-chunk raw probabilities
          4. chunk_timeline.png  — Probability timeline plot
          5. results_bar.png     — Aggregated results bar chart
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

        # Get raw chunk events directly (before aggregation) so we can
        # pass them to save_results() for per-chunk JSON + plots
        chunk_events = process_audio_chunks(
            audio_tagger=self._tagger,
            samples=samples,
            sample_rate=sample_rate,
            expected_frames=self.EXPECTED_FRAMES,
        )

        # Aggregate for the result
        aggregated = aggregate_chunk_results(chunk_events, self.top_k)
        elapsed = time.time() - start_time

        window_samples = self.EXPECTED_FRAMES * HOP_LENGTH
        hop_samples = window_samples // 2
        total_samples = max(len(samples), window_samples)
        chunk_count = max(1, (total_samples - window_samples) // hop_samples + 1)

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

        # Pass chunk_events to enable all 5 output files
        save_results(result, output_dir, chunk_events=chunk_events)
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
