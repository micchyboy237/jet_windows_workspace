from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import sherpa_onnx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

try:
    from services.audio_utils import AudioInput, load_audio
    from services.audio_config import (
        FRAME_PER_SECONDS,
        HOP_STEP_MS,
        SAMPLE_RATE,
    )
    from services.custom_logging import linkify
    from services.audio_tagger_defaults import (
        BASE_DIR,
        AUDIO_TAGGING_MODEL,
        CLASS_LABELS_INDICES_CSV,
        DEFAULT_BASE_DIR,
        DEFAULT_MODEL_PATH,
        DEFAULT_LABELS_PATH,
        DEFAULT_TOP_K,
        DEFAULT_NUM_THREADS,
        DEFAULT_PROVIDER,
        DEFAULT_MIN_SPEECH_PROB_THRESHOLD,
        DEFAULT_SPEECH_PROB_THRESHOLD,
        DEFAULT_SPEECH_TOP_N,
        DEFAULT_CHUNK_DURATION,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_MIN_CHUNK_DURATION,
        DEFAULT_MIN_SILENCE_DURATION_SEC,
        DEFAULT_MIN_SPEECH_DURATION_SEC,
        DEFAULT_RESOLUTION_MS,
        DEFAULT_CONFIDENCE_VERY_SHORT_MAX,
        DEFAULT_CONFIDENCE_SHORT_MAX,
        DEFAULT_CONFIDENCE_NORMAL_MAX,
        DEFAULT_HIGH_CONFIDENCE_VERY_SHORT,
        DEFAULT_HIGH_CONFIDENCE_SHORT,
        DEFAULT_HIGH_CONFIDENCE_NORMAL,
        DEFAULT_HIGH_CONFIDENCE_LONG,
        DEFAULT_MEDIUM_CONFIDENCE_VERY_SHORT,
        DEFAULT_MEDIUM_CONFIDENCE_SHORT,
        DEFAULT_MEDIUM_CONFIDENCE_NORMAL,
        DEFAULT_MEDIUM_CONFIDENCE_LONG,
        SPEECH_CLASS_NAMES,
    )
    from services.audio_tagger_types import (
        TaggingResult,
        ChunkTaggingResult,
        AudioChunksTaggingSummary,
        AudioTaggerConfig,
        AudioTaggingSummary,
        SpeechSegmentTimeline,
        SpeechSegmentResult,
        AudioSegmentsResult,
    )
    from services.dtype_conversion import convert_audio_dtype
except ImportError:
    from audio_utils import AudioInput, load_audio
    from audio_config import (
        FRAME_PER_SECONDS,
        HOP_STEP_MS,
        SAMPLE_RATE,
    )
    from custom_logging import linkify
    from audio_tagger_defaults import (
        BASE_DIR,
        AUDIO_TAGGING_MODEL,
        CLASS_LABELS_INDICES_CSV,
        DEFAULT_BASE_DIR,
        DEFAULT_MODEL_PATH,
        DEFAULT_LABELS_PATH,
        DEFAULT_TOP_K,
        DEFAULT_NUM_THREADS,
        DEFAULT_PROVIDER,
        DEFAULT_MIN_SPEECH_PROB_THRESHOLD,
        DEFAULT_SPEECH_PROB_THRESHOLD,
        DEFAULT_SPEECH_TOP_N,
        DEFAULT_CHUNK_DURATION,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_MIN_CHUNK_DURATION,
        DEFAULT_MIN_SILENCE_DURATION_SEC,
        DEFAULT_MIN_SPEECH_DURATION_SEC,
        DEFAULT_RESOLUTION_MS,
        DEFAULT_CONFIDENCE_VERY_SHORT_MAX,
        DEFAULT_CONFIDENCE_SHORT_MAX,
        DEFAULT_CONFIDENCE_NORMAL_MAX,
        DEFAULT_HIGH_CONFIDENCE_VERY_SHORT,
        DEFAULT_HIGH_CONFIDENCE_SHORT,
        DEFAULT_HIGH_CONFIDENCE_NORMAL,
        DEFAULT_HIGH_CONFIDENCE_LONG,
        DEFAULT_MEDIUM_CONFIDENCE_VERY_SHORT,
        DEFAULT_MEDIUM_CONFIDENCE_SHORT,
        DEFAULT_MEDIUM_CONFIDENCE_NORMAL,
        DEFAULT_MEDIUM_CONFIDENCE_LONG,
        SPEECH_CLASS_NAMES,
    )
    from audio_tagger_types import (
        TaggingResult,
        ChunkTaggingResult,
        AudioChunksTaggingSummary,
        AudioTaggerConfig,
        AudioTaggingSummary,
        SpeechSegmentTimeline,
        SpeechSegmentResult,
        AudioSegmentsResult,
    )
    from dtype_conversion import convert_audio_dtype

install_rich_traceback(show_locals=True)

console = Console()



def calculate_confidence_tier(
    avg_prob: float,
    speech_density: float,
    duration: float,
    speech_chunk_ratio: float,
) -> tuple[str, str, bool, bool]:
    """
    Duration-aware tiered confidence calculation.
    Confidence Levels:
        ✨ High   - Strong, consistent speech with sufficient duration
        ⚠ Medium - Probable speech with some uncertainty
        — Low    - Weak or inconsistent signal
    Duration adjustments:
        < 0.5s  : Very strict (require near-perfect metrics)
        0.5-1.5s: Elevated (short segments need strong signal)
        1.5-5.0s: Standard thresholds
        > 5.0s  : Slightly relaxed (more reliable with length)
    Args:
        avg_prob: Average speech probability across segment
        speech_density: Percentage of timeline cells above threshold
        duration: Segment duration in seconds
        speech_chunk_ratio: Ratio of speech-detected chunks to total chunks
    Returns:
        Tuple of (confidence_tier, confidence_label, is_high, is_medium)
    """
    if duration < DEFAULT_CONFIDENCE_VERY_SHORT_MAX:
        high_prob_threshold = DEFAULT_HIGH_CONFIDENCE_VERY_SHORT["prob_threshold"]
        high_density_threshold = DEFAULT_HIGH_CONFIDENCE_VERY_SHORT["density_threshold"]
        high_chunk_ratio_threshold = DEFAULT_HIGH_CONFIDENCE_VERY_SHORT["chunk_ratio_threshold"]
        medium_prob_threshold = DEFAULT_MEDIUM_CONFIDENCE_VERY_SHORT["prob_threshold"]
        medium_density_threshold = DEFAULT_MEDIUM_CONFIDENCE_VERY_SHORT["density_threshold"]
        duration_category = "very_short"
    elif duration < DEFAULT_CONFIDENCE_SHORT_MAX:
        high_prob_threshold = DEFAULT_HIGH_CONFIDENCE_SHORT["prob_threshold"]
        high_density_threshold = DEFAULT_HIGH_CONFIDENCE_SHORT["density_threshold"]
        high_chunk_ratio_threshold = DEFAULT_HIGH_CONFIDENCE_SHORT["chunk_ratio_threshold"]
        medium_prob_threshold = DEFAULT_MEDIUM_CONFIDENCE_SHORT["prob_threshold"]
        medium_density_threshold = DEFAULT_MEDIUM_CONFIDENCE_SHORT["density_threshold"]
        duration_category = "short"
    elif duration <= DEFAULT_CONFIDENCE_NORMAL_MAX:
        high_prob_threshold = DEFAULT_HIGH_CONFIDENCE_NORMAL["prob_threshold"]
        high_density_threshold = DEFAULT_HIGH_CONFIDENCE_NORMAL["density_threshold"]
        high_chunk_ratio_threshold = DEFAULT_HIGH_CONFIDENCE_NORMAL["chunk_ratio_threshold"]
        medium_prob_threshold = DEFAULT_MEDIUM_CONFIDENCE_NORMAL["prob_threshold"]
        medium_density_threshold = DEFAULT_MEDIUM_CONFIDENCE_NORMAL["density_threshold"]
        duration_category = "normal"
    else:
        high_prob_threshold = DEFAULT_HIGH_CONFIDENCE_LONG["prob_threshold"]
        high_density_threshold = DEFAULT_HIGH_CONFIDENCE_LONG["density_threshold"]
        high_chunk_ratio_threshold = DEFAULT_HIGH_CONFIDENCE_LONG["chunk_ratio_threshold"]
        medium_prob_threshold = DEFAULT_MEDIUM_CONFIDENCE_LONG["prob_threshold"]
        medium_density_threshold = DEFAULT_MEDIUM_CONFIDENCE_LONG["density_threshold"]
        duration_category = "long"
    is_high_confidence = (
        avg_prob >= high_prob_threshold and
        speech_density >= high_density_threshold and
        speech_chunk_ratio >= high_chunk_ratio_threshold
    )
    is_medium_confidence = not is_high_confidence and (
        avg_prob >= medium_prob_threshold and
        speech_density >= medium_density_threshold
    )
    if is_high_confidence:
        confidence_tier = "high"
        confidence_label = "✨ High"
        console.print(
            f"[green]   🟢 High Confidence: "
            f"avg_prob={avg_prob:.3f}≥{high_prob_threshold}, "
            f"density={speech_density:.1%}≥{high_density_threshold:.0%}, "
            f"chunk_ratio={speech_chunk_ratio:.1%}≥{high_chunk_ratio_threshold:.0%} "
            f"({duration_category})[/green]"
        )
    elif is_medium_confidence:
        confidence_tier = "medium"
        confidence_label = "⚠ Medium"
        console.print(
            f"[yellow]   🟡 Medium Confidence: "
            f"avg_prob={avg_prob:.3f}≥{medium_prob_threshold}, "
            f"density={speech_density:.1%}≥{medium_density_threshold:.0%} "
            f"({duration_category})[/yellow]"
        )
    else:
        confidence_tier = "low"
        confidence_label = "— Low"
        console.print(
            f"[dim]   ⚪ Low Confidence: "
            f"avg_prob={avg_prob:.3f}<{medium_prob_threshold} or "
            f"density={speech_density:.1%}<{medium_density_threshold:.0%} "
            f"({duration_category})[/dim]"
        )
    return confidence_tier, confidence_label, is_high_confidence, is_medium_confidence


class AudioTagger:
    """
    Reusable audio tagging class using Sherpa-ONNX models.

    Features:
    - Tags audio with type AudioInput (reuses load_audio)
    - Checks if audio contains high-probability speech
    - Process long audio in overlapping chunks with tag_audio_chunks()
    - Calculates speech_duration: sum of consecutive speech chunks
    - Configurable chunking parameters from jet.audio.helpers.config

    Example:
        >>> tagger = AudioTagger()
        >>> results = tagger.tag_audio("path/to/audio.wav")
        >>> is_speech = tagger.contains_speech("path/to/audio.wav")
        >>> chunked = tagger.tag_audio_chunks("long_audio.wav", chunk_duration=5.0)
    """



    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = AUDIO_TAGGING_MODEL,
        labels_path: Optional[Union[str, Path]] = CLASS_LABELS_INDICES_CSV,
        top_k: int = DEFAULT_TOP_K,
        num_threads: int = DEFAULT_NUM_THREADS,
        provider: str = DEFAULT_PROVIDER,
        debug: bool = False,
        speech_prob_threshold: Optional[float] = None,
        speech_top_n: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        chunk_overlap: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
    ) -> None:
        """
        Initialize the AudioTagger with model configuration.
        Args:
            model_path: Path to ONNX model file
            labels_path: Path to class labels CSV
            top_k: Number of top predictions to return
            num_threads: Number of CPU threads
            provider: Computation provider ("cpu", "cuda", etc.)
            debug: Enable debug logging for Sherpa-ONNX
            speech_prob_threshold: Minimum speech probability (default: 0.5)
            speech_top_n: Check the top N predictions for speech classes (default: 3)
            chunk_duration: Default chunk duration in seconds (default: 1.0s)
            chunk_overlap: Default overlap between chunks in seconds (default: 0.5s)
            min_chunk_duration: Minimum valid chunk duration (default: 0.5s)
        """
        self.model_path: Path = (
            Path(model_path) if model_path else DEFAULT_MODEL_PATH
        )
        self.labels_path: Path = (
            Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        )
        self.top_k: int = top_k
        self.num_threads: int = num_threads
        self.provider: str = provider
        self.debug: bool = debug
        
        # Set speech detection parameters with validation
        self.speech_prob_threshold: float = (
            speech_prob_threshold
            if speech_prob_threshold is not None
            else DEFAULT_SPEECH_PROB_THRESHOLD
        )
        self.speech_top_n: int = (
            speech_top_n if speech_top_n is not None else DEFAULT_SPEECH_TOP_N
        )
        
        # Validate speech threshold - prevent overly low values that cause false positives
        if self.speech_prob_threshold < DEFAULT_MIN_SPEECH_PROB_THRESHOLD:
            console.print(
                f"[yellow]⚠ Speech probability threshold {self.speech_prob_threshold} "
                f"is below minimum valid value {DEFAULT_MIN_SPEECH_PROB_THRESHOLD}. "
                f"Using {DEFAULT_SPEECH_PROB_THRESHOLD} to prevent false positives.[/yellow]"
            )
            self.speech_prob_threshold = DEFAULT_SPEECH_PROB_THRESHOLD
        
        self.chunk_duration: float = (
            chunk_duration
            if chunk_duration is not None
            else DEFAULT_CHUNK_DURATION
        )
        self.chunk_overlap: float = (
            chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        )
        self.min_chunk_duration: float = (
            min_chunk_duration
            if min_chunk_duration is not None
            else DEFAULT_MIN_CHUNK_DURATION
        )
        
        self._validate_chunking_config()
        self._tagger: Optional[sherpa_onnx.AudioTagging] = None
        self._labels_map: Optional[Dict[int, str]] = None
        
        console.print(
            Panel.fit(
                f"[bold green]AudioTagger Initialized[/bold green]\n"
                f"Model: {linkify(str(self.model_path))}\n"
                f"Labels: {linkify(str(self.labels_path))}\n"
                f"Speech Threshold: {self.speech_prob_threshold}\n"
                f"Speech Top N: {self.speech_top_n}\n"
                f"Chunk Duration: {self.chunk_duration}s\n"
                f"Chunk Overlap: {self.chunk_overlap}s\n"
                f"Min Chunk Duration: {self.min_chunk_duration}s",
                title="AudioTagger Configuration",
                border_style="blue",
            )
        )

    def _validate_chunking_config(self) -> None:
        """Validate chunking parameters from config."""
        if self.chunk_duration < self.min_chunk_duration:
            self.chunk_duration = self.min_chunk_duration

        if self.chunk_overlap >= self.chunk_duration:
            self.chunk_overlap = self.chunk_duration / 2.0

        window_samples = int(self.chunk_duration * SAMPLE_RATE)
        hop_samples = int(self.chunk_overlap * SAMPLE_RATE)

    def _validate_model_files(self) -> None:
        """Validate that required model files exist."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
            )
        if not self.labels_path.is_file():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}\n"
                "Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
            )

    def _load_labels(self) -> Dict[int, str]:
        """Load class labels from CSV file."""
        labels: Dict[int, str] = {}
        with open(self.labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        index = int(row[0])
                        labels[index] = row[1].strip('"').strip()
                    except (ValueError, IndexError):
                        pass  # Silently ignore invalid rows
        return labels

    @property
    def tagger(self) -> sherpa_onnx.AudioTagging:
        """Lazy-load the Sherpa-ONNX AudioTagging instance."""
        if self._tagger is None:
            self._validate_model_files()
            config = sherpa_onnx.AudioTaggingConfig(
                model=sherpa_onnx.AudioTaggingModelConfig(
                    zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                        model=str(self.model_path),
                    ),
                    num_threads=self.num_threads,
                    debug=self.debug,
                    provider=self.provider,
                ),
                labels=str(self.labels_path),
                top_k=self.top_k,
            )
            if not config.validate():
                raise ValueError(f"Invalid AudioTaggingConfig: {config}")
            self._tagger = sherpa_onnx.AudioTagging(config)
            self._labels_map = self._load_labels()
        return self._tagger

    @property
    def labels_map(self) -> Dict[int, str]:
        """Get the labels mapping."""
        if self._labels_map is None:
            self._labels_map = self._load_labels()
        return self._labels_map

    # ── Speech detection helper for chunks ───────────────────────
    def _chunk_has_speech(
        self, predictions: List[TaggingResult], top_n: Optional[int] = None
    ) -> tuple[bool, float]:
        """
        Check if chunk predictions indicate speech.
        Args:
            predictions: List of tagging results for a chunk
            top_n: Number of top predictions to check (default: self.speech_top_n)
        Returns:
            Tuple of (speech_detected: bool, chunk_speech_prob: float)
        Debug logs trace:
            - Top-N predictions checked
            - Speech classes found
            - Final speech probability and detection result
            - Threshold comparison
        """
        n_to_check = top_n if top_n is not None else self.speech_top_n
        chunk_speech_prob = 0.0
        speech_classes_found = []
        
        # Log the threshold being used
        console.print(
            f"[dim]🔍 _chunk_has_speech: checking top {n_to_check} predictions "
            f"against threshold {self.speech_prob_threshold}[/dim]"
        )
        
        for result in predictions[:n_to_check]:
            name = result.get("name", "")
            prob = result.get("prob", 0.0)
            if name in SPEECH_CLASS_NAMES:
                speech_classes_found.append(f"{name}({prob:.3f})")
                if prob > chunk_speech_prob:
                    chunk_speech_prob = prob
        
        # Validate threshold is reasonable
        effective_threshold = max(self.speech_prob_threshold, DEFAULT_MIN_SPEECH_PROB_THRESHOLD)  # Minimum 0.1 to prevent false positives
        if effective_threshold != self.speech_prob_threshold:
            console.print(
                f"[yellow]⚠ Speech threshold {self.speech_prob_threshold} is too low, "
                f"using minimum {effective_threshold}[/yellow]"
            )
        
        speech_detected = chunk_speech_prob >= effective_threshold
        
        # Log the decision
        if speech_classes_found:
            console.print(
                f"[dim]   Speech classes found: {', '.join(speech_classes_found)} | "
                f"max_prob={chunk_speech_prob:.4f} | "
                f"threshold={effective_threshold} | "
                f"detected={speech_detected}[/dim]"
            )
        else:
            console.print(
                f"[dim]   No speech classes in top {n_to_check} | "
                f"detected={speech_detected}[/dim]"
            )
        
        return speech_detected, chunk_speech_prob

    # ── Calculate speech duration from consecutive speech chunks ──
    def _calculate_speech_duration(
        self,
        chunks: List[ChunkTaggingResult],
        overlap_duration: float,
    ) -> float:
        """
        Calculate total speech duration by merging consecutive speech chunks.

        Accounts for overlap between consecutive chunks to avoid double-counting.

        Args:
            chunks: List of chunk results with speech_detected flags
            overlap_duration: Overlap between consecutive chunks in seconds

        Returns:
            Total speech duration in seconds (sum of consecutive speech chunks)
        """
        if not chunks:
            return 0.0

        # Sort chunks by start time (should already be sorted, but ensure)
        sorted_chunks = sorted(chunks, key=lambda c: c["start_time"])

        speech_segments = []  # List of (start, end) tuples
        current_start = None
        current_end = None

        for chunk in sorted_chunks:
            if chunk.get("speech_detected", False):
                chunk_start = chunk["start_time"]
                chunk_end = chunk["end_time"]

                if current_start is None:
                    # Start new speech segment
                    current_start = chunk_start
                    current_end = chunk_end
                elif chunk_start <= current_end:
                    # Extend current segment (accounting for overlap)
                    current_end = max(current_end, chunk_end)
                else:
                    # Gap detected, save previous segment
                    speech_segments.append((current_start, current_end))
                    current_start = chunk_start
                    current_end = chunk_end
            else:
                if current_start is not None:
                    # End of speech segment
                    speech_segments.append((current_start, current_end))
                    current_start = None
                    current_end = None

        # Don't forget the last segment
        if current_start is not None:
            speech_segments.append((current_start, current_end))

        # Calculate total speech duration
        total_speech_duration = sum(end - start for start, end in speech_segments)

        return total_speech_duration

    def _calculate_chunk_positions(
        self,
        total_samples: int,
        chunk_samples: int,
        hop_samples: int,
        min_chunk_duration: float,
        sample_rate: int,
    ) -> List[tuple[int, int]]:
        """
        Calculate (start, end) sample indices for overlapping chunks.

        Chunks are evenly spaced with the given hop. The last chunk may be
        shorter than chunk_samples but must be at least min_chunk_duration.

        Args:
            total_samples: Total number of audio samples
            chunk_samples: Number of samples per full chunk
            hop_samples: Number of samples between chunk starts
            min_chunk_duration: Minimum duration for the last chunk in seconds
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_sample, end_sample) tuples

        Debug logs trace:
            - Input parameters
            - Number of chunks calculated
            - Start/end indices for each chunk
            - Whether last chunk meets minimum duration
        """

        positions: List[tuple[int, int]] = []

        if total_samples <= chunk_samples:
            # Audio fits in one chunk
            min_samples = int(min_chunk_duration * sample_rate)
            if total_samples >= min_samples:
                positions.append((0, total_samples))
            return positions

        # Calculate chunk starts
        start = 0
        while start + chunk_samples <= total_samples:
            end = start + chunk_samples
            positions.append((start, end))
            start += hop_samples

        # Handle remaining tail
        remaining_samples = total_samples - start
        min_samples = int(min_chunk_duration * sample_rate)

        if remaining_samples > 0:
            if remaining_samples >= min_samples:
                # Include the tail as a final chunk
                positions.append((start, total_samples))

        return positions

    def _tag_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> List[TaggingResult]:
        """
        Tag a waveform array and return top-K results.

        Args:
            waveform: Audio samples (mono, float32)
            sample_rate: Sample rate in Hz

        Returns:
            List of TaggingResult dicts

        Debug logs trace:
            - Waveform shape, dtype, value range
            - Stream creation
            - Inference completion
            - Result count
        """

        try:
            stream = self.tagger.create_stream()

            stream.accept_waveform(sample_rate=sample_rate, waveform=waveform)

            raw_results = self.tagger.compute(stream)
        except Exception:
            raise

        results: List[TaggingResult] = []
        for i, event in enumerate(raw_results):
            result: TaggingResult = {
                "index": i,
                "name": getattr(event, "name", "Unknown"),
                "class_index": getattr(event, "index", -1),
                "prob": getattr(event, "prob", 0.0),
            }
            results.append(result)

        return results

    def _aggregate_chunk_predictions(
        self,
        all_predictions: Dict[str, List[float]],
        top_k: int,
    ) -> List[TaggingResult]:
        """
        Aggregate per-chunk predictions into overall top-K results.

        For each unique label name, compute the mean probability across
        all chunks where it appeared. Sort by mean probability descending.

        Args:
            all_predictions: Dict mapping label name to list of probabilities
            top_k: Number of top results to return

        Returns:
            List of TaggingResult sorted by mean probability

        Debug logs trace:
            - Number of unique labels
            - Mean probability for each label
            - Final top-K selection
        """

        if not all_predictions:
            return []

        aggregated = []
        for name, probs in all_predictions.items():
            mean_prob = float(np.mean(probs))
            max_prob = float(np.max(probs))
            aggregated.append(
                {
                    "name": name,
                    "mean_prob": mean_prob,
                    "max_prob": max_prob,
                    "count": len(probs),
                }
            )

        # Sort by mean probability descending
        aggregated.sort(key=lambda x: x["mean_prob"], reverse=True)

        # Convert to TaggingResult format
        results = []
        for i, item in enumerate(aggregated[:top_k]):
            results.append(
                TaggingResult(
                    index=i,
                    name=item["name"],
                    class_index=-1,  # Not tracked in aggregation
                    prob=round(item["mean_prob"], 4),
                )
            )

        return results

    def _save_speech_chunk(
        self,
        chunk_waveform: np.ndarray,
        sample_rate: int,
        chunk_index: int,
        start_time: float,
        end_time: float,
        speech_probability: float,
        predictions: List[TaggingResult],
        base_dir: Path,
    ) -> Path:
        """
        Save a speech chunk's audio and metadata to disk.
        
        Creates directory structure:
            base_dir / "chunk_<chunk_index + 1>" / sound.wav
            base_dir / "chunk_<chunk_index + 1>" / meta.json
        
        Args:
            chunk_waveform: Audio samples for the chunk (mono, float32)
            sample_rate: Sample rate in Hz
            chunk_index: Zero-based index of the chunk
            start_time: Start time of chunk in seconds
            end_time: End time of chunk in seconds
            speech_probability: Detected speech probability for this chunk
            predictions: List of tagging predictions for this chunk
            base_dir: Base directory for speech chunks
        
        Returns:
            Path to the created chunk directory
        
        Debug logs trace:
            - Directory creation
            - WAV file saving
            - meta.json writing
            - File sizes
        """
        import soundfile as sf
        
        # Create chunk subdirectory with 1-based index in name
        chunk_dir = base_dir / f"chunk_{chunk_index + 1}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(
            f"[dim]💾 Saving speech chunk {chunk_index + 1} to: "
            f"{chunk_dir}[/dim]"
        )
        
        # Save audio as WAV
        wav_path = chunk_dir / "sound.wav"
        try:
            # Ensure waveform is in correct format for soundfile
            # soundfile expects float32 in range [-1, 1] or int16
            if chunk_waveform.dtype != np.float32:
                chunk_waveform = chunk_waveform.astype(np.float32)
            
            sf.write(
                str(wav_path),
                chunk_waveform,
                samplerate=sample_rate,
                subtype='PCM_16',  # Use 16-bit PCM for compatibility
            )
            
            wav_size = wav_path.stat().st_size
            console.print(
                f"[dim]   ✅ sound.wav saved ({wav_size:,} bytes)[/dim]"
            )
        except Exception as e:
            console.print(f"[red]   ❌ Failed to save WAV: {e}[/red]")
            raise
        
        # Build metadata
        duration = end_time - start_time
        meta = {
            "chunk_index": chunk_index,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(duration, 3),
            "sample_rate": sample_rate,
            "total_samples": len(chunk_waveform),
            "speech_probability": round(speech_probability, 4),
            "top_prediction": predictions[0]["name"] if predictions else "Unknown",
            "top_probability": round(predictions[0]["prob"], 4) if predictions else 0.0,
            "predictions": [
                {
                    "name": p["name"],
                    "class_index": p.get("class_index", -1),
                    "prob": round(p["prob"], 4),
                }
                for p in predictions[:10]  # Save top 10 predictions
            ],
        }
        
        # Save metadata as JSON
        meta_path = chunk_dir / "meta.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            meta_size = meta_path.stat().st_size
            console.print(
                f"[dim]   ✅ meta.json saved ({meta_size:,} bytes)[/dim]"
            )
        except Exception as e:
            console.print(f"[red]   ❌ Failed to save meta.json: {e}[/red]")
            raise
        
        console.print(
            f"[green]   📁 Speech chunk saved to: {linkify(chunk_dir)}[/green]"
        )
        
        return chunk_dir

    # ── Public methods ───────────────────────────────────

    def tag_audio(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
    ) -> List[TaggingResult]:
        """
        Tag audio with predicted labels and probabilities.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (ignored for file paths)

        Returns:
            List of TaggingResult dicts with keys: index, name, class_index, prob

        Example:
            >>> tagger = AudioTagger()
            >>> results = tagger.tag_audio("speech.wav")
            >>> for r in results:
            ...     print(f"{r['name']}: {r['prob']:.3f}")
        """
        start_time = time.time()

        try:
            waveform, actual_sr = load_audio(audio, sr=sample_rate or SAMPLE_RATE, mono=True)
        except Exception:
            raise

        try:
            stream = self.tagger.create_stream()
            stream.accept_waveform(sample_rate=actual_sr, waveform=waveform)
            raw_results = self.tagger.compute(stream)
        except Exception:
            raise

        results: List[TaggingResult] = []
        for i, event in enumerate(raw_results):
            result: TaggingResult = {
                "index": i,
                "name": getattr(event, "name", "Unknown"),
                "class_index": getattr(event, "index", -1),
                "prob": getattr(event, "prob", 0.0),
            }
            results.append(result)

        return results

    def tag_audio_chunks(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
    ) -> AudioChunksTaggingSummary:
        """
        Process long audio by splitting into overlapping chunks and tagging each.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: 16000)
            chunk_duration: Duration of each chunk in seconds.
            overlap_duration: Overlap between chunks in seconds.
            min_chunk_duration: Minimum duration for the last chunk.

        Returns:
            AudioChunksTaggingSummary with per-chunk results, overall aggregation,
            speech_duration, and avg_speech_probability
        """
        _chunk_dur = chunk_duration if chunk_duration is not None else self.chunk_duration
        _overlap = overlap_duration if overlap_duration is not None else self.chunk_overlap
        _min_chunk = (
            min_chunk_duration if min_chunk_duration is not None else self.min_chunk_duration
        )

        if _chunk_dur < _min_chunk:
            console.print(
                f"[yellow]⚠ Chunk duration {_chunk_dur}s < min {_min_chunk}s, "
                f"using min value[/yellow]"
            )
            _chunk_dur = _min_chunk

        if _overlap >= _chunk_dur:
            console.print(
                f"[yellow]⚠ Overlap {_overlap}s >= chunk duration {_chunk_dur}s, "
                f"using half chunk duration[/yellow]"
            )
            _overlap = _chunk_dur / 2.0

        overall_start = time.time()
        try:
            waveform, actual_sr = load_audio(
                audio, sr=sample_rate or SAMPLE_RATE, mono=True
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to load audio: {e}[/red]")
            raise

        total_samples = len(waveform)
        total_duration = total_samples / actual_sr
        console.print(
            f"[dim]📊 Audio loaded: {total_duration:.2f}s, "
            f"{actual_sr}Hz, {total_samples} samples[/dim]"
        )

        if isinstance(audio, (str, Path)):
            audio_path_str = str(audio)
        elif isinstance(audio, bytes):
            audio_path_str = f"bytes_input_{len(audio)}bytes"
        else:
            audio_path_str = f"array_input_{waveform.shape}"

        chunk_samples = int(_chunk_dur * actual_sr)
        hop_samples = int((_chunk_dur - _overlap) * actual_sr)
        if hop_samples < 1:
            hop_samples = 1

        console.print(
            f"[dim]🔧 Chunk config: {_chunk_dur}s chunks, "
            f"{_overlap}s overlap, hop={hop_samples} samples[/dim]"
        )

        chunk_positions = self._calculate_chunk_positions(
            total_samples=total_samples,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            min_chunk_duration=_min_chunk,
            sample_rate=actual_sr,
        )
        console.print(f"[dim]📏 Calculated {len(chunk_positions)} chunk positions[/dim]")

        if not chunk_positions:
            elapsed = time.time() - overall_start
            console.print("[yellow]⚠ No valid chunk positions found[/yellow]")
            return AudioChunksTaggingSummary(
                audio_path=audio_path_str,
                total_duration=total_duration,
                sample_rate=actual_sr,
                chunk_duration=_chunk_dur,
                overlap_duration=_overlap,
                total_chunks=0,
                chunks=[],
                overall_top_predictions=[],
                total_processing_time=elapsed,
                real_time_factor=elapsed / total_duration if total_duration > 0 else 0.0,
                speech_duration=0.0,
                speech_detected=False,
                max_speech_probability=0.0,
                avg_speech_probability=0.0,
            )

        chunks: List[ChunkTaggingResult] = []
        all_predictions: Dict[str, List[float]] = {}
        any_speech_detected = False
        global_max_speech_prob = 0.0
        speech_probabilities: List[float] = []

        for idx, (start_sample, end_sample) in enumerate(chunk_positions):
            chunk_start_time = time.time()
            start_sec = start_sample / actual_sr
            end_sec = end_sample / actual_sr

            console.print(
                f"[dim]🔍 Processing chunk {idx + 1}/{len(chunk_positions)}: "
                f"{start_sec:.2f}s - {end_sec:.2f}s[/dim]"
            )

            chunk_waveform = waveform[start_sample:end_sample].copy()

            try:
                chunk_predictions = self._tag_waveform(chunk_waveform, actual_sr)
                console.print(
                    f"[dim]   ✅ Tagged successfully: "
                    f"{len(chunk_predictions)} predictions[/dim]"
                )
            except Exception as e:
                console.print(f"[red]   ❌ Tagging failed: {e}[/red]")
                chunk_predictions = []

            speech_detected, chunk_speech_prob = self._chunk_has_speech(chunk_predictions)

            if speech_detected:
                any_speech_detected = True
                speech_probabilities.append(chunk_speech_prob)
                console.print(
                    f"[green]   🎤 Speech detected! "
                    f"speech_prob={chunk_speech_prob:.4f}[/green]"
                )
            else:
                console.print(
                    f"[dim]   🔇 No speech detected "
                    f"(speech_prob={chunk_speech_prob:.4f})[/dim]"
                )

            if chunk_speech_prob > global_max_speech_prob:
                global_max_speech_prob = chunk_speech_prob

            chunk_elapsed = time.time() - chunk_start_time

            for pred in chunk_predictions:
                name = pred["name"]
                if name not in all_predictions:
                    all_predictions[name] = []
                all_predictions[name].append(pred["prob"])

            chunk_result = ChunkTaggingResult(
                chunk_index=idx,
                start_time=round(start_sec, 3),
                end_time=round(end_sec, 3),
                duration=round(end_sec - start_sec, 3),
                predictions=chunk_predictions,
                processing_time=round(chunk_elapsed, 4),
                speech_detected=speech_detected,
                speech_probability=round(chunk_speech_prob, 4),
            )
            chunks.append(chunk_result)

        speech_duration = self._calculate_speech_duration(chunks, _overlap)

        if speech_probabilities:
            avg_speech_prob = float(np.mean(speech_probabilities))
            console.print(
                f"[dim]📊 Avg speech probability: {avg_speech_prob:.4f} "
                f"(from {len(speech_probabilities)} speech chunks)[/dim]"
            )
        else:
            avg_speech_prob = 0.0
            console.print("[dim]📊 No speech chunks for avg calculation[/dim]")

        overall_top = self._aggregate_chunk_predictions(all_predictions, self.top_k)
        total_elapsed = time.time() - overall_start
        rtf = total_elapsed / total_duration if total_duration > 0 else 0.0

        console.print(
            f"[dim]⏱ Total processing: {total_elapsed:.2f}s, "
            f"RTF: {rtf:.3f}x[/dim]"
        )

        summary = AudioChunksTaggingSummary(
            audio_path=audio_path_str,
            total_duration=round(total_duration, 3),
            sample_rate=actual_sr,
            chunk_duration=_chunk_dur,
            overlap_duration=_overlap,
            total_chunks=len(chunks),
            chunks=chunks,
            overall_top_predictions=overall_top,
            total_processing_time=round(total_elapsed, 4),
            real_time_factor=round(rtf, 4),
            speech_duration=round(speech_duration, 3),
            speech_detected=any_speech_detected,
            max_speech_probability=round(global_max_speech_prob, 4),
            avg_speech_probability=round(avg_speech_prob, 4),
        )
        return summary

    def tag_audio_segments(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        speech_threshold: Optional[float] = None,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_DURATION_SEC,
        resolution_ms: float = DEFAULT_RESOLUTION_MS,
        include_non_speech: bool = False,
    ) -> AudioSegmentsResult:
        """
        Tag audio by splitting into chunks, detecting speech, and identifying
        continuous speech/non-speech segments.
        This combines tag_audio_chunks() with timeline-based segment detection
        into a single call that returns structured segment data without writing
        to disk (use save_speech_segments() for persistence).
        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor).
            sample_rate: Sample rate for raw audio data (default: SAMPLE_RATE).
            chunk_duration: Duration of each analysis chunk in seconds.
            overlap_duration: Overlap between consecutive chunks.
            min_chunk_duration: Minimum duration for the last chunk.
            speech_threshold: Speech probability threshold (default: self.speech_prob_threshold).
            min_silence_duration_sec: Continuous non-speech gap to close a segment (default: 1.0s).
            min_speech_duration_sec: Minimum duration for a valid speech segment (default: 1.0s).
            resolution_ms: Timeline resolution in ms (default: HOP_STEP_MS).
            include_non_speech: If True, also detect non-speech segments (default: False).
        Returns:
            AudioSegmentsResult with chunks, speech_segments, non_speech_segments,
            and aggregate statistics.
        Example:
            >>> tagger = AudioTagger()
            >>> result = tagger.tag_audio_segments("recording.wav", min_silence_duration_sec=1.0)
            >>> for seg in result["speech_segments"]:
            ...     print(f"Speech: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
        Debug logs trace:
            - Chunk tagging progress (from tag_audio_chunks)
            - Timeline building statistics
            - Speech/non-speech transition detection
            - Segment count and duration summary
        """
        _speech_threshold = speech_threshold if speech_threshold is not None else self.speech_prob_threshold
        if _speech_threshold <= 0.0 or _speech_threshold > 1.0:
            console.print(
                f"[yellow]⚠ Invalid speech threshold {_speech_threshold}, using {DEFAULT_SPEECH_PROB_THRESHOLD}[/yellow]"
            )
            _speech_threshold = DEFAULT_SPEECH_PROB_THRESHOLD
        overall_start = time.time()
        console.print(
            Panel.fit(
                f"[bold cyan]tag_audio_segments[/bold cyan]\n"
                f"speech_threshold={_speech_threshold:.2f} | "
                f"min_silence={min_silence_duration_sec}s | "
                f"min_speech={min_speech_duration_sec}s | "
                f"resolution={resolution_ms}ms | "
                f"include_non_speech={include_non_speech}",
                title="Segment-Based Audio Tagging",
                border_style="cyan",
            )
        )
        # Step 1: Run chunk-level tagging
        chunk_summary = self.tag_audio_chunks(
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
        )
        chunks = chunk_summary.get("chunks", [])
        actual_sr = chunk_summary.get("sample_rate", SAMPLE_RATE)
        total_duration = chunk_summary.get("total_duration", 0.0)
        audio_path_str = chunk_summary.get("audio_path", "unknown")
        if not chunks:
            elapsed = time.time() - overall_start
            console.print("[yellow]⚠ No chunks produced, returning empty result[/yellow]")
            return AudioSegmentsResult(
                audio_path=audio_path_str,
                total_duration=total_duration,
                sample_rate=actual_sr,
                chunk_duration=chunk_summary.get("chunk_duration", self.chunk_duration),
                overlap_duration=chunk_summary.get("overlap_duration", self.chunk_overlap),
                total_chunks=0,
                speech_threshold=_speech_threshold,
                min_silence_duration_sec=min_silence_duration_sec,
                min_speech_duration_sec=min_speech_duration_sec,
                resolution_ms=resolution_ms,
                chunks=[],
                speech_segments=[],
                non_speech_segments=[],
                total_speech_duration=0.0,
                total_non_speech_duration=0.0,
                overall_top_predictions=[],
                total_processing_time=round(elapsed, 4),
                real_time_factor=round(elapsed / total_duration, 4) if total_duration > 0 else 0.0,
            )
        # Step 2: Build probability timeline
        times, probs = self._build_prob_timeline(chunks, resolution_ms=resolution_ms)
        if len(times) == 0:
            elapsed = time.time() - overall_start
            console.print("[yellow]⚠ Empty probability timeline[/yellow]")
            return AudioSegmentsResult(
                audio_path=audio_path_str,
                total_duration=total_duration,
                sample_rate=actual_sr,
                chunk_duration=chunk_summary.get("chunk_duration", self.chunk_duration),
                overlap_duration=chunk_summary.get("overlap_duration", self.chunk_overlap),
                total_chunks=len(chunks),
                speech_threshold=_speech_threshold,
                min_silence_duration_sec=min_silence_duration_sec,
                min_speech_duration_sec=min_speech_duration_sec,
                resolution_ms=resolution_ms,
                chunks=chunks,
                speech_segments=[],
                non_speech_segments=[],
                total_speech_duration=0.0,
                total_non_speech_duration=0.0,
                overall_top_predictions=chunk_summary.get("overall_top_predictions", []),
                total_processing_time=round(elapsed, 4),
                real_time_factor=round(elapsed / total_duration, 4) if total_duration > 0 else 0.0,
            )
        console.print(f"[dim]🎚 Using speech threshold: {_speech_threshold}[/dim]")
        # Step 3: Detect speech segments from timeline
        step = resolution_ms / 1000.0
        min_silence_cells = max(1, int(np.ceil(min_silence_duration_sec / step)))
        min_speech_cells = max(1, int(np.ceil(min_speech_duration_sec / step)))
        is_speech = probs >= _speech_threshold
        speech_cell_count = np.sum(is_speech)
        total_cells = len(is_speech)
        console.print(
            f"[dim]📊 Timeline: {speech_cell_count}/{total_cells} cells above threshold "
            f"({speech_cell_count/total_cells*100:.1f}%)[/dim]"
        )
        raw_segments: List[Tuple[float, float]] = []
        in_speech = False
        seg_start_idx = 0
        silence_run = 0
        speech_cells_in_current = 0
        for i, sp in enumerate(is_speech):
            if not in_speech:
                if sp:
                    in_speech = True
                    seg_start_idx = i
                    silence_run = 0
                    speech_cells_in_current = 1
                    console.print(
                        f"[dim]🎤 Speech start at cell {i} (time={times[i]:.3f}s)[/dim]"
                    )
            else:
                if sp:
                    silence_run = 0
                    speech_cells_in_current += 1
                else:
                    silence_run += 1
                    if silence_run >= min_silence_cells:
                        seg_end_idx = i - silence_run + 1
                        seg_start_time = times[seg_start_idx]
                        seg_end_time = times[seg_end_idx - 1]
                        raw_segments.append((seg_start_time, seg_end_time))
                        console.print(
                            f"[dim]🔇 Speech end at cell {i} (time={times[i]:.3f}s) | "
                            f"segment: {seg_start_time:.3f}s-{seg_end_time:.3f}s "
                            f"(silence={silence_run*step:.3f}s)[/dim]"
                        )
                        in_speech = False
                        silence_run = 0
                        speech_cells_in_current = 0
        if in_speech:
            seg_start_time = times[seg_start_idx]
            seg_end_time = times[-1]
            raw_segments.append((seg_start_time, seg_end_time))
            console.print(
                f"[dim]🎤 Trailing speech segment: {seg_start_time:.3f}s-{seg_end_time:.3f}s[/dim]"
            )
        # Step 4: Filter by minimum speech duration
        speech_segments: List[Tuple[float, float]] = []
        for s, e in raw_segments:
            duration = e - s
            if duration >= min_speech_duration_sec:
                speech_segments.append((s, e))
            else:
                console.print(
                    f"[dim]⏭ Discarding short segment: {s:.3f}s-{e:.3f}s "
                    f"(dur={duration:.3f}s < min_speech={min_speech_duration_sec}s)[/dim]"
                )
        console.print(f"[bold green]✅ {len(speech_segments)} speech segment(s) detected[/bold green]")
        # Step 5: Detect non-speech segments (if requested)
        non_speech_segments: List[Tuple[float, float]] = []
        if include_non_speech:
            all_segments_sorted = sorted(speech_segments, key=lambda x: x[0])
            prev_end = 0.0
            total_end = times[-1] if len(times) > 0 else max(c["end_time"] for c in chunks)
            for seg_start, seg_end in all_segments_sorted:
                if seg_start > prev_end:
                    gap_duration = seg_start - prev_end
                    if gap_duration >= min_silence_duration_sec:
                        non_speech_segments.append((prev_end, seg_start))
                prev_end = max(prev_end, seg_end)
            if prev_end < total_end:
                gap_duration = total_end - prev_end
                if gap_duration >= min_silence_duration_sec:
                    non_speech_segments.append((prev_end, total_end))
            console.print(
                f"[dim]🔇 {len(non_speech_segments)} non-speech segment(s) detected[/dim]"
            )
        # Step 6: Build structured segment results
        speech_segment_results: List[SpeechSegmentResult] = []
        for seg_num, (seg_start, seg_end) in enumerate(speech_segments):
            result = self._build_segment_result(
                seg_num=seg_num,
                seg_start=seg_start,
                seg_end=seg_end,
                is_speech=True,
                times=times,
                probs=probs,
                chunks=chunks,
                speech_threshold=_speech_threshold,
            )
            speech_segment_results.append(result)
        non_speech_segment_results: List[SpeechSegmentResult] = []
        if include_non_speech:
            for seg_num, (seg_start, seg_end) in enumerate(non_speech_segments):
                result = self._build_segment_result(
                    seg_num=seg_num,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    is_speech=False,
                    times=times,
                    probs=probs,
                    chunks=chunks,
                    speech_threshold=_speech_threshold,
                )
                non_speech_segment_results.append(result)
        total_speech_duration = sum(e - s for s, e in speech_segments)
        total_non_speech_duration = sum(e - s for s, e in non_speech_segments)
        total_elapsed = time.time() - overall_start
        rtf = total_elapsed / total_duration if total_duration > 0 else 0.0
        console.print(
            f"[dim]⏱ Segment detection complete: {total_elapsed:.2f}s, RTF: {rtf:.3f}x[/dim]"
        )
        final_result: AudioSegmentsResult = {
            "audio_path": audio_path_str,
            "total_duration": round(total_duration, 3),
            "sample_rate": actual_sr,
            "chunk_duration": chunk_summary.get("chunk_duration", self.chunk_duration),
            "overlap_duration": chunk_summary.get("overlap_duration", self.chunk_overlap),
            "total_chunks": len(chunks),
            "speech_threshold": _speech_threshold,
            "min_silence_duration_sec": min_silence_duration_sec,
            "min_speech_duration_sec": min_speech_duration_sec,
            "resolution_ms": resolution_ms,
            "chunks": chunks,
            "speech_segments": speech_segment_results,
            "non_speech_segments": non_speech_segment_results,
            "total_speech_duration": round(total_speech_duration, 3),
            "total_non_speech_duration": round(total_non_speech_duration, 3),
            "overall_top_predictions": chunk_summary.get("overall_top_predictions", []),
            "total_processing_time": round(total_elapsed, 4),
            "real_time_factor": round(rtf, 4),
        }
        return final_result

    def _build_prob_timeline(
        self,
        chunks: List[ChunkTaggingResult],
        resolution_ms: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a continuous speech-probability timeline from overlapping chunks.
        For each (time, speech_prob) observation contributed by a chunk, the
        probability is spread uniformly over the chunk's interval.  Where chunks
        overlap the per-cell values are accumulated via a weighted sum and then
        divided by the total weight (coverage count), giving a coverage-weighted
        average — the correct formula for non-uniform overlap grids.
        Formula for each timeline cell t covered by chunk i:
            prob_timeline[t] += chunk_speech_prob[i]   # accumulate
            weight[t]        += 1                       # count coverage
        Final:
            prob_timeline[t] /= weight[t]              # weighted mean
        Args:
            chunks: Chunk results with start_time, end_time, speech_probability.
            resolution_ms: Timeline resolution in milliseconds (default 10 ms).
        Returns:
            (times_sec, probs) arrays of equal length:
                times_sec — centre of each cell in seconds
                probs     — consolidated speech probability at that time
        Debug logs trace:
            - Number of cells in timeline
            - Total time span covered
        """
        if not chunks:
            return np.array([]), np.array([])
        total_end = max(c["end_time"] for c in chunks)
        step = resolution_ms / 1000.0
        n_cells = max(1, int(np.ceil(total_end / step)))
        prob_acc = np.zeros(n_cells, dtype=np.float64)
        weight   = np.zeros(n_cells, dtype=np.float64)
        for chunk in chunks:
            t0  = chunk["start_time"]
            t1  = chunk["end_time"]
            sp  = chunk.get("speech_probability", 0.0)
            i0 = int(t0 / step)
            i1 = min(int(np.ceil(t1 / step)), n_cells)
            prob_acc[i0:i1] += sp
            weight[i0:i1]   += 1.0
        covered = weight > 0
        probs = np.where(covered, prob_acc / np.where(covered, weight, 1.0), 0.0)
        times = (np.arange(n_cells) + 0.5) * step
        console.print(
            f"[dim]🕑 Built prob timeline: {n_cells} cells @ {resolution_ms}ms "
            f"resolution, total_end={total_end:.3f}s[/dim]"
        )
        return times.astype(np.float32), probs.astype(np.float32)

    def _build_segment_result(
        self,
        seg_num: int,
        seg_start: float,
        seg_end: float,
        is_speech: bool,
        times: np.ndarray,
        probs: np.ndarray,
        chunks: List[ChunkTaggingResult],
        speech_threshold: float,
    ) -> SpeechSegmentResult:
        """
        Build a SpeechSegmentResult with duration-aware tiered confidence.
        
        Confidence considers:
            1. Average speech probability (quality of speech signal)
            2. Speech density (consistency of speech throughout segment)
            3. Segment duration (reliability increases with duration)
            4. Speech chunk ratio (agreement among overlapping chunks)
        
        Duration adjustments:
            < 0.5s  : Very strict thresholds (likely noise if weak)
            0.5-1.5s: Elevated thresholds (short segments need strong signal)
            1.5-5.0s: Standard thresholds (optimal range)
            > 5.0s  : Relaxed thresholds (long segments are more reliable)
        
        Args:
            seg_num: 0-based segment index.
            seg_start: Segment start time in seconds.
            seg_end: Segment end time in seconds.
            is_speech: True for speech segments, False for non-speech.
            times: Full probability timeline time array.
            probs: Full probability timeline probability array.
            chunks: All chunk results from tag_audio_chunks.
            speech_threshold: Threshold used for speech detection.
        
        Returns:
            SpeechSegmentResult with all computed statistics including confidence tier.
        """
        seg_duration = seg_end - seg_start
        segment_type = "speech" if is_speech else "non-speech"
        
        console.print(
            f"[dim]🔍 _build_segment_result: {segment_type} segment {seg_num + 1} "
            f"{seg_start:.3f}s–{seg_end:.3f}s (dur={seg_duration:.3f}s)[/dim]"
        )
        
        # Extract segment-specific timeline data
        mask = (times >= seg_start) & (times <= seg_end)
        seg_times = times[mask]
        seg_probs = probs[mask]
        
        # Calculate probability statistics
        avg_prob = float(np.mean(seg_probs)) if len(seg_probs) else 0.0
        max_prob = float(np.max(seg_probs)) if len(seg_probs) else 0.0
        min_prob = float(np.min(seg_probs)) if len(seg_probs) else 0.0
        
        # Calculate speech density (percentage of timeline cells above threshold)
        speech_density = float(np.mean(seg_probs >= speech_threshold)) if len(seg_probs) else 0.0
        
        # Find overlapping chunks
        seg_chunks = [
            c for c in chunks
            if c["start_time"] < seg_end and c["end_time"] > seg_start
        ]
        
        console.print(
            f"[dim]   Timeline cells: {len(seg_times)} | "
            f"Overlapping chunks: {len(seg_chunks)} | "
            f"Duration: {seg_duration:.3f}s[/dim]"
        )
        
        # Aggregate predictions from overlapping chunks
        pred_acc: Dict[str, List[float]] = {}
        for c in seg_chunks:
            for p in c.get("predictions", []):
                pred_acc.setdefault(p["name"], []).append(p["prob"])
        
        top_preds = sorted(
            [
                TaggingResult(
                    index=-1,
                    name=name,
                    class_index=-1,
                    prob=round(float(np.mean(ps)), 4),
                )
                for name, ps in pred_acc.items()
            ],
            key=lambda x: x["prob"],
            reverse=True,
        )[:10]
        
        # Count speech chunks
        speech_chunk_count = sum(
            1 for c in seg_chunks
            if c.get("speech_probability", 0.0) >= speech_threshold
        )
        
        # Calculate speech chunk ratio
        speech_chunk_ratio = (
            round(speech_chunk_count / len(seg_chunks), 4)
            if seg_chunks else 0.0
        )
        
        # DURATION-AWARE TIERED CONFIDENCE SYSTEM
        confidence_tier, confidence_label, is_high_confidence, is_medium_confidence = \
            calculate_confidence_tier(
                avg_prob=avg_prob,
                speech_density=speech_density,
                duration=seg_duration,
                speech_chunk_ratio=speech_chunk_ratio,
            )
        
        # Get duration-specific notes
        duration_note = self._get_duration_confidence_note(
            duration=seg_duration,
            confidence_tier=confidence_tier,
        )
        
        if duration_note:
            console.print(f"[dim]   📝 Duration note: {duration_note}[/dim]")
        
        result: SpeechSegmentResult = {
            "segment_index": seg_num,
            "segment_type": segment_type,
            "start_time": round(float(seg_start), 3),
            "end_time": round(float(seg_end), 3),
            "duration": round(float(seg_duration), 3),
            "avg_speech_probability": round(avg_prob, 4),
            "max_speech_probability": round(max_prob, 4),
            "min_speech_probability": round(min_prob, 4),
            "speech_density": round(speech_density, 4),
            "speech_chunk_count": speech_chunk_count,
            "total_chunk_count": len(seg_chunks),
            "speech_chunk_ratio": speech_chunk_ratio,
            "threshold_used": speech_threshold,
            "is_high_confidence": is_high_confidence,
            "is_medium_confidence": is_medium_confidence,
            "confidence_tier": confidence_tier,
            "confidence_label": confidence_label,
            "confidence_duration_note": duration_note,  # NEW: explains duration's impact
            "is_dense_speech": speech_density >= 0.8,
            "top_prediction": top_preds[0]["name"] if top_preds else "Unknown",
            "top_prediction_prob": top_preds[0]["prob"] if top_preds else 0.0,
            "top_predictions": top_preds[:5],
            "overlapping_chunks": seg_chunks,
            "timeline": SpeechSegmentTimeline(
                times=[round(float(t), 4) for t in seg_times],
                probs=[round(float(p), 4) for p in seg_probs],
            ),
        }
        
        return result

    def _get_duration_confidence_note(
        self,
        duration: float,
        confidence_tier: str,
    ) -> str:
        """
        Generate explanatory notes about duration's impact on confidence.
        
        Args:
            duration: Segment duration in seconds
            confidence_tier: Current confidence tier
        
        Returns:
            Human-readable note about duration impact
        """
        notes = []
        
        if duration < 0.3 and confidence_tier in ("high", "medium"):
            notes.append("⚠ Very short - verify manually")
        elif duration < 0.5 and confidence_tier == "medium":
            notes.append("Short duration may be noise")
        elif duration < 1.0 and confidence_tier == "low":
            notes.append("Too short for reliable classification")
        
        if duration > 10.0:
            notes.append("Long segment - check for mixed content")
        
        if duration > 30.0 and confidence_tier == "high":
            notes.append("Unusually long - may contain multiple speakers")
        
        return " | ".join(notes) if notes else ""

    def contains_speech(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = SAMPLE_RATE,
        prob_threshold: Optional[float] = DEFAULT_SPEECH_PROB_THRESHOLD,
        top_n: Optional[int] = DEFAULT_SPEECH_TOP_N,
    ) -> bool:
        """
        Check if audio contains speech with high probability.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data
            prob_threshold: Override default speech probability threshold
            top_n: Override default number of top predictions to check

        Returns:
            True if speech is detected with probability >= threshold

        Example:
            >>> tagger = AudioTagger()
            >>> if tagger.contains_speech("meeting.wav"):
            ...     print("Speech detected!")
        """
        threshold = (
            prob_threshold if prob_threshold is not None else self.speech_prob_threshold
        )
        n_to_check = top_n if top_n is not None else self.speech_top_n

        try:
            results = self.tag_audio(audio, sample_rate=sample_rate)
        except Exception:
            return False

        if not results:
            return False

        for result in results[:n_to_check]:
            name = result.get("name", "")
            prob = result.get("prob", 0.0)
            if name in SPEECH_CLASS_NAMES and prob >= threshold:
                return True

        top_result = results[0]
        top_name = top_result.get("name", "")
        top_prob = top_result.get("prob", 0.0)
        if top_name == "Speech" and top_prob >= threshold:
            return True

        return False

    def get_speech_probability(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = SAMPLE_RATE,
    ) -> float:
        """
        Get the maximum speech probability from tagging results.

        Args:
            audio: Audio input
            sample_rate: Sample rate for raw audio data

        Returns:
            Maximum probability for any speech-related class (0.0 to 1.0)
        """
        try:
            results = self.tag_audio(audio, sample_rate=sample_rate)
        except Exception:
            return 0.0

        max_speech_prob = 0.0
        for result in results:
            if result.get("name", "") in SPEECH_CLASS_NAMES:
                prob = result.get("prob", 0.0)
                if prob > max_speech_prob:
                    max_speech_prob = prob
        return max_speech_prob

    def get_tagging_summary(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = SAMPLE_RATE,
        audio_path: str = "unknown",
    ) -> AudioTaggingSummary:
        """
        Get a comprehensive summary of audio tagging results.
        
        Args:
            audio: Audio input
            sample_rate: Sample rate for raw audio data
            audio_path: Identifier for the audio source
        
        Returns:
            AudioTaggingSummary with complete analysis including speech_duration
        
        Debug logs trace:
            - Audio loading time
            - Tagging duration
            - Speech probability calculation
            - Final metrics
        """
        start_time = time.time()
        
        console.print(f"[dim]📊 Loading audio for summary: {audio_path}[/dim]")
        try:
            waveform, actual_sr = load_audio(audio, sr=sample_rate or SAMPLE_RATE, mono=True)
            audio_duration = len(waveform) / actual_sr if actual_sr > 0 else 0
            console.print(
                f"[dim]   Loaded: {audio_duration:.2f}s, {actual_sr}Hz[/dim]"
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to load audio: {e}[/red]")
            raise
        
        console.print("[dim]🏷 Tagging audio...[/dim]")
        results = self.tag_audio(audio, sample_rate=sample_rate)
        console.print(f"[dim]   Got {len(results)} predictions[/dim]")
        
        console.print("[dim]🔍 Checking speech probability...[/dim]")
        max_speech_prob = self.get_speech_probability(audio, sample_rate=sample_rate)
        speech_detected = max_speech_prob >= self.speech_prob_threshold
        speech_duration = audio_duration if speech_detected else 0.0
        
        console.print(
            f"[dim]   Speech prob: {max_speech_prob:.4f}, "
            f"detected: {speech_detected}[/dim]"
        )
        
        elapsed = time.time() - start_time
        rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
        
        console.print(
            f"[dim]⏱ Summary processing: {elapsed:.3f}s, RTF: {rtf:.3f}x[/dim]"
        )
        
        summary: AudioTaggingSummary = {
            "audio_path": audio_path,
            "duration_seconds": audio_duration,
            "sample_rate": actual_sr,
            "num_results": len(results),
            "top_predictions": results[: self.top_k],
            "speech_detected": speech_detected,
            "max_speech_probability": max_speech_prob,
            "processing_time_seconds": elapsed,
            "real_time_factor": rtf,
            "speech_duration": speech_duration,
        }
        
        return summary

    def extract_speech_only(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = SAMPLE_RATE,
        edges_only: bool = False,
        prob_threshold: Optional[float] = DEFAULT_SPEECH_PROB_THRESHOLD,
        chunk_duration: Optional[float] = DEFAULT_CHUNK_DURATION,
        overlap_duration: Optional[float] = DEFAULT_CHUNK_OVERLAP,
        min_chunk_duration: Optional[float] = DEFAULT_MIN_CHUNK_DURATION,
        top_n: Optional[int] = DEFAULT_SPEECH_TOP_N,
    ) -> np.ndarray:
        """
        Extract speech-only audio by removing non-speech segments.
        
        Uses tag_audio_chunks internally to detect speech regions,
        then trims the audio to keep only speech portions.
        
        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: SAMPLE_RATE)
            edges_only: If True, only trim leading/trailing non-speech;
                    if False (default), remove all non-speech segments
            prob_threshold: Override default speech probability threshold
            chunk_duration: Override default chunk duration in seconds
            overlap_duration: Override default overlap between chunks
            min_chunk_duration: Override minimum chunk duration
            top_n: Override number of top predictions to check for speech
        
        Returns:
            Trimmed numpy audio array containing only speech portions
        
        Example:
            >>> tagger = AudioTagger()
            >>> speech_audio = tagger.extract_speech_only("recording.wav")
            >>> trimmed = tagger.extract_speech_only("recording.wav", edges_only=True)
        
        Debug logs trace:
            - Input parameters
            - Chunk tagging results
            - Identified speech segments
            - Trimmed audio duration vs original
        """
        console.print(
            Panel.fit(
                f"[bold cyan]extract_speech_only[/bold cyan]\n"
                f"edges_only={edges_only}\n"
                f"prob_threshold={prob_threshold or self.speech_prob_threshold}",
                title="Speech Extraction",
                border_style="cyan",
            )
        )
        
        # Convert dtype
        audio_int16 = convert_audio_dtype(audio, "int16")
        audio = audio_int16

        waveform = audio
        
        total_samples = len(waveform)
        total_duration = total_samples / sample_rate
        console.print(
            f"[dim]📊 Audio loaded: {total_duration:.2f}s, "
            f"{sample_rate}Hz, {total_samples} samples[/dim]"
        )
        
        # Tag chunks to identify speech regions
        summary = self.tag_audio_chunks(
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
        )
        
        chunks = summary.get("chunks", [])
        if not chunks:
            console.print("[yellow]⚠ No chunks produced, returning empty array[/yellow]")
            return np.array([], dtype=np.float32)
        
        # Override speech detection threshold if provided
        threshold = prob_threshold if prob_threshold is not None else self.speech_prob_threshold
        n_check = top_n if top_n is not None else self.speech_top_n
        
        # Re-evaluate speech detection with potentially overridden threshold
        # (tag_audio_chunks uses instance defaults; we may need stricter/looser)
        if prob_threshold is not None or top_n is not None:
            console.print(
                f"[dim]🔧 Re-evaluating speech detection with "
                f"threshold={threshold}, top_n={n_check}[/dim]"
            )
            for chunk in chunks:
                predictions = chunk.get("predictions", [])
                speech_detected, chunk_prob = self._chunk_has_speech(
                    predictions, top_n=n_check
                )
                # Override with custom threshold
                speech_detected = chunk_prob >= threshold
                chunk["speech_detected"] = speech_detected
                chunk["speech_probability"] = round(chunk_prob, 4)
        
        # Identify speech segments from chunks
        speech_segments = self._identify_speech_segments(
            chunks=chunks,
            total_duration=total_duration,
            edges_only=edges_only,
        )
        
        if not speech_segments:
            console.print("[yellow]⚠ No speech segments found, returning empty array[/yellow]")
            return np.array([], dtype=np.float32)
        
        console.print(f"[green]🎤 Found {len(speech_segments)} speech segment(s):[/green]")
        for i, (start, end) in enumerate(speech_segments):
            console.print(f"[green]   Segment {i + 1}: {start:.3f}s - {end:.3f}s "
                        f"(duration: {end - start:.3f}s)[/green]")
        
        # Extract speech portions from waveform
        trimmed_waveforms = []
        for start_sec, end_sec in speech_segments:
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            if end_sample > start_sample:
                trimmed_waveforms.append(waveform[start_sample:end_sample].copy())
        
        if not trimmed_waveforms:
            console.print("[yellow]⚠ No valid speech samples extracted[/yellow]")
            return np.array([], dtype=np.float32)
        
        result = np.concatenate(trimmed_waveforms)
        result_duration = len(result) / sample_rate
        reduction_pct = (1 - len(result) / total_samples) * 100 if total_samples > 0 else 0
        
        console.print(
            f"[bold green]✅ Speech extracted: {result_duration:.2f}s "
            f"(removed {reduction_pct:.1f}% of audio)[/bold green]"
        )
        
        return result.astype(np.float32)

    def _identify_speech_segments(
        self,
        chunks: List[ChunkTaggingResult],
        total_duration: float,
        edges_only: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Identify speech segments from chunk tagging results.
        
        Args:
            chunks: List of chunk results with speech_detected flags
            total_duration: Total audio duration in seconds
            edges_only: If True, return a single segment covering from
                    first speech to last speech (trim edges only);
                    if False, return all speech segments with gaps removed
        
        Returns:
            List of (start_time, end_time) tuples in seconds
        
        Debug logs trace:
            - Number of chunks processed
            - Edges_only mode
            - Identified segment boundaries
        """
        if not chunks:
            return []
        
        sorted_chunks = sorted(chunks, key=lambda c: c["start_time"])
        
        if edges_only:
            # Find first and last speech chunks
            first_speech_start = None
            last_speech_end = None
            
            for chunk in sorted_chunks:
                if chunk.get("speech_detected", False):
                    if first_speech_start is None:
                        first_speech_start = chunk["start_time"]
                    last_speech_end = chunk["end_time"]
            
            if first_speech_start is not None and last_speech_end is not None:
                console.print(
                    f"[dim]🔍 Edges-only: first speech at {first_speech_start:.3f}s, "
                    f"last speech at {last_speech_end:.3f}s[/dim]"
                )
                return [(first_speech_start, last_speech_end)]
            else:
                return []
        
        # Full mode: merge consecutive speech chunks, remove all gaps
        speech_segments = []
        current_start = None
        current_end = None
        
        for chunk in sorted_chunks:
            if chunk.get("speech_detected", False):
                chunk_start = chunk["start_time"]
                chunk_end = chunk["end_time"]
                
                if current_start is None:
                    current_start = chunk_start
                    current_end = chunk_end
                elif chunk_start <= current_end:
                    # Overlapping or adjacent - merge
                    current_end = max(current_end, chunk_end)
                else:
                    # Gap detected - save previous segment
                    speech_segments.append((current_start, current_end))
                    current_start = chunk_start
                    current_end = chunk_end
        
        # Don't forget the last segment
        if current_start is not None:
            speech_segments.append((current_start, current_end))
        
        # Merge very close segments (within 1 chunk duration)
        merged = self._merge_close_segments(speech_segments)
        
        return merged

    def _merge_close_segments(
        self,
        segments: List[Tuple[float, float]],
        max_gap: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        Merge speech segments that are very close together.
        
        Small gaps (e.g., brief pauses) between speech segments
        are likely still part of the same speech event.
        
        Args:
            segments: List of (start, end) time tuples
            max_gap: Maximum gap in seconds to merge (default: chunk_duration / 2)
        
        Returns:
            Merged list of (start, end) tuples
        """
        if not segments:
            return []
        
        gap_threshold = max_gap if max_gap is not None else self.chunk_duration / 2.0
        
        merged = [list(segments[0])]
        
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            gap = start - prev_end
            
            if gap <= gap_threshold:
                # Merge: extend previous segment
                merged[-1][1] = end
                console.print(
                    f"[dim]🔗 Merged close segments: gap={gap:.3f}s "
                    f"(threshold={gap_threshold:.3f}s)[/dim]"
                )
            else:
                merged.append([start, end])
        
        return [(s, e) for s, e in merged]

    def extract_high_confidence_speech_segments(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        min_duration: float = 1.5,
        require_confidence: Optional[List[str]] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        speech_threshold: Optional[float] = None,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_DURATION_SEC,
    ) -> Tuple[List[SpeechSegmentResult], List[np.ndarray]]:
        """
        Extract high speech segments and their audio from the input.
        A segment qualifies if duration > min_duration and segment_type == "speech".
        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: SAMPLE_RATE)
            min_duration: Minimum segment duration in seconds to include (default: 2.0)
            require_confidence: (deprecated) No longer used. Kept for backward compatibility.
            chunk_duration: Duration of each analysis chunk in seconds
            overlap_duration: Overlap between consecutive chunks
            min_chunk_duration: Minimum duration for the last chunk
            speech_threshold: Speech probability threshold
            min_silence_duration_sec: Continuous non-speech gap to close a segment
            min_speech_duration_sec: Minimum duration for a valid speech segment
        Returns:
            Tuple of:
                - List[SpeechSegmentResult]: Filtered speech segments
                - List[np.ndarray]: Corresponding audio arrays for each segment
        Example:
            >>> tagger = AudioTagger()
            >>> segments, audios = tagger.extract_high_confidence_speech_segments(
            ...     "recording.wav", min_duration=2.0
            ... )
            >>> for seg, aud in zip(segments, audios):
            ...     print(f"{seg['start_time']:.1f}s-{seg['end_time']:.1f}s: {len(aud)} samples")
        """
        import soundfile as sf
        overall_start = time.time()
        console.print(
            Panel.fit(
                f"[bold cyan]extract_high_confidence_speech_segments[/bold cyan]\n"
                f"min_duration={min_duration}s | "
                f"filter: duration > {min_duration}s AND segment_type == 'speech'",
                title="High Speech Segments Extraction",
                border_style="cyan",
            )
        )
        console.print("[dim]🔍 Running tag_audio_segments...[/dim]")
        segments_result = self.tag_audio_segments(
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
            speech_threshold=speech_threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            include_non_speech=False,
        )
        speech_segments = segments_result.get("speech_segments", [])
        console.print(f"[dim]📊 Found {len(speech_segments)} total speech segments[/dim]")
        high_speech_segments: List[SpeechSegmentResult] = []
        for segment in speech_segments:
            duration = segment.get("duration", 0.0)
            segment_type = segment.get("segment_type", "")
            is_high_speech = duration > min_duration and segment_type == "speech"
            if is_high_speech:
                high_speech_segments.append(segment)
                console.print(
                    f"[green]   ✅ Segment {segment['segment_index']}: "
                    f"{segment['start_time']:.2f}s-{segment['end_time']:.2f}s "
                    f"(dur={duration:.2f}s, type={segment_type})[/green]"
                )
            else:
                reasons = []
                if duration <= min_duration:
                    reasons.append(f"duration {duration:.2f}s <= {min_duration}s")
                if segment_type != "speech":
                    reasons.append(f"segment_type is '{segment_type}'")
                console.print(
                    f"[dim]   ⏭ Skipped segment {segment['segment_index']}: "
                    f"{', '.join(reasons) if reasons else 'unknown reason'}[/dim]"
                )
        console.print(
            f"[bold green]✅ Filtered {len(high_speech_segments)} "
            f"high speech segments (duration > {min_duration}s, type=speech)[/bold green]"
        )
        high_speech_audios: List[np.ndarray] = []
        if high_speech_segments:
            try:
                audio_data, actual_sr = load_audio(
                    audio, sr=sample_rate or SAMPLE_RATE, mono=True
                )
                console.print(
                    f"[dim]📂 Loaded audio for extraction: "
                    f"{len(audio_data)/actual_sr:.2f}s @ {actual_sr}Hz[/dim]"
                )
            except Exception as e:
                console.print(f"[red]❌ Failed to load audio for extraction: {e}[/red]")
                audio_data = np.array([], dtype=np.float32)
                actual_sr = sample_rate or SAMPLE_RATE
            for segment in high_speech_segments:
                start_sample = int(segment["start_time"] * actual_sr)
                end_sample = int(segment["end_time"] * actual_sr)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                if end_sample > start_sample:
                    segment_audio = audio_data[start_sample:end_sample].copy()
                    high_speech_audios.append(segment_audio)
                    seg_dur = len(segment_audio) / actual_sr
                    console.print(
                        f"[dim]   ✂ Extracted {seg_dur:.2f}s audio "
                        f"({len(segment_audio)} samples)[/dim]"
                    )
                else:
                    console.print(
                        f"[yellow]⚠ Empty audio range for segment "
                        f"{segment['segment_index']}: "
                        f"{start_sample}-{end_sample} samples[/yellow]"
                    )
                    high_speech_audios.append(np.array([], dtype=np.float32))
        total_elapsed = time.time() - overall_start
        total_extracted_duration = sum(
            len(a) / actual_sr for a in high_speech_audios if len(a) > 0
        )
        console.print(
            f"[dim]⏱ Extraction complete: {total_elapsed:.2f}s | "
            f"Total extracted: {total_extracted_duration:.2f}s[/dim]"
        )
        return high_speech_segments, high_speech_audios

    def reset(self) -> None:
        """Reset the tagger instance (useful for testing or model updates)."""
        self._tagger = None
        self._labels_map = None
        console.print("[yellow]AudioTagger reset[/yellow]")

    def save_results(
        self,
        results: List[TaggingResult],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> Path:
        """
        Save tagging results to a file.

        Args:
            results: List of result dictionaries from tag_audio
            output_path: Path to save output file
            format: Output format ("json" or "txt")

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(
                        f"{result['index']}: {result['name']} "
                        f"(class_index={result['class_index']}) "
                        f"- prob={result['prob']:.4f}\n"
                    )
        else:
            raise ValueError(f"Unsupported format: {format}")

        console.print(f"[green]Results saved to: {linkify(str(output_path))}[/green]")
        return output_path

    def display_results(self, results: List[TaggingResult]) -> None:
        """
        Display tagging results in a rich table.

        Args:
            results: List of tagging results to display
        """
        table = Table(title="Audio Tagging Results", border_style="blue")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Name", style="green")
        table.add_column("Class Index", style="yellow", justify="right")
        table.add_column("Probability", style="magenta", justify="right")

        for result in results:
            prob_color = "green" if result["prob"] >= 0.5 else "yellow"
            table.add_row(
                str(result["index"]),
                result["name"],
                str(result["class_index"]),
                f"[{prob_color}]{result['prob']:.4f}[/{prob_color}]",
            )

        console.print(table)


if __name__ == "__main__":
    from main._main_audio_tagger import main

    main()
