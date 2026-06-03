from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
import sherpa_onnx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

try:
    from services.audio_utils import AudioInput, load_audio
    from services.custom_logging import linkify
    from services.config import (
        FRAME_PER_SECONDS,
        FRAME_SHIFT_S,
        SAMPLE_RATE,
    )
except ImportError:
    from audio_utils import AudioInput, load_audio
    from custom_logging import linkify
    from config import (
        FRAME_PER_SECONDS,
        FRAME_SHIFT_S,
        SAMPLE_RATE,
    )

install_rich_traceback(show_locals=True)

console = Console()

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
AUDIO_TAGGING_MODEL = (
    BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
)
CLASS_LABELS_INDICES_CSV = (
    BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
)


class TaggingResult(TypedDict):
    """Typed dictionary for audio tagging results."""

    index: int
    name: str
    class_index: int
    prob: float


class ChunkTaggingResult(TypedDict):
    """Per-chunk tagging result with timing metadata."""

    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    predictions: List[TaggingResult]
    processing_time: float


class AudioChunksTaggingSummary(TypedDict):
    """Complete summary of chunked audio tagging."""

    audio_path: str
    total_duration: float
    sample_rate: int
    chunk_duration: float
    overlap_duration: float
    total_chunks: int
    chunks: List[ChunkTaggingResult]
    overall_top_predictions: List[TaggingResult]
    total_processing_time: float
    real_time_factor: float


class AudioTaggerConfig(TypedDict, total=False):
    """Typed dictionary for AudioTagger configuration."""

    model_path: Optional[Union[str, Path]]
    labels_path: Optional[Union[str, Path]]
    top_k: int
    num_threads: int
    provider: str
    debug: bool
    speech_prob_threshold: float
    speech_top_n: int
    # Chunking defaults (from jet.audio.helpers.config)
    chunk_duration: float  # seconds
    chunk_overlap: float  # seconds
    min_chunk_duration: float  # seconds


class AudioTaggingSummary(TypedDict):
    """Typed dictionary for audio tagging summary."""

    audio_path: str
    duration_seconds: float
    sample_rate: int
    num_results: int
    top_predictions: List[TaggingResult]
    speech_detected: bool
    max_speech_probability: float
    processing_time_seconds: float
    real_time_factor: float


class AudioTagger:
    """
    Reusable audio tagging class using Sherpa-ONNX models.

    Features:
    - Tags audio with type AudioInput (reuses load_audio)
    - Checks if audio contains high-probability speech
    - Process long audio in overlapping chunks with tag_audio_chunks()
    - Configurable chunking parameters from jet.audio.helpers.config

    Example:
        >>> tagger = AudioTagger()
        >>> results = tagger.tag_audio("path/to/audio.wav")
        >>> is_speech = tagger.contains_speech("path/to/audio.wav")
        >>> chunked = tagger.tag_audio_chunks("long_audio.wav", chunk_duration=5.0)
    """

    DEFAULT_BASE_DIR: Path = (
        Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
    )
    DEFAULT_MODEL_PATH: Path = (
        DEFAULT_BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
    )
    DEFAULT_LABELS_PATH: Path = (
        DEFAULT_BASE_DIR
        / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
    )

    SPEECH_CLASS_NAMES: List[str] = [
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
    ]

    # Default chunking constants from jet.audio.helpers.config
    # Chunk duration: 100 frames * 0.010s = 1.0s (same as process_audio_chunks window)
    DEFAULT_CHUNK_DURATION: float = FRAME_PER_SECONDS * FRAME_SHIFT_S  # 1.0s
    DEFAULT_CHUNK_OVERLAP: float = DEFAULT_CHUNK_DURATION / 2.0  # 0.5s (50%)
    MIN_CHUNK_DURATION: float = 0.5  # Minimum chunk size in seconds

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = AUDIO_TAGGING_MODEL,
        labels_path: Optional[Union[str, Path]] = CLASS_LABELS_INDICES_CSV,
        top_k: int = 5,
        num_threads: int = 1,
        provider: str = "cpu",
        debug: bool = False,
        speech_prob_threshold: float = 0.5,
        speech_top_n: int = 3,
        # Chunking defaults
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
            speech_prob_threshold: Minimum probability to consider as speech
            speech_top_n: Check top N predictions for speech classes
            chunk_duration: Default chunk duration in seconds (default: 1.0s)
            chunk_overlap: Default overlap between chunks in seconds (default: 0.5s)
            min_chunk_duration: Minimum valid chunk duration (default: 0.5s)
        """
        self.model_path: Path = (
            Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        )
        self.labels_path: Path = (
            Path(labels_path) if labels_path else self.DEFAULT_LABELS_PATH
        )
        self.top_k: int = top_k
        self.num_threads: int = num_threads
        self.provider: str = provider
        self.debug: bool = debug
        self.speech_prob_threshold: float = speech_prob_threshold
        self.speech_top_n: int = speech_top_n

        # Chunking configuration (from jet.audio.helpers.config)
        self.chunk_duration: float = (
            chunk_duration
            if chunk_duration is not None
            else self.DEFAULT_CHUNK_DURATION
        )
        self.chunk_overlap: float = (
            chunk_overlap if chunk_overlap is not None else self.DEFAULT_CHUNK_OVERLAP
        )
        self.min_chunk_duration: float = (
            min_chunk_duration
            if min_chunk_duration is not None
            else self.MIN_CHUNK_DURATION
        )

        # Validate chunking parameters
        self._validate_chunking_config()

        self._tagger: Optional[sherpa_onnx.AudioTagging] = None
        self._labels_map: Optional[Dict[int, str]] = None

        console.print(
            Panel.fit(
                f"[bold green]AudioTagger Initialized[/bold green]\n"
                f"Model: {linkify(str(self.model_path))}\n"
                f"Labels: {linkify(str(self.labels_path))}\n"
                f"Speech Threshold: {self.speech_prob_threshold}\n"
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
            waveform, actual_sr = load_audio(audio, sr=sample_rate or 16000, mono=True)
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

    # ── NEW METHOD: tag_audio_chunks ─────────────────────────────────────

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

        This method splits audio into fixed-duration overlapping chunks,
        tags each independently, and aggregates results. Useful for:
        - Very long recordings that exceed model context windows
        - Tracking how audio content changes over time
        - Speech/music segmentation at coarse granularity

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: 16000)
            chunk_duration: Duration of each chunk in seconds.
                           Default: self.chunk_duration (from config, typically 1.0s)
            overlap_duration: Overlap between chunks in seconds.
                             Default: self.chunk_overlap (typically 0.5s)
            min_chunk_duration: Minimum duration for the last chunk.
                               Default: self.min_chunk_duration (0.5s)

        Returns:
            AudioChunksTaggingSummary with per-chunk results and overall aggregation

        Example:
            >>> tagger = AudioTagger()
            >>> summary = tagger.tag_audio_chunks("long_speech.wav", chunk_duration=5.0)
            >>> print(f"Processed {summary['total_chunks']} chunks")
            >>> for chunk in summary['chunks']:
            ...     print(f"  Chunk {chunk['chunk_index']}: "
            ...           f"{chunk['predictions'][0]['name']}")
        """
        # ── Parameter resolution with config defaults ──────────────────
        _chunk_dur = (
            chunk_duration if chunk_duration is not None else self.chunk_duration
        )
        _overlap = (
            overlap_duration if overlap_duration is not None else self.chunk_overlap
        )
        _min_chunk = (
            min_chunk_duration
            if min_chunk_duration is not None
            else self.min_chunk_duration
        )

        # Validate parameters
        if _chunk_dur < _min_chunk:
            _chunk_dur = _min_chunk

        if _overlap >= _chunk_dur:
            _overlap = _chunk_dur / 2.0

        # ── Load audio ─────────────────────────────────────────────────
        overall_start = time.time()

        try:
            waveform, actual_sr = load_audio(
                audio, sr=sample_rate or SAMPLE_RATE, mono=True
            )
        except Exception:
            raise

        total_samples = len(waveform)
        total_duration = total_samples / actual_sr

        # Determine audio path identifier
        if isinstance(audio, (str, Path)):
            audio_path_str = str(audio)
        elif isinstance(audio, bytes):
            audio_path_str = f"bytes_input_{len(audio)}bytes"
        else:
            audio_path_str = f"array_input_{waveform.shape}"

        # ── Calculate chunk boundaries ──────────────────────────────────
        chunk_samples = int(_chunk_dur * actual_sr)
        hop_samples = int((_chunk_dur - _overlap) * actual_sr)

        # Ensure hop is at least 1 sample
        if hop_samples < 1:
            hop_samples = 1

        # ── Generate chunk positions ────────────────────────────────────
        chunk_positions = self._calculate_chunk_positions(
            total_samples=total_samples,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            min_chunk_duration=_min_chunk,
            sample_rate=actual_sr,
        )

        if not chunk_positions:
            # Return empty summary
            elapsed = time.time() - overall_start
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
                real_time_factor=elapsed / total_duration
                if total_duration > 0
                else 0.0,
            )

        # ── Process each chunk ──────────────────────────────────────────
        chunks: List[ChunkTaggingResult] = []
        all_predictions: Dict[str, List[float]] = {}  # name -> list of probs

        for idx, (start_sample, end_sample) in enumerate(chunk_positions):
            chunk_start_time = time.time()

            start_sec = start_sample / actual_sr
            end_sec = end_sample / actual_sr
            chunk_waveform = waveform[start_sample:end_sample].copy()

            # Tag this chunk
            try:
                chunk_predictions = self._tag_waveform(chunk_waveform, actual_sr)
            except Exception:
                # Create error entry
                chunk_predictions = []

            chunk_elapsed = time.time() - chunk_start_time

            # Collect predictions for aggregation
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
            )
            chunks.append(chunk_result)

        # ── Aggregate overall top predictions ───────────────────────────
        overall_top = self._aggregate_chunk_predictions(all_predictions, self.top_k)

        # ── Build summary ───────────────────────────────────────────────
        total_elapsed = time.time() - overall_start
        rtf = total_elapsed / total_duration if total_duration > 0 else 0.0

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
        )

        return summary

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

    # ── Existing methods (unchanged below this line) ───────────────────

    def contains_speech(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        prob_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
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
            if name in self.SPEECH_CLASS_NAMES and prob >= threshold:
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
        sample_rate: Optional[int] = None,
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
            if result.get("name", "") in self.SPEECH_CLASS_NAMES:
                prob = result.get("prob", 0.0)
                if prob > max_speech_prob:
                    max_speech_prob = prob
        return max_speech_prob

    def get_tagging_summary(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        audio_path: str = "unknown",
    ) -> AudioTaggingSummary:
        """
        Get a comprehensive summary of audio tagging results.

        Args:
            audio: Audio input
            sample_rate: Sample rate for raw audio data
            audio_path: Identifier for the audio source

        Returns:
            AudioTaggingSummary with complete analysis
        """
        start_time = time.time()

        try:
            waveform, actual_sr = load_audio(audio, sr=sample_rate or 16000, mono=True)
            audio_duration = len(waveform) / actual_sr if actual_sr > 0 else 0
        except Exception:
            raise

        results = self.tag_audio(audio, sample_rate=sample_rate)
        max_speech_prob = self.get_speech_probability(audio, sample_rate=sample_rate)
        speech_detected = max_speech_prob >= self.speech_prob_threshold

        elapsed = time.time() - start_time
        rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

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
        }

        return summary

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
