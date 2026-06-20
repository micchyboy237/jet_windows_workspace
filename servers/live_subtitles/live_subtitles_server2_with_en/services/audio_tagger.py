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
        FRAME_SHIFT_S,
        SAMPLE_RATE,
    )
    from services.custom_logging import linkify
except ImportError:
    from audio_utils import AudioInput, load_audio
    from audio_config import (
        FRAME_PER_SECONDS,
        FRAME_SHIFT_S,
        SAMPLE_RATE,
    )
    from custom_logging import linkify

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
    speech_detected: bool
    # REMOVED: max_speech_probability: float


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
    speech_duration: float
    speech_detected: bool
    max_speech_probability: float
    avg_speech_probability: float  # average only across speech chunks


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
    # NEW: For consistency, also add speech_duration to single-tag summary
    speech_duration: float


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

    DEFAULT_SPEECH_PROB_THRESHOLD: float = 0.5
    DEFAULT_SPEECH_TOP_N: int = 3

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
        speech_prob_threshold: Optional[float] = None,
        speech_top_n: Optional[int] = None,
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
            speech_prob_threshold: Minimum speech probability (default: 0.5)
            speech_top_n: Check the top N predictions for speech classes (default: 3)
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

        # Use defaults if not provided
        self.speech_prob_threshold: float = (
            speech_prob_threshold
            if speech_prob_threshold is not None
            else self.DEFAULT_SPEECH_PROB_THRESHOLD
        )
        self.speech_top_n: int = (
            speech_top_n if speech_top_n is not None else self.DEFAULT_SPEECH_TOP_N
        )

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

    # ── NEW: Speech detection helper for chunks ───────────────────────
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
        """
        n_to_check = top_n if top_n is not None else self.speech_top_n
        chunk_speech_prob = 0.0

        for result in predictions[:n_to_check]:
            name = result.get("name", "")
            prob = result.get("prob", 0.0)
            if name in self.SPEECH_CLASS_NAMES and prob > chunk_speech_prob:
                chunk_speech_prob = prob

        speech_detected = chunk_speech_prob >= self.speech_prob_threshold
        return speech_detected, chunk_speech_prob

    # ── NEW: Calculate speech duration from consecutive speech chunks ──
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

    def tag_audio_chunks(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,  # NEW parameter
    ) -> AudioChunksTaggingSummary:
        """
        Process long audio by splitting into overlapping chunks and tagging each.
        
        This method splits audio into fixed-duration overlapping chunks,
        tags each independently, and aggregates results. Useful for:
        - Very long recordings that exceed model context windows
        - Tracking how audio content changes over time
        - Speech/music segmentation at coarse granularity
        - Computing speech_duration (sum of consecutive speech chunks)
        
        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: 16000)
            chunk_duration: Duration of each chunk in seconds.
                        Default: self.chunk_duration (from config, typically 1.0s)
            overlap_duration: Overlap between chunks in seconds.
                            Default: self.chunk_overlap (typically 0.5s)
            min_chunk_duration: Minimum duration for the last chunk.
                            Default: self.min_chunk_duration (0.5s)
            output_dir: Optional directory to save speech chunks.
                    If provided, speech chunks will be saved under
                    output_dir / "speech_chunks" / "chunk_<index+1>" /
                    as sound.wav and meta.json
        
        Returns:
            AudioChunksTaggingSummary with per-chunk results, overall aggregation,
            speech_duration, and avg_speech_probability
        
        Example:
            >>> tagger = AudioTagger()
            >>> summary = tagger.tag_audio_chunks("long_speech.wav", chunk_duration=5.0)
            >>> print(f"Processed {summary['total_chunks']} chunks")
            >>> print(f"Speech duration: {summary['speech_duration']:.2f}s")
            >>> print(f"Avg speech probability: {summary['avg_speech_probability']:.4f}")
            >>> for chunk in summary['chunks']:
            ...     print(f"  Chunk {chunk['chunk_index']}: "
            ...           f"{chunk['predictions'][0]['name']}")
        """
        import soundfile as sf  # For saving WAV files
        
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
        
        # Validate chunking parameters
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
        
        # Load audio
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
        
        # Determine audio path string for metadata
        if isinstance(audio, (str, Path)):
            audio_path_str = str(audio)
        elif isinstance(audio, bytes):
            audio_path_str = f"bytes_input_{len(audio)}bytes"
        else:
            audio_path_str = f"array_input_{waveform.shape}"
        
        # Calculate chunk parameters
        chunk_samples = int(_chunk_dur * actual_sr)
        hop_samples = int((_chunk_dur - _overlap) * actual_sr)
        if hop_samples < 1:
            hop_samples = 1
        
        console.print(
            f"[dim]🔧 Chunk config: {_chunk_dur}s chunks, "
            f"{_overlap}s overlap, hop={hop_samples} samples[/dim]"
        )
        
        # Calculate chunk positions
        chunk_positions = self._calculate_chunk_positions(
            total_samples=total_samples,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            min_chunk_duration=_min_chunk,
            sample_rate=actual_sr,
        )
        
        console.print(f"[dim]📏 Calculated {len(chunk_positions)} chunk positions[/dim]")
        
        # Handle empty audio case
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
                real_time_factor=elapsed / total_duration
                if total_duration > 0
                else 0.0,
                speech_duration=0.0,
                speech_detected=False,
                max_speech_probability=0.0,
                avg_speech_probability=0.0,  # NEW
            )
        
        # Setup output directory for speech chunks if specified
        speech_chunks_base_dir = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            speech_chunks_base_dir = output_dir / "speech_chunks"
            speech_chunks_base_dir.mkdir(parents=True, exist_ok=True)
            console.print(
                f"[dim]💾 Speech chunks will be saved to: "
                f"{speech_chunks_base_dir}[/dim]"
            )
        
        # Process each chunk
        chunks: List[ChunkTaggingResult] = []
        all_predictions: Dict[str, List[float]] = {}
        any_speech_detected = False
        global_max_speech_prob = 0.0
        speech_probabilities: List[float] = []  # NEW: collect probs for speech chunks
        
        for idx, (start_sample, end_sample) in enumerate(chunk_positions):
            chunk_start_time = time.time()
            start_sec = start_sample / actual_sr
            end_sec = end_sample / actual_sr
            
            console.print(
                f"[dim]🔍 Processing chunk {idx + 1}/{len(chunk_positions)}: "
                f"{start_sec:.2f}s - {end_sec:.2f}s[/dim]"
            )
            
            # Extract chunk waveform
            chunk_waveform = waveform[start_sample:end_sample].copy()
            
            # Tag the chunk
            try:
                chunk_predictions = self._tag_waveform(chunk_waveform, actual_sr)
                console.print(
                    f"[dim]   ✅ Tagged successfully: "
                    f"{len(chunk_predictions)} predictions[/dim]"
                )
            except Exception as e:
                console.print(f"[red]   ❌ Tagging failed: {e}[/red]")
                chunk_predictions = []
            
            # Check for speech
            speech_detected, chunk_speech_prob = self._chunk_has_speech(chunk_predictions)
            
            if speech_detected:
                any_speech_detected = True
                speech_probabilities.append(chunk_speech_prob)  # NEW: collect for avg
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
            
            # Aggregate predictions for overall stats
            for pred in chunk_predictions:
                name = pred["name"]
                if name not in all_predictions:
                    all_predictions[name] = []
                all_predictions[name].append(pred["prob"])
            
            # Build chunk result (with speech_probability instead of max_speech_probability)
            chunk_result = ChunkTaggingResult(
                chunk_index=idx,
                start_time=round(start_sec, 3),
                end_time=round(end_sec, 3),
                duration=round(end_sec - start_sec, 3),
                predictions=chunk_predictions,
                processing_time=round(chunk_elapsed, 4),
                speech_detected=speech_detected,
                speech_probability=round(chunk_speech_prob, 4),  # RENAMED: was max_speech_probability
            )
            chunks.append(chunk_result)
            
            # NEW: Save speech chunk to disk if output_dir specified
            if output_dir is not None and speech_detected and speech_chunks_base_dir:
                self._save_speech_chunk(
                    chunk_waveform=chunk_waveform,
                    sample_rate=actual_sr,
                    chunk_index=idx,
                    start_time=start_sec,
                    end_time=end_sec,
                    speech_probability=chunk_speech_prob,
                    predictions=chunk_predictions,
                    base_dir=speech_chunks_base_dir,
                )
        
        # Calculate aggregate metrics
        speech_duration = self._calculate_speech_duration(chunks, _overlap)
        
        # NEW: Calculate average speech probability
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
        
        # Build final summary with new field
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
            avg_speech_probability=round(avg_speech_prob, 4),  # NEW
        )
        
        return summary

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
            waveform, actual_sr = load_audio(audio, sr=sample_rate or 16000, mono=True)
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

    def split_speech_audio(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        prob_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        min_speech_duration: float = 0.5,
        merge_gap: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Split audio into individual speech segments, returning each as a numpy array.

        Uses chunk-based audio tagging to detect speech regions, merges overlapping
        speech chunks into continuous segments, and extracts the audio for each segment.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: SAMPLE_RATE)
            prob_threshold: Minimum speech probability threshold (default: self.speech_prob_threshold)
            top_n: Number of top predictions to check for speech classes (default: self.speech_top_n)
            chunk_duration: Duration of each analysis chunk in seconds (default: self.chunk_duration)
            overlap_duration: Overlap between consecutive chunks (default: self.chunk_overlap)
            min_chunk_duration: Minimum duration for the last chunk (default: self.min_chunk_duration)
            min_speech_duration: Minimum duration in seconds for a valid speech segment (default: 0.5s)
            merge_gap: Maximum gap in seconds to merge nearby segments (default: chunk_duration/2)

        Returns:
            List of numpy arrays, each containing a continuous speech segment
            (mono, float32, at the given sample_rate)

        Example:
            >>> tagger = AudioTagger()
            >>> speech_segments = tagger.split_speech_audio("recording.wav")
            >>> for i, segment in enumerate(speech_segments):
            ...     print(f"Segment {i+1}: {len(segment)/16000:.2f}s")
            >>> # Use with ASR
            >>> for segment in speech_segments:
            ...     transcription = asr_model.transcribe(segment)

        Debug logs trace:
            - Audio loading details
            - Chunk-based speech detection
            - Identified speech segments with timestamps
            - Filtering of segments below min_speech_duration
            - Final segment count and total speech duration
        """
        console.print(
            Panel.fit(
                f"[bold cyan]split_speech_audio[/bold cyan]\n"
                f"prob_threshold={prob_threshold or self.speech_prob_threshold}\n"
                f"min_speech_duration={min_speech_duration}s\n"
                f"merge_gap={merge_gap or (chunk_duration or self.chunk_duration) / 2}s",
                title="Speech Splitting",
                border_style="cyan",
            )
        )
        
        # Step 1: Load audio
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
        
        # Step 2: Run chunk-based speech detection
        summary = self.tag_audio_chunks(
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
        )
        
        chunks = summary.get("chunks", [])
        if not chunks:
            console.print("[yellow]⚠ No chunks produced, returning empty list[/yellow]")
            return []
        
        # Step 3: Re-evaluate speech detection with custom parameters if needed
        threshold = prob_threshold if prob_threshold is not None else self.speech_prob_threshold
        n_check = top_n if top_n is not None else self.speech_top_n
        
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
                chunk["speech_detected"] = speech_detected and chunk_prob >= threshold
                chunk["speech_probability"] = round(chunk_prob, 4)
        
        # Step 4: Identify continuous speech segments from chunk results
        speech_segments = self._identify_speech_segments(
            chunks=chunks,
            total_duration=total_duration,
            edges_only=False,  # We want all segments, not just edges
        )
        
        if not speech_segments:
            console.print("[yellow]⚠ No speech segments detected[/yellow]")
            return []
        
        # Step 5: Merge nearby segments if gap is small
        _chunk_dur = chunk_duration if chunk_duration is not None else self.chunk_duration
        merge_threshold = merge_gap if merge_gap is not None else _chunk_dur / 2.0
        
        merged_segments = self._merge_close_segments(
            speech_segments, max_gap=merge_threshold
        )
        
        console.print(
            f"[dim]🔗 Merged {len(speech_segments)} raw segments into "
            f"{len(merged_segments)} segments (gap threshold: {merge_threshold:.3f}s)[/dim]"
        )
        
        # Step 6: Extract audio for each segment, filtering by minimum duration
        speech_arrays: List[np.ndarray] = []
        total_speech_duration = 0.0
        
        for i, (start_sec, end_sec) in enumerate(merged_segments):
            duration = end_sec - start_sec
            
            if duration < min_speech_duration:
                console.print(
                    f"[dim]⏭ Skipping segment {i+1}: {start_sec:.3f}s - {end_sec:.3f}s "
                    f"(duration {duration:.3f}s < min {min_speech_duration}s)[/dim]"
                )
                continue
            
            start_sample = int(start_sec * actual_sr)
            end_sample = int(end_sec * actual_sr)
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            
            if end_sample <= start_sample:
                console.print(
                    f"[yellow]⚠ Segment {i+1} has no valid samples, skipping[/yellow]"
                )
                continue
            
            segment_audio = waveform[start_sample:end_sample].copy()
            speech_arrays.append(segment_audio)
            total_speech_duration += duration
            
            console.print(
                f"[green]🎤 Segment {len(speech_arrays)}: {start_sec:.3f}s - {end_sec:.3f}s "
                f"(duration: {duration:.3f}s, samples: {len(segment_audio)})[/green]"
            )
        
        # Step 7: Log summary
        if speech_arrays:
            console.print(
                f"[bold green]✅ Extracted {len(speech_arrays)} speech segment(s) "
                f"totaling {total_speech_duration:.2f}s[/bold green]"
            )
        else:
            console.print(
                "[yellow]⚠ No speech segments met the minimum duration criteria[/yellow]"
            )
        
        return speech_arrays

    def extract_speech_only(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        edges_only: bool = False,
        prob_threshold: Optional[float] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        top_n: Optional[int] = None,
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
        
        # Load the full waveform first (we need sample-accurate trimming)
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
            start_sample = int(start_sec * actual_sr)
            end_sample = int(end_sec * actual_sr)
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            if end_sample > start_sample:
                trimmed_waveforms.append(waveform[start_sample:end_sample].copy())
        
        if not trimmed_waveforms:
            console.print("[yellow]⚠ No valid speech samples extracted[/yellow]")
            return np.array([], dtype=np.float32)
        
        result = np.concatenate(trimmed_waveforms)
        result_duration = len(result) / actual_sr
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
