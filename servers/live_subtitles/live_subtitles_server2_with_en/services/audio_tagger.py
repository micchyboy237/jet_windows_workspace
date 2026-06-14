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
    from services.config import (
        FRAME_PER_SECONDS,
        FRAME_SHIFT_S,
        SAMPLE_RATE,
    )
    from services.custom_logging import linkify
except ImportError:
    from audio_utils import AudioInput, load_audio
    from config import (
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
