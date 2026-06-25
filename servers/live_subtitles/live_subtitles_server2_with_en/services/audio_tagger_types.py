from pathlib import Path
from typing import List, Optional, TypedDict, Union


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


class SpeechSegmentTimeline(TypedDict):
    """Probability timeline for a single segment."""

    times: List[float]
    probs: List[float]


class SpeechSegmentResult(TypedDict):
    """A single detected speech or non-speech segment."""

    segment_index: int
    segment_type: str
    start_time: float
    end_time: float
    duration: float
    avg_speech_probability: float
    max_speech_probability: float
    min_speech_probability: float
    speech_density: float
    speech_chunk_count: int
    total_chunk_count: int
    speech_chunk_ratio: float
    threshold_used: float
    is_high_confidence: bool
    is_medium_confidence: bool
    confidence_tier: str
    confidence_label: str
    confidence_duration_note: str  # NEW: explains duration's impact
    is_dense_speech: bool
    top_prediction: str
    top_prediction_prob: float
    top_predictions: List[TaggingResult]
    overlapping_chunks: List[ChunkTaggingResult]
    timeline: SpeechSegmentTimeline


class AudioSegmentsResult(TypedDict):
    """Complete result of segment-based audio tagging."""

    audio_path: str
    total_duration: float
    sample_rate: int
    chunk_duration: float
    overlap_duration: float
    total_chunks: int
    speech_threshold: float
    min_silence_duration_sec: float
    min_speech_duration_sec: float
    resolution_ms: float
    chunks: List[ChunkTaggingResult]
    speech_segments: List[SpeechSegmentResult]
    non_speech_segments: List[SpeechSegmentResult]
    total_speech_duration: float
    total_non_speech_duration: float
    overall_top_predictions: List[TaggingResult]
    total_processing_time: float
    real_time_factor: float
