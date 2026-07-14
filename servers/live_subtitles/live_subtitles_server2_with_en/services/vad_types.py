from typing import Any, Dict, List, Optional, TypedDict


class VADSegment(TypedDict):
    frame_start: int
    frame_end: int
    frame_length: int
    start_s: float
    end_s: float
    duration_s: float
    details: Dict[str, Any]


class ValleyInfo(TypedDict):
    frame_start: int
    frame_end: int
    frame_length: int
    start_s: float
    end_s: float
    duration_s: float
    valley_score: float
    trough_score: float
    final_score: float
    global_frame_start: int
    global_frame_end: int
    global_start_s: float
    global_end_s: float
    global_duration_s: float
    global_valley_score: float
    global_trough_score: float
    global_final_score: float
    is_last: bool


# vad_types.py
class ValleyTrough(TypedDict):
    frame: int
    global_frame: int
    prob: float
    time_s: float
    global_time_s: float
    valley: ValleyInfo
    prominence: Optional[float]
    width: Optional[float]


class TroughToTroughScores(TypedDict):
    """Component scores for a TroughToTroughSegment's quality."""

    median_prob_score: float  # Median VAD probability (^1.3 non-linear)
    speech_ratio_score: float  # Fraction of frames ≥ speech threshold
    duration_score: float  # Trapezoidal duration optimality
    consistency_score: float  # IQR-based stability (1.0 - IQR/0.5)
    boundary_quality_score: float  # MIN of start/end boundary scores
    content_score: float  # Weighted composite of above 4


class TroughToTroughSegment(TypedDict, total=False):
    """A segment between two consecutive valley troughs (or start/end sentinels)."""

    start_s: float
    end_s: float
    duration_s: float
    start_frame: int
    end_frame: int
    trough_start: Optional[ValleyTrough]
    trough_end: Optional[ValleyTrough]
    segment_probs: Optional[List[float]]
    prob_stats: Optional[Dict[str, float]]
    scores: TroughToTroughScores  # Component quality scores
    final_score: float  # Composite: content_score * boundary_quality_score


class StreamVadFrame(TypedDict):
    """Typed structure for accumulated VAD probability frame."""

    frame_idx: int
    raw_prob: float
    smoothed_prob: float
    is_speech: bool
    is_speech_start: bool
    is_speech_end: bool
    speech_start_frame: Optional[int]
    speech_end_frame: Optional[int]
