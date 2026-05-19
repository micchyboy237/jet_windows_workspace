from typing import Any, Dict, Optional, TypedDict


class VADSegment(TypedDict):
    frame_start: int  # Starting frame index (inclusive)
    frame_end: int  # Ending frame index (inclusive)
    frame_length: int  # Number of frames
    start_s: float  # Start time in seconds
    end_s: float  # End time in seconds
    duration_s: float  # Duration in seconds
    details: Dict[str, Any]  # Additional insights (peak/trough properties)


class ValleyInfo(TypedDict):
    frame_start: int
    frame_end: int
    frame_length: int
    start_s: float
    end_s: float
    duration_s: float

    # Local scores
    valley_score: float
    trough_score: float
    final_score: float

    # Global fields (for consistency with time/frame offsets)
    global_frame_start: int
    global_frame_end: int
    global_start_s: float
    global_end_s: float
    global_duration_s: float

    # Global score references (same values as local, duplicated for API consistency)
    global_valley_score: float
    global_trough_score: float
    global_final_score: float
    is_last: bool


class ValleyTrough(TypedDict):
    frame: int
    global_frame: int
    prob: float
    time_s: float
    global_time_s: float
    valley: ValleyInfo


class StreamVadFrame(TypedDict):
    """Typed structure for accumulated VAD probability frame."""

    frame_idx: int
    raw_prob: float
    smoothed_prob: float
    is_speech: bool
    is_speech_start: bool
    is_speech_end: bool
    # Optional extended fields (can be added later)
    speech_start_frame: Optional[int]
    speech_end_frame: Optional[int]
