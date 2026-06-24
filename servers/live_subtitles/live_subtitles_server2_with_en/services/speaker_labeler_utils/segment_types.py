"""Segment labeling result types."""
from typing import List, TypedDict


class SegmentMatch(TypedDict, total=False):
    """A single speaker match within a segment."""
    label: str
    confidence: float
    match_type: str
    is_primary: bool
    is_new_speaker: bool
    is_outlier: bool
    segment_count: int
    last_seen: float
    segment_id: str
    promoted_from_outlier: bool
    original_outlier_label: str
    resolution_method: str


class SegmentGroup(TypedDict):
    """A processed audio segment with its speaker matches."""
    timestamp: float
    audio_duration: float
    matches: List[SegmentMatch]


class SegmentGroupsResult(TypedDict):
    """Complete result from label_segments() containing all processed segments."""
    segments: List[SegmentGroup]
    total_segments: int
    speaker_count: int
    outlier_count: int
