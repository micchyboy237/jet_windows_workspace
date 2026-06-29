"""Segment labeling result types."""

from typing import List, Optional, TypedDict


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

    segment_id: str
    timestamp: float
    audio_duration: float
    matches: List[SegmentMatch]


class SegmentGroupsResult(TypedDict):
    """Complete result from label_segments() containing all processed segments."""

    segments: List[SegmentGroup]
    total_segments: int
    speaker_count: int
    outlier_count: int


class TopKMatch(TypedDict):
    """A single match from find_top_k_matches."""

    label: str
    confidence: float
    match_type: str
    is_primary: bool
    segment_count: int
    last_seen: float
    centroid_quality: float


class SmartMaintenanceResult(TypedDict):
    """Result from smart maintenance run."""

    consolidations: dict  # ConsolidationResult
    reevaluations: dict  # ReevaluationResult
    orphans_cleaned: int
    total_changes: int


class ReevaluationResult(TypedDict):
    """Result from reevaluating young speakers."""

    merges_performed: int
    speakers_removed: int
    speakers_promoted: int
    details: List[str]
    dry_run: bool


class ConsolidationResult(TypedDict):
    """Result from speaker consolidation."""

    merges_performed: int
    speakers_removed: int
    merge_details: List[dict]
    dry_run: bool


class SpeakerInfo(TypedDict):
    """Information about a single speaker."""

    label: str
    segment_count: int
    first_seen: float
    last_seen: float
    active_duration: float
    has_valid_centroid: bool
    centroid_quality: float
    centroid_coordinates: Optional[list]
    centroid_shape: Optional[list]


class HealthStatus(TypedDict):
    """Complete health status of the labeler."""

    total_speakers: int
    total_segments_processed: int
    total_speakers_created: int
    rejected_updates: int
    outlier_stats: dict  # OutlierStats
    speaker_categories: dict
    missing_speaker_ids: List[str]


class OutlierStats(TypedDict):
    """Statistics about the outlier pool."""

    enabled: bool
    active_outliers: int
    total_promotions: int
    resolved_outliers: int
    outlier_labels: List[str]
    promotion_history: List[dict]


class PotentialMerge(TypedDict):
    """A potential speaker merge pair."""

    speaker_1: str
    speaker_2: str
    similarity: float
    segments_1: int
    segments_2: int
    total_segments: int


class SpeakerSimilarityMatrix(TypedDict):
    """Similarity matrix between all speakers."""

    labels: List[str]
    similarities: List[List[float]]
    potential_merges: List[dict]


class CentroidHealthStats(TypedDict):
    """Statistics about centroid health."""

    total_speakers: int
    speakers_with_valid_centroids: int
    average_quality: float
    min_quality: float
    max_quality: float
    speakers_needing_attention: List[str]


class CentroidStats(TypedDict):
    """Detailed centroid statistics per speaker."""

    speaker_label: str
    centroid_shape: Optional[tuple]
    centroid_quality: float
    embedding_count: int
    has_valid_centroid: bool


class SpeakerHealthReport(TypedDict):
    """Health report for all speakers."""

    total_speakers: int
    mature_speakers: int
    young_speakers: int
    orphan_speakers: int
    average_centroid_quality: float
    speakers_with_issues: List[str]
    recommendations: List[str]
