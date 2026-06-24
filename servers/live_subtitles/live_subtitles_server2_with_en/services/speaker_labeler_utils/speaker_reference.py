"""Speaker reference data structures and segment info types."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict
import numpy as np


class SpeakerSegmentInfo(TypedDict):
    label: str
    segment_count: int
    first_seen: float
    last_seen: float
    active_duration: float
    has_valid_centroid: bool
    centroid_quality: float
    centroid_shape: Optional[List[int]]
    embedding_count: int
    embeddings: List[List[float]]


@dataclass
class SpeakerReference:
    """Maintains reference data for a single speaker."""
    label: str
    all_embeddings: List[np.ndarray] = field(default_factory=list)
    core_embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_metadata: List[Dict] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    first_seen: Optional[float] = None
    last_seen: float = 0.0
    segment_count: int = 0

    def add_embedding(
        self,
        embedding: np.ndarray,
        timestamp: float,
        segment_id: Optional[str] = None,
        audio_duration: float = 0.0,
        is_core: bool = True,
    ) -> None:
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        self.all_embeddings.append(embedding)
        self.embedding_metadata.append({
            'segment_id': segment_id or f"unknown_{len(self.all_embeddings)}",
            'timestamp': timestamp,
            'index': len(self.all_embeddings) - 1,
            'added_at': timestamp,
            'audio_duration': audio_duration,
            'is_core': is_core,
        })
        self.segment_count += 1
        if is_core:
            self.core_embeddings.append(embedding)
        self._recompute_centroid()
        if self.first_seen is None:
            self.first_seen = timestamp
        if timestamp > self.last_seen:
            self.last_seen = timestamp

    def _recompute_centroid(self) -> None:
        """Recompute centroid from core_embeddings only."""
        src = self.core_embeddings if self.core_embeddings else self.all_embeddings
        if not src:
            return
        stacked = np.vstack(src)
        if len(src) >= 3:
            self.centroid = np.median(stacked, axis=0, keepdims=True)
        else:
            self.centroid = np.mean(stacked, axis=0, keepdims=True)

    @property
    def embeddings(self) -> List[np.ndarray]:
        return self.all_embeddings

    @property
    def active_duration(self) -> float:
        """Total active duration of this speaker."""
        if self.first_seen is None:
            return 0.0
        return self.last_seen - self.first_seen

    @property
    def has_valid_centroid(self) -> bool:
        """Check if centroid is valid."""
        return self.centroid is not None and not np.any(np.isnan(self.centroid))

    @property
    def centroid_quality(self) -> float:
        """Estimate centroid quality based on segment count (0.0 to 1.0)."""
        if self.segment_count >= 10:
            return 1.0
        elif self.segment_count >= 5:
            return 0.8
        elif self.segment_count >= 3:
            return 0.6
        else:
            return 0.3
