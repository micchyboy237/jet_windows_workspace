"""Outlier buffer for speaker embedding validation.

Provides a temporary holding area for unconfirmed speaker candidates.
Segments that don't match any existing speaker are placed in the outlier
pool. They're only promoted to permanent SPEAKER_XX when:
  - Two outliers match each other (mutual confirmation)
  - An outlier matches an existing speaker (retroactive merge)
  - Outliers that expire without matches are garbage collected

This two-phase approach prevents centroid contamination from single
noisy observations and reduces speaker proliferation.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from rich.console import Console

console = Console()
_LOGGER_PREFIX = "[dim cyan]OutlierPool[/dim cyan]"

# ── Default configuration ─────────────────────────────────────
DEFAULT_OUTLIER_PREFIX = "OUTLIER"
DEFAULT_MIN_OUTLIER_MATCHES = 2       # Outliers needed to form a speaker
DEFAULT_OUTLIER_TTL = 60.0            # Seconds before stale outlier expires
DEFAULT_OUTLIER_PROMOTION_THRESHOLD = 0.55  # Similarity to promote


@dataclass
class OutlierEntry:
    """A single unconfirmed speaker candidate in the outlier pool.
    
    Attributes
    ----------
    label : str
        Unique label (e.g., 'OUTLIER_03').
    embedding : np.ndarray
        Speaker embedding vector for this segment.
    timestamp : float
        When this outlier was created (application time).
    segment_id : str
        External segment identifier for traceability.
    audio_duration : float
        Duration of the original audio segment.
    match_attempts : int
        How many times this outlier has been checked against others.
    created_at : float
        Monotonic creation time for TTL calculations (not in __init__).
    """
    label: str
    embedding: np.ndarray
    timestamp: float
    segment_id: str
    audio_duration: float = 0.0
    match_attempts: int = 0
    # Use init=False so it's auto-set and not required by constructor
    created_at: float = field(default_factory=time.monotonic, init=False)
    
    @property
    def age(self) -> float:
        """Age of this outlier in seconds (monotonic clock)."""
        return time.monotonic() - self.created_at
    
    def is_expired(self, ttl: float) -> bool:
        """Check if this outlier has exceeded its TTL.
        
        Parameters
        ----------
        ttl : float
            Time-to-live in seconds.
            
        Returns
        -------
        bool
            True if age > ttl.
        """
        return self.age > ttl
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for persistence."""
        return {
            "label": self.label,
            "embedding": self.embedding.tolist(),
            "timestamp": self.timestamp,
            "segment_id": self.segment_id,
            "audio_duration": self.audio_duration,
            "match_attempts": self.match_attempts,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "OutlierEntry":
        """Deserialize from dictionary.
        
        Note: created_at is NOT restored from serialized data.
        Instead, the current time is used so restored outliers
        don't immediately expire.
        """
        return cls(
            label=data["label"],
            embedding=np.array(data["embedding"]),
            timestamp=data["timestamp"],
            segment_id=data["segment_id"],
            audio_duration=data.get("audio_duration", 0.0),
            match_attempts=data.get("match_attempts", 0),
        )


@dataclass
class OutlierMatch:
    """Result of matching a query embedding against an outlier.
    
    Attributes
    ----------
    outlier_label : str
        Label of the matching outlier.
    confidence : float
        Cosine similarity score.
    outlier_entry : OutlierEntry
        Reference to the full outlier entry.
    """
    outlier_label: str
    confidence: float
    outlier_entry: OutlierEntry


@dataclass
class PromotionEvent:
    """Record of an outlier being promoted to a full speaker.
    
    Attributes
    ----------
    type : str
        'single' — promoted alone (force promotion).
        'pair' — promoted with another matching outlier.
        'merge' — merged into an existing speaker.
    outlier_labels : List[str]
        Labels of outliers involved.
    target_speaker : str
        The SPEAKER_XX label created or merged into.
    confidence : float
        Similarity score that triggered the promotion.
    timestamp : float
        When the promotion occurred.
    """
    type: str
    outlier_labels: List[str]
    target_speaker: str
    confidence: float
    timestamp: float


class OutlierPool:
    """Manages the lifecycle of unconfirmed speaker candidates.
    
    Flow:
    ┌──────────┐     ┌──────────────┐     ┌──────────────┐
    │  Segment  │────▶│  OutlierPool  │────▶│  SPEAKER_XX  │
    │  (no match)│    │  (temporary)  │     │  (permanent)  │
    └──────────┘     └──────────────┘     └──────────────┘
                           │
                     ┌─────┴─────┐
                     │  Expired?  │──▶ Garbage Collected
                     └───────────┘
    
    Parameters
    ----------
    prefix : str
        Label prefix for outliers (default: 'OUTLIER').
    promotion_threshold : float
        Minimum cosine similarity to consider two outliers a match.
    ttl : float
        Time-to-live in seconds. Outliers older than this are removed.
    debug : bool
        Enable debug logging.
    """
    
    def __init__(
        self,
        prefix: str = DEFAULT_OUTLIER_PREFIX,
        promotion_threshold: float = DEFAULT_OUTLIER_PROMOTION_THRESHOLD,
        ttl: float = DEFAULT_OUTLIER_TTL,
        debug: bool = False,
    ):
        self.prefix = prefix
        self.promotion_threshold = promotion_threshold
        self.ttl = ttl
        self.debug = debug
        
        self._outliers: Dict[str, OutlierEntry] = {}
        self._next_id: int = 1
        self._promotions: List[PromotionEvent] = []
    
    # ── Properties ─────────────────────────────────────────
    @property
    def count(self) -> int:
        """Number of active outliers."""
        return len(self._outliers)
    
    @property
    def labels(self) -> List[str]:
        """Labels of all active outliers."""
        return list(self._outliers.keys())
    
    @property
    def promotion_count(self) -> int:
        """Total number of promotions performed."""
        return len(self._promotions)
    
    @property
    def is_empty(self) -> bool:
        """True if no outliers are present."""
        return len(self._outliers) == 0
    
    # ── Core operations ────────────────────────────────────
    def add(
        self,
        embedding: np.ndarray,
        timestamp: float,
        segment_id: str,
        audio_duration: float = 0.0,
    ) -> str:
        """Add a new outlier to the pool.
        
        Parameters
        ----------
        embedding : np.ndarray
            Speaker embedding vector.
        timestamp : float
            Segment timestamp.
        segment_id : str
            External segment identifier.
        audio_duration : float
            Duration of the audio segment.
        
        Returns
        -------
        str
            Label of the new outlier (e.g., 'OUTLIER_03').
        """
        label = f"{self.prefix}_{self._next_id:02d}"
        self._next_id += 1
        
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        self._outliers[label] = OutlierEntry(
            label=label,
            embedding=embedding.copy(),
            timestamp=timestamp,
            segment_id=segment_id,
            audio_duration=audio_duration,
        )
        
        if self.debug:
            console.print(
                f"{_LOGGER_PREFIX} 📦 Added {label} "
                f"(segment: {segment_id}, total: {self.count})"
            )
        
        return label
    
    def find_matches(
        self,
        query_embedding: np.ndarray,
        min_similarity: Optional[float] = None,
    ) -> List[OutlierMatch]:
        """Find outliers similar to a query embedding.
        
        Parameters
        ----------
        query_embedding : np.ndarray
            Embedding to match against the pool.
        min_similarity : float, optional
            Minimum similarity threshold. Defaults to promotion_threshold.
        
        Returns
        -------
        List[OutlierMatch]
            Matching outliers sorted by confidence descending.
        """
        if self.is_empty:
            return []
        
        if min_similarity is None:
            min_similarity = self.promotion_threshold
        
        # Prepare embedding matrix
        labels = []
        embeddings = []
        for label, entry in self._outliers.items():
            labels.append(label)
            emb = entry.embedding
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            embeddings.append(emb)
        
        embeddings_array = np.vstack(embeddings)
        
        # Compute cosine similarities
        from scipy.spatial.distance import cdist
        query_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
        distances = cdist(query_2d, embeddings_array, metric="cosine")
        similarities = 1.0 - distances.flatten()
        
        # Collect matches above threshold
        matches = []
        for i, label in enumerate(labels):
            sim = float(similarities[i])
            if sim >= min_similarity:
                self._outliers[label].match_attempts += 1
                matches.append(OutlierMatch(
                    outlier_label=label,
                    confidence=sim,
                    outlier_entry=self._outliers[label],
                ))
        
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        if self.debug and matches:
            console.print(
                f"{_LOGGER_PREFIX} 🔍 Found {len(matches)} outlier matches "
                f"(best: {matches[0].outlier_label} "
                f"sim={matches[0].confidence:.3f})"
            )
        
        return matches
    
    def remove(self, label: str) -> Optional[OutlierEntry]:
        """Remove a specific outlier from the pool.
        
        Parameters
        ----------
        label : str
            Outlier label to remove.
        
        Returns
        -------
        Optional[OutlierEntry]
            The removed entry, or None if not found.
        """
        entry = self._outliers.pop(label, None)
        if entry and self.debug:
            console.print(
                f"{_LOGGER_PREFIX} 🗑️  Removed {label} "
                f"(remaining: {self.count})"
            )
        return entry
    
    def remove_many(self, labels: List[str]) -> int:
        """Remove multiple outliers at once.
        
        Parameters
        ----------
        labels : List[str]
            Labels to remove.
        
        Returns
        -------
        int
            Number of outliers actually removed.
        """
        count = 0
        for label in labels:
            if self.remove(label) is not None:
                count += 1
        return count
    
    def cleanup_expired(self) -> int:
        """Remove all outliers that have exceeded their TTL.
        
        Returns
        -------
        int
            Number of outliers removed.
        """
        expired_labels = [
            label
            for label, entry in self._outliers.items()
            if entry.is_expired(self.ttl)
        ]
        
        for label in expired_labels:
            entry = self._outliers[label]
            if self.debug:
                console.print(
                    f"{_LOGGER_PREFIX} ⏰ Expired {label} "
                    f"(age: {entry.age:.1f}s > ttl: {self.ttl}s)"
                )
            del self._outliers[label]
        
        return len(expired_labels)
    
    def record_promotion(
        self,
        type_: str,
        outlier_labels: List[str],
        target_speaker: str,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Record a promotion event for history/debugging.
        
        Parameters
        ----------
        type_ : str
            'single', 'pair', or 'merge'.
        outlier_labels : List[str]
            Labels of outliers involved.
        target_speaker : str
            SPEAKER_XX label created or merged into.
        confidence : float
            Similarity score.
        timestamp : float
            When the promotion occurred.
        """
        event = PromotionEvent(
            type=type_,
            outlier_labels=outlier_labels,
            target_speaker=target_speaker,
            confidence=confidence,
            timestamp=timestamp,
        )
        self._promotions.append(event)
        
        if self.debug:
            console.print(
                f"{_LOGGER_PREFIX} 🎉 Promotion [{type_}]: "
                f"{outlier_labels} → {target_speaker} "
                f"(sim={confidence:.3f})"
            )
    
    # ── Pairwise matching ──────────────────────────────────
    def find_internal_matches(
        self,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of outliers that match each other.
        
        This is used for periodic maintenance to find outlier pairs
        that should be promoted even without a new incoming segment.
        
        Parameters
        ----------
        min_similarity : float, optional
            Minimum similarity threshold.
        
        Returns
        -------
        List[Tuple[str, str, float]]
            List of (label1, label2, similarity) tuples.
        """
        if self.count < 2:
            return []
        
        if min_similarity is None:
            min_similarity = self.promotion_threshold
        
        labels = list(self._outliers.keys())
        embeddings = []
        for label in labels:
            emb = self._outliers[label].embedding
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            embeddings.append(emb)
        
        embeddings_array = np.vstack(embeddings)
        
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings_array, embeddings_array, metric="cosine")
        similarities = 1.0 - distances
        
        pairs = []
        seen = set()
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = float(similarities[i, j])
                if sim >= min_similarity:
                    pair_key = tuple(sorted([labels[i], labels[j]]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append((labels[i], labels[j], sim))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
    
    def promote_pair(
        self,
        label1: str,
        label2: str,
        similarity: float,
        timestamp: float,
        target_speaker: str,
    ) -> List[OutlierEntry]:
        """Promote a pair of matching outliers and remove them from pool.
        
        Parameters
        ----------
        label1, label2 : str
            Labels of the two outliers to promote.
        similarity : float
            Similarity score between them.
        timestamp : float
            Current timestamp.
        target_speaker : str
            SPEAKER_XX label they're being merged into.
        
        Returns
        -------
        List[OutlierEntry]
            The removed outlier entries (for adding to speaker).
        """
        entries = []
        for label in [label1, label2]:
            entry = self.remove(label)
            if entry:
                entries.append(entry)
        
        self.record_promotion(
            type_="pair",
            outlier_labels=[label1, label2],
            target_speaker=target_speaker,
            confidence=similarity,
            timestamp=timestamp,
        )
        
        return entries
    
    def promote_single(
        self,
        label: str,
        timestamp: float,
        target_speaker: str,
        confidence: float = 1.0,
    ) -> Optional[OutlierEntry]:
        """Promote a single outlier (force promotion).
        
        Parameters
        ----------
        label : str
            Outlier label to promote.
        timestamp : float
            Current timestamp.
        target_speaker : str
            SPEAKER_XX label being created.
        confidence : float
            Confidence score (default 1.0 for forced).
        
        Returns
        -------
        Optional[OutlierEntry]
            The removed entry, or None if not found.
        """
        entry = self.remove(label)
        if entry:
            self.record_promotion(
                type_="single",
                outlier_labels=[label],
                target_speaker=target_speaker,
                confidence=confidence,
                timestamp=timestamp,
            )
        return entry
    
    # ── Query & stats ──────────────────────────────────────
    def get(self, label: str) -> Optional[OutlierEntry]:
        """Get an outlier entry by label."""
        return self._outliers.get(label)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the outlier pool."""
        return {
            "total_outliers": self.count,
            "outlier_labels": self.labels,
            "total_promotions": self.promotion_count,
            "recent_promotions": [
                {
                    "type": p.type,
                    "outliers": p.outlier_labels,
                    "target": p.target_speaker,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp,
                }
                for p in self._promotions[-10:]  # Last 10
            ],
            "outlier_details": {
                label: {
                    "age": entry.age,
                    "timestamp": entry.timestamp,
                    "segment_id": entry.segment_id,
                    "match_attempts": entry.match_attempts,
                    "audio_duration": entry.audio_duration,
                }
                for label, entry in self._outliers.items()
            },
            "config": {
                "prefix": self.prefix,
                "promotion_threshold": self.promotion_threshold,
                "ttl": self.ttl,
            },
        }
    
    def to_dict(self) -> Dict:
        """Serialize the full pool state for persistence."""
        return {
            "outliers": {
                label: entry.to_dict()
                for label, entry in self._outliers.items()
            },
            "next_id": self._next_id,
            "promotions": [
                {
                    "type": p.type,
                    "outlier_labels": p.outlier_labels,
                    "target_speaker": p.target_speaker,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp,
                }
                for p in self._promotions
            ],
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict,
        prefix: str = DEFAULT_OUTLIER_PREFIX,
        promotion_threshold: float = DEFAULT_OUTLIER_PROMOTION_THRESHOLD,
        ttl: float = DEFAULT_OUTLIER_TTL,
        debug: bool = False,
    ) -> "OutlierPool":
        """Deserialize from a dictionary.
        
        Parameters
        ----------
        data : Dict
            Serialized state from to_dict().
        prefix, promotion_threshold, ttl, debug
            Configuration parameters (must match original).
        
        Returns
        -------
        OutlierPool
            Restored pool instance.
        """
        pool = cls(
            prefix=prefix,
            promotion_threshold=promotion_threshold,
            ttl=ttl,
            debug=debug,
        )
        
        # Restore outliers
        for label, entry_data in data.get("outliers", {}).items():
            pool._outliers[label] = OutlierEntry.from_dict(entry_data)
        
        pool._next_id = data.get("next_id", 1)
        
        # Restore promotions
        for p_data in data.get("promotions", []):
            pool._promotions.append(PromotionEvent(
                type=p_data["type"],
                outlier_labels=p_data["outlier_labels"],
                target_speaker=p_data["target_speaker"],
                confidence=p_data["confidence"],
                timestamp=p_data["timestamp"],
            ))
        
        if pool.debug:
            console.print(
                f"{_LOGGER_PREFIX} 📂 Restored pool: "
                f"{pool.count} outliers, {pool.promotion_count} promotions"
            )
        
        return pool
    
    def reset(self) -> None:
        """Clear all outliers and promotion history."""
        self._outliers.clear()
        self._promotions.clear()
        self._next_id = 1
        
        if self.debug:
            console.print(f"{_LOGGER_PREFIX} 🔄 Pool reset")
    
    def __len__(self) -> int:
        return self.count
    
    def __contains__(self, label: str) -> bool:
        return label in self._outliers
    
    def __repr__(self) -> str:
        return (
            f"OutlierPool(count={self.count}, "
            f"promotions={self.promotion_count}, "
            f"ttl={self.ttl}s)"
        )
