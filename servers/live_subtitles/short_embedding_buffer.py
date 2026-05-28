"""Short Embedding Buffer for progressive merging of short-duration audio embeddings.

When audio segments are shorter than a minimum duration threshold, their embeddings
are less reliable. This module provides a buffering system that:
1. Temporarily stores short embeddings
2. Groups compatible short embeddings from the same speaker
3. Merges them into a single higher-quality embedding once enough audio is accumulated
4. Promotes the merged embedding to the main speaker labeling system
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from rich.console import Console

console = Console()

# Default configuration constants
DEFAULT_MIN_DURATION_SECONDS: float = 4.0
DEFAULT_SIMILARITY_THRESHOLD: float = 0.80
DEFAULT_MAX_SLOTS: int = 10
DEFAULT_SLOT_TTL_SECONDS: float = 15.0
DEFAULT_CLEANUP_INTERVAL: int = 10


@dataclass
class BufferSlot:
    """A single buffer slot holding one or more merged short embeddings.
    
    Each slot represents a potential speaker that we're accumulating
    short segments for before promoting to the main system.
    
    Attributes:
        embeddings: List of short embeddings merged into this slot
        timestamps: Timestamps of each contributing segment
        total_duration: Sum of all contributing segment durations
        last_updated: Timestamp of the most recent addition
        centroid: Weighted centroid of all merged embeddings
        segment_count: Number of segments merged into this slot
        label: Optional label assigned if slot was checked against main system
    """
    embeddings: List[np.ndarray] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    total_duration: float = 0.0
    last_updated: float = 0.0
    centroid: Optional[np.ndarray] = None
    segment_count: int = 0
    label: Optional[str] = None

    def add_embedding(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
    ) -> None:
        """Add a short embedding to this buffer slot.
        
        Args:
            embedding: Speaker embedding array
            duration: Duration of the audio segment in seconds
            timestamp: Time when the segment occurred
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        self.embeddings.append(embedding)
        self.durations.append(duration)
        self.timestamps.append(timestamp)
        self.total_duration += duration
        self.last_updated = timestamp
        self.segment_count += 1
        
        self._update_centroid()

    def _update_centroid(self) -> None:
        """Update centroid using duration-weighted average for robustness."""
        if not self.embeddings:
            self.centroid = None
            return
        
        if len(self.embeddings) == 1:
            self.centroid = self.embeddings[0].copy()
            return
        
        # Duration-weighted average
        total_weight = sum(self.durations)
        weighted_sum = np.zeros_like(self.embeddings[0])
        for emb, dur in zip(self.embeddings, self.durations):
            weighted_sum += emb * (dur / total_weight)
        
        self.centroid = weighted_sum

    def get_merged_embedding(self) -> np.ndarray:
        """Get the merged embedding representing all accumulated segments.
        
        Returns the centroid (already duration-weighted).
        """
        if self.centroid is None:
            raise ValueError("No embeddings in buffer slot")
        return self.centroid.copy()

    @property
    def is_ready(self) -> bool:
        """Check if this slot has accumulated enough audio for promotion."""
        return self.total_duration >= DEFAULT_MIN_DURATION_SECONDS

    @property
    def age(self) -> float:
        """Calculate age in seconds since last update."""
        import time
        return time.time() - self.last_updated


class ShortEmbeddingBuffer:
    """Manages a pool of buffer slots for accumulating short-duration embeddings.
    
    This class intercepts embeddings from short audio segments, groups compatible
    ones together, and promotes them to the main speaker labeling system when
    enough audio has been accumulated.
    
    Args:
        min_duration: Minimum total duration required before promotion (seconds)
        similarity_threshold: Cosine similarity threshold for merging into same slot
        max_slots: Maximum number of concurrent buffer slots
        slot_ttl: Time-to-live for inactive slots before cleanup (seconds)
        cleanup_interval: Number of add operations between cleanup checks
        debug: Enable debug logging
    """
    
    def __init__(
        self,
        min_duration: float = DEFAULT_MIN_DURATION_SECONDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_slots: int = DEFAULT_MAX_SLOTS,
        slot_ttl: float = DEFAULT_SLOT_TTL_SECONDS,
        cleanup_interval: int = DEFAULT_CLEANUP_INTERVAL,
        debug: bool = False,
    ):
        self.min_duration = min_duration
        self.similarity_threshold = similarity_threshold
        self.max_slots = max_slots
        self.slot_ttl = slot_ttl
        self.cleanup_interval = cleanup_interval
        self.debug = debug
        
        self._slots: Dict[int, BufferSlot] = {}
        self._next_slot_id: int = 1
        self._add_count: int = 0
        
        # Statistics
        self.total_promoted: int = 0
        self.total_discarded: int = 0
        self.total_merged_into_slot: int = 0
        self.total_new_slots: int = 0

    @property
    def slot_count(self) -> int:
        """Current number of active buffer slots."""
        return len(self._slots)

    @property
    def active_slots(self) -> List[Dict]:
        """Get summary of all active slots."""
        return [
            {
                "slot_id": sid,
                "total_duration": slot.total_duration,
                "segment_count": slot.segment_count,
                "last_updated": slot.last_updated,
                "is_ready": slot.is_ready,
                "label": slot.label,
            }
            for sid, slot in self._slots.items()
        ]

    def find_best_slot(
        self,
        embedding: np.ndarray,
        timestamp: float,
    ) -> Tuple[Optional[int], float]:
        """Find the best matching buffer slot for a new short embedding.
        
        Args:
            embedding: New speaker embedding to match
            timestamp: Current timestamp
            
        Returns:
            Tuple of (slot_id or None, similarity score)
        """
        if not self._slots:
            return None, 0.0
        
        best_id = None
        best_similarity = 0.0
        
        for slot_id, slot in self._slots.items():
            if slot.centroid is None:
                continue
            
            # Compute cosine similarity
            centroid = slot.centroid
            if centroid.ndim == 1:
                centroid = centroid.reshape(1, -1)
            if embedding.ndim == 1:
                embedding_2d = embedding.reshape(1, -1)
            else:
                embedding_2d = embedding
            
            # Cosine similarity = 1 - cosine distance
            from scipy.spatial.distance import cdist
            distance = cdist(embedding_2d, centroid, metric="cosine")
            similarity = 1.0 - float(distance.flatten()[0])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = slot_id
        
        return best_id, best_similarity

    def add_embedding(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
    ) -> Optional[Tuple[np.ndarray, str]]:
        """Add a short embedding to the buffer.
        
        Args:
            embedding: Speaker embedding from short segment
            duration: Duration of the audio segment in seconds
            timestamp: Time when the segment occurred
            
        Returns:
            None if the embedding was buffered (not ready yet)
            Tuple of (merged_embedding, "promoted") if a slot is ready for promotion
        """
        self._add_count += 1
        
        # Periodic cleanup
        if self._add_count % self.cleanup_interval == 0:
            self._cleanup_expired_slots(timestamp)
        
        # Find best matching slot
        best_slot_id, similarity = self.find_best_slot(embedding, timestamp)
        
        if best_slot_id is not None and similarity >= self.similarity_threshold:
            # Merge into existing slot
            slot = self._slots[best_slot_id]
            slot.add_embedding(embedding, duration, timestamp)
            self.total_merged_into_slot += 1
            
            if self.debug:
                console.print(
                    f"[dim]🔗 Buffer merge: slot {best_slot_id} "
                    f"(sim={similarity:.3f}, dur={slot.total_duration:.1f}s, "
                    f"segs={slot.segment_count})[/dim]"
                )
            
            # Check if slot is now ready for promotion
            if slot.is_ready:
                return self._promote_slot(best_slot_id)
        
        else:
            # Create new slot (evict oldest if at capacity)
            if len(self._slots) >= self.max_slots:
                self._evict_oldest_slot()
            
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            
            slot = BufferSlot()
            slot.add_embedding(embedding, duration, timestamp)
            self._slots[slot_id] = slot
            self.total_new_slots += 1
            
            if self.debug:
                console.print(
                    f"[dim]🆕 Buffer new slot: {slot_id} "
                    f"(dur={duration:.1f}s, "
                    f"total slots={len(self._slots)}, "
                    f"best sim={similarity:.3f})[/dim]"
                )
        
        return None

    def _promote_slot(self, slot_id: int) -> Tuple[np.ndarray, str]:
        """Promote a ready slot to a full embedding and remove from buffer.
        
        Args:
            slot_id: ID of the slot to promote
            
        Returns:
            Tuple of (merged_embedding, "promoted")
        """
        slot = self._slots.pop(slot_id)
        self.total_promoted += 1
        
        merged_embedding = slot.get_merged_embedding()
        
        if self.debug:
            console.print(
                f"[green]🚀 Promoted slot {slot_id}: "
                f"dur={slot.total_duration:.1f}s, "
                f"segs={slot.segment_count}, "
                f"label={slot.label or 'none'}[/green]"
            )
        
        return merged_embedding, "promoted"

    def _evict_oldest_slot(self) -> None:
        """Remove the oldest slot by last_updated time."""
        if not self._slots:
            return
        
        oldest_id = min(self._slots.keys(), key=lambda sid: self._slots[sid].last_updated)
        evicted = self._slots.pop(oldest_id)
        self.total_discarded += 1
        
        if self.debug:
            console.print(
                f"[yellow]🗑️  Evicted slot {oldest_id}: "
                f"dur={evicted.total_duration:.1f}s, "
                f"segs={evicted.segment_count}[/yellow]"
            )

    def _cleanup_expired_slots(self, current_time: float) -> int:
        """Remove slots that haven't been updated within TTL.
        
        Args:
            current_time: Current timestamp for age calculation
            
        Returns:
            Number of slots removed
        """
        expired_ids = []
        for slot_id, slot in self._slots.items():
            age = current_time - slot.last_updated
            if age > self.slot_ttl:
                expired_ids.append(slot_id)
        
        for slot_id in expired_ids:
            del self._slots[slot_id]
            self.total_discarded += 1
        
        if expired_ids and self.debug:
            console.print(
                f"[dim]🧹 Cleaned {len(expired_ids)} expired slots "
                f"(remaining: {len(self._slots)})[/dim]"
            )
        
        return len(expired_ids)

    def check_for_ready_slots(self) -> List[Tuple[np.ndarray, str]]:
        """Check all slots and promote any that are ready.
        
        Returns:
            List of (merged_embedding, "promoted") tuples for each promoted slot
        """
        promoted = []
        ready_ids = [
            sid for sid, slot in self._slots.items() if slot.is_ready
        ]
        
        for slot_id in ready_ids:
            result = self._promote_slot(slot_id)
            promoted.append(result)
        
        return promoted

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        return {
            "active_slots": len(self._slots),
            "max_slots": self.max_slots,
            "total_promoted": self.total_promoted,
            "total_discarded": self.total_discarded,
            "total_merged_into_slot": self.total_merged_into_slot,
            "total_new_slots_created": self.total_new_slots,
            "slots_ready": sum(1 for s in self._slots.values() if s.is_ready),
        }

    def clear(self) -> None:
        """Clear all buffer slots and reset statistics."""
        self._slots.clear()
        self._next_slot_id = 1
        self._add_count = 0
        self.total_promoted = 0
        self.total_discarded = 0
        self.total_merged_into_slot = 0
        self.total_new_slots = 0
        
        if self.debug:
            console.print("[yellow]ShortEmbeddingBuffer cleared[/yellow]")

    def to_dict(self) -> Dict:
        """Serialize buffer state."""
        slots_data = {}
        for slot_id, slot in self._slots.items():
            slots_data[str(slot_id)] = {
                "embeddings": [emb.tolist() for emb in slot.embeddings],
                "durations": slot.durations,
                "timestamps": slot.timestamps,
                "total_duration": slot.total_duration,
                "last_updated": slot.last_updated,
                "segment_count": slot.segment_count,
                "label": slot.label,
            }
        
        return {
            "slots": slots_data,
            "next_slot_id": self._next_slot_id,
            "total_promoted": self.total_promoted,
            "total_discarded": self.total_discarded,
            "total_merged_into_slot": self.total_merged_into_slot,
            "total_new_slots": self.total_new_slots,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        **kwargs,
    ) -> "ShortEmbeddingBuffer":
        """Create buffer from serialized state."""
        buffer = cls(**kwargs)
        buffer._next_slot_id = data.get("next_slot_id", 1)
        buffer.total_promoted = data.get("total_promoted", 0)
        buffer.total_discarded = data.get("total_discarded", 0)
        buffer.total_merged_into_slot = data.get("total_merged_into_slot", 0)
        buffer.total_new_slots = data.get("total_new_slots", 0)
        
        for slot_id_str, slot_data in data.get("slots", {}).items():
            slot_id = int(slot_id_str)
            slot = BufferSlot(
                total_duration=slot_data.get("total_duration", 0.0),
                last_updated=slot_data.get("last_updated", 0.0),
                segment_count=slot_data.get("segment_count", 0),
                label=slot_data.get("label"),
            )
            slot.embeddings = [np.array(emb) for emb in slot_data.get("embeddings", [])]
            slot.durations = slot_data.get("durations", [])
            slot.timestamps = slot_data.get("timestamps", [])
            slot._update_centroid()
            buffer._slots[slot_id] = slot
        
        return buffer
