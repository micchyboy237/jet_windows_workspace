"""Progressive segment speaker labeling with dynamic reference maintenance."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from scipy.spatial.distance import cdist
from rich.console import Console

console = Console()

DEFAULT_THRESHOLD_SAME: float = 0.75
DEFAULT_THRESHOLD_POSSIBLE: float = 0.60
DEFAULT_MIN_SEGMENTS_FOR_REFERENCE: int = 2
DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER: int = 50
DEFAULT_TEMPORAL_SMOOTHING_WINDOW: float = 3.0


@dataclass
class SpeakerReference:
    """Maintains reference data for a single speaker."""
    
    label: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    segment_count: int = 0
    
    def add_embedding(self, embedding: np.ndarray, timestamp: float) -> None:
        """Add a new embedding and update centroid progressively.
        
        Uses cumulative moving average for numerical stability.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        self.embeddings.append(embedding)
        self.segment_count += 1
        
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            # Progressive centroid update: C_new = C_old + (x - C_old) / n
            self.centroid += (embedding - self.centroid) / self.segment_count
        
        self.last_seen = timestamp
        if self.first_seen == 0.0:
            self.first_seen = timestamp
    
    @property
    def active_duration(self) -> float:
        """Total active duration of this speaker."""
        return self.last_seen - self.first_seen
    
    @property
    def has_valid_centroid(self) -> bool:
        """Check if centroid is valid."""
        return self.centroid is not None and not np.any(np.isnan(self.centroid))


class SegmentSpeakerLabeler:
    """Dynamically labels speaker segments with progressive reference building.
    
    This class maintains a running database of speaker embeddings and assigns
    labels to incoming segments based on cosine similarity matching against
    known speaker centroids.
    
    Parameters
    ----------
    embedding_model : Inference
        Pyannote Inference instance for computing embeddings.
    threshold_same : float
        Cosine similarity threshold to classify as same speaker (0.0 to 1.0).
    threshold_possible : float
        Lower threshold for possible match, triggers context check.
    min_segments_for_reference : int
        Minimum segments before a speaker reference is considered reliable.
    max_embeddings_per_speaker : int
        Maximum number of embeddings stored per speaker (FIFO).
    temporal_smoothing_window : float
        Time window (seconds) for temporal smoothing of labels.
    debug : bool
        Enable debug logging.
    
    Examples
    --------
    >>> from pyannote.audio import Inference, Model
    >>> model = Model.from_pretrained("pyannote/embedding")
    >>> inference = Inference(model, window="whole")
    >>> labeler = SegmentSpeakerLabeler(embedding_model=inference)
    >>> 
    >>> # Process a segment
    >>> label = labeler.label_segment(
    ...     waveform=audio_tensor,
    ...     sample_rate=16000,
    ...     timestamp=1.5
    ... )
    """
    
    def __init__(
        self,
        embedding_model,
        threshold_same: float = DEFAULT_THRESHOLD_SAME,
        threshold_possible: float = DEFAULT_THRESHOLD_POSSIBLE,
        min_segments_for_reference: int = DEFAULT_MIN_SEGMENTS_FOR_REFERENCE,
        max_embeddings_per_speaker: int = DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER,
        temporal_smoothing_window: float = DEFAULT_TEMPORAL_SMOOTHING_WINDOW,
        debug: bool = False,
    ):
        self.embedding_model = embedding_model
        self.threshold_same = threshold_same
        self.threshold_possible = threshold_possible
        self.min_segments_for_reference = min_segments_for_reference
        self.max_embeddings_per_speaker = max_embeddings_per_speaker
        self.temporal_smoothing_window = temporal_smoothing_window
        self.debug = debug
        
        # Speaker database: label -> SpeakerReference
        self._speakers: Dict[str, SpeakerReference] = {}
        
        # Label history for temporal smoothing: (timestamp, label)
        self._label_history: List[Tuple[float, str]] = []
        
        # Speaker counter for generating unique labels
        self._next_speaker_id = 1
        
        # Statistics
        self.total_segments_processed = 0
        self.total_speakers_created = 0
    
    @property
    def known_speakers(self) -> List[str]:
        """Return list of known speaker labels."""
        return sorted(self._speakers.keys())
    
    @property
    def speaker_count(self) -> int:
        """Number of known speakers."""
        return len(self._speakers)
    
    def compute_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute speaker embedding from waveform segment.
        
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape (channels, samples).
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Speaker embedding vector of shape (1, dimension).
        """
        try:
            # Ensure correct shape for pyannote
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            embedding = self.embedding_model({
                "waveform": waveform,
                "sample_rate": sample_rate,
            })
            
            # Convert to numpy if tensor
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
            
            # Ensure 2D
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            return embedding
            
        except Exception as e:
            if self.debug:
                console.print(f"[red]Error computing embedding: {e}[/]")
            raise
    
    def find_best_match(
        self,
        embedding: np.ndarray,
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Find the best matching speaker for an embedding.
        
        Parameters
        ----------
        embedding : np.ndarray
            Speaker embedding to match.
        
        Returns
        -------
        Tuple[Optional[str], float, Dict[str, float]]
            - Best match label (None if no match)
            - Best similarity score
            - All similarity scores per speaker
        """
        if not self._speakers:
            return None, 0.0, {}
        
        # Compute similarities against all known centroids
        speaker_labels = []
        centroids = []
        
        for label, ref in self._speakers.items():
            if ref.has_valid_centroid:
                speaker_labels.append(label)
                centroids.append(ref.centroid)
        
        if not centroids:
            return None, 0.0, {}
        
        centroids_array = np.vstack(centroids)
        distances = cdist(embedding, centroids_array, metric="cosine")
        similarities = 1.0 - distances.flatten()
        
        # Create similarity dictionary
        sim_dict = {
            label: float(sim)
            for label, sim in zip(speaker_labels, similarities)
        }
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_label = speaker_labels[best_idx]
        best_score = float(similarities[best_idx])
        
        return best_label, best_score, sim_dict
    
    def apply_temporal_smoothing(
        self,
        candidate_label: str,
        timestamp: float,
        similarity: float,
    ) -> str:
        """Apply temporal smoothing to prevent rapid label switching.
        
        Parameters
        ----------
        candidate_label : str
            Proposed label.
        timestamp : float
            Current timestamp.
        similarity : float
            Similarity score for candidate label.
        
        Returns
        -------
        str
            Smoothed label assignment.
        """
        # Clean old history entries
        cutoff = timestamp - self.temporal_smoothing_window
        self._label_history = [
            (t, l) for t, l in self._label_history
            if t >= cutoff
        ]
        
        if not self._label_history:
            self._label_history.append((timestamp, candidate_label))
            return candidate_label
        
        # Count recent labels
        recent_labels = [l for t, l in self._label_history]
        most_common = max(set(recent_labels), key=recent_labels.count)
        most_common_count = recent_labels.count(most_common)
        total_recent = len(recent_labels)
        
        # If recent history strongly favors a different label, smooth
        if (
            most_common != candidate_label
            and most_common_count > total_recent * 0.6
            and similarity < self.threshold_same + 0.05
        ):
            if self.debug:
                console.print(
                    f"[yellow]Temporal smoothing: keeping '{most_common}' "
                    f"over '{candidate_label}' (sim={similarity:.3f})[/]"
                )
            candidate_label = most_common
        
        self._label_history.append((timestamp, candidate_label))
        return candidate_label
    
    def create_new_speaker(
        self,
        embedding: np.ndarray,
        timestamp: float,
    ) -> str:
        """Create a new speaker reference.
        
        Parameters
        ----------
        embedding : np.ndarray
            Speaker embedding.
        timestamp : float
            Current timestamp.
        
        Returns
        -------
        str
            New speaker label.
        """
        label = f"SPEAKER_{self._next_speaker_id:02d}"
        self._next_speaker_id += 1
        self.total_speakers_created += 1
        
        ref = SpeakerReference(label=label)
        ref.add_embedding(embedding, timestamp)
        self._speakers[label] = ref
        
        if self.debug:
            console.print(f"[green]Created new speaker: {label}[/]")
        
        return label
    
    def update_reference(
        self,
        label: str,
        embedding: np.ndarray,
        timestamp: float,
    ) -> None:
        """Update speaker reference with new embedding.
        
        Parameters
        ----------
        label : str
            Speaker label to update.
        embedding : np.ndarray
            New embedding to add.
        timestamp : float
            Current timestamp.
        """
        if label not in self._speakers:
            self._speakers[label] = SpeakerReference(label=label)
        
        ref = self._speakers[label]
        ref.add_embedding(embedding, timestamp)
        
        # Enforce max embeddings limit (FIFO)
        if len(ref.embeddings) > self.max_embeddings_per_speaker:
            ref.embeddings = ref.embeddings[-self.max_embeddings_per_speaker:]
            # Recompute centroid from remaining embeddings
            if ref.embeddings:
                ref.centroid = np.mean(
                    np.vstack(ref.embeddings), axis=0, keepdims=True
                )
    
    def label_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        timestamp: float,
        context: Optional[Dict] = None,
    ) -> Tuple[str, float, Dict]:
        """Label a speech segment with a speaker identity.
        
        This is the main entry point for processing segments.
        
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape (channels, samples) or (samples,).
        sample_rate : int
            Sample rate of the audio.
        timestamp : float
            Timestamp of the segment in seconds.
        context : dict, optional
            Additional context information (e.g., previous speaker,
            segment duration, etc.) for improved matching.
        
        Returns
        -------
        Tuple[str, float, Dict]
            - Assigned speaker label
            - Confidence score
            - Additional metadata dictionary
        """
        self.total_segments_processed += 1
        
        # Compute embedding for this segment
        embedding = self.compute_embedding(waveform, sample_rate)
        
        # Find best match among known speakers
        best_label, best_score, all_scores = self.find_best_match(embedding)
        
        metadata = {
            "timestamp": timestamp,
            "all_scores": all_scores,
            "is_new_speaker": False,
            "match_type": "none",
        }
        
        assigned_label = None
        confidence = 0.0
        
        # Decision logic
        if best_label is None:
            # No speakers known yet
            assigned_label = self.create_new_speaker(embedding, timestamp)
            metadata["is_new_speaker"] = True
            metadata["match_type"] = "first_speaker"
            confidence = 1.0
            
        elif best_score >= self.threshold_same:
            # Strong match
            ref = self._speakers[best_label]
            if ref.segment_count >= self.min_segments_for_reference:
                # Reference is reliable
                assigned_label = best_label
                confidence = best_score
                metadata["match_type"] = "strong_match"
            else:
                # Reference still building, but treat as match
                assigned_label = best_label
                confidence = best_score * 0.9  # Slightly lower confidence
                metadata["match_type"] = "early_match"
            
        elif best_score >= self.threshold_possible:
            # Possible match - check context and temporal smoothing
            smoothed_label = self.apply_temporal_smoothing(
                best_label, timestamp, best_score
            )
            
            if context and "previous_speaker" in context:
                prev_speaker = context["previous_speaker"]
                if prev_speaker in self._speakers:
                    prev_similarity = all_scores.get(prev_speaker, 0.0)
                    if prev_similarity >= self.threshold_same - 0.05:
                        # Context suggests same speaker
                        assigned_label = prev_speaker
                        confidence = prev_similarity
                        metadata["match_type"] = "context_match"
                    else:
                        assigned_label = smoothed_label
                        confidence = best_score
                        metadata["match_type"] = "possible_match"
                else:
                    assigned_label = smoothed_label
                    confidence = best_score
                    metadata["match_type"] = "possible_match"
            else:
                assigned_label = smoothed_label
                confidence = best_score
                metadata["match_type"] = "possible_match"
            
        else:
            # No match - create new speaker
            assigned_label = self.create_new_speaker(embedding, timestamp)
            metadata["is_new_speaker"] = True
            metadata["match_type"] = "new_speaker"
            confidence = 1.0 - best_score  # Confidence in "newness"
        
        # Update reference with this embedding
        self.update_reference(assigned_label, embedding, timestamp)
        
        if self.debug:
            console.print(
                f"[dim]Segment {self.total_segments_processed}: "
                f"t={timestamp:.2f}s → {assigned_label} "
                f"(confidence: {confidence:.3f}, type: {metadata['match_type']})[/]"
            )
        
        return assigned_label, confidence, metadata
    
    def get_speaker_info(self, label: str) -> Optional[Dict]:
        """Get information about a specific speaker.
        
        Parameters
        ----------
        label : str
            Speaker label.
        
        Returns
        -------
        Optional[Dict]
            Speaker information dictionary or None if not found.
        """
        if label not in self._speakers:
            return None
        
        ref = self._speakers[label]
        return {
            "label": ref.label,
            "segment_count": ref.segment_count,
            "first_seen": ref.first_seen,
            "last_seen": ref.last_seen,
            "active_duration": ref.active_duration,
            "has_valid_centroid": ref.has_valid_centroid,
        }
    
    def get_all_speakers_info(self) -> Dict[str, Dict]:
        """Get information about all known speakers.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping speaker labels to their information.
        """
        return {
            label: self.get_speaker_info(label)
            for label in self._speakers
        }
    
    def merge_speakers(
        self,
        label1: str,
        label2: str,
    ) -> Optional[str]:
        """Merge two speaker references.
        
        Parameters
        ----------
        label1 : str
            First speaker label to merge.
        label2 : str
            Second speaker label to merge.
        
        Returns
        -------
        Optional[str]
            The merged speaker label, or None if merge failed.
        """
        if label1 not in self._speakers or label2 not in self._speakers:
            return None
        
        ref1 = self._speakers[label1]
        ref2 = self._speakers[label2]
        
        # Merge into the one with more segments
        if ref1.segment_count >= ref2.segment_count:
            primary, secondary = ref1, ref2
            primary_label = label1
        else:
            primary, secondary = ref2, ref1
            primary_label = label2
        
        # Add all embeddings from secondary to primary
        for emb in secondary.embeddings:
            primary.embeddings.append(emb)
        
        primary.segment_count += secondary.segment_count
        primary.last_seen = max(primary.last_seen, secondary.last_seen)
        primary.first_seen = min(primary.first_seen, secondary.first_seen)
        
        # Recompute centroid
        if primary.embeddings:
            primary.centroid = np.mean(
                np.vstack(primary.embeddings), axis=0, keepdims=True
            )
        
        # Remove secondary
        del self._speakers[label1 if primary_label == label2 else label2]
        
        if self.debug:
            console.print(
                f"[yellow]Merged speakers: {label1} + {label2} → {primary_label}[/]"
            )
        
        return primary_label
    
    def reset(self) -> None:
        """Reset the labeler to initial state."""
        self._speakers.clear()
        self._label_history.clear()
        self._next_speaker_id = 1
        self.total_segments_processed = 0
        self.total_speakers_created = 0
        
        if self.debug:
            console.print("[yellow]SegmentSpeakerLabeler reset[/]")
    
    def to_dict(self) -> Dict:
        """Serialize the labeler state to a dictionary.
        
        Returns
        -------
        Dict
            Serializable state dictionary.
        """
        speakers_data = {}
        for label, ref in self._speakers.items():
            speakers_data[label] = {
                "label": ref.label,
                "embeddings": [emb.tolist() for emb in ref.embeddings],
                "centroid": ref.centroid.tolist() if ref.centroid is not None else None,
                "first_seen": ref.first_seen,
                "last_seen": ref.last_seen,
                "segment_count": ref.segment_count,
            }
        
        return {
            "speakers": speakers_data,
            "next_speaker_id": self._next_speaker_id,
            "total_segments_processed": self.total_segments_processed,
            "total_speakers_created": self.total_speakers_created,
            "threshold_same": self.threshold_same,
            "threshold_possible": self.threshold_possible,
        }
    
    @classmethod
    def from_dict(cls, data: Dict, embedding_model) -> "SegmentSpeakerLabeler":
        """Create a labeler from a serialized state dictionary.
        
        Parameters
        ----------
        data : Dict
            Serialized state from to_dict().
        embedding_model : Inference
            Pyannote Inference instance.
        
        Returns
        -------
        SegmentSpeakerLabeler
            Reconstructed labeler instance.
        """
        labeler = cls(
            embedding_model=embedding_model,
            threshold_same=data.get("threshold_same", DEFAULT_THRESHOLD_SAME),
            threshold_possible=data.get("threshold_possible", DEFAULT_THRESHOLD_POSSIBLE),
        )
        
        labeler._next_speaker_id = data.get("next_speaker_id", 1)
        labeler.total_segments_processed = data.get("total_segments_processed", 0)
        labeler.total_speakers_created = data.get("total_speakers_created", 0)
        
        for label, ref_data in data.get("speakers", {}).items():
            ref = SpeakerReference(
                label=ref_data["label"],
                first_seen=ref_data["first_seen"],
                last_seen=ref_data["last_seen"],
                segment_count=ref_data["segment_count"],
            )
            ref.embeddings = [np.array(emb) for emb in ref_data["embeddings"]]
            if ref_data["centroid"] is not None:
                ref.centroid = np.array(ref_data["centroid"])
            labeler._speakers[label] = ref
        
        return labeler
