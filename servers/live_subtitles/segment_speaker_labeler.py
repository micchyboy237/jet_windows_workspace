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
DEFAULT_TOP_K_SPEAKERS: int = 3
DEFAULT_MIN_SIMILARITY_FOR_LIST: float = 0.40
DEFAULT_CONSOLIDATION_THRESHOLD: float = 0.85


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
    top_k_speakers : int
        Number of top speakers to return in label_segments.
    min_similarity_for_list : float
        Minimum similarity threshold to include a speaker in results list.
    consolidation_threshold : float
        Threshold above which two speakers are automatically merged.
    debug : bool
        Enable debug logging.
    
    Examples
    --------
    >>> from pyannote.audio import Inference, Model
    >>> model = Model.from_pretrained("pyannote/embedding")
    >>> inference = Inference(model, window="whole")
    >>> labeler = SegmentSpeakerLabeler(embedding_model=inference)
    >>> 
    >>> # Process a segment with multiple possible speakers
    >>> results = labeler.label_segments(
    ...     waveform=audio_tensor,
    ...     sample_rate=16000,
    ...     timestamp=1.5
    ... )
    >>> for r in results:
    ...     print(f"{r['label']}: {r['confidence']:.3f}")
    """
    
    def __init__(
        self,
        embedding_model,
        threshold_same: float = DEFAULT_THRESHOLD_SAME,
        threshold_possible: float = DEFAULT_THRESHOLD_POSSIBLE,
        min_segments_for_reference: int = DEFAULT_MIN_SEGMENTS_FOR_REFERENCE,
        max_embeddings_per_speaker: int = DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER,
        temporal_smoothing_window: float = DEFAULT_TEMPORAL_SMOOTHING_WINDOW,
        top_k_speakers: int = DEFAULT_TOP_K_SPEAKERS,
        min_similarity_for_list: float = DEFAULT_MIN_SIMILARITY_FOR_LIST,
        consolidation_threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
        debug: bool = False,
    ):
        self.embedding_model = embedding_model
        self.threshold_same = threshold_same
        self.threshold_possible = threshold_possible
        self.min_segments_for_reference = min_segments_for_reference
        self.max_embeddings_per_speaker = max_embeddings_per_speaker
        self.temporal_smoothing_window = temporal_smoothing_window
        self.top_k_speakers = top_k_speakers
        self.min_similarity_for_list = min_similarity_for_list
        self.consolidation_threshold = consolidation_threshold
        self.debug = debug
        
        self._speakers: Dict[str, SpeakerReference] = {}
        self._label_history: List[Tuple[float, str]] = []
        self._next_speaker_id = 1
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
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            embedding = self.embedding_model({
                "waveform": waveform,
                "sample_rate": sample_rate,
            })
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
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

        sim_dict = {
            label: float(sim)
            for label, sim in zip(speaker_labels, similarities)
        }

        best_idx = np.argmax(similarities)
        best_label = speaker_labels[best_idx]
        best_score = float(similarities[best_idx])

        return best_label, best_score, sim_dict

    def find_top_k_matches(
        self,
        embedding: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Dict]:
        """Find top-K matching speakers for an embedding.
        
        Parameters
        ----------
        embedding : np.ndarray
            Speaker embedding to match.
        k : int, optional
            Number of top matches to return. Defaults to self.top_k_speakers.
            
        Returns
        -------
        List[Dict]
            List of speaker results sorted by similarity (descending).
            Each dict contains: label, confidence, match_type, is_primary
        """
        if k is None:
            k = self.top_k_speakers
        
        if not self._speakers:
            return []
        
        speaker_labels = []
        centroids = []
        for label, ref in self._speakers.items():
            if ref.has_valid_centroid:
                speaker_labels.append(label)
                centroids.append(ref.centroid)
        
        if not centroids:
            return []
        
        centroids_array = np.vstack(centroids)
        distances = cdist(embedding, centroids_array, metric="cosine")
        similarities = 1.0 - distances.flatten()
        
        # Sort by similarity descending
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices[:k]:
            sim = float(similarities[idx])
            if sim < self.min_similarity_for_list:
                continue
            
            label = speaker_labels[idx]
            ref = self._speakers[label]
            
            # Determine match type based on similarity
            if sim >= self.threshold_same:
                if ref.segment_count >= self.min_segments_for_reference:
                    match_type = "strong_match"
                else:
                    match_type = "early_match"
            elif sim >= self.threshold_possible:
                match_type = "possible_match"
            else:
                match_type = "weak_match"
            
            results.append({
                "label": label,
                "confidence": sim,
                "match_type": match_type,
                "is_primary": (idx == sorted_indices[0]),
                "segment_count": ref.segment_count,
                "last_seen": ref.last_seen,
            })
        
        # If no speakers or all below min_similarity, create new speaker
        if not results:
            return []
        
        return results

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
        cutoff = timestamp - self.temporal_smoothing_window
        self._label_history = [
            (t, l) for t, l in self._label_history
            if t >= cutoff
        ]

        if not self._label_history:
            self._label_history.append((timestamp, candidate_label))
            return candidate_label

        recent_labels = [l for t, l in self._label_history]
        most_common = max(set(recent_labels), key=recent_labels.count)
        most_common_count = recent_labels.count(most_common)
        total_recent = len(recent_labels)

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

        # Maintain max embeddings limit (FIFO)
        if len(ref.embeddings) > self.max_embeddings_per_speaker:
            ref.embeddings = ref.embeddings[-self.max_embeddings_per_speaker:]
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
        
        This is the main entry point for processing segments. Returns a single
        best speaker label. For multiple possible speakers, use label_segments().
        
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

        embedding = self.compute_embedding(waveform, sample_rate)
        best_label, best_score, all_scores = self.find_best_match(embedding)

        metadata = {
            "timestamp": timestamp,
            "all_scores": all_scores,
            "is_new_speaker": False,
            "match_type": "none",
        }

        assigned_label = None
        confidence = 0.0

        if best_label is None:
            # No speakers exist yet
            assigned_label = self.create_new_speaker(embedding, timestamp)
            metadata["is_new_speaker"] = True
            metadata["match_type"] = "first_speaker"
            confidence = 1.0

        elif best_score >= self.threshold_same:
            # Strong match
            ref = self._speakers[best_label]
            if ref.segment_count >= self.min_segments_for_reference:
                assigned_label = best_label
                confidence = best_score
                metadata["match_type"] = "strong_match"
            else:
                assigned_label = best_label
                confidence = best_score * 0.9
                metadata["match_type"] = "early_match"

        elif best_score >= self.threshold_possible:
            # Possible match - apply temporal smoothing and context
            smoothed_label = self.apply_temporal_smoothing(
                best_label, timestamp, best_score
            )

            if context and "previous_speaker" in context:
                prev_speaker = context["previous_speaker"]
                if prev_speaker in self._speakers:
                    prev_similarity = all_scores.get(prev_speaker, 0.0)
                    if prev_similarity >= self.threshold_same - 0.05:
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
            confidence = 1.0 - best_score

        # Update the assigned speaker's reference
        self.update_reference(assigned_label, embedding, timestamp)

        if self.debug:
            console.print(
                f"[dim]Segment {self.total_segments_processed}: "
                f"t={timestamp:.2f}s → {assigned_label} "
                f"(confidence: {confidence:.3f}, type: {metadata['match_type']})[/]"
            )

        return assigned_label, confidence, metadata

    def label_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        timestamp: float,
        context: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Label a speech segment with multiple possible speaker identities.
        
        This method returns a list of potential speakers for segments that may
        contain multiple speakers or ambiguous audio. Each result includes
        confidence scores and metadata.
        
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape (channels, samples) or (samples,).
        sample_rate : int
            Sample rate of the audio.
        timestamp : float
            Timestamp of the segment in seconds.
        context : dict, optional
            Additional context information for improved matching.
        top_k : int, optional
            Maximum number of speaker results to return.
            Defaults to self.top_k_speakers.
            
        Returns
        -------
        List[Dict]
            List of speaker results sorted by confidence (descending).
            Each dict contains:
            - label: Speaker label
            - confidence: Similarity score
            - match_type: Type of match (strong_match, possible_match, etc.)
            - is_primary: Whether this is the best match
            - is_new_speaker: Whether this is a newly created speaker
            - segment_count: Number of segments for this speaker
            - last_seen: Last timestamp this speaker was active
        """
        self.total_segments_processed += 1
        
        if top_k is None:
            top_k = self.top_k_speakers
        
        embedding = self.compute_embedding(waveform, sample_rate)
        
        # Get top-K matches
        top_matches = self.find_top_k_matches(embedding, k=top_k)
        
        results = []
        
        if not top_matches:
            # No existing speakers or no matches above threshold
            # Create a new speaker
            new_label = self.create_new_speaker(embedding, timestamp)
            self.update_reference(new_label, embedding, timestamp)
            
            results.append({
                "label": new_label,
                "confidence": 1.0,
                "match_type": "first_speaker",
                "is_primary": True,
                "is_new_speaker": True,
                "segment_count": 1,
                "last_seen": timestamp,
            })
            
            return results
        
        # Process top matches
        primary_label = top_matches[0]["label"]
        
        for i, match in enumerate(top_matches):
            label = match["label"]
            confidence = match["confidence"]
            match_type = match["match_type"]
            
            # Apply temporal smoothing for the primary match
            if i == 0 and match_type == "possible_match":
                smoothed_label = self.apply_temporal_smoothing(
                    label, timestamp, confidence
                )
                if smoothed_label != label:
                    # Recalculate confidence for smoothed label if different
                    all_scores = {
                        m["label"]: m["confidence"] 
                        for m in top_matches
                    }
                    label = smoothed_label
                    confidence = all_scores.get(smoothed_label, confidence)
            
            # Context-based adjustment for primary speaker
            if i == 0 and context and "previous_speaker" in context:
                prev_speaker = context["previous_speaker"]
                all_scores = {m["label"]: m["confidence"] for m in top_matches}
                
                if prev_speaker and prev_speaker in all_scores:
                    prev_sim = all_scores[prev_speaker]
                    if prev_sim >= self.threshold_same - 0.05:
                        label = prev_speaker
                        confidence = prev_sim
                        match_type = "context_match"
            
            results.append({
                "label": label,
                "confidence": round(confidence, 4),
                "match_type": match_type,
                "is_primary": (i == 0),
                "is_new_speaker": False,
                "segment_count": match["segment_count"],
                "last_seen": match["last_seen"],
            })
        
        # Update reference for the primary speaker
        primary_result = results[0]
        self.update_reference(primary_result["label"], embedding, timestamp)
        
        if self.debug:
            speakers_str = ", ".join(
                f"{r['label']}({r['confidence']:.3f})" 
                for r in results[:3]
            )
            console.print(
                f"[dim]Segment {self.total_segments_processed}: "
                f"t={timestamp:.2f}s → [{speakers_str}] "
                f"(primary: {primary_result['label']})[/]"
            )
        
        return results

    def consolidate_speakers(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict:
        """Consolidate similar speakers by merging those with high similarity.
        
        This method compares all speaker centroids and merges speakers whose
        cosine similarity exceeds the consolidation threshold. This is useful
        for cleaning up speaker fragmentation that can occur over long sessions.
        
        Parameters
        ----------
        threshold : float, optional
            Similarity threshold above which speakers are merged.
            Defaults to self.consolidation_threshold.
        dry_run : bool
            If True, returns proposed merges without executing them.
            
        Returns
        -------
        Dict
            Results dictionary containing:
            - merges_performed: List of (label1, label2, similarity) tuples
            - speakers_before: Number of speakers before consolidation
            - speakers_after: Number of speakers after consolidation
            - dry_run: Whether this was a dry run
        """
        if threshold is None:
            threshold = self.consolidation_threshold
        
        speakers_before = len(self._speakers)
        
        if speakers_before < 2:
            return {
                "merges_performed": [],
                "speakers_before": speakers_before,
                "speakers_after": speakers_before,
                "dry_run": dry_run,
            }
        
        # Get all valid centroids
        speaker_labels = []
        centroids = []
        for label, ref in self._speakers.items():
            if ref.has_valid_centroid:
                speaker_labels.append(label)
                centroids.append(ref.centroid)
        
        if len(centroids) < 2:
            return {
                "merges_performed": [],
                "speakers_before": speakers_before,
                "speakers_after": speakers_before,
                "dry_run": dry_run,
            }
        
        centroids_array = np.vstack(centroids)
        
        # Compute pairwise similarities
        distances = cdist(centroids_array, centroids_array, metric="cosine")
        similarities = 1.0 - distances
        
        # Find merges (upper triangle only, excluding diagonal)
        merges_to_perform = []
        already_merged = set()
        
        for i in range(len(speaker_labels)):
            if speaker_labels[i] in already_merged:
                continue
            for j in range(i + 1, len(speaker_labels)):
                if speaker_labels[j] in already_merged:
                    continue
                sim = float(similarities[i, j])
                if sim >= threshold:
                    merges_to_perform.append((
                        speaker_labels[i],
                        speaker_labels[j],
                        round(sim, 4),
                    ))
                    already_merged.add(speaker_labels[j])
        
        # Execute merges if not a dry run
        if not dry_run:
            for label1, label2, sim in merges_to_perform:
                self.merge_speakers(label1, label2)
                if self.debug:
                    console.print(
                        f"[yellow]Consolidated: {label1} + {label2} "
                        f"(similarity: {sim:.3f})[/]"
                    )
        
        speakers_after = len(self._speakers)
        
        return {
            "merges_performed": merges_to_perform,
            "speakers_before": speakers_before,
            "speakers_after": speakers_after,
            "dry_run": dry_run,
        }

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
        
        if label1 == label2:
            return label1
        
        ref1 = self._speakers[label1]
        ref2 = self._speakers[label2]
        
        # Keep the speaker with more segments as primary
        if ref1.segment_count >= ref2.segment_count:
            primary, secondary = ref1, ref2
            primary_label = label1
            secondary_label = label2
        else:
            primary, secondary = ref2, ref1
            primary_label = label2
            secondary_label = label1
        
        # Merge embeddings
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
        
        # Remove secondary speaker
        del self._speakers[secondary_label]
        
        # Update label history
        self._label_history = [
            (t, primary_label if l == secondary_label else l)
            for t, l in self._label_history
        ]
        
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
            "top_k_speakers": self.top_k_speakers,
            "consolidation_threshold": self.consolidation_threshold,
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
            top_k_speakers=data.get("top_k_speakers", DEFAULT_TOP_K_SPEAKERS),
            consolidation_threshold=data.get(
                "consolidation_threshold", DEFAULT_CONSOLIDATION_THRESHOLD
            ),
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


if __name__ == "__main__":
    from _main_segment_speaker_labeler import main
    main()
