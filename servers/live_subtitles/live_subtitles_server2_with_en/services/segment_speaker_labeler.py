# servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py

"""Progressive segment speaker labeling with dynamic reference maintenance."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np
import torch
import uuid
from rich.console import Console
from scipy.spatial.distance import cdist
try:
    from services.config import SAMPLE_RATE
    from services.embedding_model_factory import BaseEmbeddingModel
    from services.speaker_metrics_mixin import SpeakerMetricsMixin
    from services.speech_waves import extract_pure_speech_audio
except ImportError:
    from config import SAMPLE_RATE
    from embedding_model_factory import BaseEmbeddingModel
    from speaker_metrics_mixin import SpeakerMetricsMixin
    from speech_waves import extract_pure_speech_audio

console = Console()

# Threshold constants remain the same
DEFAULT_THRESHOLD_SAME: float = 0.75
DEFAULT_THRESHOLD_POSSIBLE: float = 0.5
DEFAULT_THRESHOLD_NEW_SPEAKER: float = 0.3
DEFAULT_MIN_SEGMENTS_FOR_REFERENCE: int = 2
DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER: int = 50
DEFAULT_TEMPORAL_SMOOTHING_WINDOW: float = 3.0
DEFAULT_TOP_K_SPEAKERS: int = 3
DEFAULT_MIN_SIMILARITY_FOR_LIST: float = 0.15
DEFAULT_CONSOLIDATION_THRESHOLD: float = 0.80
DEFAULT_MATURE_SEGMENT_COUNT: int = 5
DEFAULT_YOUNG_SEGMENT_COUNT: int = 2
DEFAULT_USE_SPEECH_WAVE_FILTERING: bool = True
DEFAULT_MIN_PROMINENCE: float = 0.05
DEFAULT_MIN_EXCURSION: float = 0.04
DEFAULT_MIN_PEAK_PROB: float = 0.9
DEFAULT_MIN_FRAMES: int = 100
DEFAULT_MIN_DURATION_SEC: float = 0.25
DEFAULT_BASELINE_THRESHOLD: float = 0.1
DEFAULT_MIN_SIMILARITY_TO_UPDATE: float = 0.25  # NEW: Minimum similarity to update centroid


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
    embeddings: List[List[float]]  # each embedding as flat list


@dataclass
class SpeakerReference:
    """Maintains reference data for a single speaker."""
    label: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_metadata: List[Dict] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    first_seen: Optional[float] = None
    last_seen: float = 0.0
    segment_count: int = 0
    
    def add_embedding(self, embedding: np.ndarray, timestamp: float,
                      segment_id: Optional[str] = None) -> None:
        """Add a new embedding and update centroid using median for robustness.
        
        Parameters
        ----------
        embedding : np.ndarray
            The embedding vector to add.
        timestamp : float
            Timestamp when this segment was processed.
        segment_id : str, optional
            Unique identifier for the segment that produced this embedding.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        self.embeddings.append(embedding)
        
        # NEW: Store segment metadata
        self.embedding_metadata.append({
            'segment_id': segment_id or f"unknown_{len(self.embeddings)}",
            'timestamp': timestamp,
            'index': len(self.embeddings) - 1,
            'added_at': timestamp
        })
        
        self.segment_count += 1
        if len(self.embeddings) >= 3:
            try:
                stacked = np.vstack(self.embeddings)
                self.centroid = np.median(stacked, axis=0, keepdims=True)
            except Exception:
                stacked = np.vstack(self.embeddings)
                self.centroid = np.mean(stacked, axis=0, keepdims=True)
        elif len(self.embeddings) == 2:
            stacked = np.vstack(self.embeddings)
            self.centroid = np.mean(stacked, axis=0, keepdims=True)
        else:
            self.centroid = embedding.copy()
        if self.first_seen is None:
            self.first_seen = timestamp
        if timestamp > self.last_seen:
            self.last_seen = timestamp

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


class SegmentSpeakerLabeler(SpeakerMetricsMixin):
    """Dynamically labels speaker segments with progressive reference building.
    
    Parameters
    ----------
    embedding_model : Inference
        Pyannote Inference instance for computing embeddings.
    threshold_same : float
        Cosine similarity threshold to classify as same speaker.
    threshold_possible : float
        Lower threshold for possible match.
    threshold_new_speaker : float
        Threshold below which a new speaker is created.
    min_segments_for_reference : int
        Minimum segments before a speaker reference is considered reliable.
    mature_segment_count : int
        Segments needed for "mature" speaker status (reliable centroid).
    young_segment_count : int
        Maximum segments considered "young" (needs re-evaluation).
    max_embeddings_per_speaker : int
        Maximum number of embeddings stored per speaker.
    temporal_smoothing_window : float
        Time window for temporal smoothing.
    top_k_speakers : int
        Number of top speakers to return in label_segments.
    min_similarity_for_list : float
        Minimum similarity to include a speaker in results.
    consolidation_threshold : float
        Threshold above which two speakers are merged.
    min_similarity_to_update : float
        Minimum similarity required to add embedding to existing speaker.
    debug : bool
        Enable debug logging.
    """
    
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        threshold_same: float = DEFAULT_THRESHOLD_SAME,
        threshold_possible: float = DEFAULT_THRESHOLD_POSSIBLE,
        threshold_new_speaker: float = DEFAULT_THRESHOLD_NEW_SPEAKER,
        min_segments_for_reference: int = DEFAULT_MIN_SEGMENTS_FOR_REFERENCE,
        mature_segment_count: int = DEFAULT_MATURE_SEGMENT_COUNT,
        young_segment_count: int = DEFAULT_YOUNG_SEGMENT_COUNT,
        max_embeddings_per_speaker: int = DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER,
        temporal_smoothing_window: float = DEFAULT_TEMPORAL_SMOOTHING_WINDOW,
        top_k_speakers: int = DEFAULT_TOP_K_SPEAKERS,
        min_similarity_for_list: float = DEFAULT_MIN_SIMILARITY_FOR_LIST,
        consolidation_threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
        min_similarity_to_update: float = DEFAULT_MIN_SIMILARITY_TO_UPDATE,  # NEW
        use_speech_wave_filtering: bool = DEFAULT_USE_SPEECH_WAVE_FILTERING,
        min_prominence: float = DEFAULT_MIN_PROMINENCE,
        min_excursion: float = DEFAULT_MIN_EXCURSION,
        min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
        min_frames: int = DEFAULT_MIN_FRAMES,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
        baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
        young_merge_threshold: float = 0.65,  # NEW: Higher threshold for young merges
        min_speaker_age_for_merge: float = 15.0,  # NEW: Min seconds before merging
        debug: bool = False,
    ):
        self.embedding_model = embedding_model
        self.threshold_same = threshold_same
        self.threshold_possible = threshold_possible
        self.threshold_new_speaker = threshold_new_speaker
        self.min_segments_for_reference = min_segments_for_reference
        self.mature_segment_count = mature_segment_count
        self.young_segment_count = young_segment_count
        self.max_embeddings_per_speaker = max_embeddings_per_speaker
        self.temporal_smoothing_window = temporal_smoothing_window
        self.top_k_speakers = top_k_speakers
        self.min_similarity_for_list = min_similarity_for_list
        self.consolidation_threshold = consolidation_threshold
        self.min_similarity_to_update = min_similarity_to_update  # NEW
        self.debug = debug
        self.use_speech_wave_filtering = use_speech_wave_filtering
        self.min_prominence = min_prominence
        self.min_excursion = min_excursion
        self.min_peak_prob = min_peak_prob
        self.min_frames = min_frames
        self.min_duration_sec = min_duration_sec
        self.baseline_threshold = baseline_threshold
        
        self._speakers: Dict[str, SpeakerReference] = {}
        self._label_history: List[Tuple[float, str]] = []
        self._next_speaker_id = 1
        self.total_segments_processed = 0
        self.total_speakers_created = 0
        
        # NEW: Track centroid contamination for debugging
        self._centroid_update_log: List[Dict] = []
        self._rejected_updates: int = 0

        self.young_merge_threshold = young_merge_threshold  # NEW
        self.min_speaker_age_for_merge = min_speaker_age_for_merge  # NEW
        self._speaker_creation_times: Dict[str, float] = {}  # NEW: Track creation times
        self._merge_history: List[Dict] = []  # NEW: Track all merges

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

        Uses the ``BaseEmbeddingModel.encode()`` interface so that any
        registered backend (pyannote, SpeechBrain, NeMo) works transparently.
        """
        try:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # ---- unified encode() call ----------------------------------------
            embedding = self.embedding_model.encode(waveform, sample_rate)
            # -------------------------------------------------------------------

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
        """Find the best matching speaker for an embedding."""
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
            label: float(sim) for label, sim in zip(speaker_labels, similarities)
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
        """Find top-K matching speakers with ALL similarities included."""
        if k is None:
            k = self.top_k_speakers
        
        if not self._speakers:
            return []
        
        speaker_labels = []
        centroids = []
        centroid_qualities = []
        segment_counts = []
        last_seens = []
        
        for label, ref in self._speakers.items():
            if ref.has_valid_centroid:
                speaker_labels.append(label)
                centroids.append(ref.centroid)
                centroid_qualities.append(ref.centroid_quality)
                segment_counts.append(ref.segment_count)
                last_seens.append(ref.last_seen)
        
        if not centroids:
            return []
        
        centroids_array = np.vstack(centroids)
        distances = cdist(embedding, centroids_array, metric="cosine")
        similarities = 1.0 - distances.flatten()
        
        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in sorted_indices[:k]:
            sim = float(similarities[idx])
            label = speaker_labels[idx]
            quality = centroid_qualities[idx]
            seg_count = segment_counts[idx]
            last_seen = last_seens[idx]
            
            # Adaptive thresholds based on centroid quality
            adaptive_same = self.threshold_same - (1.0 - quality) * 0.15
            adaptive_possible = self.threshold_possible - (1.0 - quality) * 0.10
            
            if sim < self.min_similarity_for_list and len(results) > 0:
                continue
            
            if sim >= adaptive_same:
                if seg_count >= self.min_segments_for_reference:
                    match_type = "strong_match"
                else:
                    match_type = "early_match"
            elif sim >= adaptive_possible:
                match_type = "possible_match"
            else:
                match_type = "weak_match"
            
            results.append(
                {
                    "label": label,
                    "confidence": sim,
                    "match_type": match_type,
                    "is_primary": (idx == sorted_indices[0]),
                    "segment_count": seg_count,
                    "last_seen": last_seen,
                    "centroid_quality": quality,
                }
            )
        
        return results
    
    def apply_temporal_smoothing(
        self,
        candidate_label: str,
        timestamp: float,
        similarity: float,
    ) -> str:
        """Apply temporal smoothing to prevent rapid label switching."""
        cutoff = timestamp - self.temporal_smoothing_window
        self._label_history = [(t, l) for t, l in self._label_history if t >= cutoff]
        
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
        segment_id: Optional[str] = None,  # NEW: Optional segment ID parameter
    ) -> str:
        """Create a new speaker reference.
        
        Parameters
        ----------
        embedding : np.ndarray
            The embedding vector for the first segment.
        timestamp : float
            Timestamp when this speaker was first detected.
        segment_id : str, optional
            Unique identifier for the segment that triggered this new speaker.
            If not provided, one will be generated automatically.
        
        Returns
        -------
        str
            Label of the newly created speaker (e.g., 'SPEAKER_03').
        
        Notes
        -----
        The speaker label format is 'SPEAKER_XX' where XX is a zero-padded
        incrementing number (01, 02, etc.).
        The embedding is stored along with its segment_id metadata for
        traceability.
        """
        label = f"SPEAKER_{self._next_speaker_id:02d}"
        self._next_speaker_id += 1
        self.total_speakers_created += 1
        
        # Generate segment_id if not provided
        if segment_id is None:
            segment_id = self._generate_segment_id()
        
        ref = SpeakerReference(label=label)
        ref.add_embedding(
            embedding=embedding,
            timestamp=timestamp,
            segment_id=segment_id  # NEW: Pass segment ID to add_embedding
        )
        self._speakers[label] = ref
        
        # Track creation time for merge protection
        self._speaker_creation_times[label] = timestamp
        
        if self.debug:
            console.print(
                f"[green]Created new speaker: {label} "
                f"(segment_id: {segment_id}, "  # NEW: Log segment ID
                f"segment_count={ref.segment_count}, "
                f"total_speakers={len(self._speakers)}, "
                f"next_id={self._next_speaker_id})[/]"
            )
        
        return label

    def _should_update_reference(
        self,
        label: str,
        similarity: float,
        match_type: str,
        timestamp: float,
        embedding: np.ndarray,  # FIXED: Added missing parameter
    ) -> Tuple[bool, str]:
        """Determine if embedding should update speaker reference.
        
        Validates that the embedding is sufficiently similar to the
        speaker's centroid before allowing the update. Prevents centroid
        contamination from dissimilar segments.
        
        Parameters
        ----------
        label : str
            Speaker label to potentially update.
        similarity : float
            Cosine similarity to the speaker's centroid.
        match_type : str
            Type of match (strong_match, possible_match, etc.).
        timestamp : float
            Current timestamp for logging.
        embedding : np.ndarray
            The embedding vector to validate.
        
        Returns:
            (should_update, reason)
        """
        if label not in self._speakers:
            return True, "new_speaker"
        
        ref = self._speakers[label]
        
        # Always allow strong matches
        if match_type in ("strong_match", "early_match"):
            return True, "strong_match"
        
        # Ensure embedding is 2D for distance calculations
        if embedding.ndim == 1:
            embedding_2d = embedding.reshape(1, -1)
        else:
            embedding_2d = embedding
        
        # Check minimum similarity threshold
        if similarity < self.min_similarity_to_update:
            if self.debug:
                console.print(
                    f"[red]⚠️  REJECTED update to {label}: similarity "
                    f"{similarity:.3f} < {self.min_similarity_to_update} "
                    f"(match_type: {match_type})[/]"
                )
            self._rejected_updates += 1
            return False, f"low_similarity_{similarity:.3f}"
        
        # For possible matches, check distance to centroid
        if match_type == "possible_match" and ref.has_valid_centroid:
            # Check if embedding is within acceptable range of centroid
            centroid_2d = ref.centroid.reshape(1, -1) if ref.centroid.ndim == 1 else ref.centroid
            
            distance_to_centroid = cdist(
                embedding_2d,
                centroid_2d,
                metric="cosine"
            )[0, 0]
            sim_to_centroid = 1.0 - distance_to_centroid
            
            if sim_to_centroid < self.min_similarity_to_update:
                if self.debug:
                    console.print(
                        f"[red]⚠️  REJECTED update to {label}: centroid similarity "
                        f"{sim_to_centroid:.3f} < {self.min_similarity_to_update} "
                        f"(match similarity: {similarity:.3f})[/]"
                    )
                self._rejected_updates += 1
                return False, f"centroid_divergence_{sim_to_centroid:.3f}"
            
            # Calculate how much this would shift the centroid
            if len(ref.embeddings) >= 2:
                current_embeddings = np.vstack(ref.embeddings)
                new_embeddings = np.vstack([current_embeddings, embedding_2d])
                
                if len(ref.embeddings) >= 3:
                    old_centroid = np.median(current_embeddings, axis=0)
                else:
                    old_centroid = np.mean(current_embeddings, axis=0)
                
                if len(ref.embeddings) + 1 >= 3:
                    new_centroid = np.median(new_embeddings, axis=0)
                else:
                    new_centroid = np.mean(new_embeddings, axis=0)
                
                # Ensure centroids are 2D for cdist
                old_centroid_2d = old_centroid.reshape(1, -1)
                new_centroid_2d = new_centroid.reshape(1, -1)
                
                centroid_shift = float(cdist(
                    old_centroid_2d,
                    new_centroid_2d,
                    metric="cosine"
                )[0, 0])
                
                # Reject if centroid would shift too much
                max_shift = 0.15 if ref.segment_count < 5 else 0.08
                if centroid_shift > max_shift:
                    if self.debug:
                        console.print(
                            f"[red]⚠️  REJECTED update to {label}: centroid shift "
                            f"{centroid_shift:.3f} > {max_shift:.3f} "
                            f"(segments: {ref.segment_count})[/]"
                        )
                    self._rejected_updates += 1
                    return False, f"centroid_shift_{centroid_shift:.3f}"
                
                # Log for monitoring
                if self.debug:
                    self._centroid_update_log.append({
                        "label": label,
                        "similarity": similarity,
                        "match_type": match_type,
                        "centroid_shift": centroid_shift,
                        "segment_count": ref.segment_count,
                        "timestamp": timestamp,
                    })
        
        return True, "passed"

    def update_reference(
        self,
        label: str,
        embedding: np.ndarray,
        timestamp: float,
        segment_id: Optional[str] = None,  # NEW parameter
    ) -> None:
        """Update speaker reference with new embedding.
        
        Parameters
        ----------
        label : str
            Speaker label to update.
        embedding : np.ndarray
            New embedding to add.
        timestamp : float
            Timestamp of the segment.
        segment_id : str, optional
            Unique segment identifier. Auto-generated if not provided.
        """
        if label not in self._speakers:
            self._speakers[label] = SpeakerReference(label=label)
        
        # Generate segment_id if not provided
        if segment_id is None:
            segment_id = self._generate_segment_id()
        
        ref = self._speakers[label]
        ref.add_embedding(embedding, timestamp, segment_id)  # Pass segment_id
        
        # Trim embeddings if exceeding max
        if len(ref.embeddings) > self.max_embeddings_per_speaker:
            # Also trim metadata
            ref.embeddings = ref.embeddings[-self.max_embeddings_per_speaker:]
            ref.embedding_metadata = ref.embedding_metadata[-self.max_embeddings_per_speaker:]
            
            # Recalculate centroid after trimming
            if ref.embeddings:
                if len(ref.embeddings) >= 3:
                    stacked = np.vstack(ref.embeddings)
                    ref.centroid = np.median(stacked, axis=0, keepdims=True)
                else:
                    stacked = np.vstack(ref.embeddings)
                    ref.centroid = np.mean(stacked, axis=0, keepdims=True)

    def _generate_segment_id(self, prefix: str = "segment") -> str:
        """Generate a unique segment identifier with UUID suffix.
        
        Parameters
        ----------
        prefix : str
            Prefix for the segment ID (default: "segment")
        
        Returns
        -------
        str
            Unique segment ID in format: '{prefix}_{uuid_short}'
            Example: 'segment_a3f2b1c4'
        
        Notes
        -----
        Uses UUID4 (random) for uniqueness across sessions/machines.
        Shortened to 8 characters for readability while maintaining
        extremely low collision probability (16^8 = ~4.3 billion combinations).
        """
        uuid_short = uuid.uuid4().hex[:8]
        return f"{prefix}_{uuid_short}"

    def _get_speaker_categories(self) -> Dict[str, Dict]:
        """Categorize speakers by reliability for maintenance decisions.
        
        Categories:
        - mature: segment_count >= mature_segment_count (default: 5+)
        - young: segment_count <= young_segment_count (default: 1-2)
        - orphan: young speakers inactive > temporal_smoothing_window
        - active_young: 3-4 segments OR young but recently active
        - newborn: speakers created within min_speaker_age_for_merge
        """
        now = max((ref.last_seen for ref in self._speakers.values()), default=0.0)
        mature = {}
        young = {}
        orphan = {}
        active_young = {}
        newborn = {}  # NEW category
        
        for label, ref in self._speakers.items():
            if not ref.has_valid_centroid:
                continue
            
            # NEW: Check if speaker is too new to merge
            creation_time = self._speaker_creation_times.get(label, 0.0)
            speaker_age = now - creation_time
            
            if speaker_age < self.min_speaker_age_for_merge:
                newborn[label] = ref
                continue  # Skip all other categorization for newborns
            
            if ref.segment_count >= self.mature_segment_count:
                mature[label] = ref
            elif ref.segment_count <= self.young_segment_count:
                young[label] = ref
                time_since_last_seen = now - ref.last_seen
                if time_since_last_seen > self.temporal_smoothing_window:
                    orphan[label] = ref
                else:
                    active_young[label] = ref
            else:
                active_young[label] = ref
        
        return {
            "mature": mature,
            "young": young,
            "orphan": orphan,
            "active_young": active_young,
            "newborn": newborn,  # NEW
        }

    def _should_run_maintenance(self, just_created_speaker: bool = False) -> bool:
        """Determine if speaker maintenance should run based on actual need.
        
        FIXED: More conservative triggering to prevent premature merging.
        """
        categories = self._get_speaker_categories()
        mature_count = len(categories["mature"])
        young_count = len(categories["young"])
        orphan_count = len(categories["orphan"])
        newborn_count = len(categories["newborn"])  # NEW
        total_count = len(self._speakers)
        
        if total_count < 2:
            return False
        
        # FIXED: Don't run immediately when 3rd speaker created
        # Only run if there are actual problems
        if just_created_speaker and total_count >= 3:
            # Only run if we have orphans or too many young speakers
            if orphan_count >= 2 or young_count >= 4:
                return True
            return False  # FIXED: Don't run just because 3rd speaker was created
        
        if orphan_count >= 3:
            return True
        
        # FIXED: More conservative ratio check
        if mature_count > 0 and young_count > mature_count * 2:  # Changed from 1x to 2x
            return True
        
        if young_count >= 5:
            return True
        
        # FIXED: Don't count newborns in the check
        if total_count >= 8 and young_count >= 3:
            return True
        
        return False

    def run_smart_maintenance(
        self, timestamp: float, just_created_speaker: bool = False
    ) -> Dict:
        """Run targeted speaker maintenance only when needed."""
        if not self._should_run_maintenance(just_created_speaker):
            return {
                "run": False,
                "reason": "no_need",
            }

        categories = self._get_speaker_categories()
        if self.debug:
            console.print(
                f"[dim]🔧 Maintenance triggered: "
                f"mature={len(categories['mature'])}, "
                f"young={len(categories['young'])}, "
                f"orphan={len(categories['orphan'])}[/dim]"
            )

        results = {
            "run": True,
            "orphans_removed": 0,
            "young_merged": [],
            "mature_merged": [],
            "speakers_before": len(self._speakers),
            "speakers_after": len(self._speakers),
        }

        # Clean up orphan speakers
        if len(categories["orphan"]) > 0:
            removed = self._cleanup_orphan_speakers(timestamp)
            results["orphans_removed"] = removed

        # Re-evaluate young speakers against mature ones
        if len(categories["young"]) > 0 and len(categories["mature"]) > 0:
            reeval = self.reevaluate_young_speakers(
                min_segments_for_mature=self.mature_segment_count,
                max_segments_for_young=self.young_segment_count,
                merge_threshold=0.50,
                dry_run=False,
            )
            results["young_merged"] = reeval.get("merges_performed", [])

        # Consolidate mature speakers if too many
        if len(self._speakers) > 5:
            consol = self.consolidate_speakers(
                threshold=self.consolidation_threshold,
                dry_run=False,
            )
            results["mature_merged"] = consol.get("merges_performed", [])

        results["speakers_after"] = len(self._speakers)

        if self.debug and results["speakers_before"] != results["speakers_after"]:
            console.print(
                f"[green]🔧 Maintenance: "
                f"{results['speakers_before']} → {results['speakers_after']} speakers "
                f"(removed {results['orphans_removed']} orphans, "
                f"merged {len(results['young_merged'])} young, "
                f"merged {len(results['mature_merged'])} mature)[/green]"
            )

        return results

    def _cleanup_orphan_speakers(self, current_timestamp: float) -> int:
        """Remove or merge orphan speakers.
        
        FIXED: More conservative cleanup with newborn protection.
        """
        removed = 0
        categories = self._get_speaker_categories()
        newborn_labels = set(categories.get("newborn", {}).keys())
        
        labels_to_check = list(self._speakers.keys())
        for label in labels_to_check:
            if label not in self._speakers:
                continue
            
            # FIXED: Never remove newborns
            if label in newborn_labels:
                if self.debug:
                    console.print(
                        f"[dim]🔒 Protecting newborn {label} from cleanup[/dim]"
                    )
                continue
            
            ref = self._speakers[label]
            time_since_last_seen = current_timestamp - ref.last_seen
            
            # FIXED: Only cleanup if truly orphaned (low segments AND long inactive)
            if (
                ref.segment_count <= self.young_segment_count
                and time_since_last_seen > self.temporal_smoothing_window * 3  # FIXED: 3x instead of 2x
            ):
                if ref.has_valid_centroid and len(self._speakers) > 1:
                    best_match, best_score, _ = self.find_best_match(ref.centroid)
                    # FIXED: Higher threshold for orphan merges
                    if best_match and best_match != label and best_score > 0.60:  # FIXED: 0.60 instead of 0.50
                        if self.debug:
                            console.print(
                                f"[yellow]Orphan merge: {label} → {best_match} "
                                f"(sim={best_score:.3f})[/]"
                            )
                        self.merge_speakers(best_match, label)
                        self._merge_history.append({
                            "type": "orphan_merge",
                            "source": label,
                            "target": best_match,
                            "similarity": best_score,
                            "timestamp": current_timestamp,
                        })
                        removed += 1
                    elif ref.segment_count == 1 and time_since_last_seen > 30.0:  # FIXED: 30s instead of 10s
                        if self.debug:
                            console.print(f"[dim]Orphan remove: {label} (inactive {time_since_last_seen:.1f}s)[/]")
                        del self._speakers[label]
                        self._speaker_creation_times.pop(label, None)  # Clean up creation time
                        removed += 1
        return removed

    def reevaluate_young_speakers(
        self,
        min_segments_for_mature: int = 5,
        max_segments_for_young: int = 2,
        merge_threshold: float = None,  # FIXED: Use instance default if None
        dry_run: bool = False,
    ) -> Dict:
        """Re-evaluate young speakers against mature speakers.
        
        FIXED: Higher merge threshold and newborn protection.
        
        Checks if young speakers (≤ max_segments_for_young) are actually
        the same as existing mature speakers (≥ min_segments_for_mature).
        """
        # FIXED: Use instance threshold if not specified
        if merge_threshold is None:
            merge_threshold = self.young_merge_threshold
        
        categories = self._get_speaker_categories()
        
        mature_speakers = {}
        young_speakers = {}
        
        for label, ref in self._speakers.items():
            if not ref.has_valid_centroid:
                continue
            
            # FIXED: Skip newborns
            if label in categories.get("newborn", {}):
                if self.debug:
                    console.print(
                        f"[dim]🔒 Skipping newborn {label} "
                        f"(age={self._speaker_creation_times.get(label, 0):.1f}s)[/dim]"
                    )
                continue
            
            if ref.segment_count >= min_segments_for_mature:
                mature_speakers[label] = ref
            elif ref.segment_count <= max_segments_for_young:
                young_speakers[label] = ref
        
        if not mature_speakers or not young_speakers:
            return {
                "merges_performed": [],
                "speakers_checked": len(young_speakers),
                "mature_speakers": len(mature_speakers),
                "newborn_skipped": len(categories.get("newborn", {})),  # NEW
                "dry_run": dry_run,
            }
        
        mature_labels = list(mature_speakers.keys())
        mature_centroids = np.vstack([ref.centroid for ref in mature_speakers.values()])
        
        merges_to_perform = []
        for young_label, young_ref in young_speakers.items():
            distances = cdist(young_ref.centroid, mature_centroids, metric="cosine")
            similarities = 1.0 - distances.flatten()
            best_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_idx])
            best_mature_label = mature_labels[best_idx]
            
            # FIXED: Use the higher merge threshold
            if best_similarity >= merge_threshold:
                merges_to_perform.append(
                    {
                        "young_speaker": young_label,
                        "mature_speaker": best_mature_label,
                        "similarity": round(best_similarity, 4),
                        "young_segments": young_ref.segment_count,
                        "mature_segments": mature_speakers[
                            best_mature_label
                        ].segment_count,
                    }
                )
                if self.debug:
                    console.print(
                        f"[yellow]🔍 Re-eval MERGE: {young_label} "
                        f"({young_ref.segment_count} segs) → "
                        f"{best_mature_label} "
                        f"({mature_speakers[best_mature_label].segment_count} segs) "
                        f"sim={best_similarity:.3f} (threshold={merge_threshold})[/yellow]"
                    )
            elif self.debug:
                # NEW: Log why merge was rejected
                console.print(
                    f"[dim]🔍 Re-eval KEEP: {young_label} "
                    f"({young_ref.segment_count} segs) vs "
                    f"{best_mature_label} "
                    f"sim={best_similarity:.3f} < {merge_threshold}[/dim]"
                )
        
        if not dry_run:
            for merge_info in merges_to_perform:
                self.merge_speakers(
                    merge_info["mature_speaker"], merge_info["young_speaker"]
                )
                # NEW: Log merge for debugging
                self._merge_history.append({
                    "type": "young_reeval",
                    "source": merge_info["young_speaker"],
                    "target": merge_info["mature_speaker"],
                    "similarity": merge_info["similarity"],
                    "timestamp": max(ref.last_seen for ref in self._speakers.values() if ref.label == merge_info["mature_speaker"]),
                })
        
        return {
            "merges_performed": [
                (m["young_speaker"], m["mature_speaker"], m["similarity"])
                for m in merges_to_perform
            ],
            "speakers_checked": len(young_speakers),
            "mature_speakers": len(mature_speakers),
            "newborn_skipped": len(categories.get("newborn", {})),  # NEW
            "dry_run": dry_run,
        }

    def _should_create_new_speaker(
        self,
        best_score: float,
        top_matches: List[Dict],
        context: Optional[Dict],
        embedding: np.ndarray,
    ) -> bool:
        """Determine if we should create a new speaker."""
        if len(self._speakers) == 0:
            return True

        if not top_matches:
            return True

        best_match = top_matches[0]
        match_type = best_match["match_type"]
        confidence = best_match["confidence"]

        if match_type in ("strong_match", "early_match"):
            return False

        if match_type == "possible_match":
            return False

        if confidence < self.threshold_new_speaker:
            if context and "previous_speaker" in context:
                prev_speaker = context["previous_speaker"]
                if prev_speaker and prev_speaker in self._speakers:
                    for match in top_matches:
                        if (
                            match["label"] == prev_speaker
                            and match["confidence"] >= self.threshold_possible
                        ):
                            return False
            return True

        if best_score < self.threshold_new_speaker:
            return True

        return False

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate speaker labels, keeping the highest confidence entry.
        
        When temporal smoothing or context matching changes the primary label to one
        that already appears in the alternatives list, we need to deduplicate and
        keep only the entry with the highest confidence score.
        Also ensures the primary result (is_primary=True) is always first.
        """
        if not results:
            return results

        best_by_label: Dict[str, Dict] = {}
        for entry in results:
            label = entry["label"]
            if (
                label not in best_by_label
                or entry["confidence"] > best_by_label[label]["confidence"]
            ):
                if label in best_by_label:
                    entry["is_primary"] = entry.get(
                        "is_primary", False
                    ) or best_by_label[label].get("is_primary", False)
                    entry["match_type"] = max(
                        [entry["match_type"], best_by_label[label]["match_type"]],
                        key=lambda mt: {
                            "strong_match": 5,
                            "early_match": 4,
                            "context_match": 4,
                            "possible_match": 3,
                            "weak_match": 2,
                            "weak_alternative": 2,
                            "new_speaker": 1,
                            "first_speaker": 1,
                        }.get(mt, 0),
                    )
                best_by_label[label] = entry

        deduped = list(best_by_label.values())
        deduped.sort(key=lambda x: (not x.get("is_primary", False), -x["confidence"]))
        return deduped

    def label_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        timestamp: float,
        context: Optional[Dict] = None,
        top_k: Optional[int] = None,
        segment_id: Optional[str] = None,  # NEW: Optional external segment ID
    ) -> List[Dict]:
        """Label a speech segment with multiple possible speaker identities.
        
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of the segment.
        sample_rate : int
            Sample rate of the audio.
        timestamp : float
            Timestamp of the segment in seconds.
        context : dict, optional
            Additional context like previous speaker info.
        top_k : int, optional
            Number of top matches to return.
        segment_id : str, optional
            External segment identifier. If not provided, one will be generated
            automatically using UUID. This allows callers to maintain their own
            segment tracking system.
        
        Returns
        -------
        list of dict
            List of speaker matches with confidence scores and segment_id.
        
        Notes
        -----
        NEW: Added centroid contamination protection. Embeddings with low
        similarity to existing speakers are no longer added to centroids.
        Each embedding now tracked with unique segment ID.
        """
        self.total_segments_processed += 1
        
        # Use provided segment_id or generate new one
        if segment_id is None:
            segment_id = self._generate_segment_id()
        
        if top_k is None:
            top_k = self.top_k_speakers
        
        # Extract pure speech audio
        waveform, was_filtered = self._extract_speech_audio(waveform=waveform)
        
        # Compute embedding
        embedding = self.compute_embedding(waveform, sample_rate)
        
        # Debug: log embedding shape
        if self.debug:
            console.print(
                f"[dim]Embedding shape: {embedding.shape}, "
                f"ndim: {embedding.ndim}, "
                f"segment_id: {segment_id}[/]"
            )
        
        # Find top matches
        top_matches = self.find_top_k_matches(embedding, k=top_k)
        
        if self.debug:
            console.print(
                f"[dim]Computed embedding for t={timestamp:.2f}s, "
                f"segment_id={segment_id}, "
                f"got {len(top_matches)} top matches[/]"
            )
        
        # Get actual best score
        actual_best_score = 0.0
        if len(self._speakers) > 0:
            _, actual_best_score, _ = self.find_best_match(embedding)
        
        # Determine if new speaker needed
        should_create = self._should_create_new_speaker(
            actual_best_score, top_matches, context, embedding
        )
        
        if self.debug:
            console.print(
                f"[dim]Actual best score: {actual_best_score:.4f}, "
                f"should_create_new_speaker: {should_create}[/]"
            )
        
        results = []
        seen_labels = set()
        just_created_speaker = False
        
        if should_create or not top_matches:
            # Create new speaker with segment ID
            new_label = self.create_new_speaker(
                embedding=embedding,
                timestamp=timestamp,
                segment_id=segment_id
            )
            just_created_speaker = True
            
            if self.debug:
                categories = self._get_speaker_categories()
                new_ref = self._speakers[new_label]
                console.print(
                    f"[yellow]⚠️  New speaker: {new_label} "
                    f"(segment_id: {segment_id}, "
                    f"segments: {new_ref.segment_count}, "
                    f"best sim: {actual_best_score:.3f}, "
                    f"mature: {len(categories['mature'])}, "
                    f"young: {len(categories['young'])}, "
                    f"total: {len(self._speakers)})[/yellow]"
                )
            
            results.append({
                "label": new_label,
                "confidence": 1.0,
                "match_type": "first_speaker" if not top_matches else "new_speaker",
                "is_primary": True,
                "is_new_speaker": True,
                "segment_count": 1,
                "last_seen": timestamp,
                "segment_id": segment_id,
            })
            seen_labels.add(new_label)
            
            # Add alternatives
            for match in top_matches:
                if match["label"] not in seen_labels and len(results) < top_k + 1:
                    results.append({
                        "label": match["label"],
                        "confidence": round(match["confidence"], 4),
                        "match_type": "weak_alternative",
                        "is_primary": False,
                        "is_new_speaker": False,
                        "segment_count": match["segment_count"],
                        "last_seen": match["last_seen"],
                        "segment_id": segment_id,
                    })
                    seen_labels.add(match["label"])
        
        else:
            # Match to existing speakers
            all_scores = {m["label"]: m["confidence"] for m in top_matches}
            
            for i, match in enumerate(top_matches):
                if match["label"] in seen_labels:
                    continue
                
                label = match["label"]
                confidence = match["confidence"]
                match_type = match["match_type"]
                is_primary = (i == 0)
                
                # Apply temporal smoothing for possible matches
                if is_primary and match_type == "possible_match":
                    smoothed_label = self.apply_temporal_smoothing(
                        label, timestamp, confidence
                    )
                    if smoothed_label != label and smoothed_label in all_scores:
                        smoothed_confidence = all_scores[smoothed_label]
                        if smoothed_confidence >= self.threshold_possible:
                            label = smoothed_label
                            confidence = smoothed_confidence
                
                # Apply context matching for primary
                if is_primary and context and "previous_speaker" in context:
                    prev_speaker = context["previous_speaker"]
                    if (prev_speaker and prev_speaker in all_scores
                        and prev_speaker not in seen_labels):
                        prev_sim = all_scores[prev_speaker]
                        if prev_sim >= self.threshold_possible and prev_sim > confidence:
                            label = prev_speaker
                            confidence = prev_sim
                            match_type = "context_match"
                            if self.debug:
                                console.print(
                                    f"[blue]Context match: {prev_speaker} "
                                    f"(segment_id: {segment_id}, "
                                    f"sim={prev_sim:.3f})[/blue]"
                                )
                
                results.append({
                    "label": label,
                    "confidence": round(confidence, 4),
                    "match_type": match_type,
                    "is_primary": is_primary,
                    "is_new_speaker": False,
                    "segment_count": match["segment_count"],
                    "last_seen": match["last_seen"],
                    "segment_id": segment_id,
                })
                seen_labels.add(label)
                
                if len(results) >= top_k:
                    break
            
            # Deduplicate results
            results = self._deduplicate_results(results)
            
            # Only update reference if similarity is sufficient
            primary_result = results[0]
            
            should_update, reason = self._should_update_reference(
                label=primary_result["label"],
                similarity=primary_result["confidence"],
                match_type=primary_result["match_type"],
                timestamp=timestamp,
                embedding=embedding,
            )
            
            if should_update:
                self.update_reference(
                    label=primary_result["label"],
                    embedding=embedding,
                    timestamp=timestamp,
                    segment_id=segment_id
                )
                if self.debug:
                    console.print(
                        f"[green]✓ Updated {primary_result['label']} "
                        f"(segment_id: {segment_id}, "
                        f"sim={primary_result['confidence']:.3f}, "
                        f"reason={reason})[/]"
                    )
            else:
                # If similarity too low for update, create new speaker instead
                if self.debug:
                    console.print(
                        f"[red]✗ Rejected update to {primary_result['label']}: "
                        f"segment_id: {segment_id}, "
                        f"{reason}. Creating new speaker.[/]"
                    )
                
                # Create new speaker with the same segment ID
                new_label = self.create_new_speaker(
                    embedding=embedding,
                    timestamp=timestamp,
                    segment_id=segment_id
                )
                just_created_speaker = True
                
                # Update primary result to new speaker
                primary_result["label"] = new_label
                primary_result["confidence"] = 1.0
                primary_result["match_type"] = "new_speaker"
                primary_result["is_new_speaker"] = True
                primary_result["segment_count"] = 1
                primary_result["segment_id"] = segment_id
                
                if self.debug:
                    console.print(
                        f"[yellow]⚠️  Created new speaker instead: {new_label} "
                        f"(segment_id: {segment_id})[/]"
                    )
        
        # Run maintenance
        self.run_smart_maintenance(timestamp, just_created_speaker=just_created_speaker)
        
        if self.debug:
            speakers_str = ", ".join(
                f"{r['label']}({r['confidence']:.3f})"
                for r in results[:3]
            )
            console.print(
                f"[dim]Segment {self.total_segments_processed}: "
                f"t={timestamp:.2f}s, "
                f"segment_id={segment_id}, "
                f"→ [{speakers_str}] "
                f"(primary: {results[0]['label']}, "
                f"speakers: {len(self._speakers)}, "
                f"rejected: {self._rejected_updates})[/]"
            )
        
        return results

    def label_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        timestamp: float,
        context: Optional[Dict] = None,
        segment_id: Optional[str] = None,  # NEW: Optional external segment ID
    ) -> Tuple[str, float, Dict]:
        """Label a speech segment with a single speaker identity.
        
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of the segment.
        sample_rate : int
            Sample rate of the audio.
        timestamp : float
            Timestamp of the segment in seconds.
        context : dict, optional
            Additional context like previous speaker info.
        segment_id : str, optional
            External segment identifier. If not provided, one will be generated
            automatically. Passed through to label_segments().
        
        Returns
        -------
        tuple
            (speaker_label, confidence_score, metadata_dict)
            metadata includes: timestamp, is_new_speaker, match_type, 
            all_scores, and segment_id
        """
        results = self.label_segments(
            waveform=waveform,
            sample_rate=sample_rate,
            timestamp=timestamp,
            context=context,
            top_k=1,
            segment_id=segment_id,  # NEW: Pass segment_id through
        )
        primary = results[0]
        metadata = {
            "timestamp": timestamp,
            "is_new_speaker": primary.get("is_new_speaker", False),
            "match_type": primary["match_type"],
            "all_scores": {},
            "segment_id": primary.get("segment_id"),  # NEW: Include segment_id in metadata
        }
        return primary["label"], primary["confidence"], metadata

    def consolidate_speakers(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict:
        """Consolidate similar speakers."""
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
        distances = cdist(centroids_array, centroids_array, metric="cosine")
        similarities = 1.0 - distances

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
                    merges_to_perform.append(
                        (
                            speaker_labels[i],
                            speaker_labels[j],
                            round(sim, 4),
                        )
                    )
                    already_merged.add(speaker_labels[j])

        if not dry_run:
            for label1, label2, sim in merges_to_perform:
                self.merge_speakers(label1, label2)
                if self.debug:
                    console.print(
                        f"[yellow]Consolidated: {label1} + {label2} (sim={sim:.3f})[/]"
                    )

        speakers_after = len(self._speakers)
        return {
            "merges_performed": merges_to_perform,
            "speakers_before": speakers_before,
            "speakers_after": speakers_after,
            "dry_run": dry_run,
        }

    def get_speaker_info(self, label: str) -> Optional[Dict]:
        """Get information about a specific speaker."""
        if label not in self._speakers:
            return None

        ref = self._speakers[label]
        centroid_coordinates = None
        centroid_shape = None
        if ref.centroid is not None:
            centroid_coordinates = ref.centroid.tolist()
            centroid_shape = list(ref.centroid.shape)

        return {
            "label": ref.label,
            "segment_count": ref.segment_count,
            "first_seen": ref.first_seen if ref.first_seen is not None else 0.0,
            "last_seen": ref.last_seen,
            "active_duration": ref.active_duration,
            "has_valid_centroid": ref.has_valid_centroid,
            "centroid_quality": ref.centroid_quality,
            "centroid_coordinates": centroid_coordinates,
            "centroid_shape": centroid_shape,
        }

    def get_all_speakers_info(self) -> Dict[str, Dict]:
        """Get information about all known speakers."""
        return {label: self.get_speaker_info(label) for label in self._speakers}

    def get_segments(
        self,
        label: Optional[str] = None,
    ) -> Union[SpeakerSegmentInfo, List[SpeakerSegmentInfo], None]:
        """Get segment info and embeddings for one or all speakers.

        Parameters
        ----------
        label : str, optional
            Speaker label (e.g. 'SPEAKER_01'). If provided, returns a single
            SpeakerSegmentInfo dict. If omitted, returns a list for all speakers.

        Returns
        -------
        SpeakerSegmentInfo
            If label is provided and found.
        List[SpeakerSegmentInfo]
            If label is omitted.
        None
            If label is provided but not found.
        """
        def _build(ref: SpeakerReference) -> SpeakerSegmentInfo:
            centroid_shape = list(ref.centroid.shape) if ref.centroid is not None else None
            embeddings_as_lists = [
                emb.flatten().tolist() for emb in ref.embeddings
            ]
            return SpeakerSegmentInfo(
                label=ref.label,
                segment_count=ref.segment_count,
                first_seen=ref.first_seen if ref.first_seen is not None else 0.0,
                last_seen=ref.last_seen,
                active_duration=ref.active_duration,
                has_valid_centroid=ref.has_valid_centroid,
                centroid_quality=ref.centroid_quality,
                centroid_shape=centroid_shape,
                embedding_count=len(ref.embeddings),
                embeddings=embeddings_as_lists,
            )

        if label is not None:
            if label not in self._speakers:
                console.print(f"[yellow]get_segments: label '{label}' not found[/]")
                return None
            result = _build(self._speakers[label])
            console.print(f"[dim]get_segments: returned info for {label} "
                        f"({result['embedding_count']} embeddings)[/]")
            return result

        results = [_build(ref) for ref in self._speakers.values()]
        console.print(f"[dim]get_segments: returned info for {len(results)} speakers[/]")
        return results

    def get_health_status(self) -> Dict:
        """Get current health status of the speaker labeler."""
        categories = self._get_speaker_categories()

        mature_count = len(categories["mature"])
        young_count = len(categories["young"])
        orphan_count = len(categories["orphan"])
        total_count = len(self._speakers)

        young_ratio = young_count / max(mature_count, 1)
        orphan_ratio = orphan_count / max(total_count, 1)

        alerts = []
        if young_count >= 5:
            alerts.append(f"⚠️  Too many young speakers: {young_count}")
        if orphan_count >= 3:
            alerts.append(f"⚠️  Too many orphans: {orphan_count}")
        if mature_count > 0 and young_ratio > 2.0:
            alerts.append(f"⚠️  Young/mature ratio: {young_ratio:.1f}")
        if total_count > 10:
            alerts.append(f"⚠️  Speaker count: {total_count}")

        if not alerts:
            alerts.append("✅ Healthy")

        return {
            "total_speakers": total_count,
            "mature_speakers": mature_count,
            "young_speakers": young_count,
            "orphan_speakers": orphan_count,
            "young_to_mature_ratio": round(young_ratio, 2),
            "orphan_ratio": round(orphan_ratio, 2),
            "alerts": alerts,
            "categories": {
                "mature": list(categories["mature"].keys()),
                "young": list(categories["young"].keys()),
                "orphan": list(categories["orphan"].keys()),
                "active_young": list(categories["active_young"].keys()),
            },
            "centroids": self.get_centroid_health_stats(),
        }

    def get_speaker_similarity_matrix(self) -> Dict:
        """Get pairwise similarity matrix between all speakers."""
        labels = []
        centroids = []
        segment_counts = []

        for label, ref in self._speakers.items():
            if ref.has_valid_centroid:
                labels.append(label)
                centroids.append(ref.centroid)
                segment_counts.append(ref.segment_count)

        if len(labels) < 2:
            return {
                "labels": labels,
                "similarities": [],
                "segment_counts": segment_counts,
            }

        centroids_array = np.vstack(centroids)
        distances = cdist(centroids_array, centroids_array, metric="cosine")
        similarities = (1.0 - distances).tolist()

        return {
            "labels": labels,
            "similarities": [[round(s, 4) for s in row] for row in similarities],
            "segment_counts": segment_counts,
        }

    def find_potential_merges(
        self,
        min_similarity: float = 0.50,
        min_segments_for_source: int = 1,
    ) -> List[Dict]:
        """Find all potential speaker merges above a similarity threshold."""
        labels = []
        centroids = []

        for label, ref in self._speakers.items():
            if ref.has_valid_centroid and ref.segment_count >= min_segments_for_source:
                labels.append(label)
                centroids.append(ref.centroid)

        if len(labels) < 2:
            return []

        centroids_array = np.vstack(centroids)
        distances = cdist(centroids_array, centroids_array, metric="cosine")
        similarities = 1.0 - distances

        potential_merges = []
        seen_pairs = set()

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pair_key = tuple(sorted([labels[i], labels[j]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                sim = float(similarities[i, j])
                if sim >= min_similarity:
                    ref_i = self._speakers[labels[i]]
                    ref_j = self._speakers[labels[j]]
                    potential_merges.append(
                        {
                            "speaker_1": labels[i],
                            "speaker_2": labels[j],
                            "similarity": round(sim, 4),
                            "segments_1": ref_i.segment_count,
                            "segments_2": ref_j.segment_count,
                            "total_segments": ref_i.segment_count + ref_j.segment_count,
                        }
                    )

        potential_merges.sort(key=lambda x: x["similarity"], reverse=True)
        return potential_merges

    def merge_speakers(
        self,
        label1: str,
        label2: str,
    ) -> Optional[str]:
        """Merge two speaker references.
        
        NEW: Added debug logging to track merges.
        """
        if label1 not in self._speakers or label2 not in self._speakers:
            return None
        if label1 == label2:
            return label1
        
        ref1 = self._speakers[label1]
        ref2 = self._speakers[label2]
        
        # Determine primary (keep the one with more segments)
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
        
        # Recalculate centroid
        if primary.embeddings:
            if len(primary.embeddings) >= 3:
                stacked = np.vstack(primary.embeddings)
                primary.centroid = np.median(stacked, axis=0, keepdims=True)
            else:
                stacked = np.vstack(primary.embeddings)
                primary.centroid = np.mean(stacked, axis=0, keepdims=True)
        
        # Remove secondary
        del self._speakers[secondary_label]
        self._speaker_creation_times.pop(secondary_label, None)  # Clean up
        
        # Update label history
        self._label_history = [
            (t, primary_label if l == secondary_label else l)
            for t, l in self._label_history
        ]
        
        if self.debug:
            console.print(
                f"[bold yellow]🔀 MERGED: {secondary_label} → {primary_label} "
                f"(kept: {primary.segment_count} segs, "
                f"removed: {secondary_label})[/bold yellow]"
            )
        
        return primary_label

    def _extract_speech_audio(
        self,
        waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Extract high-confidence speech portions using the labeler's configured thresholds.
        """
        if not self.use_speech_wave_filtering:
            return waveform, False
        
        # Convert tensor to numpy
        if waveform.dim() == 2:
            waveform_np = waveform.squeeze(0).cpu().numpy()
        else:
            waveform_np = waveform.cpu().numpy()
        
        # Convert float32 [-1,1] back to int16 for VAD processing
        audio_int16 = (waveform_np * 32768.0).astype(np.int16)
        
        # Call the self-contained extraction function
        pure_speech = extract_pure_speech_audio(
            audio=audio_int16,
            sampling_rate=SAMPLE_RATE,  # assumed, could be parameterized
            vad_threshold=0.3,
            min_prominence=self.min_prominence,
            min_excursion=self.min_excursion,
            min_peak_prob=self.min_peak_prob,
            min_frames=self.min_frames,
            min_duration_sec=self.min_duration_sec,
            baseline_threshold=self.baseline_threshold,
        )
        
        if pure_speech.size == 0:
            if self.debug:
                console.print("[warning]No pure speech extracted, using original[/warning]")
            return waveform, False
        
        # Convert back to float32 tensor
        waveform_float = pure_speech.astype(np.float32) / 32768.0
        result = torch.from_numpy(waveform_float)
        if waveform.dim() == 2:
            result = result.unsqueeze(0)
        
        if self.debug:
            orig_s = len(waveform_np) / SAMPLE_RATE
            filt_s = len(pure_speech) / SAMPLE_RATE
            console.print(f"[info]🎯 Speech filtered: {filt_s:.2f}s from {orig_s:.2f}s[/info]")
        
        return result, True

    def get_merge_history(self) -> List[Dict]:
        """NEW: Get history of all speaker merges for debugging."""
        return self._merge_history.copy()
    
    def get_speaker_health_report(self) -> Dict:
        """NEW: Comprehensive speaker health report with merge tracking."""
        health = self.get_health_status()
        health["merge_history"] = self._merge_history
        health["speaker_creation_times"] = {
            label: {"created_at": time, "age": max(
                ref.last_seen for ref in self._speakers.values()
            ) - time if self._speakers else 0}
            for label, time in self._speaker_creation_times.items()
            if label in self._speakers
        }
        health["missing_speaker_ids"] = self._find_missing_speaker_ids()
        return health
    
    def _find_missing_speaker_ids(self) -> List[str]:
        """NEW: Find speaker IDs that were skipped/removed."""
        existing_ids = set()
        for label in self._speakers.keys():
            # Extract number from SPEAKER_XX format
            if label.startswith("SPEAKER_"):
                try:
                    num = int(label.split("_")[1])
                    existing_ids.add(num)
                except (IndexError, ValueError):
                    pass
        
        missing = []
        for i in range(1, self._next_speaker_id):
            if i not in existing_ids:
                missing.append(f"SPEAKER_{i:02d}")
        
        return missing

    def get_centroid_health_stats(self) -> Dict:
        """NEW: Get statistics about centroid contamination prevention."""
        return {
            "total_updates_rejected": self._rejected_updates,
            "centroid_update_log": self._centroid_update_log[-10:],  # Last 10 updates
            "min_similarity_to_update": self.min_similarity_to_update,
            "total_segments_processed": self.total_segments_processed,
            "rejection_rate": (
                self._rejected_updates / max(self.total_segments_processed, 1)
            ),
        }

    def get_centroid_arrays(self) -> Dict[str, np.ndarray]:
        """Get raw centroid arrays for all speakers.
        
        Returns:
            Dict mapping speaker labels to their centroid numpy arrays
        """
        centroids = {}
        for label, ref in self._speakers.items():
            if ref.has_valid_centroid:
                centroids[label] = ref.centroid.copy()
        return centroids

    def get_centroid_stats(self) -> Dict:
        """Get comprehensive centroid statistics for visualization.
        
        Returns data suitable for PCA/t-SNE plotting or direct visualization.
        Includes per-dimension statistics and quality metrics.
        """
        centroids = self.get_centroid_arrays()
        if not centroids:
            return {"error": "No valid centroids available"}
        
        labels = list(centroids.keys())
        centroid_matrix = np.vstack([centroids[label] for label in labels])
        
        # Basic stats
        stats = {
            "labels": labels,
            "centroid_shape": list(centroid_matrix.shape),
            "embedding_dimension": centroid_matrix.shape[1],
            "total_speakers": len(labels),
            "total_segments": sum(ref.segment_count for ref in self._speakers.values()),
        }
        
        # Per-speaker details with centroid vectors flattened for frontend
        speaker_details = {}
        for i, label in enumerate(labels):
            ref = self._speakers[label]
            centroid = centroids[label]
            
            # Flatten centroid
            flat = centroid.flatten()
            
            # Compute centroid norm (magnitude)
            norm = float(np.linalg.norm(flat))
            
            # Get the strongest dimensions (for interpretability)
            top_dims = np.argsort(np.abs(flat))[-5:][::-1]  # Top 5 dimensions
            
            # Full centroid vector for frontend (all dimensions)
            centroid_vector = flat.tolist()
            
            speaker_details[label] = {
                "centroid_vector": centroid_vector[:50],  # First 50 dims (enough for visualization)
                "centroid_norm": round(norm, 4),
                "top_dimensions": [
                    {"dim": int(d), "value": round(float(flat[d]), 6)}
                    for d in top_dims
                ],
                "segment_count": ref.segment_count,
                "centroid_quality": ref.centroid_quality,
                "first_seen": ref.first_seen if ref.first_seen else 0,
                "last_seen": ref.last_seen,
                "active_duration": ref.active_duration,
                "embedding_count": len(ref.embeddings),
            }
        
        stats["speakers"] = speaker_details
        
        # If we have enough centroids, compute inter-centroid distances
        if len(labels) >= 2:
            distances = cdist(centroid_matrix, centroid_matrix, metric="cosine")
            similarities = 1.0 - distances
            
            # Full similarity matrix for heatmap
            stats["similarity_matrix"] = similarities.tolist()
            stats["distance_matrix"] = distances.tolist()
            
            # Average distance from each centroid to all others
            for i, label in enumerate(labels):
                other_distances = [distances[i, j] for j in range(len(labels)) if j != i]
                other_similarities = [similarities[i, j] for j in range(len(labels)) if j != i]
                
                nearest_idx = min(
                    (j for j in range(len(labels)) if j != i),
                    key=lambda j: distances[i, j]
                )
                
                speaker_details[label]["avg_distance_to_others"] = round(float(np.mean(other_distances)), 4)
                speaker_details[label]["avg_similarity_to_others"] = round(float(np.mean(other_similarities)), 4)
                speaker_details[label]["nearest_neighbor"] = labels[nearest_idx]
                speaker_details[label]["nearest_distance"] = round(float(distances[i, nearest_idx]), 4)
                speaker_details[label]["nearest_similarity"] = round(float(similarities[i, nearest_idx]), 4)
        
        return stats

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
        """Serialize the labeler state."""
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
            "threshold_new_speaker": self.threshold_new_speaker,
            "mature_segment_count": self.mature_segment_count,
            "young_segment_count": self.young_segment_count,
            "top_k_speakers": self.top_k_speakers,
            "consolidation_threshold": self.consolidation_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict, embedding_model) -> "SegmentSpeakerLabeler":
        """Create a labeler from serialized state."""
        labeler = cls(
            embedding_model=embedding_model,
            threshold_same=data.get("threshold_same", DEFAULT_THRESHOLD_SAME),
            threshold_possible=data.get("threshold_possible", DEFAULT_THRESHOLD_POSSIBLE),
            threshold_new_speaker=data.get("threshold_new_speaker", DEFAULT_THRESHOLD_NEW_SPEAKER),
            mature_segment_count=data.get("mature_segment_count", DEFAULT_MATURE_SEGMENT_COUNT),
            young_segment_count=data.get("young_segment_count", DEFAULT_YOUNG_SEGMENT_COUNT),
            top_k_speakers=data.get("top_k_speakers", DEFAULT_TOP_K_SPEAKERS),
            consolidation_threshold=data.get("consolidation_threshold", DEFAULT_CONSOLIDATION_THRESHOLD),
        )
        labeler._next_speaker_id = data.get("next_speaker_id", 1)
        labeler.total_segments_processed = data.get("total_segments_processed", 0)
        labeler.total_speakers_created = data.get("total_speakers_created", 0)

        for label, ref_data in data.get("speakers", {}).items():
            raw_first_seen = ref_data.get("first_seen")
            first_seen = None if raw_first_seen in (None, 0.0) else raw_first_seen
            
            ref = SpeakerReference(
                label=ref_data["label"],
                first_seen=first_seen,
                last_seen=ref_data.get("last_seen", 0.0),
                segment_count=ref_data["segment_count"],
            )
            ref.embeddings = [np.array(emb) for emb in ref_data.get("embeddings", [])]
            if ref_data.get("centroid") is not None:
                ref.centroid = np.array(ref_data["centroid"])
            labeler._speakers[label] = ref

        return labeler


if __name__ == "__main__":
    from main._main_segment_speaker_labeler import main
    main()
