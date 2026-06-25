# servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py

"""Progressive segment speaker labeling with dynamic reference maintenance."""
import uuid
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from rich.console import Console
from scipy.spatial.distance import cdist
try:
    from services.audio_config import SAMPLE_RATE
    from services.embedding_model_factory import BaseEmbeddingModel, EmbeddingThresholdProvider
    from services.speaker_metrics_mixin import SpeakerMetricsMixin
    from services.audio_tagger import AudioTagger
    from services.speaker_labeler_utils.speaker_reference import SpeakerReference, SpeakerSegmentInfo
    from services.speaker_labeler_utils.segment_types import SegmentMatch, SegmentGroup, SegmentGroupsResult
    from services.speaker_labeler_utils.outlier_pool import (
        OutlierPool,
        OutlierEntry,
        OutlierMatch,
        DEFAULT_OUTLIER_PREFIX,
        # DEFAULT_OUTLIER_PROMOTION_THRESHOLD,
        # DEFAULT_OUTLIER_TTL,
        DEFAULT_OUTLIER_MAX_COUNT,
    )
    from services.speaker_labeler_utils.segment_speaker_labeler_defaults import (
        DEFAULT_THRESHOLD_SAME,
        DEFAULT_THRESHOLD_POSSIBLE,
        DEFAULT_THRESHOLD_NEW_SPEAKER,
        DEFAULT_MIN_SEGMENTS_FOR_REFERENCE,
        DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER,
        DEFAULT_TEMPORAL_SMOOTHING_WINDOW,
        DEFAULT_TOP_K_SPEAKERS,
        DEFAULT_MIN_SIMILARITY_FOR_LIST,
        DEFAULT_CONSOLIDATION_THRESHOLD,
        DEFAULT_MATURE_SEGMENT_COUNT,
        DEFAULT_YOUNG_SEGMENT_COUNT,
        DEFAULT_USE_SPEECH_WAVE_FILTERING,
        DEFAULT_MIN_PROMINENCE,
        DEFAULT_MIN_EXCURSION,
        DEFAULT_MIN_PEAK_PROB,
        DEFAULT_MIN_FRAMES,
        DEFAULT_MIN_DURATION_SEC,
        DEFAULT_BASELINE_THRESHOLD,
        DEFAULT_MIN_SIMILARITY_TO_UPDATE,
    )
    from services.speaker_labeler_utils.speaker_maintenance import SpeakerMaintenance
    from services.speaker_labeler_utils.outlier_orchestrator import OutlierOrchestrator
    from services.speaker_labeler_utils.speaker_labeler_serializer import SpeakerLabelerSerializer
    from services.speech_waves import extract_pure_speech_audio
except ImportError:
    from audio_config import SAMPLE_RATE
    from embedding_model_factory import BaseEmbeddingModel, EmbeddingThresholdProvider
    from speaker_metrics_mixin import SpeakerMetricsMixin
    from audio_tagger import AudioTagger
    from speaker_labeler_utils.speaker_reference import SpeakerReference, SpeakerSegmentInfo
    from speaker_labeler_utils.segment_types import SegmentMatch, SegmentGroup, SegmentGroupsResult
    from speaker_labeler_utils.outlier_pool import (
        OutlierPool,
        OutlierEntry,
        OutlierMatch,
        DEFAULT_OUTLIER_PREFIX,
        # DEFAULT_OUTLIER_PROMOTION_THRESHOLD,
        # DEFAULT_OUTLIER_TTL,
        DEFAULT_OUTLIER_MAX_COUNT,
    )
    from speaker_labeler_utils.segment_speaker_labeler_defaults import (
        DEFAULT_THRESHOLD_SAME,
        DEFAULT_THRESHOLD_POSSIBLE,
        DEFAULT_THRESHOLD_NEW_SPEAKER,
        DEFAULT_MIN_SEGMENTS_FOR_REFERENCE,
        DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER,
        DEFAULT_TEMPORAL_SMOOTHING_WINDOW,
        DEFAULT_TOP_K_SPEAKERS,
        DEFAULT_MIN_SIMILARITY_FOR_LIST,
        DEFAULT_CONSOLIDATION_THRESHOLD,
        DEFAULT_MATURE_SEGMENT_COUNT,
        DEFAULT_YOUNG_SEGMENT_COUNT,
        DEFAULT_USE_SPEECH_WAVE_FILTERING,
        DEFAULT_MIN_PROMINENCE,
        DEFAULT_MIN_EXCURSION,
        DEFAULT_MIN_PEAK_PROB,
        DEFAULT_MIN_FRAMES,
        DEFAULT_MIN_DURATION_SEC,
        DEFAULT_BASELINE_THRESHOLD,
        DEFAULT_MIN_SIMILARITY_TO_UPDATE,
    )
    from speaker_labeler_utils.speaker_maintenance import SpeakerMaintenance
    from speaker_labeler_utils.outlier_orchestrator import OutlierOrchestrator
    from speaker_labeler_utils.speaker_labeler_serializer import SpeakerLabelerSerializer
    from speech_waves import extract_pure_speech_audio

console = Console()

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
        threshold_same: Optional[float] = DEFAULT_THRESHOLD_SAME,
        threshold_possible: Optional[float] = DEFAULT_THRESHOLD_POSSIBLE,
        threshold_new_speaker: Optional[float] = DEFAULT_THRESHOLD_NEW_SPEAKER,
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
        young_merge_threshold: float = 0.65,  # Higher threshold for young merges
        min_speaker_age_for_merge: float = 15.0,  # Min seconds before merging
        audio_tagger: AudioTagger | None = None,  # NEW: Enable pure speech extraction

        # NEW: Outlier management
        use_outlier_buffer: bool = True,
        outlier_pool: Optional[OutlierPool] = None,  # Can inject custom pool
        # outlier_promotion_threshold: float = DEFAULT_OUTLIER_PROMOTION_THRESHOLD,
        # outlier_ttl: float = DEFAULT_OUTLIER_TTL,

        outlier_promotion_threshold: Optional[float] = None,  # None = use model default
        outlier_max_count: int = DEFAULT_OUTLIER_MAX_COUNT,  # NEW: replaces TTL with max count

        debug: bool = False,
    ):
        self.embedding_model = embedding_model
        
        # Resolve thresholds (now includes promotion)
        resolved = EmbeddingThresholdProvider.resolve_thresholds(
            model_type=embedding_model.model_type,
            threshold_same=threshold_same,
            threshold_possible=threshold_possible,
            threshold_new_speaker=threshold_new_speaker,
            threshold_promotion=outlier_promotion_threshold,  # NEW
        )
        self.threshold_same = resolved.same
        self.threshold_possible = resolved.possible
        self.threshold_new_speaker = resolved.new_speaker
        self.threshold_promotion = resolved.promotion  # NEW

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

        self.audio_tagger = audio_tagger

        # NEW: Outlier pool integration
        self.use_outlier_buffer = use_outlier_buffer
        # Outlier pool now uses model-specific promotion threshold and max count
        if outlier_pool is not None:
            self.outlier_pool = outlier_pool
        else:
            self.outlier_pool = OutlierPool(
                prefix=DEFAULT_OUTLIER_PREFIX,
                promotion_threshold=self.threshold_promotion,  # MODEL-SPECIFIC
                max_count=outlier_max_count,  # NEW: replaces TTL
                debug=debug,
            )

        # NEW: Internal segment storage
        self._segment_groups: List[Dict] = []
        """Internal storage for all processed segments with their matches."""
        
        self._labels_finalized: bool = False
        """Whether finalize_labels() has been called."""

        self._maintenance = SpeakerMaintenance(
            labeler=self,
            young_merge_threshold=young_merge_threshold,
            min_speaker_age_for_merge=min_speaker_age_for_merge,
            debug=debug,
        )
        self._outlier_orchestrator = OutlierOrchestrator(
            labeler=self,
            debug=debug,
        )
        self._serializer = SpeakerLabelerSerializer(labeler=self)

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
        """Compute speaker embedding from waveform segment."""
        try:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            embedding = self.embedding_model.encode(waveform, sample_rate)
            
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
            
            # Ensure 2D shape (1, dim)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                embedding = embedding.reshape(1, -1)
            
            # Validate shape
            if embedding.ndim != 2:
                raise ValueError(f"Expected 2D embedding, got shape {embedding.shape}")
            
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
        
        # FIX: Ensure similarities length matches speakers count
        if len(similarities) != len(speaker_labels):
            if self.debug:
                console.print(
                    f"[red]WARNING: Mismatch in find_top_k_matches - "
                    f"similarities: {len(similarities)}, "
                    f"speaker_labels: {len(speaker_labels)}. "
                    f"embedding shape: {embedding.shape}, "
                    f"centroids_array shape: {centroids_array.shape}[/]"
                )
            # Take only the first N similarities where N = number of speakers
            similarities = similarities[:len(speaker_labels)]
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Limit k to actual number of speakers
        k = min(k, len(speaker_labels))
        
        results = []
        for idx in sorted_indices[:k]:
            # Safety check: ensure idx is within bounds
            if idx >= len(speaker_labels):
                if self.debug:
                    console.print(
                        f"[red]WARNING: Index {idx} out of bounds for "
                        f"speaker_labels (len={len(speaker_labels)}). Skipping.[/]"
                    )
                continue
                
            sim = float(similarities[idx])
            label = speaker_labels[idx]
            quality = centroid_qualities[idx]
            seg_count = segment_counts[idx]
            last_seen = last_seens[idx]
            
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
            
            results.append({
                "label": label,
                "confidence": sim,
                "match_type": match_type,
                "is_primary": (len(results) == 0),  # FIX: First result is primary
                "segment_count": seg_count,
                "last_seen": last_seen,
                "centroid_quality": quality,
            })
        
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
        segment_id: Optional[str] = None,
        audio_duration: float = 0.0,
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
            segment_id=segment_id,
            audio_duration=audio_duration,
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
        segment_id: Optional[str] = None,
        audio_duration: float = 0.0,  # NEW parameter
        match_type: str = "strong_match",      # ← new param
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
        if segment_id is None:
            segment_id = self._generate_segment_id()

        # Only strong/early matches shape the centroid
        is_core = match_type in ("strong_match", "early_match", "new_speaker", "first_speaker")

        ref = self._speakers[label]
        ref.add_embedding(
            embedding, timestamp, segment_id,
            audio_duration=audio_duration,
            is_core=is_core,
        )

        # Trim all_embeddings but preserve core_embeddings (they're the source of truth)
        if len(ref.all_embeddings) > self.max_embeddings_per_speaker:
            ref.all_embeddings = ref.all_embeddings[-self.max_embeddings_per_speaker:]
            ref.embedding_metadata = ref.embedding_metadata[-self.max_embeddings_per_speaker:]
            # Re-sync core from metadata truth
            ref.core_embeddings = [
                emb for emb, meta in zip(ref.all_embeddings, ref.embedding_metadata)
                if meta.get('is_core', True)
            ]
            ref._recompute_centroid()
            console.print(
                f"[dim]Trimmed {label}: {len(ref.all_embeddings)} total, "
                f"{len(ref.core_embeddings)} core embeddings[/dim]"
            )

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
        return self._maintenance.get_speaker_categories()

    def _should_run_maintenance(self, just_created_speaker: bool = False) -> bool:
        return self._maintenance.should_run_maintenance(just_created_speaker)

    def run_smart_maintenance(
        self, timestamp: float, just_created_speaker: bool = False
    ) -> Dict:
        return self._maintenance.run_smart_maintenance(timestamp, just_created_speaker)

    def _cleanup_orphan_speakers(self, current_timestamp: float) -> int:
        return self._maintenance.cleanup_orphan_speakers(current_timestamp)

    def reevaluate_young_speakers(
        self,
        min_segments_for_mature: int = 5,
        max_segments_for_young: int = 2,
        merge_threshold: float = None,
        dry_run: bool = False,
    ) -> Dict:
        return self._maintenance.reevaluate_young_speakers(
            min_segments_for_mature=min_segments_for_mature,
            max_segments_for_young=max_segments_for_young,
            merge_threshold=merge_threshold,
            dry_run=dry_run,
        )

    def _should_create_new_speaker(
        self,
        best_score: float,
        top_matches: List[Dict],
        context: Optional[Dict],
        embedding: np.ndarray,
    ) -> bool:
        return self._outlier_orchestrator.should_create_new_speaker(
            best_score, top_matches, context, embedding
        )

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
        segment_id: Optional[str] = None,
    ) -> List[SegmentGroup]:
        """Label a speech segment and return ALL processed segments with resolved labels.
        
        Each call adds the new segment internally, resolves any OUTLIER_XX → SPEAKER_XX
        retroactively, and returns the complete segment history.
        
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
            External segment identifier. Auto-generated if not provided.
        
        Returns
        -------
        List[Dict]
            ALL processed segment groups with resolved labels. Each group has:
            - timestamp: float
            - audio_duration: float
            - matches: List[Dict] with label, confidence, match_type, etc.
            OUTLIER_XX labels are retroactively replaced with SPEAKER_XX when
            promotions occur.
        """
        self.total_segments_processed += 1
        
        if segment_id is None:
            segment_id = self._generate_segment_id()
        if top_k is None:
            top_k = self.top_k_speakers
        
        # Extract speech and compute embedding
        waveform, was_filtered = self._extract_speech_audio(waveform=waveform)
        if isinstance(waveform, torch.Tensor):
            audio_duration = waveform.shape[-1] / sample_rate if waveform.dim() > 0 else 0.0
        else:
            audio_duration = len(waveform) / sample_rate
        
        embedding = self.compute_embedding(waveform, sample_rate)
        
        if self.debug:
            console.print(
                f"[dim]Embedding: shape={embedding.shape}, "
                f"segment_id={segment_id}, "
                f"speakers={self.speaker_count}, "
                f"outliers={self.outlier_pool.count}[/]"
            )
        
        # Find matches among existing speakers
        top_matches = self.find_top_k_matches(embedding, k=top_k)
        
        actual_best_score = 0.0
        if len(self._speakers) > 0:
            _, actual_best_score, _ = self.find_best_match(embedding)
        
        just_created_speaker = False
        
        if self.use_outlier_buffer:
            results, just_created_speaker = self._label_with_outlier_buffer(
                embedding=embedding,
                top_matches=top_matches,
                actual_best_score=actual_best_score,
                timestamp=timestamp,
                context=context,
                segment_id=segment_id,
                audio_duration=audio_duration,
            )
        else:
            results, just_created_speaker = self._label_without_outlier_buffer(
                embedding=embedding,
                top_matches=top_matches,
                actual_best_score=actual_best_score,
                timestamp=timestamp,
                context=context,
                segment_id=segment_id,
                audio_duration=audio_duration,
            )
        
        # Run maintenance
        self.run_smart_maintenance(timestamp, just_created_speaker=just_created_speaker)
        
        # Store new segment
        self._segment_groups.append({
            "timestamp": timestamp,
            "audio_duration": audio_duration,
            "matches": results,
        })
        
        # Resolve all labels retroactively (catches promotions that just happened)
        if self.use_outlier_buffer:
            self._segment_groups = self.finalize_labels(self._segment_groups)
        
        if self.debug:
            speakers_str = ", ".join(
                f"{r['label']}({r['confidence']:.3f})"
                for r in self._segment_groups[-1]["matches"][:3]
            )
            console.print(
                f"[dim]Segment {self.total_segments_processed}: "
                f"t={timestamp:.2f}s, segment_id={segment_id}, "
                f"→ [{speakers_str}] "
                f"(speakers: {self.speaker_count}, "
                f"outliers: {self.outlier_pool.count}, "
                f"rejected: {self._rejected_updates})[/]"
            )
        
        import copy
        return copy.deepcopy(self._segment_groups)

    def label_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        timestamp: float,
        context: Optional[Dict] = None,
        segment_id: Optional[str] = None,
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
        segment_groups = self.label_segments(
            waveform=waveform,
            sample_rate=sample_rate,
            timestamp=timestamp,
            context=context,
            top_k=1,
            segment_id=segment_id,
        )
        # label_segments now returns List[SegmentGroup]
        # Each SegmentGroup has: timestamp, audio_duration, matches (List[SegmentMatch])
        # Get the latest (most recent) segment group
        latest_group = segment_groups[-1] if segment_groups else None

        if latest_group is None or not latest_group.get("matches"):
            # Fallback if no results
            console.print(
                f"[yellow]⚠️  label_segment: No matches found for segment_id={segment_id}[/yellow]"
            )
            return "SPEAKER_UNKNOWN", 0.0, {
                "timestamp": timestamp,
                "is_new_speaker": False,
                "match_type": "unknown",
                "all_scores": {},
                "segment_id": segment_id,
            }

        # Get the primary match from the matches list
        matches = latest_group["matches"]
        primary = matches[0]

        metadata = {
            "timestamp": timestamp,
            "is_new_speaker": primary.get("is_new_speaker", False),
            "match_type": primary.get("match_type", "unknown"),
            "all_scores": {},
            "segment_id": primary.get("segment_id", segment_id),
            "audio_duration": latest_group.get("audio_duration", 0.0),
        }
        return primary["label"], primary["confidence"], metadata

    def get_outlier_resolution_map(self) -> Dict[str, str]:
        """Build a map from outlier labels to their final speaker labels.
        
        Combines two sources:
        1. Promotion history: OUTLIER_03 → SPEAKER_01
        2. Segment metadata: segment_id → speaker_label
        
        Returns
        -------
        Dict[str, str]
            Mapping of OUTLIER_XX → SPEAKER_XX for all resolved outliers.
        """
        resolution_map: Dict[str, str] = {}
        
        if not self.use_outlier_buffer:
            return resolution_map
        
        # Source 1: Promotion history from outlier pool
        for promo in self.outlier_pool._promotions:
            for outlier_label in promo.outlier_labels:
                resolution_map[outlier_label] = promo.target_speaker
        
        # Source 2: Remaining outliers still in pool — check if any
        # have been merged into speakers via segment_id matching
        segment_speaker_map: Dict[str, str] = {}
        for speaker_label, ref in self._speakers.items():
            for meta in ref.embedding_metadata:
                seg_id = meta.get("segment_id", "")
                if seg_id:
                    segment_speaker_map[seg_id] = speaker_label
        
        if self.debug and resolution_map:
            console.print(
                f"[dim]📋 Outlier resolution map: "
                f"{len(resolution_map)} outliers → speakers[/]"
            )
        
        return resolution_map

    def resolve_segment_label(
        self,
        label: str,
        segment_id: str = "",
    ) -> Tuple[str, bool, str]:
        """Resolve a segment label to its final speaker label.
        
        If the label is an OUTLIER_XX that was later promoted to a speaker,
        returns the final SPEAKER_XX label. Otherwise returns the label as-is.
        
        Parameters
        ----------
        label : str
            The label to resolve (e.g., 'OUTLIER_03' or 'SPEAKER_01').
        segment_id : str
            The segment's unique ID for fallback lookup.
        
        Returns
        -------
        Tuple[str, bool, str]
            (resolved_label, was_resolved, resolution_method)
            - resolved_label: Final speaker label
            - was_resolved: True if the label changed
            - resolution_method: How it was resolved ('promotion_map', 
            'segment_lookup', 'already_speaker', 'unresolved_outlier')
        """
        # Already a speaker label
        if label.startswith("SPEAKER_"):
            return label, False, "already_speaker"
        
        # Not an outlier — return as-is
        if not label.startswith("OUTLIER_"):
            return label, False, "unknown_format"
        
        # Get resolution map
        resolution_map = self.get_outlier_resolution_map()
        
        # Method 1: Direct promotion lookup
        if label in resolution_map:
            return resolution_map[label], True, "promotion_map"
        
        # Method 2: Check if segment_id is in a speaker's metadata
        if segment_id:
            for speaker_label, ref in self._speakers.items():
                for meta in ref.embedding_metadata:
                    if meta.get("segment_id") == segment_id:
                        return speaker_label, True, "segment_lookup"
        
        # Method 3: Still in outlier pool — truly unresolved
        if self.use_outlier_buffer and label in self.outlier_pool:
            return label, False, "unresolved_outlier"
        
        # Outlier was removed (expired/merged) without promotion record
        return label, False, "removed_outlier"

    def resolve_segment_results(
        self,
        results: List[Dict],
    ) -> List[Dict]:
        """Resolve all outlier labels in a list of segment results.
        
        Walks through results and replaces OUTLIER_XX labels with their
        final SPEAKER_XX labels. Also updates match_type for resolved entries.
        
        Parameters
        ----------
        results : List[Dict]
            Results from label_segments() or label_segment().
        
        Returns
        -------
        List[Dict]
            Results with resolved labels.
        """
        resolved_results = []
        
        for result in results:
            label = result.get("label", "")
            segment_id = result.get("segment_id", "")
            
            resolved_label, was_resolved, method = self.resolve_segment_label(
                label=label,
                segment_id=segment_id,
            )
            
            # Create updated result
            updated = {**result}
            updated["label"] = resolved_label
            
            if was_resolved:
                updated["is_outlier"] = False
                updated["original_outlier_label"] = label  # Preserve history
                updated["resolution_method"] = method
                
                # Update match type
                if result.get("match_type") == "outlier_pending":
                    updated["match_type"] = "resolved_from_outlier"
                
                if self.debug:
                    console.print(
                        f"[green]✅ Resolved: {label} → {resolved_label} "
                        f"(via {method}, segment_id: {segment_id})[/]"
                    )
            
            resolved_results.append(updated)
        
        return resolved_results

    def get_outlier_stats_for_display(self) -> Dict:
        """Get outlier statistics formatted for display/summary.
        
        Returns
        -------
        Dict
            Stats including resolved/unresolved counts.
        """
        if not self.use_outlier_buffer:
            return {"enabled": False}
        
        resolution_map = self.get_outlier_resolution_map()
        
        return {
            "enabled": True,
            "active_outliers": self.outlier_pool.count,
            "total_promotions": self.outlier_pool.promotion_count,
            "resolved_outliers": len(resolution_map),
            "outlier_labels": self.outlier_pool.labels,
            "promotion_history": [
                {
                    "type": p.type,
                    "outliers": p.outlier_labels,
                    "target": p.target_speaker,
                    "confidence": p.confidence,
                }
                for p in self.outlier_pool._promotions
            ],
        }

    def finalize_labels(
        self,
        segment_groups: List[Dict],
    ) -> List[Dict]:
        """Resolve OUTLIER_XX labels to final SPEAKER_XX labels retroactively.
        
        Call this ONCE after all segments have been processed via label_segments().
        
        Parameters
        ----------
        segment_groups : List[Dict]
            List of segment groups collected by the caller. Each must have
            a "matches" list where each match has "label" and "segment_id".
        
        Returns
        -------
        List[Dict]
            Same structure with OUTLIER_XX replaced by SPEAKER_XX where resolved.
        """
        if not self.use_outlier_buffer:
            return segment_groups
        
        if self.debug:
            console.print(
                f"\n[bold yellow]🔍 Finalizing labels: "
                f"{len(segment_groups)} segments, "
                f"{self.speaker_count} speakers, "
                f"{self.outlier_pool.count} active outliers[/]"
            )
        
        resolution_map = self.get_outlier_resolution_map()
        
        segment_speaker_map: Dict[str, str] = {}
        for speaker_label, ref in self._speakers.items():
            for meta in ref.embedding_metadata:
                seg_id = meta.get("segment_id", "")
                if seg_id:
                    segment_speaker_map[seg_id] = speaker_label
        
        resolved_count = 0
        unresolved_count = 0
        
        for group in segment_groups:
            for match in group.get("matches", []):
                label = match.get("label", "")
                
                if not label.startswith("OUTLIER_"):
                    continue
                
                new_label = None
                resolution_method = "unknown"
                
                if label in resolution_map:
                    new_label = resolution_map[label]
                    resolution_method = "promotion_map"
                
                if new_label is None:
                    seg_id = match.get("segment_id", "")
                    if seg_id and seg_id in segment_speaker_map:
                        new_label = segment_speaker_map[seg_id]
                        resolution_method = "segment_lookup"
                
                if new_label is None:
                    if label in self.outlier_pool:
                        unresolved_count += 1
                        continue
                    else:
                        unresolved_count += 1
                        continue
                
                original_label = match["label"]
                match["label"] = new_label
                match["is_outlier"] = False
                match["original_outlier_label"] = original_label
                match["resolution_method"] = resolution_method
                
                if match.get("match_type") == "outlier_pending":
                    match["match_type"] = "resolved_from_outlier"
                
                resolved_count += 1
        
        if self.debug:
            parts = [f"[green]{resolved_count} resolved[/green]"]
            if unresolved_count > 0:
                parts.append(f"[yellow]{unresolved_count} unresolved[/yellow]")
            console.print(f"[bold]Label finalization: {', '.join(parts)}[/bold]")
        
        return segment_groups

    def _label_with_outlier_buffer(
        self,
        embedding: np.ndarray,
        top_matches: List[Dict],
        actual_best_score: float,
        timestamp: float,
        context: Optional[Dict],
        segment_id: str,
        audio_duration: float,
    ) -> Tuple[List[Dict], bool]:
        return self._outlier_orchestrator.label_with_outlier_buffer(
            embedding=embedding,
            top_matches=top_matches,
            actual_best_score=actual_best_score,
            timestamp=timestamp,
            context=context,
            segment_id=segment_id,
            audio_duration=audio_duration,
        )

    def _label_without_outlier_buffer(
        self,
        embedding: np.ndarray,
        top_matches: List[Dict],
        actual_best_score: float,
        timestamp: float,
        context: Optional[Dict],
        segment_id: str,
        audio_duration: float,
    ) -> Tuple[List[Dict], bool]:
        """Original labeling logic (outlier buffer disabled)."""
        should_create = self._should_create_new_speaker(
            actual_best_score, top_matches, context, embedding
        )
        
        just_created_speaker = False
        
        if should_create or not top_matches:
            new_label = self.create_new_speaker(
                embedding=embedding,
                timestamp=timestamp,
                segment_id=segment_id,
                audio_duration=audio_duration,
            )
            just_created_speaker = True
            
            results = [{
                "label": new_label,
                "confidence": 1.0,
                "match_type": "first_speaker" if not top_matches else "new_speaker",
                "is_primary": True,
                "is_new_speaker": True,
                "segment_count": 1,
                "last_seen": timestamp,
                "segment_id": segment_id,
            }]
            
            # Add alternatives
            seen_labels = {new_label}
            for match in top_matches:
                if match["label"] not in seen_labels:
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
            results = self._build_standard_results(
                top_matches=top_matches,
                embedding=embedding,
                timestamp=timestamp,
                context=context,
                segment_id=segment_id,
                audio_duration=audio_duration,
            )
        
        return results, just_created_speaker

    def _handle_outlier_promotion(
        self,
        outlier_matches: List[OutlierMatch],
        embedding: np.ndarray,
        timestamp: float,
        segment_id: str,
        audio_duration: float,
    ) -> Tuple[List[Dict], bool]:
        return self._outlier_orchestrator.handle_outlier_promotion(
            outlier_matches=outlier_matches,
            embedding=embedding,
            timestamp=timestamp,
            segment_id=segment_id,
            audio_duration=audio_duration,
        )

    def _handle_new_outlier(
        self,
        embedding: np.ndarray,
        timestamp: float,
        segment_id: str,
        audio_duration: float,
    ) -> List[Dict]:
        return self._outlier_orchestrator.handle_new_outlier(
            embedding=embedding,
            timestamp=timestamp,
            segment_id=segment_id,
            audio_duration=audio_duration,
        )

    def _merge_outlier_into_speaker(
        self,
        outlier_label: str,
        speaker_label: str,
        similarity: float,
        timestamp: float,
    ) -> bool:
        return self._outlier_orchestrator.merge_outlier_into_speaker(
            outlier_label=outlier_label,
            speaker_label=speaker_label,
            similarity=similarity,
            timestamp=timestamp,
        )

    def _build_standard_results(
        self,
        top_matches: List[Dict],
        embedding: np.ndarray,
        timestamp: float,
        context: Optional[Dict],
        segment_id: str,
        audio_duration: float,
    ) -> List[Dict]:
        """Build standard results list with update/rejection logic."""
        all_scores = {m["label"]: m["confidence"] for m in top_matches}
        results = []
        seen_labels = set()
        
        for i, match in enumerate(top_matches):
            if match["label"] in seen_labels:
                continue
            
            label = match["label"]
            confidence = match["confidence"]
            match_type = match["match_type"]
            is_primary = (i == 0)
            
            if is_primary and match_type == "possible_match":
                smoothed_label = self.apply_temporal_smoothing(label, timestamp, confidence)
                if smoothed_label != label and smoothed_label in all_scores:
                    smoothed_confidence = all_scores[smoothed_label]
                    if smoothed_confidence >= self.threshold_possible:
                        label = smoothed_label
                        confidence = smoothed_confidence
            
            if is_primary and context and "previous_speaker" in context:
                prev_speaker = context["previous_speaker"]
                if (prev_speaker and prev_speaker in all_scores
                    and prev_speaker not in seen_labels):
                    prev_sim = all_scores[prev_speaker]
                    if prev_sim >= self.threshold_possible and prev_sim > confidence:
                        label = prev_speaker
                        confidence = prev_sim
                        match_type = "context_match"
            
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
            
            if len(results) >= self.top_k_speakers:
                break
        
        results = self._deduplicate_results(results)
        
        # Handle primary match update/rejection
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
                segment_id=segment_id,
                audio_duration=audio_duration,
                match_type=primary_result["match_type"],
            )
        else:
            # Rejected → use outlier pool instead of immediate speaker creation
            if self.use_outlier_buffer:
                outlier_label = self.outlier_pool.add(
                    embedding=embedding,
                    timestamp=timestamp,
                    segment_id=segment_id,
                    audio_duration=audio_duration,
                )
                primary_result["label"] = outlier_label
                primary_result["confidence"] = 1.0
                primary_result["match_type"] = "outlier_pending"
                primary_result["is_new_speaker"] = False
                primary_result["is_outlier"] = True
                primary_result["segment_count"] = 1
                
                if self.debug:
                    console.print(
                        f"[yellow]📦 Rejected update → outlier: {outlier_label} "
                        f"(reason: {reason})[/]"
                    )
            else:
                new_label = self.create_new_speaker(
                    embedding=embedding,
                    timestamp=timestamp,
                    segment_id=segment_id,
                    audio_duration=audio_duration,
                )
                primary_result["label"] = new_label
                primary_result["confidence"] = 1.0
                primary_result["match_type"] = "new_speaker"
                primary_result["is_new_speaker"] = True
                primary_result["segment_count"] = 1
        
        return results

    def consolidate_speakers(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict:
        return self._maintenance.consolidate_speakers(threshold=threshold, dry_run=dry_run)

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
        return self._serializer.get_health_status()

    def get_speaker_similarity_matrix(self) -> Dict:
        return self._serializer.get_speaker_similarity_matrix()

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
        return self._maintenance.merge_speakers(label1, label2)

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

        if self.audio_tagger:
            pure_speech = self.audio_tagger.extract_speech_only(
                audio_int16,
                sample_rate=SAMPLE_RATE,
                edges_only=True,
            )
        else:
            # pure_speech = audio_int16
        
            # Call the self-contained extraction function
            pure_speech = extract_pure_speech_audio(
                audio=audio_int16,
                sampling_rate=SAMPLE_RATE,  # assumed, could be parameterized
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
        return self._serializer.get_speaker_health_report()

    def _find_missing_speaker_ids(self) -> List[str]:
        return self._serializer.find_missing_speaker_ids()

    def get_centroid_health_stats(self) -> Dict:
        return self._serializer.get_centroid_health_stats()

    def get_centroid_arrays(self) -> Dict[str, np.ndarray]:
        return self._serializer.get_centroid_arrays()

    def get_centroid_stats(self) -> Dict:
        return self._serializer.get_centroid_stats()

    def reset(self) -> None:
        """Reset the labeler to initial state."""
        self._speakers.clear()
        self._label_history.clear()
        self._next_speaker_id = 1
        self.total_segments_processed = 0
        self.total_speakers_created = 0
        self._rejected_updates = 0
        self._centroid_update_log.clear()
        self._merge_history.clear()
        self._speaker_creation_times.clear()
        self.outlier_pool.reset()  # NEW
        
        if self.debug:
            console.print("[yellow]SegmentSpeakerLabeler reset[/]")

    def to_dict(self) -> Dict:
        return self._serializer.to_dict()

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        embedding_model,
        audio_tagger=None,
    ) -> "SegmentSpeakerLabeler":
        return SpeakerLabelerSerializer.from_dict(
            cls=cls,
            data=data,
            embedding_model=embedding_model,
            audio_tagger=audio_tagger,
        )


if __name__ == "__main__":
    from main._main_segment_speaker_labeler import main
    main()
