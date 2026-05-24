"""Progressive segment speaker labeling with dynamic reference maintenance."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from scipy.spatial.distance import cdist

console = Console()
DEFAULT_THRESHOLD_SAME: float = 0.65
DEFAULT_THRESHOLD_POSSIBLE: float = 0.35
DEFAULT_THRESHOLD_NEW_SPEAKER: float = 0.20
DEFAULT_MIN_SEGMENTS_FOR_REFERENCE: int = 3
DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER: int = 50
DEFAULT_TEMPORAL_SMOOTHING_WINDOW: float = 3.0
DEFAULT_TOP_K_SPEAKERS: int = 3
DEFAULT_MIN_SIMILARITY_FOR_LIST: float = 0.15
DEFAULT_CONSOLIDATION_THRESHOLD: float = 0.80
DEFAULT_MATURE_SEGMENT_COUNT: int = 5
DEFAULT_YOUNG_SEGMENT_COUNT: int = 2


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
        """Add a new embedding and update centroid using median for robustness."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        self.embeddings.append(embedding)
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


class SegmentSpeakerLabeler:
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
    debug : bool
        Enable debug logging.
    """

    def __init__(
        self,
        embedding_model,
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
        """Compute speaker embedding from waveform segment."""
        try:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            embedding = self.embedding_model(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                }
            )
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
        """Find top-K matching speakers with ALL similarities included.
        Key fix: Returns ALL matches above a very low threshold, not just
        those above min_similarity_for_list. This ensures we always have
        similarity data for debugging and decision making.
        """
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
    ) -> str:
        """Create a new speaker reference."""
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
        """Update speaker reference with new embedding."""
        if label not in self._speakers:
            self._speakers[label] = SpeakerReference(label=label)
        ref = self._speakers[label]
        ref.add_embedding(embedding, timestamp)
        if len(ref.embeddings) > self.max_embeddings_per_speaker:
            ref.embeddings = ref.embeddings[-self.max_embeddings_per_speaker :]
            if ref.embeddings:
                if len(ref.embeddings) >= 3:
                    stacked = np.vstack(ref.embeddings)
                    ref.centroid = np.median(stacked, axis=0, keepdims=True)
                else:
                    stacked = np.vstack(ref.embeddings)
                    ref.centroid = np.mean(stacked, axis=0, keepdims=True)

    def _get_speaker_categories(self) -> Dict[str, Dict]:
        """Categorize speakers by reliability for maintenance decisions.
        Uses mature_segment_count and young_segment_count instead of
        min_segments_for_reference to properly distinguish mature vs young.
        """
        now = max((ref.last_seen for ref in self._speakers.values()), default=0.0)
        mature = {}
        young = {}
        orphan = {}
        active_young = {}
        for label, ref in self._speakers.items():
            if not ref.has_valid_centroid:
                continue
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
        }

    def _should_run_maintenance(self, just_created_speaker: bool = False) -> bool:
        """Determine if speaker maintenance should run based on actual need."""
        categories = self._get_speaker_categories()
        mature_count = len(categories["mature"])
        young_count = len(categories["young"])
        orphan_count = len(categories["orphan"])
        total_count = len(self._speakers)
        if total_count < 2:
            return False
        if just_created_speaker and total_count >= 3:
            return True
        if orphan_count >= 3:
            return True
        if mature_count > 0 and young_count > mature_count:
            return True
        if young_count >= 5:
            return True
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
        if len(categories["orphan"]) > 0:
            removed = self._cleanup_orphan_speakers(timestamp)
            results["orphans_removed"] = removed
        if len(categories["young"]) > 0 and len(categories["mature"]) > 0:
            reeval = self.reevaluate_young_speakers(
                min_segments_for_mature=self.mature_segment_count,
                max_segments_for_young=self.young_segment_count,
                merge_threshold=0.50,
                dry_run=False,
            )
            results["young_merged"] = reeval.get("merges_performed", [])
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
        """Remove or merge orphan speakers."""
        removed = 0
        labels_to_check = list(self._speakers.keys())
        for label in labels_to_check:
            if label not in self._speakers:
                continue
            ref = self._speakers[label]
            time_since_last_seen = current_timestamp - ref.last_seen
            if (
                ref.segment_count <= self.young_segment_count
                and time_since_last_seen > self.temporal_smoothing_window * 2
            ):
                if ref.has_valid_centroid and len(self._speakers) > 1:
                    best_match, best_score, _ = self.find_best_match(ref.centroid)
                    if best_match and best_match != label and best_score > 0.50:
                        if self.debug:
                            console.print(
                                f"[yellow]Orphan merge: {label} → {best_match} "
                                f"(sim={best_score:.3f})[/]"
                            )
                        self.merge_speakers(best_match, label)
                        removed += 1
                    elif ref.segment_count == 1 and time_since_last_seen > 10.0:
                        if self.debug:
                            console.print(f"[dim]Orphan remove: {label}[/]")
                        del self._speakers[label]
                        removed += 1
        return removed

    def reevaluate_young_speakers(
        self,
        min_segments_for_mature: int = 5,
        max_segments_for_young: int = 2,
        merge_threshold: float = 0.50,
        dry_run: bool = False,
    ) -> Dict:
        """Re-evaluate young speakers against mature speakers."""
        mature_speakers = {}
        young_speakers = {}
        for label, ref in self._speakers.items():
            if not ref.has_valid_centroid:
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
                        f"[yellow]🔍 Re-eval: {young_label} "
                        f"({young_ref.segment_count} segs) → "
                        f"{best_mature_label} "
                        f"({mature_speakers[best_mature_label].segment_count} segs) "
                        f"sim={best_similarity:.3f}[/yellow]"
                    )
        if not dry_run:
            for merge_info in merges_to_perform:
                self.merge_speakers(
                    merge_info["mature_speaker"], merge_info["young_speaker"]
                )
        return {
            "merges_performed": [
                (m["young_speaker"], m["mature_speaker"], m["similarity"])
                for m in merges_to_perform
            ],
            "speakers_checked": len(young_speakers),
            "mature_speakers": len(mature_speakers),
            "dry_run": dry_run,
        }

    def _should_create_new_speaker(
        self,
        best_score: float,
        top_matches: List[Dict],
        context: Optional[Dict],
        embedding: np.ndarray,
    ) -> bool:
        """Determine if we should create a new speaker.
        FIXED: Now properly uses the actual best_score from top_matches
        instead of requiring matches to pass min_similarity_for_list filter.
        """
        if len(self._speakers) == 0:
            return True
        if not top_matches:
            return True
        best_match = top_matches[0]
        match_type = best_match["match_type"]
        if match_type in ("strong_match", "early_match"):
            return False
        if best_score >= self.threshold_possible:
            return False
        if len(self._speakers) <= 3:
            if best_score >= 0.15:
                return False
        if context and "previous_speaker" in context:
            prev_speaker = context["previous_speaker"]
            if prev_speaker and prev_speaker in self._speakers:
                for match in top_matches:
                    if match["label"] == prev_speaker and match["confidence"] >= 0.12:
                        return False
        for match in top_matches[:3]:
            if match["segment_count"] <= self.young_segment_count:
                if best_score >= 0.15:
                    return False
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

        # Use dict to deduplicate by label, keeping highest confidence
        best_by_label: Dict[str, Dict] = {}
        for entry in results:
            label = entry["label"]
            if (
                label not in best_by_label
                or entry["confidence"] > best_by_label[label]["confidence"]
            ):
                # Merge: keep the higher confidence, but preserve is_primary if either was primary
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

        # Sort: primary first, then by confidence descending
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
    ) -> List[Dict]:
        """Label a speech segment with multiple possible speaker identities.
        
        FIXED: Temporal smoothing and context matching now only override the
        primary label when confidence meets threshold_possible. Weak matches
        no longer influence label changes.
        """
        self.total_segments_processed += 1
        if top_k is None:
            top_k = self.top_k_speakers
        
        embedding = self.compute_embedding(waveform, sample_rate)
        top_matches = self.find_top_k_matches(embedding, k=top_k)
        
        if self.debug:
            console.print(
                f"[dim]Computed embedding for t={timestamp:.2f}s, got {len(top_matches)} top matches[/dim]"
            )
        
        actual_best_score = 0.0
        if len(self._speakers) > 0:
            _, actual_best_score, _ = self.find_best_match(embedding)
        
        should_create = self._should_create_new_speaker(
            actual_best_score, top_matches, context, embedding
        )
        
        if self.debug:
            console.print(
                f"[dim]Actual best score: {actual_best_score:.4f}, should_create_new_speaker: {should_create}[/dim]"
            )
        
        results = []
        seen_labels = set()
        just_created_speaker = False
        
        if should_create or not top_matches:
            # NEW SPEAKER PATH
            new_label = self.create_new_speaker(embedding, timestamp)
            self.update_reference(new_label, embedding, timestamp)
            just_created_speaker = True
            
            if self.debug:
                categories = self._get_speaker_categories()
                console.print(
                    f"[yellow]⚠️  New speaker: {new_label} "
                    f"(best sim: {actual_best_score:.3f}, "
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
            })
            seen_labels.add(new_label)
            
            # Add alternatives (existing speakers that this could have matched)
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
                    })
                    seen_labels.add(match["label"])
        else:
            # EXISTING SPEAKER PATH
            all_scores = {m["label"]: m["confidence"] for m in top_matches}
            
            for i, match in enumerate(top_matches):
                if match["label"] in seen_labels:
                    continue
                
                label = match["label"]
                confidence = match["confidence"]
                match_type = match["match_type"]
                is_primary = (i == 0)
                
                # ── TEMPORAL SMOOTHING ──────────────────────────────────
                # FIXED: Only apply smoothing when confidence meets threshold_possible.
                # Previously: triggered on "possible_match" AND "weak_match",
                # which allowed low-confidence labels to override the top match.
                # Now: only triggers for "possible_match" or better,
                # meaning confidence >= threshold_possible (0.35 default).
                if is_primary and match_type == "possible_match":
                    smoothed_label = self.apply_temporal_smoothing(
                        label, timestamp, confidence
                    )
                    if smoothed_label != label and smoothed_label in all_scores:
                        smoothed_confidence = all_scores[smoothed_label]
                        # Only accept smoothed label if ITS confidence also meets threshold
                        if smoothed_confidence >= self.threshold_possible:
                            label = smoothed_label
                            confidence = smoothed_confidence
                
                # ── CONTEXT MATCHING ────────────────────────────────────
                # FIXED: Context matching now requires the context speaker's
                # confidence to meet threshold_possible (not threshold_possible - 0.10).
                # Previously: prev_sim >= threshold_possible - 0.10 (e.g. 0.25)
                # Now: prev_sim >= threshold_possible (e.g. 0.35)
                # AND prev_sim must be HIGHER than the current best match.
                if is_primary and context and "previous_speaker" in context:
                    prev_speaker = context["previous_speaker"]
                    if prev_speaker and prev_speaker in all_scores and prev_speaker not in seen_labels:
                        prev_sim = all_scores[prev_speaker]
                        # Must meet threshold_possible AND be higher than current best
                        if prev_sim >= self.threshold_possible and prev_sim > confidence:
                            label = prev_speaker
                            confidence = prev_sim
                            match_type = "context_match"
                            if self.debug:
                                console.print(
                                    f"[blue]Context match: {prev_speaker} "
                                    f"(sim={prev_sim:.3f})[/blue]"
                                )
                
                results.append({
                    "label": label,
                    "confidence": round(confidence, 4),
                    "match_type": match_type,
                    "is_primary": is_primary,
                    "is_new_speaker": False,
                    "segment_count": match["segment_count"],
                    "last_seen": match["last_seen"],
                })
                seen_labels.add(label)
                
                if len(results) >= top_k:
                    break
            
            # Deduplicate: if smoothing/context changed the primary to a label
            # that was already added as an alternative, keep higher confidence
            results = self._deduplicate_results(results)
            
            # Update reference for the primary speaker
            primary_result = results[0]
            self.update_reference(primary_result["label"], embedding, timestamp)
        
        # Run maintenance if needed
        self.run_smart_maintenance(timestamp, just_created_speaker=just_created_speaker)
        
        if self.debug:
            speakers_str = ", ".join(
                f"{r['label']}({r['confidence']:.3f})"
                for r in results[:3]
            )
            console.print(
                f"[dim]Segment {self.total_segments_processed}: "
                f"t={timestamp:.2f}s → [{speakers_str}] "
                f"(primary: {results[0]['label']}, "
                f"speakers: {len(self._speakers)})[/]"
            )
        
        return results

    def label_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        timestamp: float,
        context: Optional[Dict] = None,
    ) -> Tuple[str, float, Dict]:
        """Label a speech segment with a single speaker identity."""
        results = self.label_segments(
            waveform, sample_rate, timestamp, context, top_k=1
        )
        primary = results[0]
        metadata = {
            "timestamp": timestamp,
            "is_new_speaker": primary.get("is_new_speaker", False),
            "match_type": primary["match_type"],
            "all_scores": {},
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
        return {
            "label": ref.label,
            "segment_count": ref.segment_count,
            "first_seen": ref.first_seen,
            "last_seen": ref.last_seen,
            "active_duration": ref.active_duration,
            "has_valid_centroid": ref.has_valid_centroid,
        }

    def get_all_speakers_info(self) -> Dict[str, Dict]:
        """Get information about all known speakers."""
        return {label: self.get_speaker_info(label) for label in self._speakers}

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
        """Merge two speaker references."""
        if label1 not in self._speakers or label2 not in self._speakers:
            return None
        if label1 == label2:
            return label1
        ref1 = self._speakers[label1]
        ref2 = self._speakers[label2]
        if ref1.segment_count >= ref2.segment_count:
            primary, secondary = ref1, ref2
            primary_label = label1
            secondary_label = label2
        else:
            primary, secondary = ref2, ref1
            primary_label = label2
            secondary_label = label1
        for emb in secondary.embeddings:
            primary.embeddings.append(emb)
        primary.segment_count += secondary.segment_count
        primary.last_seen = max(primary.last_seen, secondary.last_seen)
        primary.first_seen = min(primary.first_seen, secondary.first_seen)
        if primary.embeddings:
            if len(primary.embeddings) >= 3:
                stacked = np.vstack(primary.embeddings)
                primary.centroid = np.median(stacked, axis=0, keepdims=True)
            else:
                stacked = np.vstack(primary.embeddings)
                primary.centroid = np.mean(stacked, axis=0, keepdims=True)
        del self._speakers[secondary_label]
        self._label_history = [
            (t, primary_label if l == secondary_label else l)
            for t, l in self._label_history
        ]
        if self.debug:
            console.print(f"[yellow]Merged: {label1} + {label2} → {primary_label}[/]")
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
            threshold_possible=data.get(
                "threshold_possible", DEFAULT_THRESHOLD_POSSIBLE
            ),
            threshold_new_speaker=data.get(
                "threshold_new_speaker", DEFAULT_THRESHOLD_NEW_SPEAKER
            ),
            mature_segment_count=data.get(
                "mature_segment_count", DEFAULT_MATURE_SEGMENT_COUNT
            ),
            young_segment_count=data.get(
                "young_segment_count", DEFAULT_YOUNG_SEGMENT_COUNT
            ),
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
