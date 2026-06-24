"""Speaker maintenance operations: categorization, cleanup, reevaluation, consolidation, merging."""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy.spatial.distance import cdist
from rich.console import Console

if TYPE_CHECKING:
    from segment_speaker_labeler import SegmentSpeakerLabeler

console = Console()


class SpeakerMaintenance:
    """Handles speaker categorization, orphan cleanup, young speaker reevaluation,
    smart maintenance triggering, consolidation, and merging.

    Extracted from SegmentSpeakerLabeler to keep the main class focused on
    core labeling logic.
    """

    def __init__(
        self,
        labeler: "SegmentSpeakerLabeler",
        young_merge_threshold: float = 0.65,
        min_speaker_age_for_merge: float = 15.0,
        debug: bool = False,
    ):
        self._labeler = labeler
        self.young_merge_threshold = young_merge_threshold
        self.min_speaker_age_for_merge = min_speaker_age_for_merge
        self.debug = debug

    # ------------------------------------------------------------------
    # Speaker categorization
    # ------------------------------------------------------------------
    def get_speaker_categories(self) -> Dict[str, Dict]:
        """Categorize speakers by reliability for maintenance decisions."""
        speakers = self._labeler._speakers
        speaker_creation_times = self._labeler._speaker_creation_times
        mature_segment_count = self._labeler.mature_segment_count
        young_segment_count = self._labeler.young_segment_count
        temporal_smoothing_window = self._labeler.temporal_smoothing_window

        now = max((ref.last_seen for ref in speakers.values()), default=0.0)
        mature = {}
        young = {}
        orphan = {}
        active_young = {}
        newborn = {}

        for label, ref in speakers.items():
            if not ref.has_valid_centroid:
                continue
            creation_time = speaker_creation_times.get(label, 0.0)
            speaker_age = now - creation_time
            if speaker_age < self.min_speaker_age_for_merge:
                newborn[label] = ref
                continue
            if ref.segment_count >= mature_segment_count:
                mature[label] = ref
            elif ref.segment_count <= young_segment_count:
                young[label] = ref
                time_since_last_seen = now - ref.last_seen
                if time_since_last_seen > temporal_smoothing_window:
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
            "newborn": newborn,
        }

    # ------------------------------------------------------------------
    # Maintenance triggering
    # ------------------------------------------------------------------
    def should_run_maintenance(self, just_created_speaker: bool = False) -> bool:
        """Determine if speaker maintenance should run based on actual need."""
        categories = self.get_speaker_categories()
        mature_count = len(categories["mature"])
        young_count = len(categories["young"])
        orphan_count = len(categories["orphan"])
        total_count = len(self._labeler._speakers)

        if total_count < 2:
            return False
        if just_created_speaker and total_count >= 3:
            if orphan_count >= 2 or young_count >= 4:
                return True
            return False
        if orphan_count >= 3:
            return True
        if mature_count > 0 and young_count > mature_count * 2:
            return True
        if young_count >= 5:
            return True
        if total_count >= 8 and young_count >= 3:
            return True
        return False

    # ------------------------------------------------------------------
    # Smart maintenance
    # ------------------------------------------------------------------
    def run_smart_maintenance(
        self, timestamp: float, just_created_speaker: bool = False
    ) -> Dict:
        """Run targeted speaker maintenance only when needed."""
        if not self.should_run_maintenance(just_created_speaker):
            return {"run": False, "reason": "no_need"}

        categories = self.get_speaker_categories()
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
            "speakers_before": len(self._labeler._speakers),
            "speakers_after": len(self._labeler._speakers),
        }

        if len(categories["orphan"]) > 0:
            removed = self.cleanup_orphan_speakers(timestamp)
            results["orphans_removed"] = removed

        if len(categories["young"]) > 0 and len(categories["mature"]) > 0:
            reeval = self.reevaluate_young_speakers(
                min_segments_for_mature=self._labeler.mature_segment_count,
                max_segments_for_young=self._labeler.young_segment_count,
                merge_threshold=0.50,
                dry_run=False,
            )
            results["young_merged"] = reeval.get("merges_performed", [])

        if len(self._labeler._speakers) > 5:
            consol = self.consolidate_speakers(
                threshold=self._labeler.consolidation_threshold,
                dry_run=False,
            )
            results["mature_merged"] = consol.get("merges_performed", [])

        results["speakers_after"] = len(self._labeler._speakers)
        if self.debug and results["speakers_before"] != results["speakers_after"]:
            console.print(
                f"[green]🔧 Maintenance: "
                f"{results['speakers_before']} → {results['speakers_after']} speakers "
                f"(removed {results['orphans_removed']} orphans, "
                f"merged {len(results['young_merged'])} young, "
                f"merged {len(results['mature_merged'])} mature)[/green]"
            )
        return results

    # ------------------------------------------------------------------
    # Orphan cleanup
    # ------------------------------------------------------------------
    def cleanup_orphan_speakers(self, current_timestamp: float) -> int:
        """Remove or merge orphan speakers with newborn protection."""
        removed = 0
        categories = self.get_speaker_categories()
        newborn_labels = set(categories.get("newborn", {}).keys())
        labels_to_check = list(self._labeler._speakers.keys())

        for label in labels_to_check:
            if label not in self._labeler._speakers:
                continue
            if label in newborn_labels:
                if self.debug:
                    console.print(f"[dim]🔒 Protecting newborn {label} from cleanup[/dim]")
                continue

            ref = self._labeler._speakers[label]
            time_since_last_seen = current_timestamp - ref.last_seen
            if (
                ref.segment_count <= self._labeler.young_segment_count
                and time_since_last_seen > self._labeler.temporal_smoothing_window * 3
            ):
                if ref.has_valid_centroid and len(self._labeler._speakers) > 1:
                    best_match, best_score, _ = self._labeler.find_best_match(ref.centroid)
                    if best_match and best_match != label and best_score > 0.60:
                        if self.debug:
                            console.print(
                                f"[yellow]Orphan merge: {label} → {best_match} "
                                f"(sim={best_score:.3f})[/]"
                            )
                        self.merge_speakers(best_match, label)
                        self._labeler._merge_history.append({
                            "type": "orphan_merge",
                            "source": label,
                            "target": best_match,
                            "similarity": best_score,
                            "timestamp": current_timestamp,
                        })
                        removed += 1
                    elif ref.segment_count == 1 and time_since_last_seen > 30.0:
                        if self.debug:
                            console.print(
                                f"[dim]Orphan remove: {label} (inactive {time_since_last_seen:.1f}s)[/]"
                            )
                        del self._labeler._speakers[label]
                        self._labeler._speaker_creation_times.pop(label, None)
                        removed += 1
        return removed

    # ------------------------------------------------------------------
    # Young speaker reevaluation
    # ------------------------------------------------------------------
    def reevaluate_young_speakers(
        self,
        min_segments_for_mature: int = 5,
        max_segments_for_young: int = 2,
        merge_threshold: float = None,
        dry_run: bool = False,
    ) -> Dict:
        """Re-evaluate young speakers against mature speakers."""
        if merge_threshold is None:
            merge_threshold = self.young_merge_threshold

        categories = self.get_speaker_categories()
        mature_speakers = {}
        young_speakers = {}

        for label, ref in self._labeler._speakers.items():
            if not ref.has_valid_centroid:
                continue
            if label in categories.get("newborn", {}):
                if self.debug:
                    console.print(
                        f"[dim]🔒 Skipping newborn {label} "
                        f"(age={self._labeler._speaker_creation_times.get(label, 0):.1f}s)[/dim]"
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
                "newborn_skipped": len(categories.get("newborn", {})),
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
                merges_to_perform.append({
                    "young_speaker": young_label,
                    "mature_speaker": best_mature_label,
                    "similarity": round(best_similarity, 4),
                    "young_segments": young_ref.segment_count,
                    "mature_segments": mature_speakers[best_mature_label].segment_count,
                })
                if self.debug:
                    console.print(
                        f"[yellow]🔍 Re-eval MERGE: {young_label} "
                        f"({young_ref.segment_count} segs) → "
                        f"{best_mature_label} "
                        f"({mature_speakers[best_mature_label].segment_count} segs) "
                        f"sim={best_similarity:.3f} (threshold={merge_threshold})[/yellow]"
                    )
            elif self.debug:
                console.print(
                    f"[dim]🔍 Re-eval KEEP: {young_label} "
                    f"({young_ref.segment_count} segs) vs "
                    f"{best_mature_label} "
                    f"sim={best_similarity:.3f} < {merge_threshold}[/dim]"
                )

        if not dry_run:
            for merge_info in merges_to_perform:
                self.merge_speakers(merge_info["mature_speaker"], merge_info["young_speaker"])
                self._labeler._merge_history.append({
                    "type": "young_reeval",
                    "source": merge_info["young_speaker"],
                    "target": merge_info["mature_speaker"],
                    "similarity": merge_info["similarity"],
                    "timestamp": max(
                        ref.last_seen for ref in self._labeler._speakers.values()
                        if ref.label == merge_info["mature_speaker"]
                    ),
                })

        return {
            "merges_performed": [
                (m["young_speaker"], m["mature_speaker"], m["similarity"])
                for m in merges_to_perform
            ],
            "speakers_checked": len(young_speakers),
            "mature_speakers": len(mature_speakers),
            "newborn_skipped": len(categories.get("newborn", {})),
            "dry_run": dry_run,
        }

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def consolidate_speakers(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict:
        """Consolidate similar speakers."""
        if threshold is None:
            threshold = self._labeler.consolidation_threshold

        speakers_before = len(self._labeler._speakers)
        if speakers_before < 2:
            return {
                "merges_performed": [],
                "speakers_before": speakers_before,
                "speakers_after": speakers_before,
                "dry_run": dry_run,
            }

        speaker_labels = []
        centroids = []
        for label, ref in self._labeler._speakers.items():
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
                    merges_to_perform.append((speaker_labels[i], speaker_labels[j], round(sim, 4)))
                    already_merged.add(speaker_labels[j])

        if not dry_run:
            for label1, label2, sim in merges_to_perform:
                self.merge_speakers(label1, label2)
                if self.debug:
                    console.print(f"[yellow]Consolidated: {label1} + {label2} (sim={sim:.3f})[/]")

        return {
            "merges_performed": merges_to_perform,
            "speakers_before": speakers_before,
            "speakers_after": len(self._labeler._speakers),
            "dry_run": dry_run,
        }

    # ------------------------------------------------------------------
    # Merge two speakers
    # ------------------------------------------------------------------
    def merge_speakers(self, label1: str, label2: str) -> Optional[str]:
        """Merge two speaker references."""
        speakers = self._labeler._speakers
        if label1 not in speakers or label2 not in speakers:
            return None
        if label1 == label2:
            return label1

        ref1 = speakers[label1]
        ref2 = speakers[label2]

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

        del speakers[secondary_label]
        self._labeler._speaker_creation_times.pop(secondary_label, None)
        self._labeler._label_history = [
            (t, primary_label if l == secondary_label else l)
            for t, l in self._labeler._label_history
        ]

        if self.debug:
            console.print(
                f"[bold yellow]🔀 MERGED: {secondary_label} → {primary_label} "
                f"(kept: {primary.segment_count} segs, "
                f"removed: {secondary_label})[/bold yellow]"
            )
        return primary_label
