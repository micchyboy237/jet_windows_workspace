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
    smart maintenance triggering, consolidation, merging, centroid validation,
    and health scoring.

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
        """Categorize speakers by reliability for maintenance decisions.
        
        Categories:
            mature: Well-established speakers (>= mature_segment_count segments)
            young: New speakers with few segments (<= young_segment_count)
            orphan: Young speakers not seen recently
            active_young: Young speakers still active
            newborn: Very young speakers (age < min_speaker_age_for_merge)
            degraded_mature: Mature speakers with low centroid quality
        """
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
        degraded_mature = {}

        for label, ref in speakers.items():
            if not ref.has_valid_centroid:
                continue
                
            creation_time = speaker_creation_times.get(label, 0.0)
            speaker_age = now - creation_time
            
            # NEW: Check newborn status first
            if speaker_age < self.min_speaker_age_for_merge:
                newborn[label] = ref
                continue
                
            # NEW: Check centroid quality for mature speakers
            centroid_quality = getattr(ref, 'centroid_quality', 1.0)
            is_degraded = centroid_quality < 0.5
            
            if ref.segment_count >= mature_segment_count:
                if is_degraded:
                    degraded_mature[label] = ref
                else:
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
            "degraded_mature": degraded_mature,
        }

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------
    def score_speaker_health(self, label: str) -> Dict:
        """Score a speaker's health for maintenance prioritization.
        
        Returns a dict with:
            - score: float (0.0-1.0, higher is better)
            - issues: list of identified problems
            - speaker: speaker label
            - segments: segment count
            - quality: centroid quality if available
        """
        if label not in self._labeler._speakers:
            return {"score": 0.0, "issues": ["not_found"], "speaker": label}

        ref = self._labeler._speakers[label]
        issues = []
        health_score = 1.0

        # Segment count score
        if ref.segment_count < self._labeler.mature_segment_count:
            deduction = 0.3 * (1 - ref.segment_count / self._labeler.mature_segment_count)
            health_score -= deduction
            issues.append("low_segments")

        # Quality score
        centroid_quality = getattr(ref, 'centroid_quality', None)
        if centroid_quality is not None:
            if centroid_quality < 0.4:
                health_score -= 0.3
                issues.append("very_low_quality")
            elif centroid_quality < 0.6:
                health_score -= 0.15
                issues.append("low_quality")

        # Overlap contamination check
        overlap_ratio = getattr(ref, 'overlap_ratio', 0.0)
        if overlap_ratio > 0.3:  # >30% segments from overlap regions
            health_score -= 0.25
            issues.append("high_overlap")
        elif overlap_ratio > 0.15:
            health_score -= 0.1
            issues.append("moderate_overlap")

        # Core embedding consistency
        if hasattr(ref, 'core_embeddings') and ref.core_embeddings:
            core_ratio = len(ref.core_embeddings) / max(1, ref.segment_count)
            if core_ratio < 0.5:
                health_score -= 0.2
                issues.append("low_core_ratio")

        # Recency score
        if ref.last_seen is not None and ref.first_seen is not None:
            time_since_last = max(0, ref.last_seen - ref.first_seen)
            if time_since_last > 120:  # Inactive for >2 minutes
                health_score -= 0.2
                issues.append("long_inactive")
            elif time_since_last > 60:  # Inactive for >1 minute
                health_score -= 0.1
                issues.append("inactive")

        return {
            "score": round(max(0.0, health_score), 3),
            "issues": issues,
            "speaker": label,
            "segments": ref.segment_count,
            "quality": centroid_quality,
            "overlap_ratio": overlap_ratio,
        }

    def get_maintenance_priority(self) -> List[Dict]:
        """Get speakers ordered by maintenance urgency (lowest health first)."""
        scores = []
        for label in self._labeler._speakers:
            scores.append(self.score_speaker_health(label))
        scores.sort(key=lambda x: x["score"])
        return scores

    # ------------------------------------------------------------------
    # Maintenance triggering
    # ------------------------------------------------------------------
    def should_run_maintenance(self, just_created_speaker: bool = False) -> bool:
        """Enhanced maintenance triggering with quality and overlap awareness."""
        categories = self.get_speaker_categories()
        mature_count = len(categories["mature"])
        young_count = len(categories["young"])
        orphan_count = len(categories["orphan"])
        degraded_count = len(categories["degraded_mature"])
        total_count = len(self._labeler._speakers)

        if total_count < 2:
            return False

        # NEW: Trigger on degraded mature speakers
        if degraded_count >= 2:
            return True
        if degraded_count >= 1 and young_count >= 3:
            return True

        # NEW: Check overall health scores
        if total_count >= 4:
            health_scores = self.get_maintenance_priority()
            unhealthy_count = sum(1 for s in health_scores if s["score"] < 0.5)
            if unhealthy_count >= 3:
                return True

        # NEW: Trigger on overlap contamination
        if hasattr(self._labeler, '_overlap_regions') and self._labeler._overlap_regions:
            overlap_count = len(self._labeler._overlap_regions)
            if overlap_count > 5 and young_count >= 2:
                return True

        # Existing triggers
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
        """Run targeted speaker maintenance with quality-aware operations."""
        if not self.should_run_maintenance(just_created_speaker):
            return {"run": False, "reason": "no_need"}

        categories = self.get_speaker_categories()
        if self.debug:
            console.print(
                f"[dim]🔧 Maintenance triggered: "
                f"mature={len(categories['mature'])}, "
                f"degraded={len(categories['degraded_mature'])}, "
                f"young={len(categories['young'])}, "
                f"orphan={len(categories['orphan'])}[/dim]"
            )

        results = {
            "run": True,
            "orphans_removed": 0,
            "young_merged": [],
            "mature_merged": [],
            "degraded_rebuilt": [],
            "speakers_before": len(self._labeler._speakers),
            "speakers_after": len(self._labeler._speakers),
        }

        # NEW: Validate and rebuild degraded centroids first
        if len(categories["degraded_mature"]) > 0:
            rebuild_results = self.validate_mature_centroids()
            results["degraded_rebuilt"] = rebuild_results.get("rebuilt", [])

        # Clean up orphans
        if len(categories["orphan"]) > 0:
            removed = self.cleanup_orphan_speakers(timestamp)
            results["orphans_removed"] = removed

        # Reevaluate young speakers
        if len(categories["young"]) > 0 and len(categories["mature"]) > 0:
            reeval = self.reevaluate_young_speakers(
                min_segments_for_mature=self._labeler.mature_segment_count,
                max_segments_for_young=self._labeler.young_segment_count,
                merge_threshold=0.50,
                dry_run=False,
            )
            results["young_merged"] = reeval.get("merges_performed", [])

        # Consolidate if too many speakers
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
                f"merged {len(results['mature_merged'])} mature, "
                f"rebuilt {len(results['degraded_rebuilt'])} centroids)[/green]"
            )
        return results

    # ------------------------------------------------------------------
    # Orphan cleanup
    # ------------------------------------------------------------------
    def cleanup_orphan_speakers(self, current_timestamp: float) -> int:
        """Remove or merge orphan speakers with newborn protection 
        and quality-aware merge decisions."""
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
                        # NEW: Quality check before merge
                        target_ref = self._labeler._speakers.get(best_match)
                        target_quality = getattr(target_ref, 'centroid_quality', 1.0)
                        
                        if target_quality < 0.3:
                            if self.debug:
                                console.print(
                                    f"[yellow]⚠️ Skipping orphan merge into low-quality "
                                    f"{best_match} (quality={target_quality:.2f})[/]"
                                )
                            # Instead, just remove the orphan
                            del self._labeler._speakers[label]
                            self._labeler._speaker_creation_times.pop(label, None)
                            removed += 1
                            continue
                        
                        if self.debug:
                            console.print(
                                f"[yellow]Orphan merge: {label} → {best_match} "
                                f"(sim={best_score:.3f}, target_quality={target_quality:.2f})[/]"
                            )
                        self.merge_speakers(best_match, label)
                        self._labeler._merge_history.append({
                            "type": "orphan_merge",
                            "source": label,
                            "target": best_match,
                            "similarity": best_score,
                            "target_quality": target_quality,
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
        """Re-evaluate young speakers against mature speakers 
        with quality filtering."""
        if merge_threshold is None:
            merge_threshold = self.young_merge_threshold

        categories = self.get_speaker_categories()
        mature_speakers = {}
        young_speakers = {}

        # NEW: Filter mature speakers by quality
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
            
            # NEW: Skip degraded mature speakers for reevaluation targets
            centroid_quality = getattr(ref, 'centroid_quality', 1.0)
            
            if ref.segment_count >= min_segments_for_mature:
                if centroid_quality >= 0.4:  # Only use healthy mature speakers as targets
                    mature_speakers[label] = ref
                elif self.debug:
                    console.print(
                        f"[yellow]⚠️ Skipping degraded mature {label} "
                        f"(quality={centroid_quality:.2f}) in reeval[/]"
                    )
            elif ref.segment_count <= max_segments_for_young:
                young_speakers[label] = ref

        if not mature_speakers or not young_speakers:
            return {
                "merges_performed": [],
                "speakers_checked": len(young_speakers),
                "mature_speakers": len(mature_speakers),
                "newborn_skipped": len(categories.get("newborn", {})),
                "degraded_skipped": len(categories.get("degraded_mature", {})),
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
                # NEW: Quality-weighted merge threshold
                target_quality = getattr(mature_speakers[best_mature_label], 'centroid_quality', 1.0)
                young_quality = getattr(young_ref, 'centroid_quality', 0.5)
                
                # Raise threshold for low-quality targets
                adjusted_threshold = merge_threshold + (1.0 - target_quality) * 0.1
                
                if best_similarity >= adjusted_threshold:
                    merges_to_perform.append({
                        "young_speaker": young_label,
                        "mature_speaker": best_mature_label,
                        "similarity": round(best_similarity, 4),
                        "young_segments": young_ref.segment_count,
                        "mature_segments": mature_speakers[best_mature_label].segment_count,
                        "target_quality": target_quality,
                        "young_quality": young_quality,
                    })
                    if self.debug:
                        console.print(
                            f"[yellow]🔍 Re-eval MERGE: {young_label} "
                            f"({young_ref.segment_count} segs, q={young_quality:.2f}) → "
                            f"{best_mature_label} "
                            f"({mature_speakers[best_mature_label].segment_count} segs, q={target_quality:.2f}) "
                            f"sim={best_similarity:.3f} (adj_threshold={adjusted_threshold:.3f})[/yellow]"
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
                    "target_quality": merge_info.get("target_quality"),
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
            "degraded_skipped": len(categories.get("degraded_mature", {})),
            "dry_run": dry_run,
        }

    # ------------------------------------------------------------------
    # Centroid validation and rebuilding
    # ------------------------------------------------------------------
    def validate_mature_centroids(
        self,
        quality_threshold: float = 0.5,
    ) -> Dict:
        """Validate mature speaker centroids and rebuild from core embeddings if degraded.
        
        This is the key improvement for overlap-aware maintenance.
        When a speaker's centroid quality drops (due to overlap contamination),
        this method attempts to rebuild it from only the clean, non-overlap
        core embeddings.
        """
        categories = self.get_speaker_categories()
        rebuilt = []
        
        # Check both mature and degraded_mature
        all_mature = {}
        all_mature.update(categories.get("mature", {}))
        all_mature.update(categories.get("degraded_mature", {}))
        
        for label, ref in all_mature.items():
            centroid_quality = getattr(ref, 'centroid_quality', 1.0)
            
            if centroid_quality >= quality_threshold:
                continue
            
            # Try to rebuild from core embeddings
            core_embs = getattr(ref, 'core_embeddings', None)
            if core_embs is None:
                # Fall back to all embeddings
                core_embs = ref.embeddings if hasattr(ref, 'embeddings') else []
            
            if not core_embs or len(core_embs) < 2:
                if self.debug:
                    console.print(
                        f"[yellow]⚠️ Cannot rebuild {label}: "
                        f"insufficient embeddings ({len(core_embs)})[/]"
                    )
                continue
            
            # Stack embeddings and compute new centroid
            stacked = np.vstack(core_embs)
            if len(core_embs) >= 3:
                new_centroid = np.median(stacked, axis=0, keepdims=True)
            else:
                new_centroid = np.mean(stacked, axis=0, keepdims=True)
            
            # Check consistency of core embeddings with new centroid
            from scipy.spatial.distance import cdist
            distances = cdist(stacked, new_centroid, metric="cosine")
            similarities = 1.0 - distances.flatten()
            
            # Use model-aware same threshold for consistency check
            same_threshold = getattr(self._labeler, 'threshold_same', 0.75)
            consistent_count = np.sum(similarities >= same_threshold)
            consistency_ratio = consistent_count / len(core_embs)
            
            if consistency_ratio >= 0.6:  # At least 60% of core embeddings are consistent
                old_quality = centroid_quality
                ref.centroid = new_centroid
                ref.centroid_quality = consistency_ratio
                
                rebuilt.append({
                    "speaker": label,
                    "old_quality": round(old_quality, 3),
                    "new_quality": round(consistency_ratio, 3),
                    "core_embeddings_used": len(core_embs),
                    "consistent_embeddings": int(consistent_count),
                    "consistency_ratio": round(consistency_ratio, 3),
                })
                
                if self.debug:
                    console.print(
                        f"[green]🔧 Rebuilt {label} centroid: "
                        f"quality {old_quality:.2f} → {consistency_ratio:.2f} "
                        f"({int(consistent_count)}/{len(core_embs)} consistent)[/]"
                    )
            else:
                if self.debug:
                    console.print(
                        f"[red]⚠️ Failed to rebuild {label}: "
                        f"consistency {consistency_ratio:.2f} < 0.6 "
                        f"({int(consistent_count)}/{len(core_embs)})[/]"
                    )
        
        return {
            "rebuilt": rebuilt,
            "total_checked": len(all_mature),
        }

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def consolidate_speakers(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict:
        """Consolidate similar speakers with quality-aware decisions."""
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
        qualities = []
        
        for label, ref in self._labeler._speakers.items():
            if ref.has_valid_centroid:
                speaker_labels.append(label)
                centroids.append(ref.centroid)
                qualities.append(getattr(ref, 'centroid_quality', 0.5))

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
                
                # NEW: Quality-adjusted threshold
                avg_quality = (qualities[i] + qualities[j]) / 2
                adjusted_threshold = threshold + (1.0 - avg_quality) * 0.1
                
                if sim >= adjusted_threshold:
                    merges_to_perform.append((
                        speaker_labels[i], 
                        speaker_labels[j], 
                        round(sim, 4),
                        round(avg_quality, 3)
                    ))
                    already_merged.add(speaker_labels[j])

        if not dry_run:
            for label1, label2, sim, quality in merges_to_perform:
                self.merge_speakers(label1, label2)
                if self.debug:
                    console.print(
                        f"[yellow]Consolidated: {label1} + {label2} "
                        f"(sim={sim:.3f}, avg_quality={quality:.3f})[/]"
                    )

        return {
            "merges_performed": [(m[0], m[1], m[2]) for m in merges_to_perform],
            "speakers_before": speakers_before,
            "speakers_after": len(self._labeler._speakers),
            "dry_run": dry_run,
        }

    # ------------------------------------------------------------------
    # Merge two speakers (quality-aware)
    # ------------------------------------------------------------------
    def merge_speakers(self, label1: str, label2: str) -> Optional[str]:
        """Merge two speaker references with quality-aware decisions.
        
        Key improvements:
        1. Considers centroid quality, not just segment count
        2. Tracks overlap contamination in merged speaker
        3. Rebuilds centroid from core embeddings when available
        4. Adjusts quality based on contamination ratio
        """
        speakers = self._labeler._speakers
        if label1 not in speakers or label2 not in speakers:
            return None
        if label1 == label2:
            return label1

        ref1 = speakers[label1]
        ref2 = speakers[label2]

        # NEW: Quality-weighted primary selection
        quality1 = getattr(ref1, 'centroid_quality', 0.5)
        quality2 = getattr(ref2, 'centroid_quality', 0.5)
        
        weighted_count1 = ref1.segment_count * max(0.1, quality1)
        weighted_count2 = ref2.segment_count * max(0.1, quality2)
        
        if weighted_count1 >= weighted_count2:
            primary, secondary = ref1, ref2
            primary_label = label1
            secondary_label = label2
        else:
            primary, secondary = ref2, ref1
            primary_label = label2
            secondary_label = label1

        # NEW: Only merge core embeddings from secondary
        core_embs_secondary = getattr(secondary, 'core_embeddings', None)
        if core_embs_secondary is not None and len(core_embs_secondary) > 0:
            emb_source = core_embs_secondary
        else:
            emb_source = secondary.embeddings if hasattr(secondary, 'embeddings') else []
        
        for emb in emb_source:
            if hasattr(primary, 'embeddings'):
                primary.embeddings.append(emb)
            if hasattr(primary, 'core_embeddings'):
                primary.core_embeddings.append(emb)
        
        # Update segment count based on core embeddings merged
        primary.segment_count += len(emb_source)
        primary.last_seen = max(primary.last_seen, secondary.last_seen)
        primary.first_seen = min(primary.first_seen, secondary.first_seen)

        # NEW: Track overlap contamination
        if hasattr(primary, 'overlap_segment_count'):
            secondary_overlap = getattr(secondary, 'overlap_segment_count', 0)
            primary.overlap_segment_count += secondary_overlap

        # Rebuild centroid from core embeddings if available
        if hasattr(primary, 'core_embeddings') and primary.core_embeddings:
            rebuild_source = primary.core_embeddings
        elif hasattr(primary, 'embeddings') and primary.embeddings:
            rebuild_source = primary.embeddings
        else:
            rebuild_source = []
        
        if rebuild_source:
            stacked = np.vstack(rebuild_source)
            if len(rebuild_source) >= 3:
                primary.centroid = np.median(stacked, axis=0, keepdims=True)
            else:
                primary.centroid = np.mean(stacked, axis=0, keepdims=True)
            
            # NEW: Update centroid quality
            if hasattr(primary, 'centroid_quality'):
                # Blend qualities with weight toward the better one
                max_quality = max(quality1, quality2)
                min_quality = min(quality1, quality2)
                blended_quality = max_quality * 0.7 + min_quality * 0.3
                
                # Apply contamination penalty
                if hasattr(primary, 'overlap_segment_count') and primary.segment_count > 0:
                    contamination_ratio = primary.overlap_segment_count / primary.segment_count
                    quality_penalty = contamination_ratio * 0.3
                    blended_quality = max(0.1, blended_quality - quality_penalty)
                
                primary.centroid_quality = min(1.0, blended_quality)

        # Clean up secondary speaker
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
                f"quality: {getattr(primary, 'centroid_quality', 'N/A')}, "
                f"removed: {secondary_label})[/bold yellow]"
            )
        return primary_label
