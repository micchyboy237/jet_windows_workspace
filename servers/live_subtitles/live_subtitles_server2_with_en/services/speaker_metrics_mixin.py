# Jet_Windows_Workspace/servers/live_subtitles/live_subtitles_server2_with_en/services/speaker_metrics_mixin.py

"""
SpeakerMetricsMixin - Provides metrics computation methods for SegmentSpeakerLabeler.

This mixin adds comprehensive speaker analytics without modifying the core labeling logic.
All methods access self._speakers, self._segment_groups, and other internal state
to compute intra-speaker cohesion, inter-speaker separation, segment group health,
outlier pool health, and overall system health summaries.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class SpeakerMetricsMixin:
    """
    Mixin providing speaker metrics and health computation methods.
    
    Designed to be mixed into SegmentSpeakerLabeler, accessing its internal state
    to compute various metrics about speaker quality, separation, and system health.
    
    Metrics Categories:
    1. Intra-speaker Cohesion - How consistent are segments within a speaker?
    2. Inter-speaker Separation - How distinct are different speakers?
    3. Segment Group Health - Quality metrics for segment labeling
    4. Outlier Pool Health - Outlier buffer statistics
    5. Overall System Health - Combined health summary
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. INTRA-SPEAKER COHESION METRICS
    # ─────────────────────────────────────────────────────────────────────
    
    def get_speaker_cohesion(self, speaker_label: str) -> Optional[Dict[str, Any]]:
        """
        Compute intra-speaker cohesion metrics for a single speaker.
        
        Measures:
        - mean_pairwise_similarity: Average cosine similarity between all pairs of embeddings
        - centroid_similarity_std: Standard deviation of similarities to centroid
        - min_similarity_to_centroid: Minimum similarity any embedding has to centroid
        - max_similarity_to_centroid: Maximum similarity any embedding has to centroid
        - embedding_count: Number of embeddings used
        - cohesion_score: Composite score (0-1) combining the above
        
        Returns None if speaker not found or has insufficient embeddings.
        """
        if speaker_label not in self._speakers:
            logger.warning(f"Speaker {speaker_label} not found")
            return None
        
        ref = self._speakers[speaker_label]
        embeddings = ref.embeddings
        
        if len(embeddings) < 2:
            logger.debug(f"Speaker {speaker_label} has insufficient embeddings ({len(embeddings)})")
            return {
                "speaker_label": speaker_label,
                "embedding_count": len(embeddings),
                "mean_pairwise_similarity": None,
                "centroid_similarity_std": None,
                "min_similarity_to_centroid": None,
                "max_similarity_to_centroid": None,
                "cohesion_score": 1.0 if len(embeddings) == 1 else 0.0,
                "status": "insufficient_data",
                "segment_count": ref.segment_count,
                "first_seen": ref.first_seen,
                "last_seen": ref.last_seen,
                "active_duration": ref.active_duration,
            }
        
        # Stack embeddings
        stacked = np.vstack(embeddings)
        
        # Mean pairwise similarity
        if len(embeddings) > 1:
            pairwise_distances = cdist(stacked, stacked, metric="cosine")
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(pairwise_distances, k=1)
            if len(triu_indices[0]) > 0:
                pairwise_sims = 1.0 - pairwise_distances[triu_indices]
                mean_pairwise = float(np.mean(pairwise_sims))
                std_pairwise = float(np.std(pairwise_sims))
                min_pairwise = float(np.min(pairwise_sims))
            else:
                mean_pairwise = 1.0
                std_pairwise = 0.0
                min_pairwise = 1.0
        else:
            mean_pairwise = 1.0
            std_pairwise = 0.0
            min_pairwise = 1.0
        
        # Similarities to centroid
        if ref.has_valid_centroid:
            centroid_2d = ref.centroid.reshape(1, -1) if ref.centroid.ndim == 1 else ref.centroid
            centroid_distances = cdist(stacked, centroid_2d, metric="cosine").flatten()
            centroid_sims = 1.0 - centroid_distances
            centroid_sim_mean = float(np.mean(centroid_sims))
            centroid_sim_std = float(np.std(centroid_sims))
            centroid_sim_min = float(np.min(centroid_sims))
            centroid_sim_max = float(np.max(centroid_sims))
        else:
            centroid_sim_mean = None
            centroid_sim_std = None
            centroid_sim_min = None
            centroid_sim_max = None
        
        # Composite cohesion score (0-1)
        # Weights: mean_pairwise (0.4), 1-std_pairwise (0.3), centroid_sim_mean (0.3)
        if mean_pairwise is not None and centroid_sim_mean is not None:
            normalized_std = max(0, 1.0 - std_pairwise * 2)  # Lower std = better
            cohesion_score = round(
                0.4 * mean_pairwise + 
                0.3 * normalized_std + 
                0.3 * centroid_sim_mean, 
                4
            )
        else:
            cohesion_score = mean_pairwise if mean_pairwise is not None else 0.0
        
        return {
            "speaker_label": speaker_label,
            "embedding_count": len(embeddings),
            "segment_count": ref.segment_count,
            "mean_pairwise_similarity": round(mean_pairwise, 4) if mean_pairwise is not None else None,
            "std_pairwise_similarity": round(std_pairwise, 4),
            "min_pairwise_similarity": round(min_pairwise, 4),
            "centroid_similarity_mean": round(centroid_sim_mean, 4) if centroid_sim_mean is not None else None,
            "centroid_similarity_std": round(centroid_sim_std, 4) if centroid_sim_std is not None else None,
            "min_similarity_to_centroid": round(centroid_sim_min, 4) if centroid_sim_min is not None else None,
            "max_similarity_to_centroid": round(centroid_sim_max, 4) if centroid_sim_max is not None else None,
            "cohesion_score": round(cohesion_score, 4),
            "status": "healthy" if cohesion_score >= 0.7 else "warning" if cohesion_score >= 0.5 else "critical",
            "centroid_quality": ref.centroid_quality,
            "first_seen": ref.first_seen,
            "last_seen": ref.last_seen,
            "active_duration": ref.active_duration,
        }
    
    def get_all_speakers_cohesion(self) -> Dict[str, Any]:
        """
        Compute cohesion metrics for all speakers.
        
        Returns a summary with per-speaker details and overall averages.
        """
        speaker_metrics = {}
        cohesion_scores = []
        
        for label in self._speakers:
            metrics = self.get_speaker_cohesion(label)
            if metrics:
                speaker_metrics[label] = metrics
                if metrics["cohesion_score"] is not None:
                    cohesion_scores.append(metrics["cohesion_score"])
        
        avg_cohesion = float(np.mean(cohesion_scores)) if cohesion_scores else 0.0
        
        # Categorize speakers
        healthy = [l for l, m in speaker_metrics.items() if m.get("status") == "healthy"]
        warning = [l for l, m in speaker_metrics.items() if m.get("status") == "warning"]
        critical = [l for l, m in speaker_metrics.items() if m.get("status") == "critical"]
        
        return {
            "total_speakers": len(speaker_metrics),
            "average_cohesion_score": round(avg_cohesion, 4),
            "healthy_count": len(healthy),
            "warning_count": len(warning),
            "critical_count": len(critical),
            "healthy_speakers": healthy,
            "warning_speakers": warning,
            "critical_speakers": critical,
            "speakers": speaker_metrics,
            "computed_at": datetime.now().isoformat(),
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. INTER-SPEAKER SEPARATION METRICS
    # ─────────────────────────────────────────────────────────────────────
    
    def get_speaker_separation_matrix(self) -> Dict[str, Any]:
        """
        Compute inter-speaker separation metrics.
        
        Measures:
        - pairwise_centroid_similarities: Similarity between all speaker centroids
        - mean_separation: Average separation between speakers (1 - similarity)
        - min_separation: Closest pair of speakers
        - max_separation: Most distant pair of speakers
        - ambiguous_pairs: Speaker pairs with similarity > threshold (potential confusion)
        - separation_health: Overall separation quality assessment
        
        Returns None for fields if < 2 speakers with valid centroids.
        """
        # Get speakers with valid centroids
        valid_speakers = {
            label: ref for label, ref in self._speakers.items() 
            if ref.has_valid_centroid
        }
        
        if len(valid_speakers) < 2:
            return {
                "total_speakers_with_centroids": len(valid_speakers),
                "pairwise_similarities": {},
                "mean_separation": None,
                "min_separation": None,
                "max_separation": None,
                "ambiguous_pairs": [],
                "separation_health": "insufficient_data",
                "computed_at": datetime.now().isoformat(),
            }
        
        labels = list(valid_speakers.keys())
        centroids = np.vstack([ref.centroid for ref in valid_speakers.values()])
        
        # Compute pairwise similarities
        distances = cdist(centroids, centroids, metric="cosine")
        similarities = 1.0 - distances
        
        # Build pairwise dict
        pairwise = {}
        separation_values = []
        ambiguous_pairs = []
        
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = float(similarities[i, j])
                pair_key = f"{labels[i]}___{labels[j]}"
                sep = 1.0 - sim
                pairwise[pair_key] = {
                    "speaker_1": labels[i],
                    "speaker_2": labels[j],
                    "cosine_similarity": round(sim, 4),
                    "separation": round(sep, 4),
                    "segments_1": valid_speakers[labels[i]].segment_count,
                    "segments_2": valid_speakers[labels[j]].segment_count,
                }
                separation_values.append(sep)
                
                # Flag ambiguous pairs (similarity > 0.7 means they're quite similar)
                if sim > 0.7:
                    ambiguous_pairs.append({
                        "speaker_1": labels[i],
                        "speaker_2": labels[j],
                        "similarity": round(sim, 4),
                        "risk": "high" if sim > 0.85 else "medium" if sim > 0.75 else "low",
                    })
        
        mean_sep = float(np.mean(separation_values)) if separation_values else 0.0
        min_sep = float(np.min(separation_values)) if separation_values else 0.0
        max_sep = float(np.max(separation_values)) if separation_values else 0.0
        
        # Health assessment
        if len(ambiguous_pairs) == 0:
            health = "excellent"
        elif mean_sep > 0.5:
            health = "good"
        elif mean_sep > 0.3:
            health = "fair"
        else:
            health = "poor"
        
        return {
            "total_speakers_with_centroids": len(valid_speakers),
            "pairwise_similarities": pairwise,
            "mean_separation": round(mean_sep, 4),
            "min_separation": round(min_sep, 4),
            "max_separation": round(max_sep, 4),
            "ambiguous_pairs": ambiguous_pairs,
            "ambiguous_count": len(ambiguous_pairs),
            "separation_health": health,
            "computed_at": datetime.now().isoformat(),
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. SEGMENT GROUP HEALTH METRICS
    # ─────────────────────────────────────────────────────────────────────
    
    def get_segment_group_health(self) -> Dict[str, Any]:
        """
        Compute health metrics for segment groups (labeling quality).
        
        Analyzes all processed segments in self._segment_groups to assess:
        - Match confidence distribution
        - Match type distribution
        - Temporal consistency
        - Label switching frequency
        - Unresolved outlier segments
        """
        if not self._segment_groups:
            return {
                "total_segments": 0,
                "confidence_distribution": {},
                "match_type_distribution": {},
                "label_switches": 0,
                "temporal_consistency_score": None,
                "unresolved_outliers": 0,
                "status": "no_data",
                "computed_at": datetime.now().isoformat(),
            }
        
        total_segments = len(self._segment_groups)
        
        # Collect all matches with confidence
        confidences = []
        match_types = defaultdict(int)
        primary_labels = []
        unresolved_outliers = 0
        
        for group in self._segment_groups:
            matches = group.get("matches", [])
            for match in matches:
                conf = match.get("confidence", 0)
                if conf is not None:
                    confidences.append(conf)
                mt = match.get("match_type", "unknown")
                match_types[mt] += 1
                
                # Track primary labels and outliers
                if match.get("is_primary"):
                    primary_labels.append(match.get("label", "UNKNOWN"))
                    # Check for unresolved outliers
                    if match.get("label", "").startswith("OUTLIER_"):
                        unresolved_outliers += 1
        
        # Confidence distribution
        bins = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0, "0.85-0.95": 0, "0.95-1.0": 0}
        for c in confidences:
            if c < 0.3:
                bins["0.0-0.3"] += 1
            elif c < 0.5:
                bins["0.3-0.5"] += 1
            elif c < 0.7:
                bins["0.5-0.7"] += 1
            elif c < 0.85:
                bins["0.7-0.85"] += 1
            elif c < 0.95:
                bins["0.85-0.95"] += 1
            else:
                bins["0.95-1.0"] += 1
        
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        # Label switch frequency
        label_switches = 0
        for i in range(1, len(primary_labels)):
            if primary_labels[i] != primary_labels[i-1]:
                label_switches += 1
        
        # Temporal consistency: ratio of same-speaker consecutive segments
        if len(primary_labels) > 1:
            same_consecutive = sum(
                1 for i in range(1, len(primary_labels)) 
                if primary_labels[i] == primary_labels[i-1]
            )
            temporal_consistency = round(same_consecutive / (len(primary_labels) - 1), 4)
        else:
            temporal_consistency = 1.0
        
        # Health status
        if mean_confidence >= 0.8 and temporal_consistency >= 0.7:
            status = "healthy"
        elif mean_confidence >= 0.6 and temporal_consistency >= 0.5:
            status = "fair"
        else:
            status = "needs_attention"
        
        return {
            "total_segments": total_segments,
            "confidence_distribution": bins,
            "mean_confidence": round(mean_confidence, 4),
            "match_type_distribution": dict(match_types),
            "label_switches": label_switches,
            "temporal_consistency_score": temporal_consistency,
            "unresolved_outliers": unresolved_outliers,
            "status": status,
            "primary_label_sequence": primary_labels[-20:] if primary_labels else [],  # Last 20 for preview
            "computed_at": datetime.now().isoformat(),
        }
    
    def get_segment_detail(self, segment_index: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific segment by index.
        
        Parameters
        ----------
        segment_index : int
            Index into self._segment_groups (0-based)
            
        Returns None if index out of range.
        """
        if segment_index < 0 or segment_index >= len(self._segment_groups):
            return None
        
        group = self._segment_groups[segment_index]
        matches = group.get("matches", [])
        
        # Enrich matches with speaker info
        enriched_matches = []
        for match in matches:
            label = match.get("label", "")
            enriched = dict(match)
            
            # Add speaker details if available
            if label in self._speakers:
                ref = self._speakers[label]
                enriched["speaker_info"] = {
                    "segment_count": ref.segment_count,
                    "first_seen": ref.first_seen,
                    "last_seen": ref.last_seen,
                    "centroid_quality": ref.centroid_quality,
                    "has_valid_centroid": ref.has_valid_centroid,
                }
            
            # Check outlier status
            if label.startswith("OUTLIER_"):
                enriched["outlier_info"] = self._get_outlier_info(label)
            
            enriched_matches.append(enriched)
        
        # Get surrounding context
        prev_segment = None
        next_segment = None
        if segment_index > 0:
            prev_group = self._segment_groups[segment_index - 1]
            prev_primary = next((m for m in prev_group.get("matches", []) if m.get("is_primary")), None)
            prev_segment = {
                "segment_id": prev_group.get("segment_id"),
                "timestamp": prev_group.get("timestamp"),
                "primary_label": prev_primary.get("label") if prev_primary else None,
            }
        if segment_index < len(self._segment_groups) - 1:
            next_group = self._segment_groups[segment_index + 1]
            next_primary = next((m for m in next_group.get("matches", []) if m.get("is_primary")), None)
            next_segment = {
                "segment_id": next_group.get("segment_id"),
                "timestamp": next_group.get("timestamp"),
                "primary_label": next_primary.get("label") if next_primary else None,
            }
        
        return {
            "segment_index": segment_index,
            "segment_id": group.get("segment_id"),
            "timestamp": group.get("timestamp"),
            "audio_duration": group.get("audio_duration"),
            "matches": enriched_matches,
            "match_count": len(enriched_matches),
            "primary_match": enriched_matches[0] if enriched_matches else None,
            "previous_segment": prev_segment,
            "next_segment": next_segment,
        }
    
    def _get_outlier_info(self, outlier_label: str) -> Optional[Dict[str, Any]]:
        """Get information about an outlier from the outlier pool."""
        if not hasattr(self, 'outlier_pool') or not self.use_outlier_buffer:
            return None
        
        try:
            # Check if outlier exists in pool
            if outlier_label in self.outlier_pool:
                entry = self.outlier_pool[outlier_label]
                return {
                    "label": outlier_label,
                    "timestamp": getattr(entry, 'timestamp', None),
                    "promoted": False,
                }
            
            # Check if it was promoted
            for promo in getattr(self.outlier_pool, '_promotions', []):
                if outlier_label in promo.outlier_labels:
                    return {
                        "label": outlier_label,
                        "promoted": True,
                        "target_speaker": promo.target_speaker,
                        "confidence": promo.confidence,
                    }
        except Exception as e:
            logger.debug(f"Error getting outlier info for {outlier_label}: {e}")
        
        return {"label": outlier_label, "status": "unknown"}
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. OUTLIER POOL HEALTH METRICS
    # ─────────────────────────────────────────────────────────────────────
    
    def get_outlier_pool_health(self) -> Dict[str, Any]:
        """
        Compute health metrics for the outlier pool.
        
        Analyzes outlier buffer state, promotion rates, and unresolved outliers.
        """
        if not hasattr(self, 'outlier_pool') or not self.use_outlier_buffer:
            return {
                "enabled": False,
                "status": "disabled",
                "computed_at": datetime.now().isoformat(),
            }
        
        outlier_count = self.outlier_pool.count if hasattr(self.outlier_pool, 'count') else 0
        promotion_count = self.outlier_pool.promotion_count if hasattr(self.outlier_pool, 'promotion_count') else 0
        
        # Get promotion history
        promotions = []
        if hasattr(self.outlier_pool, '_promotions'):
            for promo in self.outlier_pool._promotions:
                promotions.append({
                    "type": getattr(promo, 'type', 'unknown'),
                    "outlier_labels": getattr(promo, 'outlier_labels', []),
                    "target_speaker": getattr(promo, 'target_speaker', ''),
                    "confidence": getattr(promo, 'confidence', 0.0),
                })
        
        # Check for old outliers that haven't been promoted
        max_age_warning = 300  # 5 minutes
        old_outliers = 0
        if hasattr(self.outlier_pool, 'entries'):
            current_time = datetime.now().timestamp()
            for entry in self.outlier_pool.entries.values():
                ts = getattr(entry, 'timestamp', 0)
                if ts and (current_time - ts) > max_age_warning:
                    old_outliers += 1
        
        # Health assessment
        if outlier_count == 0 and promotion_count > 0:
            status = "healthy"
        elif outlier_count <= 3:
            status = "normal"
        elif outlier_count <= 10:
            status = "elevated"
        else:
            status = "high"
        
        if old_outliers > 0:
            status = "stale_outliers" if status == "normal" else status
        
        return {
            "enabled": True,
            "active_outliers": outlier_count,
            "total_promotions": promotion_count,
            "promotion_rate": round(promotion_count / max(1, promotion_count + outlier_count), 4),
            "promotions": promotions[-10:],  # Last 10 promotions
            "old_outliers": old_outliers,
            "max_capacity": getattr(self.outlier_pool, 'max_count', None),
            "status": status,
            "computed_at": datetime.now().isoformat(),
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # 5. OVERALL SYSTEM HEALTH SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    
    def get_speaker_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive system health summary.
        
        Aggregates all metrics into a single overview combining:
        - Speaker cohesion stats
        - Speaker separation stats
        - Segment group health
        - Outlier pool health
        - System statistics
        """
        cohesion = self.get_all_speakers_cohesion()
        separation = self.get_speaker_separation_matrix()
        segment_health = self.get_segment_group_health()
        outlier_health = self.get_outlier_pool_health()
        
        # Get speaker categories
        speaker_categories = {}
        if hasattr(self, '_get_speaker_categories'):
            try:
                speaker_categories = self._get_speaker_categories()
            except Exception as e:
                logger.debug(f"Could not get speaker categories: {e}")
        
        # Compute overall health score (weighted)
        scores = []
        weights = []
        
        if cohesion.get("average_cohesion_score") is not None:
            scores.append(cohesion["average_cohesion_score"])
            weights.append(0.25)
        
        if separation.get("mean_separation") is not None:
            scores.append(separation["mean_separation"])
            weights.append(0.25)
        
        if segment_health.get("temporal_consistency_score") is not None:
            scores.append(segment_health["temporal_consistency_score"])
            weights.append(0.25)
        
        if segment_health.get("mean_confidence") is not None:
            scores.append(segment_health["mean_confidence"])
            weights.append(0.25)
        
        if scores:
            total_weight = sum(weights)
            overall_score = round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 4)
        else:
            overall_score = 0.0
        
        # Determine overall status
        if overall_score >= 0.8:
            overall_status = "excellent"
        elif overall_score >= 0.65:
            overall_status = "good"
        elif overall_score >= 0.5:
            overall_status = "fair"
        else:
            overall_status = "needs_attention"
        
        # Generate recommendations
        recommendations = []
        if cohesion.get("critical_count", 0) > 0:
            recommendations.append(f"{cohesion['critical_count']} speaker(s) have critical cohesion - consider merging or reviewing")
        if separation.get("ambiguous_count", 0) > 0:
            recommendations.append(f"{separation['ambiguous_count']} ambiguous speaker pair(s) detected - possible over-segmentation")
        if segment_health.get("unresolved_outliers", 0) > 0:
            recommendations.append(f"{segment_health['unresolved_outliers']} unresolved outlier segment(s) - consider running consolidation")
        if outlier_health.get("status") == "stale_outliers":
            recommendations.append(f"Stale outliers detected - consider reviewing outlier pool")
        if self.total_segments_processed < 10:
            recommendations.append("Low segment count - metrics will improve with more data")
        
        return {
            "overall_health_score": overall_score,
            "overall_status": overall_status,
            "system_stats": {
                "total_segments_processed": self.total_segments_processed,
                "total_speakers_created": self.total_speakers_created,
                "active_speakers": self.speaker_count if hasattr(self, 'speaker_count') else len(self._speakers),
                "rejected_updates": getattr(self, '_rejected_updates', 0),
                "merge_count": len(getattr(self, '_merge_history', [])),
                "speaker_categories": speaker_categories,
            },
            "cohesion": {
                "average_score": cohesion.get("average_cohesion_score"),
                "healthy": cohesion.get("healthy_count", 0),
                "warning": cohesion.get("warning_count", 0),
                "critical": cohesion.get("critical_count", 0),
            },
            "separation": {
                "mean_separation": separation.get("mean_separation"),
                "ambiguous_pairs": separation.get("ambiguous_count", 0),
                "health": separation.get("separation_health"),
            },
            "segment_health": {
                "mean_confidence": segment_health.get("mean_confidence"),
                "temporal_consistency": segment_health.get("temporal_consistency_score"),
                "status": segment_health.get("status"),
            },
            "outlier_health": {
                "active_outliers": outlier_health.get("active_outliers", 0),
                "total_promotions": outlier_health.get("total_promotions", 0),
                "status": outlier_health.get("status"),
            },
            "recommendations": recommendations,
            "computed_at": datetime.now().isoformat(),
        }
    
    def get_speaker_timeline(self) -> Dict[str, Any]:
        """
        Build a timeline of speaker activity across segments.
        
        Returns speaker labels with their segment timestamps for visualization.
        """
        if not self._segment_groups:
            return {"timeline": [], "speakers": []}
        
        timeline = []
        for i, group in enumerate(self._segment_groups):
            primary = next((m for m in group.get("matches", []) if m.get("is_primary")), None)
            timeline.append({
                "index": i,
                "segment_id": group.get("segment_id"),
                "timestamp": group.get("timestamp"),
                "audio_duration": group.get("audio_duration"),
                "primary_label": primary.get("label") if primary else "UNKNOWN",
                "primary_confidence": primary.get("confidence") if primary else 0,
                "match_type": primary.get("match_type") if primary else "unknown",
            })
        
        # Get unique speaker labels
        speakers = list(set(
            entry["primary_label"] for entry in timeline 
            if entry["primary_label"] != "UNKNOWN"
        ))
        speakers.sort()
        
        return {
            "total_segments": len(timeline),
            "unique_speakers": len(speakers),
            "speakers": speakers,
            "timeline": timeline,
            "computed_at": datetime.now().isoformat(),
        }
    
    def get_speaker_segment_list(
        self, 
        speaker_label: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get list of segments, optionally filtered by speaker.
        
        Parameters
        ----------
        speaker_label : str, optional
            Filter segments where this speaker is the primary match
        limit : int
            Maximum number of segments to return
        offset : int
            Number of segments to skip
            
        Returns paginated segment list with metadata.
        """
        if not self._segment_groups:
            return {
                "total": 0,
                "limit": limit,
                "offset": offset,
                "segments": [],
            }
        
        filtered = []
        for i, group in enumerate(self._segment_groups):
            primary = next((m for m in group.get("matches", []) if m.get("is_primary")), None)
            primary_label = primary.get("label") if primary else "UNKNOWN"
            
            if speaker_label is None or primary_label == speaker_label:
                filtered.append({
                    "index": i,
                    "segment_id": group.get("segment_id"),
                    "timestamp": group.get("timestamp"),
                    "audio_duration": group.get("audio_duration"),
                    "primary_label": primary_label,
                    "primary_confidence": primary.get("confidence") if primary else 0,
                    "match_type": primary.get("match_type") if primary else "unknown",
                })
        
        total = len(filtered)
        paginated = filtered[offset:offset + limit]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "segments": paginated,
            "filter_speaker": speaker_label,
        }
