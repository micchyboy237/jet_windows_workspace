"""
SpeakerMetricsMixin - adds intra/inter speaker metrics and segment group health
computation to SegmentSpeakerLabeler.

Mixed into SegmentSpeakerLabeler to access:
    - self._speakers: Dict[str, SpeakerReference]
    - self._segment_groups: List[Dict] (segment group history)
    - self.outlier_pool: OutlierPool
    - self._rejected_updates: int
    - self.total_segments_processed: int
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from rich.console import Console

try:
    from services.helpers.speaker_metrics import (
        HealthStatus,
        InterSpeakerInput,
        IntraSpeakerInput,
        compute_inter_speaker_separation,
        compute_intra_speaker_variance,
        cosine_distance,
    )
except ImportError:
    from helpers.speaker_metrics import (
        HealthStatus,
        InterSpeakerInput,
        IntraSpeakerInput,
        compute_inter_speaker_separation,
        compute_intra_speaker_variance,
        cosine_distance,
    )

console = Console()


class SpeakerMetricsMixin:
    """
    Mixin providing speaker metrics methods for SegmentSpeakerLabeler.
    
    Requires the host class to have:
        - self._speakers: Dict[str, SpeakerReference]
        - self._segment_groups: List[Dict]
        - self.outlier_pool: OutlierPool (optional)
        - self._rejected_updates: int
        - self.total_segments_processed: int
        
    SpeakerReference must have:
        - .label: str
        - .embeddings: List[np.ndarray]  (each shape (1, dim) or (dim,))
        - .segment_count: int
        - .has_valid_centroid: bool
        - .centroid: np.ndarray
        - .embedding_metadata: List[Dict] with keys: segment_id, timestamp
        - .first_seen: float
        - .last_seen: float
        - .active_duration: float
        - .centroid_quality: float
    """
    
    # ------------------------------------------------------------------
    # INTRA-SPEAKER METRICS
    # ------------------------------------------------------------------
    
    def compute_intra_speaker_metrics(
        self,
        label: Optional[str] = None,
        healthy_threshold: float = 0.70,
        warning_threshold: float = 0.55,
    ) -> Dict:
        """
        Compute intra-speaker cohesion metrics for one or all speakers.
        
        Uses cosine similarity to centroid (higher = more cohesive).
        Health classification:
            - similarity >= healthy_threshold → HEALTHY
            - similarity >= warning_threshold  → WARNING
            - otherwise                         → UNHEALTHY
        
        Parameters
        ----------
        label : str, optional
            Specific speaker label. If None, computes for all speakers.
        healthy_threshold : float
            Mean similarity above this is "healthy" (default 0.70).
        warning_threshold : float
            Mean similarity above this is "warning" (default 0.55).
            
        Returns
        -------
        dict with keys:
            - speakers: list of per-speaker metrics dicts
            - overall_status: HealthStatus for the worst speaker
            - total_speakers_analyzed: int
            - error: str if something went wrong
        """
        speakers_to_analyze = {}
        
        if label is not None:
            if label not in self._speakers:
                console.print(
                    f"[warning]compute_intra_speaker_metrics: '{label}' not found in "
                    f"{list(self._speakers.keys())}[/]"
                )
                return {
                    "speakers": [],
                    "overall_status": HealthStatus.UNHEALTHY.value,
                    "total_speakers_analyzed": 0,
                    "error": f"Speaker '{label}' not found",
                }
            speakers_to_analyze[label] = self._speakers[label]
        else:
            speakers_to_analyze = self._speakers
        
        results = []
        worst_status = HealthStatus.HEALTHY
        
        for spk_label, ref in speakers_to_analyze.items():
            if not ref.embeddings or len(ref.embeddings) == 0:
                console.print(f"[dim]Skipping {spk_label}: no embeddings[/]")
                continue
            
            # Build intra-speaker input for the shared helper
            embeddings_array = np.vstack([
                emb.reshape(1, -1) if emb.ndim == 1 else emb
                for emb in ref.embeddings
            ]).astype(np.float64)
            
            # Get other centroids for silhouette calculation
            other_centroids = {}
            for other_label, other_ref in self._speakers.items():
                if other_label != spk_label and other_ref.has_valid_centroid:
                    other_centroids[other_label] = other_ref.centroid.flatten()
            
            try:
                intra_input: IntraSpeakerInput = {
                    "label": spk_label,
                    "embeddings": embeddings_array,
                }
                
                intra_result = compute_intra_speaker_variance(
                    speaker_input=intra_input,
                    healthy_threshold=healthy_threshold,
                    warning_threshold=warning_threshold,
                    min_embeddings_for_mature=getattr(self, 'mature_segment_count', 5),
                    other_centroids=other_centroids if other_centroids else None,
                )
            except Exception as e:
                console.print(f"[error]Intra-speaker computation failed for {spk_label}: {e}[/]")
                continue
            
            # Build segment-level data for detailed view
            centroid = np.mean(embeddings_array, axis=0)
            segments_data = []
            for i in range(len(ref.embeddings)):
                emb = embeddings_array[i]
                sim_to_centroid = float(1.0 - cosine_distance(emb, centroid))
                
                meta = {}
                if hasattr(ref, 'embedding_metadata') and i < len(ref.embedding_metadata):
                    meta = ref.embedding_metadata[i]
                
                seg_id = meta.get('segment_id', f"segment_{i}")
                timestamp = meta.get('timestamp', 0.0)
                duration = self._estimate_segment_duration(ref, i, meta)
                is_core = meta.get('is_core', True)
                
                segments_data.append({
                    "id": seg_id,
                    "similarity": round(sim_to_centroid, 4),
                    "distance": round(1.0 - sim_to_centroid, 4),
                    "timestamp": timestamp,
                    "duration": round(duration, 4),
                    "is_core": is_core,
                })
            
            # Sort segments by timestamp for timeline visualization
            segments_data.sort(key=lambda s: s["timestamp"])
            
            # Build per-speaker result
            speaker_result = {
                "label": spk_label,
                "segmentsCount": len(ref.embeddings),
                "health": intra_result["status"].value,
                "meanSimilarity": intra_result["mean_similarity"],
                "stdSimilarity": intra_result["std_similarity"],
                "minSimilarity": intra_result["min_similarity"],
                "silhouetteScore": intra_result["silhouette_score"],
                "isMature": intra_result["is_mature"],
                "firstSeen": ref.first_seen if ref.first_seen else 0.0,
                "lastSeen": ref.last_seen,
                "activeDuration": ref.active_duration,
                "centroidQuality": ref.centroid_quality,
                "segments": segments_data,
            }
            
            results.append(speaker_result)
            
            # Track worst status
            status_order = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.WARNING: 1,
                HealthStatus.UNHEALTHY: 2,
            }
            current_status = intra_result["status"]
            if status_order.get(current_status, 0) > status_order.get(worst_status, 0):
                worst_status = current_status
        
        # Sort results: unhealthy first, then warning, then healthy
        results.sort(key=lambda r: {
            HealthStatus.UNHEALTHY.value: 0,
            HealthStatus.WARNING.value: 1,
            HealthStatus.HEALTHY.value: 2,
        }.get(r["health"], 3))
        
        console.print(
            f"[info]✓ Intra-speaker metrics: {len(results)} speakers analyzed, "
            f"worst_status={worst_status.value}[/]"
        )
        
        return {
            "speakers": results,
            "overall_status": worst_status.value,
            "total_speakers_analyzed": len(results),
        }
    
    # ------------------------------------------------------------------
    # INTER-SPEAKER METRICS
    # ------------------------------------------------------------------
    
    def compute_inter_speaker_metrics(
        self,
        healthy_threshold: float = 0.5,
        warning_threshold: float = 0.3,
    ) -> Dict:
        """
        Compute inter-speaker separation using speaker centroids.
        
        Higher distance = better separation between speakers.
        Health classification:
            - mean_separation >= healthy_threshold AND min >= 0.3 → HEALTHY
            - mean_separation >= warning_threshold AND min >= 0.15  → WARNING
            - otherwise                                              → UNHEALTHY
        
        Parameters
        ----------
        healthy_threshold : float
            Mean separation distance above this is "healthy" (default 0.5).
        warning_threshold : float
            Mean separation above this is "warning" (default 0.3).
            
        Returns
        -------
        dict with:
            - meanSeparation, stdSeparation, minSeparation, maxSeparation
            - health: overall health status
            - pairwise: list of {speaker1, speaker2, distance, similarity}
            - num_speakers: int
            - closest_pair: dict with speaker1, speaker2, distance
            - error: str if something went wrong
        """
        # Collect valid speaker embeddings
        speaker_embeddings = {}
        for spk_label, ref in self._speakers.items():
            if not ref.has_valid_centroid:
                continue
            if not ref.embeddings:
                continue
            embeddings_list = [emb.flatten() for emb in ref.embeddings]
            if not embeddings_list:
                continue
            speaker_embeddings[spk_label] = np.array(embeddings_list, dtype=np.float64)
        
        if len(speaker_embeddings) < 2:
            console.print(
                f"[warning]Inter-speaker metrics: need >= 2 speakers with centroids, "
                f"got {len(speaker_embeddings)}[/]"
            )
            return {
                "meanSeparation": 0.0,
                "stdSeparation": 0.0,
                "minSeparation": 0.0,
                "maxSeparation": 0.0,
                "health": HealthStatus.UNHEALTHY.value,
                "pairwise": [],
                "num_speakers": len(speaker_embeddings),
                "closest_pair": None,
                "error": "Need at least 2 speakers with valid centroids",
            }
        
        try:
            inter_input: InterSpeakerInput = {
                "speakers": speaker_embeddings,
            }
            result = compute_inter_speaker_separation(
                speaker_input=inter_input,
                healthy_threshold=healthy_threshold,
                warning_threshold=warning_threshold,
            )
            
            # Build pairwise with both distance and similarity for UI flexibility
            pairwise = []
            for p in result["pairwise_distances"]:
                pairwise.append({
                    "speaker1": p["speaker_id_1"],
                    "speaker2": p["speaker_id_2"],
                    "distance": round(p["distance"], 4),
                    "similarity": round(1.0 - p["distance"], 4),
                })
            
            # Sort pairwise: most similar (closest) first - these are the risk pairs
            pairwise.sort(key=lambda p: p["distance"])
            
            closest = None
            if result.get("closest_pair"):
                cp = result["closest_pair"]
                closest = {
                    "speaker1": cp[0],
                    "speaker2": cp[1],
                    "distance": round(result["min_separation"], 4),
                    "similarity": round(1.0 - result["min_separation"], 4),
                }
            
            console.print(
                f"[info]✓ Inter-speaker metrics: {result['num_speakers']} speakers, "
                f"mean_sep={result['mean_separation']:.4f}, "
                f"status={result['status'].value}[/]"
            )
            
            return {
                "meanSeparation": round(result["mean_separation"], 4),
                "stdSeparation": round(result["std_separation"], 4),
                "minSeparation": round(result["min_separation"], 4),
                "maxSeparation": round(result["max_separation"], 4),
                "health": result["status"].value,
                "pairwise": pairwise,
                "num_speakers": result["num_speakers"],
                "closest_pair": closest,
            }
            
        except Exception as e:
            console.print(f"[error]Inter-speaker metrics failed: {e}[/]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
            return {
                "meanSeparation": 0.0,
                "stdSeparation": 0.0,
                "minSeparation": 0.0,
                "maxSeparation": 0.0,
                "health": HealthStatus.UNHEALTHY.value,
                "pairwise": [],
                "num_speakers": len(speaker_embeddings),
                "closest_pair": None,
                "error": str(e),
            }
    
    # ------------------------------------------------------------------
    # SEGMENT GROUP HEALTH METRICS (NEW)
    # ------------------------------------------------------------------
    
    def compute_segment_group_health(self) -> Dict:
        """
        Analyze the health of the segment labeling process itself.
        
        Examines self._segment_groups to provide insights on:
        - Label stability over time
        - Outlier ratio (how many segments went to outlier pool)
        - Rejection rate (how many updates were rejected)
        - Match type distribution (strong vs weak matches)
        - Temporal coherence (label switching frequency)
        
        Returns
        -------
        dict with:
            - totalSegments: int
            - segmentGroupsCount: int
            - outlierRatio: float (0-1)
            - rejectionRate: float (0-1)
            - matchTypeDistribution: dict
            - labelSwitches: int
            - labelStability: float (0-1)
            - health: HealthStatus
            - timeline: list of {timestamp, primaryLabel, matchType, isOutlier}
        """
        groups = getattr(self, '_segment_groups', [])
        if not groups:
            console.print("[dim]Segment group health: no segment groups available[/]")
            return {
                "totalSegments": 0,
                "segmentGroupsCount": 0,
                "outlierRatio": 0.0,
                "rejectionRate": 0.0,
                "matchTypeDistribution": {},
                "labelSwitches": 0,
                "labelStability": 1.0,
                "health": HealthStatus.HEALTHY.value,
                "timeline": [],
            }
        
        total = len(groups)
        outlier_count = 0
        match_types = {}
        label_switches = 0
        prev_primary = None
        timeline = []
        
        for group in groups:
            matches = group.get("matches", [])
            if not matches:
                continue
            
            primary = matches[0]
            primary_label = primary.get("label", "UNKNOWN")
            match_type = primary.get("match_type", "unknown")
            is_outlier = primary.get("is_outlier", False)
            timestamp = group.get("timestamp", 0.0)
            
            if is_outlier or primary_label.startswith("OUTLIER_"):
                outlier_count += 1
            
            match_types[match_type] = match_types.get(match_type, 0) + 1
            
            if prev_primary is not None and primary_label != prev_primary:
                label_switches += 1
            prev_primary = primary_label
            
            timeline.append({
                "timestamp": timestamp,
                "primaryLabel": primary_label,
                "matchType": match_type,
                "isOutlier": is_outlier or primary_label.startswith("OUTLIER_"),
            })
        
        outlier_ratio = outlier_count / max(total, 1)
        rejection_rate = getattr(self, '_rejected_updates', 0) / max(
            getattr(self, 'total_segments_processed', 1), 1
        )
        label_stability = 1.0 - (label_switches / max(total - 1, 1))
        
        # Health classification based on segment group quality
        if outlier_ratio <= 0.1 and rejection_rate <= 0.1 and label_stability >= 0.9:
            health = HealthStatus.HEALTHY
        elif outlier_ratio <= 0.3 and rejection_rate <= 0.2 and label_stability >= 0.7:
            health = HealthStatus.WARNING
        else:
            health = HealthStatus.UNHEALTHY
        
        console.print(
            f"[info]✓ Segment group health: {total} groups, "
            f"outlier_ratio={outlier_ratio:.2%}, "
            f"rejection_rate={rejection_rate:.2%}, "
            f"stability={label_stability:.2%}, "
            f"health={health.value}[/]"
        )
        
        return {
            "totalSegments": getattr(self, 'total_segments_processed', total),
            "segmentGroupsCount": total,
            "outlierRatio": round(outlier_ratio, 4),
            "rejectionRate": round(rejection_rate, 4),
            "matchTypeDistribution": match_types,
            "labelSwitches": label_switches,
            "labelStability": round(label_stability, 4),
            "health": health.value,
            "timeline": timeline,
        }
    
    # ------------------------------------------------------------------
    # OUTLIER HEALTH METRICS (NEW)
    # ------------------------------------------------------------------
    
    def compute_outlier_health(self) -> Dict:
        """
        Get health metrics for the outlier pool.
        
        Provides visibility into:
        - Active outlier count and age distribution
        - Promotion history (outlier → speaker transitions)
        - Pool turnover rate
        
        Returns
        -------
        dict with:
            - enabled: bool
            - activeCount: int
            - totalPromotions: int
            - promotionHistory: list
            - outlierDetails: list
            - health: HealthStatus
        """
        if not hasattr(self, 'outlier_pool') or not getattr(self, 'use_outlier_buffer', False):
            return {
                "enabled": False,
                "activeCount": 0,
                "totalPromotions": 0,
                "promotionHistory": [],
                "outlierDetails": [],
                "health": HealthStatus.HEALTHY.value,
            }
        
        pool = self.outlier_pool
        stats = pool.get_stats()
        
        active_count = stats["total_outliers"]
        total_promotions = stats["total_promotions"]
        
        # Build outlier details for UI
        outlier_details = []
        for label, detail in stats.get("outlier_details", {}).items():
            outlier_details.append({
                "label": label,
                "age": round(detail["age"], 1),
                "timestamp": detail["timestamp"],
                "segmentId": detail["segment_id"],
                "matchAttempts": detail["match_attempts"],
                "audioDuration": detail["audio_duration"],
            })
        
        # Sort by age (oldest first - these are at risk of expiry)
        outlier_details.sort(key=lambda o: o["age"], reverse=True)
        
        # Health: too many active outliers or old outliers is concerning
        old_outliers = sum(1 for o in outlier_details if o["age"] > pool.ttl * 0.8)
        if active_count == 0:
            health = HealthStatus.HEALTHY
        elif active_count <= 3 and old_outliers == 0:
            health = HealthStatus.HEALTHY
        elif active_count <= 5 and old_outliers <= 1:
            health = HealthStatus.WARNING
        else:
            health = HealthStatus.UNHEALTHY
        
        console.print(
            f"[info]✓ Outlier health: {active_count} active, "
            f"{total_promotions} promotions, "
            f"health={health.value}[/]"
        )
        
        return {
            "enabled": True,
            "activeCount": active_count,
            "totalPromotions": total_promotions,
            "promotionThreshold": pool.promotion_threshold,
            "ttl": pool.ttl,
            "promotionHistory": stats.get("recent_promotions", []),
            "outlierDetails": outlier_details,
            "health": health.value,
        }
    
    # ------------------------------------------------------------------
    # COMBINED METRICS (primary API endpoint)
    # ------------------------------------------------------------------
    
    def get_speaker_metrics(
        self,
        label: Optional[str] = None,
    ) -> Dict:
        """
        Combined metrics endpoint returning ALL speaker health data.
        
        This is the primary method called by the API route. It returns
        intra-speaker, inter-speaker, segment group health, and outlier
        health in a single structured response.
        
        Parameters
        ----------
        label : str, optional
            Filter intra-speaker metrics to a specific speaker.
            
        Returns
        -------
        dict with:
            - intra_speaker: dict (from compute_intra_speaker_metrics)
            - inter_speaker: dict (from compute_inter_speaker_metrics)
            - segment_groups: dict (from compute_segment_group_health)
            - outliers: dict (from compute_outlier_health)
            - summary: dict (overall health summary)
            - timestamp: ISO datetime
        """
        console.print(
            f"[bold blue]📊 Computing speaker metrics (label={label or 'all'})[/]"
        )
        
        intra = self.compute_intra_speaker_metrics(label=label)
        inter = self.compute_inter_speaker_metrics()
        segment_groups = self.compute_segment_group_health()
        outliers = self.compute_outlier_health()
        
        # Compute overall summary health
        health_scores = {
            HealthStatus.HEALTHY.value: 3,
            HealthStatus.WARNING.value: 2,
            HealthStatus.UNHEALTHY.value: 1,
        }
        
        all_healths = [
            intra.get("overall_status", "healthy"),
            inter.get("health", "healthy"),
            segment_groups.get("health", "healthy"),
            outliers.get("health", "healthy"),
        ]
        
        min_score = min(health_scores.get(h, 1) for h in all_healths)
        if min_score >= 3:
            overall = HealthStatus.HEALTHY.value
        elif min_score >= 2:
            overall = HealthStatus.WARNING.value
        else:
            overall = HealthStatus.UNHEALTHY.value
        
        summary = {
            "overall": overall,
            "intra": intra.get("overall_status", "healthy"),
            "inter": inter.get("health", "healthy"),
            "segmentGroups": segment_groups.get("health", "healthy"),
            "outliers": outliers.get("health", "healthy"),
            "totalSpeakers": intra.get("total_speakers_analyzed", 0),
            "totalSegments": segment_groups.get("totalSegments", 0),
            "outlierCount": outliers.get("activeCount", 0),
        }
        
        console.print(
            f"[bold green]✅ Speaker metrics complete: overall={overall}[/]"
        )
        
        return {
            "intra_speaker": intra,
            "inter_speaker": inter,
            "segment_groups": segment_groups,
            "outliers": outliers,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }
    
    # ------------------------------------------------------------------
    # SEGMENT DETAIL (existing, enhanced)
    # ------------------------------------------------------------------
    
    def get_segment_detail(self, segment_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific segment by its ID.
        
        Searches through all speakers AND segment groups to find the segment.
        
        Parameters
        ----------
        segment_id : str
            The unique segment identifier (e.g., 'segment_a3f2b1c4')
            
        Returns
        -------
        dict or None
            Segment detail with keys:
            - segment_id, speaker_label, timestamp, embedding_index
            - segment_duration, match_type, confidence, is_outlier
            - speaker_segment_count, centroid_quality
            Returns None if segment_id not found.
        """
        console.print(
            f"[info]get_segment_detail: searching for segment_id='{segment_id}'[/]"
        )
        
        # First, search in speaker references
        for spk_label, ref in self._speakers.items():
            if not hasattr(ref, 'embedding_metadata'):
                continue
            for i, meta in enumerate(ref.embedding_metadata):
                if meta.get('segment_id') == segment_id:
                    return self._build_segment_detail(
                        segment_id=segment_id,
                        speaker_label=spk_label,
                        ref=ref,
                        meta=meta,
                        embedding_index=i,
                    )
        
        # Fallback: search in segment groups
        for group in getattr(self, '_segment_groups', []):
            for match in group.get("matches", []):
                if match.get("segment_id") == segment_id:
                    console.print(
                        f"[success]get_segment_detail: found {segment_id} "
                        f"in segment groups (label: {match.get('label')})[/]"
                    )
                    return {
                        "segment_id": segment_id,
                        "speaker_label": match.get("label", "UNKNOWN"),
                        "timestamp": group.get("timestamp", 0.0),
                        "match_type": match.get("match_type", "unknown"),
                        "confidence": match.get("confidence", 0.0),
                        "is_outlier": match.get("is_outlier", False),
                        "source": "segment_groups",
                    }
        
        console.print(f"[warning]get_segment_detail: '{segment_id}' not found[/]")
        return None
    
    def _build_segment_detail(
        self,
        segment_id: str,
        speaker_label: str,
        ref,
        meta: Dict,
        embedding_index: int,
    ) -> Dict:
        """Build comprehensive segment detail from speaker reference data."""
        duration = self._estimate_segment_duration(ref, embedding_index, meta)
        
        detail = {
            "segment_id": segment_id,
            "speaker_label": speaker_label,
            "timestamp": meta.get('timestamp', 0.0),
            "added_at": meta.get('added_at', meta.get('timestamp', 0.0)),
            "embedding_index": embedding_index,
            "speaker_segment_count": ref.segment_count,
            "embedding_dim": (
                ref.embeddings[embedding_index].shape[0]
                if embedding_index < len(ref.embeddings)
                else 0
            ),
            "segment_duration": round(duration, 4),
            "speaker_first_seen": ref.first_seen if ref.first_seen else 0.0,
            "speaker_last_seen": ref.last_seen,
            "speaker_active_duration": ref.active_duration,
            "centroid_quality": ref.centroid_quality,
            "is_core": meta.get('is_core', True),
            "source": "speaker_reference",
        }
        
        console.print(
            f"[success]get_segment_detail: found {segment_id} "
            f"in speaker '{speaker_label}' (index {embedding_index})[/]"
        )
        return detail
    
    def _estimate_segment_duration(
        self,
        ref,
        embedding_index: int,
        meta: Dict,
    ) -> float:
        """Estimate segment duration from metadata or gaps between segments."""
        duration = meta.get('audio_duration', 0.0)
        if duration > 0.0:
            return duration
        
        timestamp = meta.get('timestamp', 0.0)
        
        if hasattr(ref, 'embedding_metadata') and len(ref.embedding_metadata) > 1:
            if embedding_index < len(ref.embedding_metadata) - 1:
                next_ts = ref.embedding_metadata[embedding_index + 1].get(
                    'timestamp', timestamp
                )
                duration = max(0.0, next_ts - timestamp)
            else:
                # Last segment: use average gap of previous segments
                gaps = []
                for j in range(len(ref.embedding_metadata) - 1):
                    t1 = ref.embedding_metadata[j].get('timestamp', 0)
                    t2 = ref.embedding_metadata[j + 1].get('timestamp', 0)
                    if t2 > t1:
                        gaps.append(t2 - t1)
                if gaps:
                    duration = sum(gaps) / len(gaps)
        
        return duration
    
    # ------------------------------------------------------------------
    # AUDIO INFO (existing, unchanged)
    # ------------------------------------------------------------------
    
    def get_segment_audio_info(self, segment_id: str) -> Dict:
        """
        Check if audio data is available for a segment.
        
        Checks the context buffer and last_n_segments directory.
        
        Parameters
        ----------
        segment_id : str
            The unique segment identifier
            
        Returns
        -------
        dict with: segment_id, has_audio, audio_source, sample_rate, duration_seconds
        """
        result = {
            "segment_id": segment_id,
            "has_audio": False,
            "audio_source": None,
            "sample_rate": None,
            "duration_seconds": 0.0,
        }
        
        # Check context buffer
        try:
            from core.state import get_context_buffer
            context_buffer = get_context_buffer()
            if context_buffer and hasattr(context_buffer, 'segments'):
                for segment_audio, metadata in context_buffer.segments:
                    if metadata.get('segment_id') == segment_id:
                        try:
                            from services.audio_utils import get_audio_duration
                            sample_rate = 16000
                            duration = get_audio_duration(segment_audio, sr=sample_rate)
                        except ImportError:
                            sample_rate = 16000
                            if hasattr(segment_audio, 'shape'):
                                duration = segment_audio.shape[-1] / sample_rate
                            else:
                                duration = len(segment_audio) / sample_rate
                        
                        result["has_audio"] = True
                        result["audio_source"] = "context_buffer"
                        result["sample_rate"] = sample_rate
                        result["duration_seconds"] = round(duration, 3)
                        console.print(
                            f"[dim]get_segment_audio_info: found {segment_id} "
                            f"in context_buffer ({duration:.3f}s)[/]"
                        )
                        return result
        except ImportError:
            console.print("[dim]get_segment_audio_info: core.state not importable[/]")
        except Exception as e:
            console.print(f"[warning]get_segment_audio_info: context buffer error: {e}[/]")
        
        # Check disk
        try:
            from services.audio_config import OUTPUT_DIR
            last_n_dir = OUTPUT_DIR / "last_50_segments"
            if last_n_dir.exists():
                audio_path = last_n_dir / f"{segment_id}.wav"
                if audio_path.exists():
                    try:
                        from services.audio_utils import get_audio_duration
                        duration = get_audio_duration(str(audio_path))
                    except ImportError:
                        import wave
                        with wave.open(str(audio_path), 'rb') as wf:
                            duration = wf.getnframes() / wf.getframerate()
                    
                    result["has_audio"] = True
                    result["audio_source"] = "disk"
                    result["sample_rate"] = 16000
                    result["duration_seconds"] = round(duration, 3)
                    console.print(
                        f"[dim]get_segment_audio_info: found {segment_id} "
                        f"on disk ({duration:.3f}s)[/]"
                    )
                    return result
        except Exception as e:
            console.print(f"[dim]get_segment_audio_info: disk check failed: {e}[/]")
        
        console.print(f"[dim]get_segment_audio_info: no audio found for {segment_id}[/]")
        return result
