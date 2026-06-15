# servers/live_subtitles/live_subtitles_server2_with_en/services/speaker_metrics_mixin.py

"""
SpeakerMetricsMixin - adds intra/inter speaker metrics computation to SegmentSpeakerLabeler.
Mixed into SegmentSpeakerLabeler to access self._speakers (Dict[str, SpeakerReference]).
"""
from typing import Dict, List, Optional
import numpy as np
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
    Mixin that provides speaker metrics methods for SegmentSpeakerLabeler.
    
    Requires the host class to have:
        - self._speakers: Dict[str, SpeakerReference]
        - SpeakerReference must have:
            - .label: str
            - .embeddings: List[np.ndarray]  (each shape (1, dim) or (dim,))
            - .segment_count: int
            - .has_valid_centroid: bool
    """

    def compute_intra_speaker_metrics(
        self,
        label: Optional[str] = None,
        healthy_threshold: float = 0.3,
        warning_threshold: float = 0.5,
    ) -> Dict:
        """
        Compute intra-speaker variance for one or all speakers.

        Parameters
        ----------
        label : str, optional
            Specific speaker label. If None, computes for all speakers.
        healthy_threshold : float
            Mean distance below this is "healthy".
        warning_threshold : float
            Mean distance below this is "warning", above is "unhealthy".

        Returns
        -------
        dict with keys:
            - speakers: list of per-speaker IntraSpeakerResult dicts
            - overall_status: HealthStatus for the worst speaker
            - total_speakers_analyzed: int
        """
        speakers_to_analyze = {}
        if label is not None:
            if label not in self._speakers:
                console.print(f"[warning]compute_intra_speaker_metrics: '{label}' not found[/]")
                return {
                    "speakers": [],
                    "overall_status": HealthStatus.HEALTHY.value,
                    "total_speakers_analyzed": 0,
                    "error": f"Speaker '{label}' not found",
                }
            speakers_to_analyze[label] = self._speakers[label]
        else:
            speakers_to_analyze = self._speakers

        results = []
        worst_status = HealthStatus.HEALTHY

        for spk_label, ref in speakers_to_analyze.items():
            if not ref.embeddings:
                console.print(f"[dim]Skipping {spk_label}: no embeddings[/]")
                continue

            # Stack embeddings into (n_embeddings, dim)
            embeddings_list = []
            segment_ids = []
            for i, emb in enumerate(ref.embeddings):
                flat = emb.flatten()
                embeddings_list.append(flat)
                # Generate segment IDs from metadata if available
                if hasattr(ref, 'embedding_metadata') and i < len(ref.embedding_metadata):
                    seg_id = ref.embedding_metadata[i].get('segment_id', f"segment_{i}")
                else:
                    seg_id = f"segment_{i}"
                segment_ids.append(seg_id)

            if len(embeddings_list) == 0:
                continue

            embeddings_array = np.array(embeddings_list, dtype=np.float64)

            try:
                intra_input: IntraSpeakerInput = {
                    "label": spk_label,
                    "embeddings": embeddings_array,
                }
                result = compute_intra_speaker_variance(
                    speaker_input=intra_input,
                    segment_ids=segment_ids,
                    healthy_threshold=healthy_threshold,
                    warning_threshold=warning_threshold,
                )

                # Build frontend-friendly format
                results.append({
                    "label": result["label"],
                    "segmentsCount": result["num_embeddings"],
                    "health": result["status"].value,
                    "meanDist": round(result["mean_distance"], 4),
                    "stdDev": round(result["std_distance"], 4),
                    "minDist": round(result["min_distance"], 4),
                    "maxDist": round(result["max_distance"], 4),
                    "segments": [
                        {"id": d["segment_id"], "d": round(d["distance"], 4)}
                        for d in result["distances"]
                    ],
                })

                # Track worst status
                status_order = {
                    HealthStatus.HEALTHY: 0,
                    HealthStatus.WARNING: 1,
                    HealthStatus.UNHEALTHY: 2,
                }
                if status_order.get(result["status"], 0) > status_order.get(worst_status, 0):
                    worst_status = result["status"]

            except Exception as e:
                console.print(f"[error]Intra-speaker metrics failed for {spk_label}: {e}[/]")
                results.append({
                    "label": spk_label,
                    "segmentsCount": len(embeddings_list),
                    "health": HealthStatus.UNHEALTHY.value,
                    "meanDist": 0.0,
                    "stdDev": 0.0,
                    "minDist": 0.0,
                    "maxDist": 0.0,
                    "segments": [],
                    "error": str(e),
                })

        console.print(
            f"[info]compute_intra_speaker_metrics: analyzed {len(results)} speakers, "
            f"worst_status={worst_status.value}[/]"
        )
        return {
            "speakers": results,
            "overall_status": worst_status.value,
            "total_speakers_analyzed": len(results),
        }

    def compute_inter_speaker_metrics(
        self,
        healthy_threshold: float = 0.5,
        warning_threshold: float = 0.3,
    ) -> Dict:
        """
        Compute inter-speaker separation using speaker centroids.

        Parameters
        ----------
        healthy_threshold : float
            Mean separation above this is "healthy".
        warning_threshold : float
            Mean separation above this is "warning", below is "unhealthy".

        Returns
        -------
        dict with keys:
            - meanSeparation, stdSeparation, minSeparation, maxSeparation
            - health: overall health status
            - pairwise: list of {from, to, distance}
            - num_speakers: int
            - error: str if something went wrong
        """
        # Collect speakers with valid centroids
        speaker_embeddings = {}
        for spk_label, ref in self._speakers.items():
            if not ref.has_valid_centroid:
                continue
            if not ref.embeddings:
                continue
            # Stack all embeddings for this speaker
            embeddings_list = [emb.flatten() for emb in ref.embeddings]
            if not embeddings_list:
                continue
            speaker_embeddings[spk_label] = np.array(embeddings_list, dtype=np.float64)

        if len(speaker_embeddings) < 2:
            console.print(
                f"[warning]compute_inter_speaker_metrics: need >=2 speakers with centroids, "
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

            pairwise = [
                {
                    "from": p["speaker_id_1"],
                    "to": p["speaker_id_2"],
                    "distance": round(p["distance"], 4),
                }
                for p in result["pairwise_distances"]
            ]

            console.print(
                f"[info]compute_inter_speaker_metrics: {result['num_speakers']} speakers, "
                f"mean_sep={result['mean_separation']:.4f}, status={result['status'].value}[/]"
            )

            return {
                "meanSeparation": round(result["mean_separation"], 4),
                "stdSeparation": round(result["std_separation"], 4),
                "minSeparation": round(result["min_separation"], 4),
                "maxSeparation": round(result["max_separation"], 4),
                "health": result["status"].value,
                "pairwise": pairwise,
                "num_speakers": result["num_speakers"],
            }

        except Exception as e:
            console.print(f"[error]compute_inter_speaker_metrics failed: {e}[/]")
            return {
                "meanSeparation": 0.0,
                "stdSeparation": 0.0,
                "minSeparation": 0.0,
                "maxSeparation": 0.0,
                "health": HealthStatus.UNHEALTHY.value,
                "pairwise": [],
                "num_speakers": len(speaker_embeddings),
                "error": str(e),
            }

    def get_speaker_metrics(
        self,
        label: Optional[str] = None,
    ) -> Dict:
        """
        Combined metrics endpoint returning intra + inter speaker data.

        This is the primary method called by the API route.
        Returns data structured for the speaker_metrics.html frontend.

        Parameters
        ----------
        label : str, optional
            Filter intra-speaker metrics to a specific speaker.

        Returns
        -------
        dict with:
            - intra_speaker: dict (from compute_intra_speaker_metrics)
            - inter_speaker: dict (from compute_inter_speaker_metrics)
            - timestamp: ISO datetime
        """
        from datetime import datetime

        console.print(f"[info]get_speaker_metrics: label={label or 'all'}[/]")
        
        intra = self.compute_intra_speaker_metrics(label=label)
        inter = self.compute_inter_speaker_metrics()

        return {
            "intra_speaker": intra,
            "inter_speaker": inter,
            "timestamp": datetime.now().isoformat(),
        }
