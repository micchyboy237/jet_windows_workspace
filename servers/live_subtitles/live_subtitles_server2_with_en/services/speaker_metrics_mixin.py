"""
SpeakerMetricsMixin - adds intra/inter speaker metrics computation to SegmentSpeakerLabeler.
Mixed into SegmentSpeakerLabeler to access self._speakers (Dict[str, SpeakerReference]).
"""
import numpy as np
import torch
from typing import Dict, List, Optional
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
            - .embedding_metadata: List[Dict] with keys: segment_id, timestamp
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
            - speakers: list of per-speaker dicts with segment-level metrics
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

            embeddings_list = []
            segment_ids = []
            segment_durations = []
            
            # Extract segment metadata including duration estimates
            for i, emb in enumerate(ref.embeddings):
                flat = emb.flatten()
                embeddings_list.append(flat)
                
                # Get segment_id and timestamp from metadata
                meta = {}
                if hasattr(ref, 'embedding_metadata') and i < len(ref.embedding_metadata):
                    meta = ref.embedding_metadata[i]
                
                seg_id = meta.get('segment_id', f"segment_{i}")
                segment_ids.append(seg_id)
                
                # Estimate segment duration from timestamps
                # If we have consecutive segments, duration = time diff to next segment
                # For the last segment, use the difference from previous to last
                duration = 0.0
                timestamp = meta.get('timestamp', 0.0)
                if hasattr(ref, 'embedding_metadata') and len(ref.embedding_metadata) > 1:
                    if i < len(ref.embedding_metadata) - 1:
                        next_ts = ref.embedding_metadata[i + 1].get('timestamp', timestamp)
                        duration = max(0.0, next_ts - timestamp)
                    else:
                        # Last segment: use the average gap between segments as estimate
                        gaps = []
                        for j in range(len(ref.embedding_metadata) - 1):
                            t1 = ref.embedding_metadata[j].get('timestamp', 0)
                            t2 = ref.embedding_metadata[j + 1].get('timestamp', 0)
                            if t2 > t1:
                                gaps.append(t2 - t1)
                        if gaps:
                            duration = sum(gaps) / len(gaps)
                        else:
                            duration = 0.0
                segment_durations.append(duration)

            if len(embeddings_list) == 0:
                continue

            embeddings_array = np.array(embeddings_list, dtype=np.float64)

            # Compute centroid for distance calculations
            centroid = np.mean(embeddings_array, axis=0)
            
            # Calculate individual distances and build segment data
            segments_data = []
            for i in range(len(embeddings_list)):
                emb = embeddings_array[i]
                # Cosine distance to centroid
                dist = float(cosine_distance(emb, centroid))
                segments_data.append({
                    "id": segment_ids[i],
                    "d": round(dist, 4),
                    "duration": round(segment_durations[i], 4),
                    "timestamp": (
                        ref.embedding_metadata[i].get('timestamp', 0.0)
                        if hasattr(ref, 'embedding_metadata') and i < len(ref.embedding_metadata)
                        else 0.0
                    ),
                })

            # Compute aggregate metrics
            distances = np.array([s["d"] for s in segments_data])
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            min_dist = float(np.min(distances))
            max_dist = float(np.max(distances))

            # Determine health status
            if mean_dist <= healthy_threshold:
                status = HealthStatus.HEALTHY
            elif mean_dist <= warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.UNHEALTHY

            results.append({
                "label": spk_label,
                "segmentsCount": len(embeddings_list),
                "health": status.value,
                "meanDist": round(mean_dist, 4),
                "stdDev": round(std_dist, 4),
                "minDist": round(min_dist, 4),
                "maxDist": round(max_dist, 4),
                "segments": segments_data,
            })

            # Track worst status
            status_order = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.WARNING: 1,
                HealthStatus.UNHEALTHY: 2,
            }
            if status_order.get(status, 0) > status_order.get(worst_status, 0):
                worst_status = status

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

    def get_segment_detail(self, segment_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific segment by its ID.
        
        Searches through all speakers to find the segment and returns
        comprehensive metadata including the embedding vector, speaker label,
        timestamp, and associated audio info.

        Parameters
        ----------
        segment_id : str
            The unique segment identifier (e.g., 'segment_a3f2b1c4')

        Returns
        -------
        dict or None
            Segment detail with keys:
            - segment_id: str
            - speaker_label: str (the speaker this segment belongs to)
            - timestamp: float (when segment was processed)
            - embedding_index: int (position in speaker's embedding list)
            - speaker_segment_count: int (total segments for this speaker)
            - embedding_dim: int (dimensionality of the embedding)
            - segment_duration: float (estimated duration in seconds)
            - speaker_first_seen: float
            - speaker_last_seen: float
            Returns None if segment_id not found.
        """
        console.print(f"[info]get_segment_detail: searching for segment_id='{segment_id}'[/]")
        
        for spk_label, ref in self._speakers.items():
            if not hasattr(ref, 'embedding_metadata'):
                continue
            for i, meta in enumerate(ref.embedding_metadata):
                if meta.get('segment_id') == segment_id:
                    # Calculate duration
                    duration = 0.0
                    if len(ref.embedding_metadata) > 1:
                        if i < len(ref.embedding_metadata) - 1:
                            next_ts = ref.embedding_metadata[i + 1].get('timestamp', meta.get('timestamp', 0))
                            duration = max(0.0, next_ts - meta.get('timestamp', 0))
                        else:
                            gaps = []
                            for j in range(len(ref.embedding_metadata) - 1):
                                t1 = ref.embedding_metadata[j].get('timestamp', 0)
                                t2 = ref.embedding_metadata[j + 1].get('timestamp', 0)
                                if t2 > t1:
                                    gaps.append(t2 - t1)
                            if gaps:
                                duration = sum(gaps) / len(gaps)
                    
                    detail = {
                        "segment_id": segment_id,
                        "speaker_label": spk_label,
                        "timestamp": meta.get('timestamp', 0.0),
                        "added_at": meta.get('added_at', meta.get('timestamp', 0.0)),
                        "embedding_index": i,
                        "speaker_segment_count": ref.segment_count,
                        "embedding_dim": ref.embeddings[i].shape[0] if i < len(ref.embeddings) else 0,
                        "segment_duration": round(duration, 4),
                        "speaker_first_seen": ref.first_seen if ref.first_seen else 0.0,
                        "speaker_last_seen": ref.last_seen,
                        "speaker_active_duration": ref.active_duration,
                        "centroid_quality": ref.centroid_quality,
                    }
                    console.print(
                        f"[success]get_segment_detail: found {segment_id} "
                        f"in speaker '{spk_label}' (index {i}, "
                        f"duration={duration:.3f}s)[/]"
                    )
                    return detail
        
        console.print(f"[warning]get_segment_detail: segment_id '{segment_id}' not found[/]")
        return None

    def get_segment_audio_info(self, segment_id: str) -> Dict:
        """
        Check if audio data is available for a segment.
        
        Checks the context buffer for audio data associated with this segment ID.
        Also checks the last_n_segments directory for saved WAV files.

        Parameters
        ----------
        segment_id : str
            The unique segment identifier

        Returns
        -------
        dict
            Audio info with keys:
            - segment_id: str
            - has_audio: bool
            - audio_source: str or None (e.g., 'context_buffer', 'disk')
            - sample_rate: int or None
            - duration_seconds: float
        """
        result = {
            "segment_id": segment_id,
            "has_audio": False,
            "audio_source": None,
            "sample_rate": None,
            "duration_seconds": 0.0,
        }
        
        # Check context buffer first (most recent segments)
        try:
            from core.state import get_context_buffer
            context_buffer = get_context_buffer()
            if context_buffer and hasattr(context_buffer, 'segments'):
                for segment_audio, metadata in context_buffer.segments:
                    if metadata.get('segment_id') == segment_id:
                        sample_rate = 16000  # Default for this system
                        if isinstance(segment_audio, torch.Tensor):
                            duration = segment_audio.shape[-1] / sample_rate if segment_audio.dim() > 0 else 0.0
                        elif isinstance(segment_audio, np.ndarray):
                            duration = len(segment_audio) / sample_rate
                        else:
                            duration = 0.0
                        
                        result["has_audio"] = True
                        result["audio_source"] = "context_buffer"
                        result["sample_rate"] = sample_rate
                        result["duration_seconds"] = round(duration, 3)
                        console.print(f"[dim]get_segment_audio_info: found {segment_id} in context_buffer ({duration:.3f}s)[/]")
                        return result
        except ImportError:
            console.print("[dim]get_segment_audio_info: core.state not importable[/]")
        except Exception as e:
            console.print(f"[warning]get_segment_audio_info: error checking context buffer: {e}[/]")
        
        # Fallback: check disk for saved WAV files
        try:
            from config import OUTPUT_DIR
            last_n_dir = OUTPUT_DIR / f"last_50_segments"  # N_SEGMENT_RESULTS = 50
            if last_n_dir.exists():
                audio_path = last_n_dir / f"{segment_id}.wav"
                if audio_path.exists():
                    file_size = audio_path.stat().st_size
                    # WAV files: 44-byte header + 16-bit mono PCM
                    duration = max(0, (file_size - 44) / (16000 * 2))
                    result["has_audio"] = True
                    result["audio_source"] = "disk"
                    result["sample_rate"] = 16000
                    result["duration_seconds"] = round(duration, 3)
                    console.print(f"[dim]get_segment_audio_info: found {segment_id} on disk ({duration:.3f}s)[/]")
                    return result
        except Exception as e:
            console.print(f"[dim]get_segment_audio_info: disk check failed: {e}[/]")
        
        console.print(f"[dim]get_segment_audio_info: no audio found for {segment_id}[/]")
        return result
