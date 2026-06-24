"""Serialization and reporting for SegmentSpeakerLabeler.

Extracted to keep the main class focused on core labeling logic.
"""
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from scipy.spatial.distance import cdist
from rich.console import Console

if TYPE_CHECKING:
    from segment_speaker_labeler import SegmentSpeakerLabeler
    from speaker_reference import SpeakerReference

console = Console()


class SpeakerLabelerSerializer:
    """Handles serialization, health reports, centroid stats, and similarity matrix."""

    def __init__(self, labeler: "SegmentSpeakerLabeler"):
        self._labeler = labeler

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------
    def get_speaker_similarity_matrix(self) -> Dict:
        """Get pairwise similarity matrix between all speakers."""
        labels = []
        centroids = []
        segment_counts = []
        for label, ref in self._labeler._speakers.items():
            if ref.has_valid_centroid:
                labels.append(label)
                centroids.append(ref.centroid)
                segment_counts.append(ref.segment_count)

        if len(labels) < 2:
            return {"labels": labels, "similarities": [], "segment_counts": segment_counts}

        centroids_array = np.vstack(centroids)
        distances = cdist(centroids_array, centroids_array, metric="cosine")
        similarities = (1.0 - distances).tolist()
        return {
            "labels": labels,
            "similarities": [[round(s, 4) for s in row] for row in similarities],
            "segment_counts": segment_counts,
        }

    # ------------------------------------------------------------------
    # Centroid arrays
    # ------------------------------------------------------------------
    def get_centroid_arrays(self) -> Dict[str, np.ndarray]:
        """Get raw centroid arrays for all speakers."""
        centroids = {}
        for label, ref in self._labeler._speakers.items():
            if ref.has_valid_centroid:
                centroids[label] = ref.centroid.copy()
        return centroids

    # ------------------------------------------------------------------
    # Centroid stats
    # ------------------------------------------------------------------
    def get_centroid_stats(self) -> Dict:
        """Get comprehensive centroid statistics for visualization."""
        centroids = self.get_centroid_arrays()
        if not centroids:
            return {"error": "No valid centroids available"}

        labels = list(centroids.keys())
        centroid_matrix = np.vstack([centroids[label] for label in labels])
        stats = {
            "labels": labels,
            "centroid_shape": list(centroid_matrix.shape),
            "embedding_dimension": centroid_matrix.shape[1],
            "total_speakers": len(labels),
            "total_segments": sum(
                ref.segment_count for ref in self._labeler._speakers.values()
            ),
        }

        speaker_details = {}
        for i, label in enumerate(labels):
            ref = self._labeler._speakers[label]
            centroid = centroids[label]
            flat = centroid.flatten()
            norm = float(np.linalg.norm(flat))
            top_dims = np.argsort(np.abs(flat))[-5:][::-1]
            centroid_vector = flat.tolist()
            speaker_details[label] = {
                "centroid_vector": centroid_vector[:50],
                "centroid_norm": round(norm, 4),
                "top_dimensions": [
                    {"dim": int(d), "value": round(float(flat[d]), 6)} for d in top_dims
                ],
                "segment_count": ref.segment_count,
                "centroid_quality": ref.centroid_quality,
                "first_seen": ref.first_seen if ref.first_seen else 0,
                "last_seen": ref.last_seen,
                "active_duration": ref.active_duration,
                "embedding_count": len(ref.embeddings),
            }
        stats["speakers"] = speaker_details

        if len(labels) >= 2:
            distances = cdist(centroid_matrix, centroid_matrix, metric="cosine")
            similarities = 1.0 - distances
            stats["similarity_matrix"] = similarities.tolist()
            stats["distance_matrix"] = distances.tolist()
            for i, label in enumerate(labels):
                other_distances = [distances[i, j] for j in range(len(labels)) if j != i]
                other_similarities = [similarities[i, j] for j in range(len(labels)) if j != i]
                nearest_idx = min(
                    (j for j in range(len(labels)) if j != i),
                    key=lambda j: distances[i, j],
                )
                speaker_details[label]["avg_distance_to_others"] = round(
                    float(np.mean(other_distances)), 4
                )
                speaker_details[label]["avg_similarity_to_others"] = round(
                    float(np.mean(other_similarities)), 4
                )
                speaker_details[label]["nearest_neighbor"] = labels[nearest_idx]
                speaker_details[label]["nearest_distance"] = round(
                    float(distances[i, nearest_idx]), 4
                )
                speaker_details[label]["nearest_similarity"] = round(
                    float(similarities[i, nearest_idx]), 4
                )
        return stats

    # ------------------------------------------------------------------
    # Health status
    # ------------------------------------------------------------------
    def get_health_status(self) -> Dict:
        """Get current health status of the speaker labeler."""
        categories = self._labeler._maintenance.get_speaker_categories()
        mature_count = len(categories["mature"])
        young_count = len(categories["young"])
        orphan_count = len(categories["orphan"])
        total_count = len(self._labeler._speakers)
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

        status = {
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
        if self._labeler.use_outlier_buffer:
            status["outliers"] = self._labeler.outlier_pool.get_stats()
        return status

    # ------------------------------------------------------------------
    # Speaker health report
    # ------------------------------------------------------------------
    def get_speaker_health_report(self) -> Dict:
        """Comprehensive speaker health report with merge tracking."""
        health = self.get_health_status()
        health["merge_history"] = self._labeler._merge_history
        health["speaker_creation_times"] = {
            label: {
                "created_at": time,
                "age": max(
                    ref.last_seen for ref in self._labeler._speakers.values()
                ) - time if self._labeler._speakers else 0,
            }
            for label, time in self._labeler._speaker_creation_times.items()
            if label in self._labeler._speakers
        }
        health["missing_speaker_ids"] = self.find_missing_speaker_ids()
        return health

    # ------------------------------------------------------------------
    # Missing speaker IDs
    # ------------------------------------------------------------------
    def find_missing_speaker_ids(self) -> List[str]:
        """Find speaker IDs that were skipped/removed."""
        existing_ids = set()
        for label in self._labeler._speakers.keys():
            if label.startswith("SPEAKER_"):
                try:
                    num = int(label.split("_")[1])
                    existing_ids.add(num)
                except (IndexError, ValueError):
                    pass
        missing = []
        for i in range(1, self._labeler._next_speaker_id):
            if i not in existing_ids:
                missing.append(f"SPEAKER_{i:02d}")
        return missing

    # ------------------------------------------------------------------
    # Centroid health stats
    # ------------------------------------------------------------------
    def get_centroid_health_stats(self) -> Dict:
        """Get statistics about centroid contamination prevention."""
        return {
            "total_updates_rejected": self._labeler._rejected_updates,
            "centroid_update_log": self._labeler._centroid_update_log[-10:],
            "min_similarity_to_update": self._labeler.min_similarity_to_update,
            "total_segments_processed": self._labeler.total_segments_processed,
            "rejection_rate": (
                self._labeler._rejected_updates
                / max(self._labeler.total_segments_processed, 1)
            ),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Serialize the labeler state including outlier pool."""
        return {
            "speakers": {
                label: {
                    "label": ref.label,
                    "embeddings": [emb.tolist() for emb in ref.embeddings],
                    "centroid": ref.centroid.tolist() if ref.centroid is not None else None,
                    "first_seen": ref.first_seen,
                    "last_seen": ref.last_seen,
                    "segment_count": ref.segment_count,
                }
                for label, ref in self._labeler._speakers.items()
            },
            "next_speaker_id": self._labeler._next_speaker_id,
            "total_segments_processed": self._labeler.total_segments_processed,
            "total_speakers_created": self._labeler.total_speakers_created,
            "threshold_same": self._labeler.threshold_same,
            "threshold_possible": self._labeler.threshold_possible,
            "threshold_new_speaker": self._labeler.threshold_new_speaker,
            "mature_segment_count": self._labeler.mature_segment_count,
            "young_segment_count": self._labeler.young_segment_count,
            "top_k_speakers": self._labeler.top_k_speakers,
            "consolidation_threshold": self._labeler.consolidation_threshold,
            "outlier_pool": self._labeler.outlier_pool.to_dict() if self._labeler.use_outlier_buffer else {},
            "use_outlier_buffer": self._labeler.use_outlier_buffer,
        }

    @staticmethod
    def from_dict(
        cls,
        data: Dict,
        embedding_model,
        audio_tagger=None,
    ) -> "SegmentSpeakerLabeler":
        """Create a labeler from serialized state."""
        from speaker_reference import SpeakerReference
        from segment_speaker_labeler_defaults import (
            DEFAULT_THRESHOLD_SAME,
            DEFAULT_THRESHOLD_POSSIBLE,
            DEFAULT_THRESHOLD_NEW_SPEAKER,
            DEFAULT_MATURE_SEGMENT_COUNT,
            DEFAULT_YOUNG_SEGMENT_COUNT,
            DEFAULT_TOP_K_SPEAKERS,
            DEFAULT_CONSOLIDATION_THRESHOLD,
        )
        from outlier_pool import OutlierPool

        labeler = cls(
            embedding_model=embedding_model,
            threshold_same=data.get("threshold_same", DEFAULT_THRESHOLD_SAME),
            threshold_possible=data.get("threshold_possible", DEFAULT_THRESHOLD_POSSIBLE),
            threshold_new_speaker=data.get("threshold_new_speaker", DEFAULT_THRESHOLD_NEW_SPEAKER),
            mature_segment_count=data.get("mature_segment_count", DEFAULT_MATURE_SEGMENT_COUNT),
            young_segment_count=data.get("young_segment_count", DEFAULT_YOUNG_SEGMENT_COUNT),
            top_k_speakers=data.get("top_k_speakers", DEFAULT_TOP_K_SPEAKERS),
            consolidation_threshold=data.get("consolidation_threshold", DEFAULT_CONSOLIDATION_THRESHOLD),
            audio_tagger=audio_tagger,
            use_outlier_buffer=data.get("use_outlier_buffer", True),
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

        if data.get("outlier_pool"):
            labeler.outlier_pool = OutlierPool.from_dict(
                data["outlier_pool"],
                debug=labeler.debug,
            )
        return labeler
