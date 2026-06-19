"""
Speaker Metrics Module
Provides functions for computing intra-speaker variance and inter-speaker
separation metrics with health status classification.
Typical usage:
    >>> from speaker_metrics import compute_intra_speaker_variance
    >>> embeddings = np.random.randn(10, 256)
    >>> result = compute_intra_speaker_variance(embeddings)
    >>> result['status']
    'healthy'
"""

from enum import Enum
from typing import Dict, List, TypedDict

import numpy as np
from numpy.typing import NDArray


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"


class DistanceItem(TypedDict):
    """Individual distance measurement with identifiers."""

    segment_id: str
    distance: float


class PairwiseDistanceItem(TypedDict):
    """Pairwise distance between two speakers."""

    speaker_id_1: str
    speaker_id_2: str
    distance: float


class IntraSpeakerInput(TypedDict):
    label: str
    embeddings: NDArray[np.float64]


class InterSpeakerInput(TypedDict):
    speakers: Dict[str, NDArray[np.float64]]


class IntraSpeakerResult(TypedDict):
    label: str
    mean_similarity: float  # Most interpretable: "85% similar to own centroid"
    std_similarity: float  # Spread: "±5% variation"
    min_similarity: float  # Worst outlier: catch contamination
    silhouette_score: float  # How distinct this speaker is from others
    num_embeddings: int
    is_mature: bool  # Has enough data to be reliable
    status: HealthStatus


class InterSpeakerResult(TypedDict):
    mean_separation: float  # Average centroid-to-centroid distance
    min_separation: float  # Closest pair — merge risk indicator
    closest_pair: tuple  # Which speakers are closest
    num_speakers: int
    status: HealthStatus


def cosine_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    similarity = np.dot(a, b) / (a_norm * b_norm)
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)


def compute_intra_speaker_variance(
    speaker_input: IntraSpeakerInput,
    healthy_threshold: float = 0.70,  # Mean similarity > 0.70 = healthy
    warning_threshold: float = 0.55,  # Mean similarity > 0.55 = warning
    min_embeddings_for_mature: int = 5,
    other_centroids: Dict[str, NDArray[np.float64]] | None = None,
) -> IntraSpeakerResult:
    """
    Compute intra-speaker health with 3 key signals:
    1. Mean similarity to centroid (is this speaker coherent?)
    2. Silhouette score (is this speaker distinct from others?)
    3. Maturity check (do we have enough data?)

    These 3 signals catch: contamination, premature clusters, and noise.
    """
    embeddings = speaker_input["embeddings"]
    speaker_label = speaker_input["label"]

    if embeddings.size == 0:
        raise ValueError("Embeddings array cannot be empty")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")

    n_embeddings = embeddings.shape[0]
    is_mature = n_embeddings >= min_embeddings_for_mature

    # Single embedding = can't measure variance
    if n_embeddings < 2:
        return IntraSpeakerResult(
            label=speaker_label,
            mean_similarity=1.0,
            std_similarity=0.0,
            min_similarity=1.0,
            silhouette_score=1.0 if not other_centroids else 0.0,
            num_embeddings=1,
            is_mature=False,
            status=HealthStatus.WARNING,
        )

    centroid = np.mean(embeddings, axis=0)
    similarities = np.array(
        [1.0 - cosine_distance(emb, centroid) for emb in embeddings]
    )

    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    min_sim = float(np.min(similarities))

    # Compute silhouette only if we have other speakers to compare against
    silhouette = 0.0
    if other_centroids and len(other_centroids) > 0:
        # Filter out own centroid
        other_centroids_filtered = {
            k: v for k, v in other_centroids.items() if k != speaker_label
        }
        if other_centroids_filtered:
            a = 1.0 - mean_sim  # intra-cluster distance
            b = min(
                cosine_distance(centroid, other_c)
                for other_c in other_centroids_filtered.values()
            )
            silhouette = float((b - a) / max(a, b)) if max(a, b) > 0 else 0.0

    # Health determination — simple, clear rules
    if not is_mature:
        status = HealthStatus.WARNING  # Can't trust immature clusters
    elif mean_sim >= healthy_threshold and silhouette >= 0.3:
        status = HealthStatus.HEALTHY
    elif mean_sim >= warning_threshold and silhouette >= 0.1:
        status = HealthStatus.WARNING
    else:
        status = HealthStatus.UNHEALTHY

    return IntraSpeakerResult(
        label=speaker_label,
        mean_similarity=round(mean_sim, 4),
        std_similarity=round(std_sim, 4),
        min_similarity=round(min_sim, 4),
        silhouette_score=round(silhouette, 4),
        num_embeddings=n_embeddings,
        is_mature=is_mature,
        status=status,
    )


def compute_inter_speaker_separation(
    speaker_input: InterSpeakerInput,
    healthy_threshold: float = 0.5,  # Mean distance > 0.5 = well separated
    warning_threshold: float = 0.3,  # Mean distance > 0.3 = borderline
) -> InterSpeakerResult:
    """
    Compute inter-speaker separation with the bare minimum:
    - Mean separation (are speakers generally distinct?)
    - Closest pair (who's at risk of merging?)

    Your labeler already handles merge detection operationally,
    so this is just for monitoring/dashboard visibility.
    """
    speaker_embeddings = speaker_input["speakers"]

    if len(speaker_embeddings) < 2:
        raise ValueError(f"Need at least 2 speakers, got {len(speaker_embeddings)}")

    centroids = {sid: np.mean(embs, axis=0) for sid, embs in speaker_embeddings.items()}

    labels = sorted(centroids.keys())
    n = len(labels)

    separations = []
    min_sep = float("inf")
    closest_pair = (labels[0], labels[1])

    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine_distance(centroids[labels[i]], centroids[labels[j]])
            separations.append(dist)
            if dist < min_sep:
                min_sep = dist
                closest_pair = (labels[i], labels[j])

    mean_sep = float(np.mean(separations))

    # Simple health rules
    if mean_sep >= healthy_threshold and min_sep >= 0.3:
        status = HealthStatus.HEALTHY
    elif mean_sep >= warning_threshold and min_sep >= 0.15:
        status = HealthStatus.WARNING
    else:
        status = HealthStatus.UNHEALTHY

    return InterSpeakerResult(
        mean_separation=round(mean_sep, 4),
        min_separation=round(min_sep, 4),
        closest_pair=closest_pair,
        num_speakers=n,
        status=status,
    )
