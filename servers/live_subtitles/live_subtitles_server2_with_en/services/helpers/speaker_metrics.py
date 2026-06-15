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
    """Health status classification for speaker metrics."""

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


class IntraSpeakerResult(TypedDict):
    """Result type for intra-speaker variance analysis."""

    mean_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    distances: List[DistanceItem]
    distance_values: NDArray[np.float64]
    status: HealthStatus
    num_embeddings: int


class InterSpeakerResult(TypedDict):
    """Result type for inter-speaker separation analysis."""

    mean_separation: float
    std_separation: float
    min_separation: float
    max_separation: float
    pairwise_distances: List[PairwiseDistanceItem]
    distance_matrix: NDArray[np.float64]
    speaker_labels: List[str]
    status: HealthStatus
    num_speakers: int


def cosine_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Compute cosine distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine distance (1 - cosine similarity)

    Examples:
        >>> import numpy as np
        >>> cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        1.0
        >>> cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        0.0
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 1.0

    similarity = np.dot(a, b) / (a_norm * b_norm)
    # Clamp to [-1, 1] to handle floating point errors
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)


def compute_intra_speaker_variance(
    embeddings: NDArray[np.float64],
    segment_ids: List[str] | None = None,
    healthy_threshold: float = 0.3,
    warning_threshold: float = 0.5,
) -> IntraSpeakerResult:
    """
    Compute intra-speaker variance by measuring distances from each embedding
    to the speaker centroid.

    Low variance = all points are close to the center → "healthy" tight cluster.

    Args:
        embeddings: Array of shape (n_embeddings, embedding_dim) containing
                   all embeddings for a single speaker
        segment_ids: Optional list of segment identifiers corresponding to each embedding.
                    If None, auto-generates IDs as "segment_0", "segment_1", etc.
        healthy_threshold: Mean distance below this is considered healthy
        warning_threshold: Mean distance below this is considered warning,
                          above is unhealthy

    Returns:
        IntraSpeakerResult with variance metrics, labeled distances, and health status

    Raises:
        ValueError: If embeddings array is empty or has invalid shape
        ValueError: If segment_ids length doesn't match number of embeddings

    Example:
        >>> embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]])
        >>> result = compute_intra_speaker_variance(embeddings)
        >>> result['status']
        'healthy'
        >>> result['distances'][0]['segment_id']
        'segment_0'
    """
    if embeddings.size == 0:
        raise ValueError("Embeddings array cannot be empty")

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")

    n_embeddings = embeddings.shape[0]

    # Handle segment IDs
    if segment_ids is None:
        segment_ids = [f"segment_{i}" for i in range(n_embeddings)]
    elif len(segment_ids) != n_embeddings:
        raise ValueError(
            f"Number of segment_ids ({len(segment_ids)}) must match "
            f"number of embeddings ({n_embeddings})"
        )

    if n_embeddings < 2:
        # With only one embedding, variance is 0
        distance_items = [DistanceItem(segment_id=segment_ids[0], distance=0.0)]
        return IntraSpeakerResult(
            mean_distance=0.0,
            std_distance=0.0,
            min_distance=0.0,
            max_distance=0.0,
            distances=distance_items,
            distance_values=np.array([0.0]),
            status=HealthStatus.HEALTHY,
            num_embeddings=1,
        )

    # Compute centroid
    centroid = np.mean(embeddings, axis=0)

    # Compute distances from each embedding to centroid with labels
    distance_values = np.array(
        [cosine_distance(embedding, centroid) for embedding in embeddings]
    )

    # Create labeled distance items
    distance_items = [
        DistanceItem(segment_id=seg_id, distance=float(dist))
        for seg_id, dist in zip(segment_ids, distance_values)
    ]

    mean_dist = float(np.mean(distance_values))
    std_dist = float(np.std(distance_values))
    min_dist = float(np.min(distance_values))
    max_dist = float(np.max(distance_values))

    # Determine health status
    if mean_dist <= healthy_threshold:
        status = HealthStatus.HEALTHY
    elif mean_dist <= warning_threshold:
        status = HealthStatus.WARNING
    else:
        status = HealthStatus.UNHEALTHY

    return IntraSpeakerResult(
        mean_distance=mean_dist,
        std_distance=std_dist,
        min_distance=min_dist,
        max_distance=max_dist,
        distances=distance_items,
        distance_values=distance_values,
        status=status,
        num_embeddings=n_embeddings,
    )


def compute_inter_speaker_separation(
    speaker_embeddings: Dict[str, NDArray[np.float64]],
    healthy_threshold: float = 0.5,
    warning_threshold: float = 0.3,
) -> InterSpeakerResult:
    """
    Compute inter-speaker separation by measuring distances between
    speaker centroids.

    High separation = centroids are far apart → "healthy" distinct speakers.

    Args:
        speaker_embeddings: Dictionary mapping speaker IDs to their embedding arrays.
                           Each array has shape (n_embeddings, embedding_dim)
        healthy_threshold: Mean separation above this is considered healthy
        warning_threshold: Mean separation above this is considered warning,
                          below is unhealthy

    Returns:
        InterSpeakerResult with separation metrics, labeled pairwise distances,
        and health status

    Raises:
        ValueError: If fewer than 2 speakers provided or embeddings are invalid

    Example:
        >>> spk_embs = {
        ...     'speaker_A': np.array([[1.0, 0.0], [0.9, 0.1]]),
        ...     'speaker_B': np.array([[-1.0, 0.0], [-0.9, -0.1]])
        ... }
        >>> result = compute_inter_speaker_separation(spk_embs)
        >>> result['pairwise_distances'][0]['speaker_id_1']
        'speaker_A'
    """
    if len(speaker_embeddings) < 2:
        raise ValueError(f"Need at least 2 speakers, got {len(speaker_embeddings)}")

    # Validate all embedding arrays
    for speaker_id, embeddings in speaker_embeddings.items():
        if embeddings.size == 0:
            raise ValueError(f"Speaker '{speaker_id}' has empty embeddings")
        if embeddings.ndim != 2:
            raise ValueError(
                f"Speaker '{speaker_id}' embeddings must be 2D, got {embeddings.ndim}D"
            )

    # Compute centroids for each speaker
    centroids = {
        speaker_id: np.mean(embeddings, axis=0)
        for speaker_id, embeddings in speaker_embeddings.items()
    }

    # Get ordered list of speaker labels for consistent matrix indexing
    speaker_labels = sorted(centroids.keys())
    n_speakers = len(speaker_labels)

    # Compute pairwise distances between all centroids
    distance_matrix = np.zeros((n_speakers, n_speakers))
    pairwise_items: List[PairwiseDistanceItem] = []

    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            dist = cosine_distance(
                centroids[speaker_labels[i]], centroids[speaker_labels[j]]
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

            # Create labeled distance item
            pairwise_items.append(
                PairwiseDistanceItem(
                    speaker_id_1=speaker_labels[i],
                    speaker_id_2=speaker_labels[j],
                    distance=float(dist),
                )
            )

    # Extract upper triangle (excluding diagonal) for statistics
    upper_triangle = distance_matrix[np.triu_indices(n_speakers, k=1)]

    mean_sep = float(np.mean(upper_triangle))
    std_sep = float(np.std(upper_triangle))
    min_sep = float(np.min(upper_triangle))
    max_sep = float(np.max(upper_triangle))

    # Determine health status
    if mean_sep >= healthy_threshold:
        status = HealthStatus.HEALTHY
    elif mean_sep >= warning_threshold:
        status = HealthStatus.WARNING
    else:
        status = HealthStatus.UNHEALTHY

    return InterSpeakerResult(
        mean_separation=mean_sep,
        std_separation=std_sep,
        min_separation=min_sep,
        max_separation=max_sep,
        pairwise_distances=pairwise_items,
        distance_matrix=distance_matrix,
        speaker_labels=speaker_labels,
        status=status,
        num_speakers=n_speakers,
    )
