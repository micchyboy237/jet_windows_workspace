"""
speaker_centroid_health.py
==========================
Reusable, fully-typed utilities for assessing the health of speaker label
centroids in a diarization system.

Each centroid is assumed to own a list of L2- or unit-normalized embeddings
(numpy arrays). All similarity/distance operations default to cosine distance,
matching the convention of ECAPA-TDNN / AM-Softmax trained encoders.

Public surface
--------------
- CentroidHealth          – dataclass holding every health metric for one centroid
- CentroidHealthReport    – dataclass holding system-wide diagnostics
- CentroidHealthChecker   – main class; call .check_all() to get the report
- HealthThresholds        – dataclass of configurable pass/fail thresholds
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Embedding = NDArray[np.float32]          # shape: (D,)
EmbeddingMatrix = NDArray[np.float32]    # shape: (N, D)
LabelID = str                            # arbitrary speaker label, e.g. "SPEAKER_00"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthFlag(Enum):
    HEALTHY = auto()
    IMMATURE = auto()        # too few embeddings to be trusted
    DIFFUSE = auto()         # high intra-cluster spread
    TOO_CLOSE = auto()       # dangerously similar to another centroid
    REDUNDANT = auto()       # likely duplicate of an existing speaker
    CONTAMINATED = auto()    # low mean cosine similarity → mixed content


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HealthThresholds:
    """
    All thresholds are tunable. Defaults are empirically common in the
    literature (cosine similarity space, unit-normalized embeddings).
    """
    min_embedding_count: int = 5
    """Centroids with fewer embeddings than this are flagged IMMATURE."""

    min_mean_cosine_sim: float = 0.55
    """Mean cosine similarity of embeddings to their centroid. Below → CONTAMINATED."""

    max_intra_spread: float = 0.45
    """Max allowed mean cosine *distance* within a cluster. Above → DIFFUSE."""

    min_silhouette_score: float = 0.10
    """Per-centroid average silhouette. Below → DIFFUSE (overlapping)."""

    max_inter_centroid_similarity: float = 0.75
    """
    If two centroids are more similar than this they risk representing the
    same speaker → TOO_CLOSE / REDUNDANT.
    """

    merge_similarity_threshold: float = 0.85
    """Above this the pair is almost certainly the same speaker → REDUNDANT."""


# ---------------------------------------------------------------------------
# Per-centroid result
# ---------------------------------------------------------------------------

@dataclass
class CentroidHealth:
    label: LabelID
    embedding_count: int

    centroid_vector: Embedding                 # mean/normalized centroid
    mean_cosine_sim_to_centroid: float         # cohesion
    intra_cluster_spread: float                # mean cosine distance within cluster
    silhouette_score: float                    # per-centroid average silhouette

    # Inter-centroid: filled in after comparing against all other centroids
    nearest_centroid_label: Optional[LabelID] = None
    nearest_centroid_similarity: Optional[float] = None

    flags: List[HealthFlag] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return len(self.flags) == 0 or self.flags == [HealthFlag.HEALTHY]


# ---------------------------------------------------------------------------
# System-wide report
# ---------------------------------------------------------------------------

@dataclass
class CentroidHealthReport:
    results: Dict[LabelID, CentroidHealth]
    thresholds: HealthThresholds

    @property
    def healthy_labels(self) -> List[LabelID]:
        return [lbl for lbl, r in self.results.items() if r.is_healthy]

    @property
    def unhealthy_labels(self) -> List[LabelID]:
        return [lbl for lbl, r in self.results.items() if not r.is_healthy]

    @property
    def merge_candidates(self) -> List[Tuple[LabelID, LabelID, float]]:
        """
        Returns pairs (label_a, label_b, similarity) that exceed the
        merge_similarity_threshold. Pairs are deduplicated.
        """
        seen: set[frozenset[LabelID]] = set()
        candidates: List[Tuple[LabelID, LabelID, float]] = []
        thresh = self.thresholds.merge_similarity_threshold
        for lbl, r in self.results.items():
            if (
                r.nearest_centroid_label is not None
                and r.nearest_centroid_similarity is not None
                and r.nearest_centroid_similarity >= thresh
            ):
                pair: frozenset[LabelID] = frozenset({lbl, r.nearest_centroid_label})
                if pair not in seen:
                    seen.add(pair)
                    candidates.append((lbl, r.nearest_centroid_label, r.nearest_centroid_similarity))
        return candidates

    def summary(self) -> str:
        lines = [
            f"=== Centroid Health Report ===",
            f"Total centroids : {len(self.results)}",
            f"Healthy         : {len(self.healthy_labels)}",
            f"Unhealthy       : {len(self.unhealthy_labels)}",
            f"Merge candidates: {len(self.merge_candidates)}",
            "",
        ]
        for lbl, r in self.results.items():
            flag_str = ", ".join(f.name for f in r.flags) or "HEALTHY"
            lines.append(
                f"  [{flag_str:<20}] {lbl:20} "
                f"n={r.embedding_count:3d}  "
                f"coh={r.mean_cosine_sim_to_centroid:.3f}  "
                f"spread={r.intra_cluster_spread:.3f}  "
                f"sil={r.silhouette_score:.3f}"
                + (
                    f"  nearest={r.nearest_centroid_label}({r.nearest_centroid_similarity:.3f})"
                    if r.nearest_centroid_label
                    else ""
                )
            )
        if self.merge_candidates:
            lines += ["", "--- Merge Candidates ---"]
            for a, b, sim in self.merge_candidates:
                lines.append(f"  {a} <-> {b}  similarity={sim:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def _l2_normalize(v: Embedding) -> Embedding:
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return (v / norm).astype(np.float32)


def _cosine_similarity(a: Embedding, b: Embedding) -> float:
    """Cosine similarity in [-1, 1]. Both inputs should be unit-normed."""
    return float(np.dot(a, b))


def _cosine_distance(a: Embedding, b: Embedding) -> float:
    return 1.0 - _cosine_similarity(a, b)


def _compute_centroid(embeddings: EmbeddingMatrix) -> Embedding:
    """Mean of embeddings, then L2-normalized (standard for cosine space)."""
    mean_vec = embeddings.mean(axis=0).astype(np.float32)
    return _l2_normalize(mean_vec)


def _mean_cosine_sim_to_centroid(embeddings: EmbeddingMatrix, centroid: Embedding) -> float:
    """Average cosine similarity of each embedding to the centroid."""
    sims = embeddings @ centroid          # shape: (N,)
    return float(sims.mean())


def _intra_cluster_spread(embeddings: EmbeddingMatrix, centroid: Embedding) -> float:
    """Mean cosine *distance* of each embedding from its centroid."""
    sims = embeddings @ centroid
    distances = 1.0 - sims
    return float(distances.mean())


def _per_centroid_silhouette(
    label: LabelID,
    embeddings_map: Dict[LabelID, EmbeddingMatrix],
    centroids_map: Dict[LabelID, Embedding],
) -> float:
    """
    Approximate per-centroid silhouette: computes silhouette for each embedding
    in the cluster and returns the mean.

    Uses centroid-based approximation for efficiency (O(N·K) vs O(N²·K)):
      a(i) ≈ mean intra-cluster cosine distance
      b(i) ≈ min over other clusters of mean cosine distance to that centroid
    """
    own_embeddings = embeddings_map[label]   # (N, D)
    other_labels = [l for l in embeddings_map if l != label]

    if not other_labels:
        return 0.0                           # only one speaker: undefined

    own_centroid = centroids_map[label]

    # a(i) for every embedding in this cluster
    a_values = 1.0 - (own_embeddings @ own_centroid)  # (N,)

    # b(i): min mean distance to any *other* centroid
    b_matrix = np.column_stack([
        1.0 - (own_embeddings @ centroids_map[other])
        for other in other_labels
    ])  # shape: (N, K-1)
    b_values = b_matrix.min(axis=1)           # (N,)

    denom = np.maximum(a_values, b_values)
    # avoid division by zero for identical vectors
    safe = denom > 1e-10
    sil = np.where(safe, (b_values - a_values) / denom, 0.0)
    return float(sil.mean())


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

class CentroidHealthChecker:
    """
    Check the health of speaker label centroids.

    Parameters
    ----------
    embeddings_per_label : dict mapping each speaker label to its list of
        unit-normalized embedding vectors (shape: (D,) each).
    thresholds : HealthThresholds (optional; uses defaults if not provided).

    Example
    -------
    >>> checker = CentroidHealthChecker(
    ...     embeddings_per_label={
    ...         "SPEAKER_00": [emb1, emb2, emb3],
    ...         "SPEAKER_01": [emb4, emb5],
    ...     }
    ... )
    >>> report = checker.check_all()
    >>> print(report.summary())
    """

    def __init__(
        self,
        embeddings_per_label: Dict[LabelID, Sequence[Embedding]],
        thresholds: Optional[HealthThresholds] = None,
    ) -> None:
        self.thresholds = thresholds or HealthThresholds()

        # Convert all embedding lists → stacked unit-normalized matrices
        self._embeddings: Dict[LabelID, EmbeddingMatrix] = {}
        for label, embs in embeddings_per_label.items():
            if len(embs) == 0:
                warnings.warn(f"Label '{label}' has no embeddings; skipping.")
                continue
            mat = np.stack([_l2_normalize(np.asarray(e, dtype=np.float32)) for e in embs])
            self._embeddings[label] = mat

        # Pre-compute centroids (needed for silhouette)
        self._centroids: Dict[LabelID, Embedding] = {
            label: _compute_centroid(mat)
            for label, mat in self._embeddings.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(self) -> CentroidHealthReport:
        """Run all health checks and return a full report."""
        results: Dict[LabelID, CentroidHealth] = {}

        for label in self._embeddings:
            results[label] = self._check_one(label)

        self._fill_inter_centroid_metrics(results)
        self._apply_flags(results)

        return CentroidHealthReport(results=results, thresholds=self.thresholds)

    def check_one(self, label: LabelID) -> CentroidHealth:
        """
        Run health checks for a single centroid in isolation.
        Inter-centroid metrics (nearest neighbour, TOO_CLOSE, REDUNDANT flags)
        are computed relative to all other registered centroids.
        """
        result = self._check_one(label)
        self._fill_inter_centroid_metrics({label: result})
        self._apply_flags({label: result})
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_one(self, label: LabelID) -> CentroidHealth:
        embeddings = self._embeddings[label]
        centroid = self._centroids[label]

        mean_sim = _mean_cosine_sim_to_centroid(embeddings, centroid)
        spread = _intra_cluster_spread(embeddings, centroid)
        silhouette = _per_centroid_silhouette(label, self._embeddings, self._centroids)

        return CentroidHealth(
            label=label,
            embedding_count=len(embeddings),
            centroid_vector=centroid,
            mean_cosine_sim_to_centroid=mean_sim,
            intra_cluster_spread=spread,
            silhouette_score=silhouette,
        )

    def _fill_inter_centroid_metrics(
        self, results: Dict[LabelID, CentroidHealth]
    ) -> None:
        """Populate nearest_centroid_* fields by comparing against all centroids."""
        all_labels = list(self._centroids.keys())

        for label, result in results.items():
            other_labels = [l for l in all_labels if l != label]
            if not other_labels:
                continue

            sims = {
                other: _cosine_similarity(result.centroid_vector, self._centroids[other])
                for other in other_labels
            }
            nearest = max(sims, key=lambda l: sims[l])
            result.nearest_centroid_label = nearest
            result.nearest_centroid_similarity = sims[nearest]

    def _apply_flags(self, results: Dict[LabelID, CentroidHealth]) -> None:
        t = self.thresholds
        for result in results.values():
            flags: List[HealthFlag] = []

            if result.embedding_count < t.min_embedding_count:
                flags.append(HealthFlag.IMMATURE)

            if result.mean_cosine_sim_to_centroid < t.min_mean_cosine_sim:
                flags.append(HealthFlag.CONTAMINATED)

            if result.intra_cluster_spread > t.max_intra_spread:
                flags.append(HealthFlag.DIFFUSE)

            if result.silhouette_score < t.min_silhouette_score:
                if HealthFlag.DIFFUSE not in flags:
                    flags.append(HealthFlag.DIFFUSE)

            if result.nearest_centroid_similarity is not None:
                sim = result.nearest_centroid_similarity
                if sim >= t.merge_similarity_threshold:
                    flags.append(HealthFlag.REDUNDANT)
                elif sim >= t.max_inter_centroid_similarity:
                    flags.append(HealthFlag.TOO_CLOSE)

            result.flags = flags if flags else [HealthFlag.HEALTHY]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def check_centroid_health(
    embeddings_per_label: Dict[LabelID, Sequence[Embedding]],
    thresholds: Optional[HealthThresholds] = None,
) -> CentroidHealthReport:
    """
    One-shot helper. Equivalent to::

        CentroidHealthChecker(embeddings_per_label, thresholds).check_all()
    """
    return CentroidHealthChecker(embeddings_per_label, thresholds).check_all()


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    D = 192  # typical ECAPA-TDNN embedding dimension

    def _make_cluster(center: Embedding, n: int, noise: float) -> List[Embedding]:
        vecs = center + rng.normal(0, noise, size=(n, D)).astype(np.float32)
        return [_l2_normalize(v) for v in vecs]

    centers = {
        "SPEAKER_00": _l2_normalize(rng.standard_normal(D).astype(np.float32)),
        "SPEAKER_01": _l2_normalize(rng.standard_normal(D).astype(np.float32)),
        "SPEAKER_02": _l2_normalize(rng.standard_normal(D).astype(np.float32)),
    }

    embeddings_per_label: Dict[LabelID, List[Embedding]] = {
        "SPEAKER_00": _make_cluster(centers["SPEAKER_00"], n=20, noise=0.05),   # healthy
        "SPEAKER_01": _make_cluster(centers["SPEAKER_01"], n=2, noise=0.05),    # IMMATURE
        "SPEAKER_02": _make_cluster(centers["SPEAKER_00"], n=15, noise=0.40),   # DIFFUSE / CONTAMINATED
        # Duplicate of SPEAKER_00 → should be flagged REDUNDANT
        "SPEAKER_03": _make_cluster(centers["SPEAKER_00"], n=10, noise=0.03),
    }

    report = check_centroid_health(embeddings_per_label)
    print(report.summary())
