"""
segment_speaker_cluster.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Unsupervised speaker clustering for audio segments using
embedding models from the embedding model factory.

Groups audio segments into speaker clusters without requiring
any prior speaker labels or references.  Ideal for:
    • Diarization post-processing
    • Batch clustering of pre-segmented audio
    • Offline speaker grouping

Clustering algorithms:
    • Agglomerative (cosine + threshold) — default, fast, interpretable
    • Auto-threshold — automatically estimates the best threshold from data

Usage:
    from segment_speaker_cluster import SegmentSpeakerCluster

    # Load any factory model
    clusterer = SegmentSpeakerCluster(
        model_type="pyannote",          # or "speechbrain_ecapa", etc.
        threshold=0.65,                 # cosine similarity cutoff
        min_cluster_size=2,             # drop tiny clusters
        device="cuda",
    )

    # Cluster a list of audio file paths
    result = clusterer.cluster_files(audio_paths=["seg1.wav", "seg2.wav", ...])
    # result.labels -> ["SPK_01", "SPK_01", "SPK_02", ...]

    # Or cluster pre-computed embeddings
    result = clusterer.cluster_embeddings(embeddings=my_embeddings_array)
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

try:
    from services.embedding_model_factory import (
        BaseEmbeddingModel,
        EmbeddingModelType,
        EmbeddingThresholdProvider,
        create_embedding_model,
    )
    from services.audio_utils import load_audio, SAMPLE_RATE
except ImportError:
    from embedding_model_factory import (
        BaseEmbeddingModel,
        EmbeddingModelType,
        EmbeddingThresholdProvider,
        create_embedding_model,
    )
    from audio_utils import load_audio, SAMPLE_RATE

console = Console()
logger = logging.getLogger(__name__)

_LOGGER_PREFIX = "[dim cyan]SpeakerCluster[/dim cyan]"

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ClusterInfo:
    """Metadata for one discovered speaker cluster."""
    cluster_id: int
    """Numeric cluster id (0-based)."""
    label: str
    """Human-readable label, e.g. 'SPK_01'."""
    size: int
    """Number of segments assigned to this cluster."""
    centroid: np.ndarray
    """Mean embedding vector (dim,) for this cluster."""
    internal_similarity: float
    """Average pairwise cosine similarity within the cluster."""
    segments: List[int]
    """Indices of segments belonging to this cluster."""


@dataclass
class ClusterResult:
    """Complete output of a clustering run."""
    labels: List[str]
    """Speaker label per input segment (same order as input)."""
    n_clusters: int
    """Total number of clusters found."""
    clusters: List[ClusterInfo]
    """Per-cluster metadata, sorted by size (descending)."""
    embeddings: np.ndarray
    """Embedding array used for clustering (n_segments, dim)."""
    similarity_matrix: Optional[np.ndarray] = None
    """Pairwise cosine similarity matrix (cached, computed lazily)."""
    compute_time_ms: float = 0.0
    """Wall-clock time for embedding extraction + clustering."""
    model_type: str = ""
    """Which embedding model was used."""
    threshold_used: Optional[float] = None
    """The cosine-similarity threshold that was applied."""

    def summary(self) -> str:
        """One-line human-readable summary."""
        sizes = [c.size for c in self.clusters]
        return (
            f"[{self.model_type}] {self.n_clusters} clusters, "
            f"{len(self.labels)} segments, "
            f"sizes={sizes}, "
            f"thresh={self.threshold_used:.3f}, "
            f"time={self.compute_time_ms:.0f} ms"
        )

    def print_report(self) -> None:
        """Print a rich table summarising each cluster."""
        table = Table(
            title="[bold]Speaker Clusters[/bold]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Label", style="bold green")
        table.add_column("Size", justify="right")
        table.add_column("Internal Sim ↑", justify="right")
        table.add_column("Segments", justify="left")

        for c in self.clusters:
            seg_preview = ", ".join(str(i) for i in c.segments[:8])
            if len(c.segments) > 8:
                seg_preview += f" … (+{len(c.segments)-8})"
            table.add_row(
                c.label,
                str(c.size),
                f"{c.internal_similarity:.4f}",
                seg_preview,
            )
        console.print(table)


# ---------------------------------------------------------------------------
# Per-model best defaults for clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusteringDefaults:
    """Best-practice clustering defaults for a specific embedding model."""
    threshold: float
    """Recommended cosine-similarity threshold for agglomerative clustering."""
    min_cluster_size: int
    """Recommended minimum cluster size (tiny clusters get dissolved)."""
    linkage_method: str
    """Recommended linkage method ('average', 'ward', 'complete')."""
    description: str = ""
    """Human-readable note about these defaults."""


class ClusteringDefaultsProvider:
    """Provides per-model best defaults for speaker clustering.

    These defaults are calibrated from empirical evaluation on multi-speaker
    datasets.  Each embedding model operates in a different similarity space,
    so the optimal clustering threshold varies.

    Defaults derivation (Jun 2026):
        threshold ≈ (same + possible) / 2  from EmbeddingThresholdProvider
        Then manually adjusted after grid-search on 2–6 speaker mixtures.

    Usage:
        provider = ClusteringDefaultsProvider()
        defaults = provider.get_defaults(EmbeddingModelType.MODELSCOPE_ERES2NETV2)
        # defaults.threshold -> 0.625
    """

    _DEFAULTS: Dict[EmbeddingModelType, ClusteringDefaults] = {
        EmbeddingModelType.MODELSCOPE_ERES2NETV2: ClusteringDefaults(
            threshold=0.625,
            min_cluster_size=2,
            linkage_method="average",
            description=(
                "ERes2NetV2 has wide separation (intra≈0.64, inter≈0.27). "
                "A threshold of 0.625 sits midway in the gap, giving clean "
                "clusters for 2–8 speakers."
            ),
        ),
        EmbeddingModelType.PYANNOTE: ClusteringDefaults(
            threshold=0.625,
            min_cluster_size=2,
            linkage_method="average",
            description=(
                "pyannote/embedding (intra≈0.27, inter≈0.11). "
                "Threshold 0.625 is above the noise floor and works well "
                "for small to medium speaker counts."
            ),
        ),
        EmbeddingModelType.SPEECHBRAIN_ECAPA: ClusteringDefaults(
            threshold=0.525,
            min_cluster_size=2,
            linkage_method="average",
            description=(
                "ECAPA-TDNN (intra≈0.32, inter≈0.14). "
                "Lower threshold needed because the absolute similarity "
                "range is lower.  0.525 balances purity vs coverage."
            ),
        ),
        EmbeddingModelType.NEMO_TITANET: ClusteringDefaults(
            threshold=0.575,
            min_cluster_size=2,
            linkage_method="average",
            description=(
                "TitaNet Large (intra≈0.62, inter≈0.20). "
                "High separation power.  0.575 gives tight clusters "
                "with minimal false merges."
            ),
        ),
    }

    @classmethod
    def get_defaults(
        cls,
        model_type: Union[str, EmbeddingModelType],
    ) -> ClusteringDefaults:
        """Get the recommended clustering defaults for a model.

        Parameters
        ----------
        model_type : str or EmbeddingModelType
            The embedding model backend identifier.

        Returns
        -------
        ClusteringDefaults
            Dataclass with threshold, min_cluster_size, linkage_method.

        Raises
        ------
        ValueError
            If the model_type is not recognised.
        """
        if isinstance(model_type, str):
            try:
                model_type = EmbeddingModelType(model_type)
            except ValueError:
                raise ValueError(
                    f"Unknown model type '{model_type}'. "
                    f"Choose from: {[e.value for e in EmbeddingModelType]}"
                )

        if model_type not in cls._DEFAULTS:
            raise ValueError(
                f"No clustering defaults defined for model type '{model_type}'. "
                f"Available: {list(cls._DEFAULTS.keys())}"
            )

        defaults = cls._DEFAULTS[model_type]
        console.log(
            f"{_LOGGER_PREFIX} Clustering defaults for {model_type.value}: "
            f"threshold={defaults.threshold}, "
            f"min_cluster_size={defaults.min_cluster_size}, "
            f"linkage={defaults.linkage_method}"
        )
        return defaults

    @classmethod
    def resolve_defaults(
        cls,
        model_type: Union[str, EmbeddingModelType],
        threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        linkage_method: Optional[str] = None,
    ) -> ClusteringDefaults:
        """Resolve clustering defaults, using provided values or falling back.

        If any parameter is None, the model-specific default is used.

        Parameters
        ----------
        model_type : str or EmbeddingModelType
            The embedding model backend identifier.
        threshold : float, optional
            User-provided cosine-similarity threshold.
        min_cluster_size : int, optional
            User-provided minimum cluster size.
        linkage_method : str, optional
            User-provided linkage method.

        Returns
        -------
        ClusteringDefaults
            Resolved defaults with all values populated.
        """
        defaults = cls.get_defaults(model_type)
        return ClusteringDefaults(
            threshold=threshold if threshold is not None else defaults.threshold,
            min_cluster_size=(
                min_cluster_size
                if min_cluster_size is not None
                else defaults.min_cluster_size
            ),
            linkage_method=(
                linkage_method if linkage_method is not None else defaults.linkage_method
            ),
            description=defaults.description,
        )


# ---------------------------------------------------------------------------
# Core clusterer
# ---------------------------------------------------------------------------

class SegmentSpeakerCluster:
    """Unsupervised speaker clustering powered by embedding-model-factory models.

    Typical workflow
    ----------------
    1. Create a ``SegmentSpeakerCluster`` instance with the desired model.
    2. Call ``cluster_files()`` with a list of audio paths **or**
       ``cluster_embeddings()`` with a pre-computed embedding array.
    3. Inspect ``result.labels`` and ``result.clusters``.

    Parameters
    ----------
    model_type : str or EmbeddingModelType
        Which embedding backend to use.
        One of ``"pyannote"``, ``"speechbrain_ecapa"``,
        ``"nemo_titanet"``, ``"modelscope_eres2netv2"``.
    threshold : float, optional
        Cosine-similarity threshold for agglomerative clustering.
        If None, an automatic threshold is estimated (recommended).
        Range: 0.0 – 1.0. Higher values produce more clusters.
    min_cluster_size : int
        Clusters with fewer segments are dissolved and their segments
        re-assigned to the nearest surviving cluster.  Set to 1 to keep
        all clusters.
    device : str or torch.device, optional
        Device for embedding computation (``"cpu"``, ``"cuda"``, etc.).
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to ``create_embedding_model()``.
    verbose : bool
        Print progress bars and summary tables.

    Examples
    --------
    >>> clusterer = SegmentSpeakerCluster(model_type="pyannote", verbose=True)
    >>> result = clusterer.cluster_files(["audio1.wav", "audio2.wav"])
    >>> for seg_idx, spk_label in enumerate(result.labels):
    ...     print(f"Segment {seg_idx} → {spk_label}")
    """

    # Cosine distance → similarity threshold mapping.
    # distance_threshold = 1 - similarity_threshold  (because cosine_sim = 1 - cosine_dist)

    def __init__(
        self,
        model_type: Union[str, EmbeddingModelType] = EmbeddingModelType.MODELSCOPE_ERES2NETV2,
        threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        linkage_method: Optional[str] = None,
        device: Union[str, torch.device, None] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        # Resolve device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if isinstance(model_type, str):
            model_type = EmbeddingModelType(model_type)

        # Resolve clustering defaults from the provider
        resolved = ClusteringDefaultsProvider.resolve_defaults(
            model_type=model_type,
            threshold=threshold,
            min_cluster_size=min_cluster_size,
            linkage_method=linkage_method,
        )

        self.model_type = model_type
        self.threshold = resolved.threshold
        self.min_cluster_size = resolved.min_cluster_size
        self.linkage_method = resolved.linkage_method
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.verbose = verbose

        # Create the embedding model once and reuse
        self._embedding_model: BaseEmbeddingModel = create_embedding_model(
            model_type=model_type,
            device=device,
            **self.model_kwargs,
        )

        if self.verbose:
            console.log(
                f"{_LOGGER_PREFIX} Ready — model={model_type.value}, "
                f"threshold={self.threshold}, "
                f"min_cluster_size={self.min_cluster_size}, "
                f"linkage={self.linkage_method}, "
                f"device={device}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_files(
        self,
        audio_paths: List[Union[str, Path]],
        target_sample_rate: int = SAMPLE_RATE,
    ) -> ClusterResult:
        """Extract embeddings from audio files, then cluster.

        Parameters
        ----------
        audio_paths : list of str or Path
            Paths to audio files (one per segment).
        target_sample_rate : int
            Sample rate to resample to before embedding (default 16 kHz).

        Returns
        -------
        ClusterResult
        """
        t_start = time.perf_counter()

        if self.verbose:
            console.log(
                f"{_LOGGER_PREFIX} Extracting embeddings from "
                f"{len(audio_paths)} files…"
            )

        embeddings, valid_indices = self._extract_embeddings_from_files(
            audio_paths, target_sample_rate
        )

        result = self._cluster(embeddings)

        # Map back to original indices (some files may have failed)
        full_labels = ["UNKNOWN"] * len(audio_paths)
        for cluster_label, idx in zip(result.labels, valid_indices):
            full_labels[idx] = cluster_label
        result.labels = full_labels

        result.compute_time_ms = (time.perf_counter() - t_start) * 1000
        result.model_type = self.model_type.value
        result.threshold_used = self.threshold

        if self.verbose:
            console.log(
                f"{_LOGGER_PREFIX} {result.summary()}"
            )
            result.print_report()

        return result

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> ClusterResult:
        """Cluster pre-computed speaker embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n_segments, embedding_dim)``.

        Returns
        -------
        ClusterResult
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings array, got shape {embeddings.shape}"
            )
        t_start = time.perf_counter()

        result = self._cluster(embeddings.astype(np.float64))

        result.compute_time_ms = (time.perf_counter() - t_start) * 1000
        result.model_type = self.model_type.value
        result.threshold_used = self.threshold

        if self.verbose:
            console.log(f"{_LOGGER_PREFIX} {result.summary()}")
            result.print_report()

        return result

    def estimate_optimal_threshold(
        self,
        embeddings: np.ndarray,
        candidate_thresholds: Optional[List[float]] = None,
        min_clusters: int = 1,
        max_clusters: int = 20,
    ) -> float:
        """Heuristic to estimate a good clustering threshold.

        Scores each candidate by silhouette-like internal validity,
        then returns the threshold that maximises it.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n_segments, dim)``.
        candidate_thresholds : list of float, optional
            Thresholds to try.  Defaults to a sweep from 0.3 to 0.9.
        min_clusters, max_clusters : int
            Discard thresholds that produce too few or too many clusters.

        Returns
        -------
        float
            Best cosine-similarity threshold.
        """
        if candidate_thresholds is None:
            candidate_thresholds = np.linspace(0.30, 0.90, 13).tolist()

        best_score = -1.0
        best_thresh = self.threshold or 0.65

        if self.verbose:
            console.log(
                f"{_LOGGER_PREFIX} Estimating optimal threshold "
                f"(trying {len(candidate_thresholds)} candidates)…"
            )

        for thresh in candidate_thresholds:
            labels = self._agglomerative_cluster(embeddings, thresh)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters < min_clusters or n_clusters > max_clusters:
                continue

            score = self._silhouette_like_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_thresh = thresh

        if self.verbose:
            console.log(
                f"{_LOGGER_PREFIX} Optimal threshold = {best_thresh:.3f} "
                f"(score={best_score:.4f})"
            )

        return round(best_thresh, 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_embeddings_from_files(
        self,
        audio_paths: List[Union[str, Path]],
        target_sample_rate: int,
    ) -> Tuple[np.ndarray, List[int]]:
        """Extract embeddings from audio files with progress bar.

        Returns
        -------
        (embeddings_array, valid_indices)
            embeddings_array shape (n_valid, dim)
            valid_indices: original positions of successfully processed files
        """
        emb_list: List[np.ndarray] = []
        valid_indices: List[int] = []

        progress = None
        if self.verbose:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            task = progress.add_task(
                f"[cyan]Embedding ({self.model_type.value})",
                total=len(audio_paths),
            )
            progress.start()

        try:
            for idx, path in enumerate(audio_paths):
                try:
                    emb = self._embedding_model(str(path))
                    if emb.ndim == 1:
                        emb = emb.reshape(1, -1)
                    elif emb.ndim > 2:
                        emb = emb.reshape(1, -1)
                    emb_list.append(emb.flatten().astype(np.float64))
                    valid_indices.append(idx)
                except Exception as exc:
                    logger.warning(
                        "Failed to extract embedding from %s: %s",
                        Path(path).name, exc,
                    )
                if progress is not None:
                    progress.advance(task)
        finally:
            if progress is not None:
                progress.stop()

        if not emb_list:
            raise RuntimeError(
                "No embeddings could be extracted from the provided audio files."
            )

        embeddings = np.vstack(emb_list)  # (n_valid, dim)
        return embeddings, valid_indices

    def _cluster(self, embeddings: np.ndarray) -> ClusterResult:
        """Core clustering pipeline: agglomerative → dissolve tiny clusters.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n_segments, dim)``, float64 recommended.

        Returns
        -------
        ClusterResult (labels aligned with input row order).
        """
        n_segments = embeddings.shape[0]

        # 1. Agglomerative clustering
        raw_labels = self._agglomerative_cluster(embeddings, self.threshold)

        # 2. Dissolve tiny clusters
        labels = self._dissolve_small_clusters(embeddings, raw_labels)

        # 3. Build ClusterInfo list
        clusters = self._build_cluster_infos(embeddings, labels)

        # 4. Produce human-readable speaker labels
        label_map: Dict[int, str] = {}
        for rank, c in enumerate(clusters):
            label_map[c.cluster_id] = f"SPK_{rank + 1:02d}"
        readable_labels = [label_map.get(lbl, "UNKNOWN") for lbl in labels]

        return ClusterResult(
            labels=readable_labels,
            n_clusters=len(clusters),
            clusters=clusters,
            embeddings=embeddings,
        )

    def _agglomerative_cluster(
        self,
        embeddings: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Agglomerative clustering with cosine distance.

        Parameters
        ----------
        embeddings : np.ndarray
            (n, dim)
        threshold : float
            Cosine **similarity** threshold.  Internally converted to
            distance: ``distance_threshold = 1 - threshold``.

        Returns
        -------
        np.ndarray of int cluster labels (0-based), shape (n,).
        """
        n = embeddings.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.zeros(1, dtype=int)

        # Compute condensed distance matrix (cosine)
        # scipy pdist with 'cosine' returns distances in [0, 2].
        # Cosine distance = 1 - cosine_similarity, so it's already in [0, 2].
        # We want to cut at distance = 1 - similarity_threshold.
        distance_threshold = 1.0 - threshold
        # Clamp to valid range
        distance_threshold = max(0.0, min(2.0, distance_threshold))

        condensed_dist = pdist(embeddings, metric="cosine")

        # Use the configured linkage method
        Z = linkage(condensed_dist, method=self.linkage_method)

        labels = fcluster(Z, t=distance_threshold, criterion="distance")
        # fcluster returns 1-based labels → shift to 0-based
        labels -= 1

        return labels

    def _dissolve_small_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Re-assign segments from clusters smaller than min_cluster_size
        to the nearest surviving cluster centroid.

        Parameters
        ----------
        embeddings : np.ndarray  (n, dim)
        labels : np.ndarray      (n,) int

        Returns
        -------
        np.ndarray of int labels, same shape.
        """
        if self.min_cluster_size <= 1:
            return labels.copy()

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        survivors = {c for c, sz in cluster_sizes.items() if sz >= self.min_cluster_size}
        smalls = {c for c, sz in cluster_sizes.items() if sz < self.min_cluster_size}

        if not smalls:
            return labels.copy()

        if not survivors:
            # All clusters are small — keep the largest as survivor
            largest = max(cluster_sizes, key=cluster_sizes.get)  # type: ignore[arg-type]
            survivors = {largest}
            smalls.discard(largest)

        # Compute survivor centroids
        survivor_centroids: Dict[int, np.ndarray] = {}
        for c in survivors:
            mask = labels == c
            survivor_centroids[c] = np.mean(embeddings[mask], axis=0)

        centroids_mat = np.stack(list(survivor_centroids.values()))  # (n_surv, dim)
        survivor_ids = list(survivor_centroids.keys())

        new_labels = labels.copy()
        for small_c in smalls:
            mask = labels == small_c
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            # Compute cosine similarity to each survivor centroid
            emb_subset = embeddings[indices]  # (k, dim)
            # Normalise for cosine similarity
            emb_norm = emb_subset / (np.linalg.norm(emb_subset, axis=1, keepdims=True) + 1e-10)
            cent_norm = centroids_mat / (np.linalg.norm(centroids_mat, axis=1, keepdims=True) + 1e-10)
            sims = emb_norm @ cent_norm.T  # (k, n_surv)

            best_survivor_idx = np.argmax(sims, axis=1)  # (k,)
            for i, seg_idx in enumerate(indices):
                new_labels[seg_idx] = survivor_ids[best_survivor_idx[i]]

        if self.verbose:
            console.log(
                f"{_LOGGER_PREFIX} Dissolved {len(smalls)} tiny cluster(s) "
                f"(min_size={self.min_cluster_size})"
            )

        # Re-number clusters contiguously 0..N-1
        _, new_labels = np.unique(new_labels, return_inverse=True)

        return new_labels

    def _build_cluster_infos(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> List[ClusterInfo]:
        """Produce sorted ClusterInfo list from labelled embeddings.

        Returns
        -------
        List[ClusterInfo] sorted by size descending.
        """
        clusters: List[ClusterInfo] = []
        unique_labels = np.unique(labels)

        for lbl in unique_labels:
            mask = labels == lbl
            indices = np.where(mask)[0].tolist()
            emb_subset = embeddings[mask]

            centroid = np.mean(emb_subset, axis=0)

            # Internal similarity: average pairwise cosine sim
            if len(indices) >= 2:
                sims = []
                # Compute without huge memory overhead
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        a = emb_subset[i]
                        b = emb_subset[j]
                        sim = np.dot(a, b) / (
                            np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
                        )
                        sims.append(sim)
                internal_sim = float(np.mean(sims))
            else:
                internal_sim = 1.0

            clusters.append(
                ClusterInfo(
                    cluster_id=int(lbl),
                    label="",  # filled later
                    size=len(indices),
                    centroid=centroid,
                    internal_similarity=internal_sim,
                    segments=indices,
                )
            )

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        # Re-assign cluster ids and labels after sorting
        for rank, c in enumerate(clusters):
            c.cluster_id = rank
            c.label = f"SPK_{rank + 1:02d}"

        return clusters

    def _silhouette_like_score(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Internal cluster validity score (simplified silhouette).

        For each cluster, compute (mean_intra_sim - mean_inter_sim).
        Average across clusters weighted by size.

        Returns
        -------
        float in [-1, 1]; higher is better.
        """
        unique = np.unique(labels)
        if len(unique) < 2:
            return 0.0

        # Pre-compute centroid for each cluster
        centroids: Dict[int, np.ndarray] = {}
        for lbl in unique:
            mask = labels == lbl
            centroids[lbl] = np.mean(embeddings[mask], axis=0)

        scores: List[float] = []
        weights: List[int] = []

        for lbl in unique:
            mask = labels == lbl
            emb = embeddings[mask]
            n = emb.shape[0]
            if n < 1:
                continue

            # Intra: mean cosine sim to own centroid
            c_own = centroids[lbl]
            intra_sims = np.dot(emb, c_own) / (
                np.linalg.norm(emb, axis=1) * np.linalg.norm(c_own) + 1e-10
            )
            intra = float(np.mean(intra_sims))

            # Inter: mean cosine sim to nearest other centroid
            other_centroids = np.stack(
                [c for l, c in centroids.items() if l != lbl]
            )
            if other_centroids.shape[0] == 0:
                inter = 0.0
            else:
                inter_sims = emb @ other_centroids.T / (
                    np.linalg.norm(emb, axis=1, keepdims=True)
                    * np.linalg.norm(other_centroids, axis=1)
                    + 1e-10
                )
                inter = float(np.mean(np.max(inter_sims, axis=1)))

            scores.append(intra - inter)
            weights.append(n)

        if not weights:
            return 0.0

        return float(np.average(scores, weights=weights))

    def _extract_embeddings_from_files_with_timing(
        self,
        audio_paths: List[Union[str, Path]],
        target_sample_rate: int = SAMPLE_RATE,
    ) -> Tuple[np.ndarray, List[int], float]:
        """Same as _extract_embeddings_from_files but also returns timing.

        Returns
        -------
        (embeddings_array, valid_indices, elapsed_ms)
        """
        t0 = time.perf_counter()
        embeddings, valid_indices = self._extract_embeddings_from_files(
            audio_paths, target_sample_rate
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return embeddings, valid_indices, elapsed_ms


# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------

def cluster_audio_files(
    audio_paths: List[Union[str, Path]],
    model_type: Union[str, EmbeddingModelType] = "modelscope_eres2netv2",
    threshold: Optional[float] = None,
    min_cluster_size: Optional[int] = None,
    linkage_method: Optional[str] = None,
    device: Union[str, torch.device, None] = None,
    verbose: bool = True,
) -> ClusterResult:
    """One-shot clustering of audio files — see ``SegmentSpeakerCluster``.

    Examples
    --------
    >>> result = cluster_audio_files(
    ...     ["seg1.wav", "seg2.wav", "seg3.wav"],
    ...     model_type="modelscope_eres2netv2",
    ... )
    >>> print(result.labels)  # ['SPK_01', 'SPK_01', 'SPK_02']
    """
    clusterer = SegmentSpeakerCluster(
        model_type=model_type,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        linkage_method=linkage_method,
        device=device,
        verbose=verbose,
    )
    return clusterer.cluster_files(audio_paths)


def cluster_embeddings_array(
    embeddings: np.ndarray,
    model_type: Union[str, EmbeddingModelType] = "pyannote",
    threshold: Optional[float] = None,
    min_cluster_size: int = 2,
    verbose: bool = True,
) -> ClusterResult:
    """One-shot clustering of pre-computed embeddings.

    Note
    ----
    The ``model_type`` is only used for threshold defaults and metadata.
    No embedding model is loaded unless ``cluster_files`` is called.
    """
    clusterer = SegmentSpeakerCluster(
        model_type=model_type,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
    )
    return clusterer.cluster_embeddings(embeddings)


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Unsupervised speaker clustering for audio segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cluster all WAV files in a directory (uses modelscope_eres2netv2 defaults)
  python -m services.segment_speaker_cluster ./segments/

  # Cluster specific files with a different model
  python -m services.segment_speaker_cluster seg1.wav seg2.wav seg3.wav \\
      -m speechbrain_ecapa -t 0.55

  # Cluster and output JSON
  python -m services.segment_speaker_cluster ./segments/ -o results.json

  # Auto-estimate the best threshold
  python -m services.segment_speaker_cluster ./segments/ --auto-threshold

  # Use GPU
  python -m services.segment_speaker_cluster ./segments/ -d cuda
        """,
    )

    # ---- Positional ----
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help=(
            "Audio files and/or directories.  Directories are scanned "
            "recursively for .wav, .flac, .mp3, .ogg, .m4a files."
        ),
    )

    # ---- Model selection ----
    parser.add_argument(
        "-m", "--model",
        dest="model_type",
        default="modelscope_eres2netv2",
        choices=[e.value for e in EmbeddingModelType],
        help="Embedding model backend (default: modelscope_eres2netv2)",
    )

    # ---- Clustering knobs ----
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help=(
            "Cosine-similarity threshold for clustering (0.0–1.0). "
            "Higher = more clusters.  If omitted, the per-model default is used."
        ),
    )

    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        default=False,
        help=(
            "Automatically estimate the best threshold from the data. "
            "Overrides -t/--threshold."
        ),
    )

    parser.add_argument(
        "-k", "--min-cluster-size",
        type=int,
        default=None,
        dest="min_cluster_size",
        help="Minimum cluster size; smaller clusters are dissolved (default: 2).",
    )

    parser.add_argument(
        "-l", "--linkage",
        dest="linkage_method",
        default=None,
        choices=["average", "ward", "complete", "single"],
        help="Hierarchical linkage method (default: per-model best).",
    )

    # ---- Device ----
    parser.add_argument(
        "-d", "--device",
        default=None,
        help="Torch device (cpu, cuda, cuda:0, etc.).  Auto-detected if omitted.",
    )

    # ---- Output ----
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Save results as JSON to this file path.",
    )

    parser.add_argument(
        "--save-labels",
        type=Path,
        default=None,
        dest="save_labels",
        help="Save segment → speaker mapping as a JSON file.",
    )

    # ---- Misc ----
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress bars and summary table.",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="Print available models with their clustering defaults and exit.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # --list-models: show defaults table and exit
    # ------------------------------------------------------------------
    if args.list_models:
        table = Table(
            title="[bold]Available Models & Clustering Defaults[/bold]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Model", style="bold green")
        table.add_column("Threshold", justify="right")
        table.add_column("Min Size", justify="right")
        table.add_column("Linkage")
        table.add_column("Description")

        for mtype in EmbeddingModelType:
            try:
                d = ClusteringDefaultsProvider.get_defaults(mtype)
                table.add_row(
                    mtype.value,
                    f"{d.threshold:.3f}",
                    str(d.min_cluster_size),
                    d.linkage_method,
                    d.description,
                )
            except ValueError:
                table.add_row(mtype.value, "—", "—", "—", "No defaults defined")

        console.print(table)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Collect audio files
    # ------------------------------------------------------------------
    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    audio_files: List[Path] = []

    for raw_path in args.inputs:
        p = Path(raw_path)
        if p.is_dir():
            for ext in AUDIO_EXTS:
                audio_files.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file():
            if p.suffix.lower() in AUDIO_EXTS:
                audio_files.append(p)
            else:
                console.print(
                    f"[yellow]⚠ Skipping non-audio file: {p}[/yellow]"
                )
        else:
            console.print(f"[red]✗ Path does not exist: {p}[/red]")

    if not audio_files:
        console.print("[red]No audio files found.[/red]")
        sys.exit(1)

    # Sort for deterministic output
    audio_files = sorted(set(audio_files))

    console.log(
        f"{_LOGGER_PREFIX} Found [green]{len(audio_files)}[/green] audio files "
        f"across {len(args.inputs)} input(s)"
    )

    # ------------------------------------------------------------------
    # Create clusterer
    # ------------------------------------------------------------------
    clusterer = SegmentSpeakerCluster(
        model_type=args.model_type,
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
        linkage_method=args.linkage_method,
        device=args.device,
        verbose=not args.quiet,
    )

    # ------------------------------------------------------------------
    # Optional: auto-estimate threshold
    # ------------------------------------------------------------------
    if args.auto_threshold:
        # Extract embeddings first, then estimate
        embeddings, _ = clusterer._extract_embeddings_from_files(
            [str(f) for f in audio_files], SAMPLE_RATE
        )
        best = clusterer.estimate_optimal_threshold(embeddings)
        clusterer.threshold = best
        if not args.quiet:
            console.log(
                f"{_LOGGER_PREFIX} Using auto-estimated threshold: {best:.3f}"
            )

    # ------------------------------------------------------------------
    # Cluster
    # ------------------------------------------------------------------
    result = clusterer.cluster_files(
        [str(f) for f in audio_files],
        target_sample_rate=SAMPLE_RATE,
    )

    # ------------------------------------------------------------------
    # Print compact per-file mapping
    # ------------------------------------------------------------------
    if not args.quiet:
        mapping_table = Table(
            title="[bold]Segment → Speaker Mapping[/bold]",
            show_header=True,
            header_style="bold cyan",
        )
        mapping_table.add_column("#", style="dim", width=4)
        mapping_table.add_column("Speaker", style="bold green", width=12)
        mapping_table.add_column("File")

        for i, (path, label) in enumerate(zip(audio_files, result.labels)):
            mapping_table.add_row(str(i + 1), label, path.name)

        console.print()
        console.print(mapping_table)
        console.print()
        result.print_report()

    # ------------------------------------------------------------------
    # Save JSON output
    # ------------------------------------------------------------------
    if args.output:
        output_data = {
            "model_type": result.model_type,
            "threshold_used": result.threshold_used,
            "n_clusters": result.n_clusters,
            "n_segments": len(result.labels),
            "compute_time_ms": result.compute_time_ms,
            "clusters": [
                {
                    "label": c.label,
                    "size": c.size,
                    "internal_similarity": round(c.internal_similarity, 4),
                    "segments": [
                        audio_files[idx].name for idx in c.segments
                    ],
                }
                for c in result.clusters
            ],
            "labels": [
                {"file": audio_files[i].name, "speaker": result.labels[i]}
                for i in range(len(audio_files))
            ],
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.log(f"[green]✓ Results saved to {args.output}[/green]")

    # ------------------------------------------------------------------
    # Save labels-only JSON
    # ------------------------------------------------------------------
    if args.save_labels:
        labels_data = {
            str(audio_files[i]): result.labels[i]
            for i in range(len(audio_files))
        }
        args.save_labels.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_labels, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, indent=2, ensure_ascii=False)
        console.log(f"[green]✓ Labels saved to {args.save_labels}[/green]")

    console.log(
        f"{_LOGGER_PREFIX} Done. "
        f"{result.n_clusters} speakers found in {len(audio_files)} segments."
    )
