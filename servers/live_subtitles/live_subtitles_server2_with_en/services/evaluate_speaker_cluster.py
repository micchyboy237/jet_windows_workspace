"""
evaluate_speaker_cluster.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluate speaker clustering models on the same multi-speaker dataset
used by evaluate_speaker_embeddings.py.

Dataset layout expected (identical to embedding eval):
    dataset_root/
        speaker_A/
            utt_001.wav
            utt_002.wav
        speaker_B/
            ...

Workflow
--------
1. Scan dataset → ground-truth labels (directory name = speaker id)
2. Run SegmentSpeakerCluster on all audio files
3. Compare predicted clusters vs ground truth
4. Report: ARI, NMI, Purity, Homogeneity, Completeness, V-measure
5. Rank models by ARI (descending)

Usage:
    python -m services.evaluate_speaker_cluster \\
        /path/to/speakers \\
        -m pyannote speechbrain_ecapa modelscope_eres2netv2 \\
        -o results/
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from segment_speaker_cluster import (
    ClusteringDefaultsProvider,
    ClusterResult,
    SegmentSpeakerCluster,
)
from embedding_model_factory import (
    EmbeddingModelType,
    create_embedding_model,
    list_available_models,
)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
log = logging.getLogger("evaluate_cluster")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ClusterEvalMetrics:
    """Evaluation metrics for one clustering run on one model."""
    model_type: str
    """Which embedding backend was used."""
    threshold_used: float
    """Cosine-similarity threshold applied."""
    min_cluster_size: int
    """Minimum cluster size setting."""
    linkage_method: str
    """Linkage method used."""

    # Dataset stats
    n_files: int
    n_true_speakers: int
    n_predicted_clusters: int

    # Core metrics (range 0–1, higher = better)
    ari: float
    """Adjusted Rand Index — chance-corrected cluster similarity."""
    nmi: float
    """Normalized Mutual Information."""
    purity: float
    """Average purity of clusters."""
    homogeneity: float
    """Are all members of a cluster from the same class?"""
    completeness: float
    """Are all members of a class assigned to the same cluster?"""
    v_measure: float
    """Harmonic mean of homogeneity and completeness."""

    # Timing
    embedding_time_ms: float
    """Time spent extracting embeddings."""
    clustering_time_ms: float
    """Time spent clustering (excluding embedding extraction)."""
    total_time_ms: float
    """End-to-end wall-clock time."""

    # Per-speaker breakdown
    per_speaker_recall: Dict[str, float] = field(default_factory=dict)
    per_speaker_precision: Dict[str, float] = field(default_factory=dict)

    # Threshold sweep (optional)
    threshold_sweep: Optional[List[Dict]] = None
    """If --sweep was used, stores ARI-vs-threshold data points."""

    def summary(self) -> str:
        return (
            f"[{self.model_type}] "
            f"ARI={self.ari:.4f} | "
            f"NMI={self.nmi:.4f} | "
            f"Purity={self.purity:.4f} | "
            f"V={self.v_measure:.4f} | "
            f"Clusters={self.n_predicted_clusters}/{self.n_true_speakers} | "
            f"Thresh={self.threshold_used:.3f} | "
            f"Time={self.total_time_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Dataset scanning (same as embedding eval)
# ---------------------------------------------------------------------------

def scan_dataset(
    dataset_root: Path,
    min_utterances: int = 1,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Scan dataset, returning flat file list + ground-truth labels.

    Parameters
    ----------
    dataset_root : Path
        Root directory where each subfolder is one speaker.
    min_utterances : int
        Minimum files required per speaker.

    Returns
    -------
    (files, y_true, speaker_names)
        files : list of Path, all audio files found
        y_true : np.ndarray of int labels (0-based speaker indices)
        speaker_names : list of str, speaker directory names
    """
    log.info(f"Scanning dataset at: [bold]{dataset_root}[/bold]")

    speaker_dirs: List[Tuple[str, List[Path]]] = []

    for spk_dir in sorted(dataset_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        files = sorted(
            f for f in spk_dir.rglob("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if len(files) >= min_utterances:
            speaker_dirs.append((spk_dir.name, files))
        else:
            log.warning(
                f"Skipping speaker '{spk_dir.name}': "
                f"only {len(files)} file(s), need >= {min_utterances}"
            )

    if len(speaker_dirs) < 2:
        raise ValueError(
            f"Need at least 2 speakers with >= {min_utterances} files. "
            f"Found: {len(speaker_dirs)}"
        )

    all_files: List[Path] = []
    y_true_list: List[int] = []
    speaker_names: List[str] = []

    for spk_idx, (spk_name, files) in enumerate(speaker_dirs):
        speaker_names.append(spk_name)
        all_files.extend(files)
        y_true_list.extend([spk_idx] * len(files))

    log.info(
        f"Found [green]{len(speaker_dirs)}[/green] speakers, "
        f"[green]{len(all_files)}[/green] total utterances"
    )

    return all_files, np.array(y_true_list, dtype=int), speaker_names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand Index — chance-corrected cluster similarity.

    ARI ∈ [-1, 1].  1 = perfect agreement, 0 = chance-level.
    """
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(y_true, y_pred))


def compute_nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Mutual Information (arithmetic mean normalisation).

    NMI ∈ [0, 1].  1 = perfect agreement.
    """
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(
        y_true, y_pred, average_method="arithmetic"
    ))


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average cluster purity.

    For each predicted cluster, find the most common true label.
    Purity = sum(majority counts) / total samples.

    Returns
    -------
    float in [0, 1]
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    total_correct = 0
    unique_pred = np.unique(y_pred)
    for cluster_id in unique_pred:
        mask = y_pred == cluster_id
        true_in_cluster = y_true[mask]
        if len(true_in_cluster) == 0:
            continue
        # Most common true label in this cluster
        counts = np.bincount(true_in_cluster)
        total_correct += counts.max()

    return float(total_correct / n)


def compute_homogeneity_completeness_v(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """Homogeneity, Completeness, and V-measure.

    Returns
    -------
    (homogeneity, completeness, v_measure)
    """
    from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
    h = float(homogeneity_score(y_true, y_pred))
    c = float(completeness_score(y_true, y_pred))
    v = float(v_measure_score(y_true, y_pred))
    return h, c, v


def compute_per_speaker_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speaker_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-speaker recall and precision.

    For each true speaker:
        recall = proportion of speaker's files in the majority cluster
    For each predicted cluster:
        precision = proportion of the cluster that is the majority speaker

    Returns
    -------
    (per_speaker_recall, per_speaker_precision)
    """
    unique_true = np.unique(y_true)

    # Per-speaker recall
    per_speaker_recall: Dict[str, float] = {}
    for spk_idx in unique_true:
        spk_name = speaker_names[spk_idx]
        spk_mask = y_true == spk_idx
        spk_preds = y_pred[spk_mask]
        if len(spk_preds) == 0:
            per_speaker_recall[spk_name] = 0.0
            continue
        # Majority cluster for this speaker
        cluster_counts = np.bincount(spk_preds)
        majority_count = cluster_counts.max()
        per_speaker_recall[spk_name] = round(
            float(majority_count / len(spk_preds)), 4
        )

    # Per-cluster precision → mapped back to the dominant speaker
    per_speaker_precision: Dict[str, float] = {}
    cluster_to_speaker: Dict[int, int] = {}
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        true_in_cluster = y_true[mask]
        counts = np.bincount(true_in_cluster)
        dominant_spk = int(counts.argmax())
        cluster_to_speaker[cluster_id] = dominant_spk
        precision = float(counts.max() / len(true_in_cluster))
        spk_name = speaker_names[dominant_spk]
        # If a speaker is dominant in multiple clusters, take the best
        if spk_name not in per_speaker_precision or precision > per_speaker_precision[spk_name]:
            per_speaker_precision[spk_name] = round(precision, 4)

    # Fill in 0.0 for speakers with no dominant cluster
    for spk_name in speaker_names:
        if spk_name not in per_speaker_precision:
            per_speaker_precision[spk_name] = 0.0

    return per_speaker_recall, per_speaker_precision


def labels_to_int_array(labels: List[str], unknown_label: str = "UNKNOWN") -> np.ndarray:
    """Convert string labels (SPK_01, SPK_02, ...) to 0-based int array.

    Parameters
    ----------
    labels : list of str
        Predicted labels from ClusterResult.
    unknown_label : str
        Label assigned to files that couldn't be clustered.

    Returns
    -------
    np.ndarray of int
    """
    unique_labels = [lbl for lbl in sorted(set(labels)) if lbl != unknown_label]
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    # Reserve the next integer for UNKNOWN
    unknown_int = len(unique_labels)
    return np.array(
        [label_to_int.get(lbl, unknown_int) for lbl in labels],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_model(
    model_type: str,
    all_files: List[Path],
    y_true: np.ndarray,
    speaker_names: List[str],
    threshold: Optional[float] = None,
    min_cluster_size: Optional[int] = None,
    linkage_method: Optional[str] = None,
    device: Optional[str] = None,
    auto_threshold: bool = False,
    verbose: bool = True,
) -> ClusterEvalMetrics:
    """Evaluate one clustering model on the full dataset.

    Parameters
    ----------
    model_type : str
        One of EmbeddingModelType values.
    all_files : list of Path
        All audio files in the dataset.
    y_true : np.ndarray
        Ground-truth speaker labels (0-based ints).
    speaker_names : list of str
        Human-readable speaker names.
    threshold : float, optional
        Override the per-model default threshold.
    min_cluster_size : int, optional
        Override the per-model default min_cluster_size.
    linkage_method : str, optional
        Override the per-model default linkage_method.
    device : str, optional
        Torch device.
    auto_threshold : bool
        If True, estimate the best threshold from the data.
    verbose : bool
        Print progress.

    Returns
    -------
    ClusterEvalMetrics
    """
    log.info(f"\n{'─'*60}")
    log.info(f"Evaluating clustering: [bold yellow]{model_type}[/bold yellow]")

    # Show defaults being used
    defaults = ClusteringDefaultsProvider.get_defaults(model_type)
    resolved_threshold = threshold if threshold is not None else defaults.threshold
    resolved_min_size = min_cluster_size if min_cluster_size is not None else defaults.min_cluster_size
    resolved_linkage = linkage_method if linkage_method is not None else defaults.linkage_method

    log.info(
        f"[{model_type}] Settings — threshold={resolved_threshold}, "
        f"min_cluster_size={resolved_min_size}, linkage={resolved_linkage}"
    )

    # Build clusterer
    clusterer = SegmentSpeakerCluster(
        model_type=model_type,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        linkage_method=linkage_method,
        device=device,
        verbose=verbose,
    )

    t_total_start = time.perf_counter()

    # Auto-threshold estimation (if requested)
    if auto_threshold:
        log.info(f"[{model_type}] Estimating optimal threshold…")
        # Extract embeddings first
        embeddings, _, emb_time = clusterer._extract_embeddings_from_files_with_timing(
            [str(f) for f in all_files]
        )
        best = clusterer.estimate_optimal_threshold(embeddings)
        clusterer.threshold = best
        log.info(f"[{model_type}] Auto-threshold = {best:.3f}")

    # Cluster
    result = clusterer.cluster_files(
        [str(f) for f in all_files],
    )

    t_total = (time.perf_counter() - t_total_start) * 1000

    # Convert predictions to int array
    y_pred = labels_to_int_array(result.labels)

    # Compute metrics
    ari = compute_ari(y_true, y_pred)
    nmi = compute_nmi(y_true, y_pred)
    purity = compute_purity(y_true, y_pred)
    hom, comp, vm = compute_homogeneity_completeness_v(y_true, y_pred)
    per_recall, per_prec = compute_per_speaker_metrics(
        y_true, y_pred, speaker_names
    )

    metrics = ClusterEvalMetrics(
        model_type=model_type,
        threshold_used=result.threshold_used or clusterer.threshold,
        min_cluster_size=clusterer.min_cluster_size,
        linkage_method=clusterer.linkage_method,
        n_files=len(all_files),
        n_true_speakers=len(speaker_names),
        n_predicted_clusters=result.n_clusters,
        ari=round(ari, 4),
        nmi=round(nmi, 4),
        purity=round(purity, 4),
        homogeneity=round(hom, 4),
        completeness=round(comp, 4),
        v_measure=round(vm, 4),
        embedding_time_ms=0.0,  # embedded in total for cluster_files
        clustering_time_ms=0.0,
        total_time_ms=round(t_total, 1),
        per_speaker_recall=per_recall,
        per_speaker_precision=per_prec,
    )

    log.info(metrics.summary())

    return metrics


def evaluate_model_with_sweep(
    model_type: str,
    all_files: List[Path],
    y_true: np.ndarray,
    speaker_names: List[str],
    thresholds: Optional[List[float]] = None,
    min_cluster_size: Optional[int] = None,
    linkage_method: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> ClusterEvalMetrics:
    """Evaluate model across a sweep of thresholds, returning the best + sweep data.

    Parameters
    ----------
    thresholds : list of float, optional
        Thresholds to sweep. Defaults to np.linspace(0.30, 0.90, 13).

    Returns
    -------
    ClusterEvalMetrics with threshold_sweep populated.
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.90, 13).tolist()

    log.info(f"\n{'─'*60}")
    log.info(
        f"Threshold sweep: [bold yellow]{model_type}[/bold yellow] "
        f"({len(thresholds)} values from {thresholds[0]:.2f} to {thresholds[-1]:.2f})"
    )

    # Extract embeddings ONCE
    clusterer = SegmentSpeakerCluster(
        model_type=model_type,
        device=device,
        verbose=False,  # suppress per-run output
    )
    log.info(f"[{model_type}] Extracting embeddings (once)…")
    embeddings, _, _ = clusterer._extract_embeddings_from_files_with_timing(
        [str(f) for f in all_files]
    )

    sweep_results: List[Dict] = []
    best_ari = -1.0
    best_threshold = thresholds[0]
    best_y_pred: Optional[np.ndarray] = None
    best_cluster_count = 0

    for thresh in thresholds:
        y_pred_int = clusterer._agglomerative_cluster(embeddings, thresh)
        y_pred_int = clusterer._dissolve_small_clusters(embeddings, y_pred_int)

        # Convert to string labels for consistency
        unique_pred = np.unique(y_pred_int)
        label_map = {uid: f"SPK_{i+1:02d}" for i, uid in enumerate(unique_pred)}
        _ = [label_map[l] for l in y_pred_int]  # unused, just for parity

        ari = compute_ari(y_true, y_pred_int)
        nmi = compute_nmi(y_true, y_pred_int)
        purity = compute_purity(y_true, y_pred_int)
        n_clusters = len(unique_pred)

        sweep_results.append({
            "threshold": round(thresh, 3),
            "ari": round(ari, 4),
            "nmi": round(nmi, 4),
            "purity": round(purity, 4),
            "n_clusters": n_clusters,
        })

        if ari > best_ari:
            best_ari = ari
            best_threshold = thresh
            best_y_pred = y_pred_int
            best_cluster_count = n_clusters

    # Compute full metrics at the best threshold
    hom, comp, vm = compute_homogeneity_completeness_v(y_true, best_y_pred)
    per_recall, per_prec = compute_per_speaker_metrics(
        y_true, best_y_pred, speaker_names
    )

    defaults = ClusteringDefaultsProvider.get_defaults(model_type)
    metrics = ClusterEvalMetrics(
        model_type=model_type,
        threshold_used=round(best_threshold, 3),
        min_cluster_size=min_cluster_size if min_cluster_size is not None else defaults.min_cluster_size,
        linkage_method=linkage_method if linkage_method is not None else defaults.linkage_method,
        n_files=len(all_files),
        n_true_speakers=len(speaker_names),
        n_predicted_clusters=best_cluster_count,
        ari=round(best_ari, 4),
        nmi=round(compute_nmi(y_true, best_y_pred), 4),
        purity=round(compute_purity(y_true, best_y_pred), 4),
        homogeneity=round(hom, 4),
        completeness=round(comp, 4),
        v_measure=round(vm, 4),
        embedding_time_ms=0.0,
        clustering_time_ms=0.0,
        total_time_ms=0.0,
        per_speaker_recall=per_recall,
        per_speaker_precision=per_prec,
        threshold_sweep=sweep_results,
    )

    log.info(metrics.summary())
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compare_models(results: List[ClusterEvalMetrics]) -> None:
    """Print a ranked comparison table."""
    ranked = sorted(results, key=lambda m: m.ari, reverse=True)

    table = Table(
        title="[bold]Speaker Clustering Model Comparison[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", min_width=22)
    table.add_column("ARI ↑", justify="right")
    table.add_column("NMI ↑", justify="right")
    table.add_column("Purity ↑", justify="right")
    table.add_column("V ↑", justify="right")
    table.add_column("H ↑", justify="right")
    table.add_column("C ↑", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Thresh", justify="right")
    table.add_column("ms ↓", justify="right")

    for rank, m in enumerate(ranked, start=1):
        style = "bold green" if rank == 1 else ""
        cluster_str = f"{m.n_predicted_clusters}/{m.n_true_speakers}"
        table.add_row(
            str(rank),
            m.model_type,
            f"{m.ari:.4f}",
            f"{m.nmi:.4f}",
            f"{m.purity:.4f}",
            f"{m.v_measure:.4f}",
            f"{m.homogeneity:.4f}",
            f"{m.completeness:.4f}",
            cluster_str,
            f"{m.threshold_used:.3f}",
            f"{m.total_time_ms:.0f}",
            style=style,
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]↑ = higher is better | ↓ = lower is better | "
        "Clusters = predicted / true | "
        "H=Homogeneity C=Completeness V=V-measure[/dim]\n"
    )


def print_per_speaker_table(
    results: List[ClusterEvalMetrics],
    speaker_names: List[str],
) -> None:
    """Print per-speaker recall table for the best model."""
    if not results:
        return

    best = max(results, key=lambda m: m.ari)

    table = Table(
        title=f"[bold]Per-Speaker Recall — {best.model_type} (ARI={best.ari:.4f})[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Speaker", style="bold green")
    table.add_column("Recall ↑", justify="right")
    table.add_column("Precision ↑", justify="right")

    for spk in speaker_names:
        recall = best.per_speaker_recall.get(spk, 0.0)
        precision = best.per_speaker_precision.get(spk, 0.0)
        r_style = "red" if recall < 0.5 else ("yellow" if recall < 0.8 else "green")
        p_style = "red" if precision < 0.5 else ("yellow" if precision < 0.8 else "green")
        table.add_row(
            spk,
            f"[{r_style}]{recall:.4f}[/{r_style}]",
            f"[{p_style}]{precision:.4f}[/{p_style}]",
        )

    console.print()
    console.print(table)


def save_results(
    results: List[ClusterEvalMetrics],
    output_dir: Path,
    speaker_names: List[str],
) -> None:
    """Save all results to JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "cluster_results.json"
    data = []
    for m in results:
        d = asdict(m)
        # Convert threshold_sweep to serializable format
        if d.get("threshold_sweep"):
            d["threshold_sweep"] = [
                {k: v for k, v in point.items()}
                for point in d["threshold_sweep"]
            ]
        data.append(d)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Results saved to [green]{json_path}[/green]")

    # Markdown
    md_path = output_dir / "cluster_summary.md"
    ranked = sorted(results, key=lambda m: m.ari, reverse=True)
    lines = [
        "# Speaker Clustering Evaluation\n",
        f"**Dataset:** {ranked[0].n_files} files, {ranked[0].n_true_speakers} speakers\n",
        "| Rank | Model | ARI ↑ | NMI ↑ | Purity ↑ | V ↑ | H ↑ | C ↑ | Clusters | Thresh | ms ↓ |",
        "|------|-------|-------|-------|----------|-----|-----|-----|----------|--------|------|",
    ]

    for rank, m in enumerate(ranked, 1):
        cluster_str = f"{m.n_predicted_clusters}/{m.n_true_speakers}"
        lines.append(
            f"| {rank} | {m.model_type} "
            f"| {m.ari:.4f} | {m.nmi:.4f} | {m.purity:.4f} "
            f"| {m.v_measure:.4f} | {m.homogeneity:.4f} | {m.completeness:.4f} "
            f"| {cluster_str} | {m.threshold_used:.3f} | {m.total_time_ms:.0f} |"
        )

    lines += [
        "",
        "> ↑ = higher is better | ↓ = lower is better",
        "> H = Homogeneity | C = Completeness | V = V-measure",
        "> Clusters = predicted / true",
        "",
        "## Per-Speaker Recall (Best Model)\n",
    ]

    if ranked:
        best = ranked[0]
        lines.append("| Speaker | Recall | Precision |")
        lines.append("|---------|--------|-----------|")
        for spk in speaker_names:
            recall = best.per_speaker_recall.get(spk, 0.0)
            precision = best.per_speaker_precision.get(spk, 0.0)
            lines.append(f"| {spk} | {recall:.4f} | {precision:.4f} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Summary saved to [green]{md_path}[/green]")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_root: Path,
    model_types: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
    min_cluster_size: Optional[int] = None,
    linkage_method: Optional[str] = None,
    auto_threshold: bool = False,
    threshold_sweep: bool = False,
    sweep_values: Optional[List[float]] = None,
    min_utterances: int = 1,
    verbose: bool = True,
) -> List[ClusterEvalMetrics]:
    """Run the full clustering evaluation pipeline.

    Parameters
    ----------
    dataset_root : Path
        Root directory with speaker subfolders.
    model_types : list of str, optional
        Models to evaluate. Defaults to all registered.
    output_dir : Path, optional
        Directory to save results.
    device : str, optional
        Torch device (auto-detected if None).
    threshold : float, optional
        Override clustering threshold for all models.
    min_cluster_size : int, optional
        Override min_cluster_size for all models.
    linkage_method : str, optional
        Override linkage method for all models.
    auto_threshold : bool
        Estimate best threshold per model from data.
    threshold_sweep : bool
        Run a full threshold sweep per model and pick best.
    sweep_values : list of float, optional
        Custom threshold values for sweep.
    min_utterances : int
        Minimum files per speaker.
    verbose : bool
        Print progress tables.

    Returns
    -------
    list of ClusterEvalMetrics
    """
    log.info("[bold green]Speaker Clustering Evaluation[/bold green]")
    log.info(f"Dataset: {dataset_root}")

    if model_types is None:
        model_types = [e.value for e in EmbeddingModelType]
    log.info(f"Models to evaluate: {model_types}")

    available = list_available_models()
    for m in model_types:
        if m not in available:
            raise ValueError(
                f"Unknown model '{m}'. Available: {list(available.keys())}"
            )

    # Scan dataset
    all_files, y_true, speaker_names = scan_dataset(
        dataset_root, min_utterances=min_utterances
    )

    results: List[ClusterEvalMetrics] = []

    for model_type in model_types:
        try:
            if threshold_sweep:
                metrics = evaluate_model_with_sweep(
                    model_type=model_type,
                    all_files=all_files,
                    y_true=y_true,
                    speaker_names=speaker_names,
                    thresholds=sweep_values,
                    min_cluster_size=min_cluster_size,
                    linkage_method=linkage_method,
                    device=device,
                    verbose=verbose,
                )
            else:
                metrics = evaluate_model(
                    model_type=model_type,
                    all_files=all_files,
                    y_true=y_true,
                    speaker_names=speaker_names,
                    threshold=threshold,
                    min_cluster_size=min_cluster_size,
                    linkage_method=linkage_method,
                    device=device,
                    auto_threshold=auto_threshold,
                    verbose=verbose,
                )
            results.append(metrics)
        except Exception as exc:
            log.error(f"[red]Failed to evaluate '{model_type}': {exc}[/red]")
            import traceback
            traceback.print_exc()

    if results:
        compare_models(results)
        print_per_speaker_table(results, speaker_names)

    if output_dir and results:
        save_results(results, output_dir, speaker_names)

    return results


if __name__ == "__main__":
    from main._main_evaluate_speaker_cluster import main

    main()
