import argparse
import json
import shutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from audio_utils import resolve_audio_paths
from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.clustering import (
    AgglomerativeClustering as PyannoteAHC,
)
from pyannote.core import SlidingWindow, SlidingWindowFeature
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity

console = Console(record=True)


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio file into waveform tensor, downmix to mono if needed."""
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate


def compute_embedding(
    inference: Inference,
    waveform: torch.Tensor,
    sample_rate: int,
) -> np.ndarray:
    """
    Compute speaker embedding from preloaded waveform.
    Returns always 2D array: (1, D).
    """
    embedding = inference({"waveform": waveform, "sample_rate": sample_rate})
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    if embedding.ndim == 1:
        embedding = embedding[None, :]
    return embedding


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClusteringResult:
    """Holds the output of cluster_speakers()."""

    labels: Dict[str, int] = field(default_factory=dict)
    clusters: Dict[int, List[str]] = field(default_factory=dict)
    n_clusters: int = 0
    algorithm: str = ""
    linkage: str = ""
    # Distance threshold in [0, 2] cosine-distance space (pyannote convention).
    # -1.0 when n_clusters was forced explicitly.
    distance_threshold: float = 0.0
    # Soft cluster scores: (N, K) — one row per file, one col per cluster.
    # Higher score → file more likely belongs to that cluster.
    soft_scores: np.ndarray | None = None
    # Centroid vectors: (K, D) — one centroid per cluster.
    centroids: np.ndarray | None = None


# ── Core clustering ────────────────────────────────────────────────────────────

def cluster_speakers(
    speaker_paths: List[str],
    embeddings: Dict[str, np.ndarray],
    *,
    distance_threshold: float = 0.7,
    method: str = "average",
    min_cluster_size: int = 1,
    n_clusters: int | None = None,
    min_clusters: int | None = None,
    max_clusters: int | None = None,
) -> ClusteringResult:
    """
    Cluster speaker embeddings using pyannote's own AgglomerativeClustering.

    Compared to a plain sklearn approach, this version adds:
      - NaN and low-activity embedding filtering   (filter_embeddings)
      - Small-cluster merging into nearest centroid (min_cluster_size)
      - Soft cluster confidence scores             (soft_clusters)
      - Per-cluster centroid vectors               (centroids)
      - min_clusters / max_clusters enforcement

    How the reshape works
    ---------------------
    Pyannote's clustering pipeline expects:
      embeddings    : (num_chunks, num_speakers, D)
      segmentations : (num_chunks, num_frames,   num_speakers)

    Since every file here is a standalone whole-file embedding (no chunked
    segmentation), we treat each file as one chunk with exactly one speaker
    slot and one always-active frame:
      embeddings    : (N, 1, D)
      segmentations : (N, 1, 1)  — all-ones (always active)

    After clustering, hard_clusters[:, 0] gives the integer label per file.

    Args:
        speaker_paths:      Ordered list of audio file paths (keys into
                            embeddings). Must contain at least 2 entries.
        embeddings:         Dict mapping path → (1, D) numpy array.
        distance_threshold: Cosine distance cut-off in [0, 2].
                            Lower → fewer merges → more clusters.
                            Typical useful range: 0.4 – 1.0.
                            Default 0.7 mirrors pyannote's tuned sweet-spot.
                            Ignored when n_clusters is provided.
        method:             Scipy linkage method: 'average', 'complete',
                            'single', 'ward', 'centroid', 'median'.
                            'average' (UPGMA) is pyannote's default and works
                            well for cosine-distance speaker embeddings.
        min_cluster_size:   Minimum number of files a cluster must contain.
                            Clusters smaller than this are dissolved and their
                            files reassigned to the nearest large-cluster
                            centroid. Default 1 (no merging).
        n_clusters:         Force an exact cluster count. Overrides
                            distance_threshold but still respects min/max.
        min_clusters:       Minimum cluster count the dendrogram cut must
                            produce. Useful when you know at least K speakers
                            are present.
        max_clusters:       Maximum cluster count.

    Returns:
        ClusteringResult dataclass with labels, clusters, soft_scores,
        centroids, and metadata.
    """
    if len(speaker_paths) < 2:
        raise ValueError("At least 2 speaker embeddings are required for clustering.")

    # ── 1. Stack embeddings into (N, 1, D) ────────────────────────────────────
    stacked = np.stack(
        [embeddings[p].squeeze(0) for p in speaker_paths], axis=0
    )  # (N, D)
    N, D = stacked.shape
    emb_3d = stacked[:, np.newaxis, :]  # (N, 1, D)

    # ── 2. Build synthetic all-ones segmentations (N, 1, 1) ───────────────────
    #    filter_embeddings checks active frames. One always-on frame per file
    #    means every embedding passes the activity filter (min_active_ratio).
    seg_data = np.ones((N, 1, 1), dtype=np.float32)
    window = SlidingWindow(start=0.0, duration=1.0, step=1.0)
    segmentations = SlidingWindowFeature(seg_data, window)

    # ── 3. Configure pyannote's AgglomerativeClustering ───────────────────────
    #    Hyper-parameters are normally set via pipeline.instantiate() from a
    #    YAML config, but we set them directly so no config file is needed.
    clusterer = PyannoteAHC(metric="cosine")
    clusterer.threshold = distance_threshold
    clusterer.method = method
    clusterer.min_cluster_size = min_cluster_size

    # ── 4. Run clustering ─────────────────────────────────────────────────────
    #    Returns:
    #      hard_clusters : (N, 1)    integer label per (chunk, speaker)
    #      soft_clusters : (N, 1, K) score per (chunk, speaker, cluster)
    #      centroids     : (K, D)
    hard_clusters, soft_clusters, centroids = clusterer(
        emb_3d,
        segmentations=segmentations,
        num_clusters=n_clusters,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    # ── 5. Flatten speaker dimension (always 0) and map back to paths ─────────
    raw_labels: np.ndarray = hard_clusters[:, 0]       # (N,)
    n_found = int(raw_labels.max()) + 1
    soft_scores: np.ndarray = soft_clusters[:, 0, :]   # (N, K)

    label_map: Dict[str, int] = {
        path: int(raw_labels[i]) for i, path in enumerate(speaker_paths)
    }
    cluster_map: Dict[int, List[str]] = {}
    for path, label in label_map.items():
        cluster_map.setdefault(label, []).append(path)

    return ClusteringResult(
        labels=label_map,
        clusters=cluster_map,
        n_clusters=n_found,
        algorithm="PyannoteAgglomerativeClustering",
        linkage=method,
        distance_threshold=distance_threshold if n_clusters is None else -1.0,
        soft_scores=soft_scores,
        centroids=centroids,
    )


# ── Display & output ───────────────────────────────────────────────────────────

def display_clustering(
    result: ClusteringResult,
    embeddings: Dict[str, np.ndarray],
) -> None:
    """
    Print a Rich table with per-file Duration, Size, Similarity to centroid,
    and (when available) the soft cluster confidence score.
    """
    has_soft = result.soft_scores is not None
    # Build a fast lookup: path → row index in soft_scores
    path_to_idx: Dict[str, int] = {
        path: i for i, path in enumerate(result.labels.keys())
    }

    table = Table(
        title=f"Speaker Clusters ({result.n_clusters} found)", show_lines=True
    )
    table.add_column("Cluster", style="yellow", justify="center")
    table.add_column("Files", style="white", no_wrap=True)
    table.add_column("Duration", style="cyan", justify="right")
    table.add_column("Size", style="magenta", justify="right")
    table.add_column("Centroid sim", style="green", justify="right")
    if has_soft:
        table.add_column("Confidence", style="blue", justify="right")

    for label, members in sorted(result.clusters.items()):
        cluster_size = len(members)
        cluster_embeddings = np.vstack([embeddings[p] for p in members])
        centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)

        for i, full_path in enumerate(members):
            p = Path(full_path)
            short_name = f"{p.parent.name}/{p.name}"
            file_link = f"file://{full_path}"
            linked_text = f"[link={file_link}]{short_name}[/link]"

            try:
                waveform, sr = torchaudio.load(full_path)
                duration_str = f"{waveform.shape[1] / sr:.2f}s"
            except Exception:
                duration_str = "—"

            try:
                size_str = f"{p.stat().st_size / (1024 * 1024):.1f} MB"
            except Exception:
                size_str = "—"

            try:
                sim = cosine_similarity(embeddings[full_path], centroid)[0][0]
                sim_str = f"{sim:.3f}"
            except Exception:
                sim_str = "—"

            cluster_cell = (
                f"Cluster {label}\n[dim]({cluster_size})[/dim]" if i == 0 else ""
            )
            row = [cluster_cell, linked_text, duration_str, size_str, sim_str]

            if has_soft:
                try:
                    idx = path_to_idx[full_path]
                    score = float(result.soft_scores[idx, label])
                    row.append(f"{score:.3f}")
                except Exception:
                    row.append("—")

            table.add_row(*row)

    console.print("\n")
    console.print(table)
    console.print(
        f"\n[dim]Algorithm: {result.algorithm} | "
        f"Linkage: {result.linkage} | "
        f"Distance threshold: {result.distance_threshold}[/]\n"
    )


def save_results(
    output_dir: Path,
    embeddings: Dict[str, np.ndarray],
    result: ClusteringResult,
) -> None:
    """
    Save clustering outputs to output_dir:
      - embeddings.npz
      - centroids.npz   (when available)
      - clustering.json
      - report.html
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "embeddings.npz"
    np.savez(npz_path, **{str(k): v for k, v in embeddings.items()})
    console.print(f"[dim]Saved embeddings  → {npz_path}[/]")

    if result.centroids is not None:
        centroids_path = output_dir / "centroids.npz"
        np.savez(centroids_path, centroids=result.centroids)
        console.print(f"[dim]Saved centroids   → {centroids_path}[/]")

    if result.soft_scores is not None:
        soft_path = output_dir / "soft_scores.npz"
        np.savez(soft_path, soft_scores=result.soft_scores)
        console.print(f"[dim]Saved soft scores → {soft_path}[/]")

    cluster_path = output_dir / "clustering.json"
    cluster_path.write_text(
        json.dumps(
            {
                "algorithm": result.algorithm,
                "linkage": result.linkage,
                "distance_threshold": result.distance_threshold,
                "n_clusters": result.n_clusters,
                "labels": {str(k): v for k, v in result.labels.items()},
                "clusters": {
                    str(label): paths for label, paths in result.clusters.items()
                },
            },
            indent=2,
        )
    )
    console.print(f"[dim]Saved clustering  → {cluster_path}[/]")

    html_path = output_dir / "report.html"
    console.save_html(str(html_path))
    console.print(f"[dim]Saved report      → {html_path}[/]")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    DEFAULT_AUDIO = str(
        Path(
            "~/Desktop/Jet_Files/Jet_Windows_Workspace/python_scripts/samples/audio"
            "/features/generated/speech_waves/waves/"
        )
        .expanduser()
        .resolve()
    )

    parser = argparse.ArgumentParser(
        description=(
            "Cluster speaker embeddings from WAV files using pyannote's "
            "AgglomerativeClustering pipeline."
        )
    )
    parser.add_argument(
        "speakers",
        nargs="*",
        default=[DEFAULT_AUDIO],
        help=(
            "Paths to speaker WAV files or directories (space-separated, at least 2 "
            "required). Directories are scanned recursively for audio files."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help=f"Output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-t",
        "--distance-threshold",
        type=float,
        default=0.7,
        help=(
            "Cosine distance threshold for merging clusters (0–2, default: 0.7). "
            "Lower → stricter merging → more clusters. "
            "Ignored if --n-clusters is set."
        ),
    )
    parser.add_argument(
        "-l",
        "--linkage",
        type=str,
        default="average",
        choices=["average", "complete", "single", "ward", "centroid", "median"],
        help="Linkage method for the dendrogram (default: average).",
    )
    parser.add_argument(
        "-n",
        "--n-clusters",
        type=int,
        default=None,
        help="Force exact number of clusters. Overrides --distance-threshold.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=None,
        help="Minimum number of clusters.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Maximum number of clusters.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help=(
            "Minimum files a cluster must contain before being dissolved and "
            "reassigned to the nearest large cluster (default: 1, no merging)."
        ),
    )
    args = parser.parse_args()

    speaker_paths = resolve_audio_paths(args.speakers, recursive=True, includes=["**/sound.wav"])
    if len(speaker_paths) < 2:
        console.print(
            "[red]Error: At least 2 speaker files are required for clustering.[/]"
        )
        sys.exit(1)

    with console.status("[bold green]Loading embedding model..."):
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")

    embeddings: Dict[str, np.ndarray] = {}
    with console.status("[bold green]Loading audio and computing embeddings..."):
        for path in speaker_paths:
            waveform, sr = load_audio(path)
            embeddings[path] = compute_embedding(inference, waveform, sr)

    with console.status("[bold green]Clustering speakers..."):
        result = cluster_speakers(
            speaker_paths,
            embeddings,
            distance_threshold=args.distance_threshold,
            method=args.linkage,
            min_cluster_size=args.min_cluster_size,
            n_clusters=args.n_clusters,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
        )

    display_clustering(result, embeddings)
    save_results(args.output_dir, embeddings, result)


if __name__ == "__main__":
    main()
