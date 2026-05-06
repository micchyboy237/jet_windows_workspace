# cluster_speakers.py

import argparse
import json
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
from rich.console import Console
from rich.table import Table
from sklearn.cluster import AgglomerativeClustering as SklearnAHC

console = Console(record=True)


def _patch_torchmetrics_compat() -> None:
    import torchmetrics.utilities.data as _tmd

    if not hasattr(_tmd, "get_num_classes"):

        def get_num_classes(pred, target=None, num_classes=None):
            """Stub replacing removed torchmetrics helper."""
            if num_classes is not None:
                return num_classes
            if target is not None:
                return int(target.max().item()) + 1
            return int(pred.max().item()) + 1

        _tmd.get_num_classes = get_num_classes
    if "pytorch_lightning.metrics" not in sys.modules:
        import pytorch_lightning as _pl

        if not hasattr(_pl, "metrics"):
            _metrics_mod = types.ModuleType("pytorch_lightning.metrics")
            sys.modules["pytorch_lightning.metrics"] = _metrics_mod
            _pl.metrics = _metrics_mod


_patch_torchmetrics_compat()


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


@dataclass
class ClusteringResult:
    """Holds the output of cluster_speakers()."""

    # Maps each input file path → cluster label (0-based int)
    labels: Dict[str, int] = field(default_factory=dict)

    # Maps cluster label → list of file paths in that cluster
    clusters: Dict[int, List[str]] = field(default_factory=dict)

    # Total number of distinct clusters found
    n_clusters: int = 0

    # Algorithm used and its key parameters
    algorithm: str = ""
    linkage: str = ""
    distance_threshold: float = 0.0


def cluster_speakers(
    speaker_paths: List[str],
    embeddings: Dict[str, np.ndarray],
    *,
    distance_threshold: float = 0.5,
    linkage: str = "average",
    n_clusters: int | None = None,
) -> ClusteringResult:
    """
    Cluster speaker embeddings using agglomerative hierarchical clustering.

    Uses sklearn's AgglomerativeClustering with cosine affinity, which mirrors
    the approach used inside pyannote's own diarization pipeline
    (pyannote.audio.pipelines.clustering.AgglomerativeClustering) but operates
    directly on the pre-computed embedding matrix so no segmentation pass is
    needed.

    Args:
        speaker_paths:      Ordered list of audio file paths (keys into embeddings).
        embeddings:         Dict mapping path → (1, D) numpy embedding array.
        distance_threshold: Cosine distance cut-off for merging clusters.
                            Ignored when n_clusters is set. Default 0.5
                            (≈ similarity ≥ 0.5, "possibly same speaker").
        linkage:            Linkage criterion – 'average', 'complete', or
                            'single'. 'average' (UPGMA) matches pyannote's
                            default centroid strategy most closely.
        n_clusters:         Force an exact cluster count. When set,
                            distance_threshold is ignored.

    Returns:
        ClusteringResult dataclass.
    """
    if len(speaker_paths) < 2:
        raise ValueError("At least 2 speaker embeddings are required for clustering.")

    # Stack into (N, D) matrix – each row is one speaker's embedding
    matrix = np.vstack([embeddings[p] for p in speaker_paths])  # (N, D)

    ahc_kwargs: dict = dict(metric="cosine", linkage=linkage)
    if n_clusters is not None:
        ahc_kwargs["n_clusters"] = n_clusters
    else:
        ahc_kwargs["n_clusters"] = None
        ahc_kwargs["distance_threshold"] = distance_threshold

    ahc = SklearnAHC(**ahc_kwargs)
    raw_labels: np.ndarray = ahc.fit_predict(matrix)  # shape (N,)

    label_map: Dict[str, int] = {
        path: int(raw_labels[i]) for i, path in enumerate(speaker_paths)
    }
    cluster_map: Dict[int, List[str]] = {}
    for path, label in label_map.items():
        cluster_map.setdefault(label, []).append(path)

    return ClusteringResult(
        labels=label_map,
        clusters=cluster_map,
        n_clusters=int(ahc.n_clusters_),
        algorithm="AgglomerativeClustering",
        linkage=linkage,
        distance_threshold=distance_threshold if n_clusters is None else -1.0,
    )


def display_clustering(result: ClusteringResult) -> None:
    """Print a Rich table summarising the clustering result."""
    table = Table(title=f"Speaker Clusters ({result.n_clusters} found)")
    table.add_column("Cluster", style="yellow", justify="center")
    table.add_column("Files", style="white")

    for label, members in sorted(result.clusters.items()):
        for i, m in enumerate(members):
            cluster_cell = f"Cluster {label}" if i == 0 else ""
            table.add_row(cluster_cell, str(m))

    console.print("\n")
    console.print(table)
    console.print(
        f"\n[dim]Algorithm: {result.algorithm} | "
        f"Linkage: {result.linkage} | "
        f"Threshold: {result.distance_threshold}[/]\n"
    )


def save_results(
    output_dir: Path,
    embeddings: Dict[str, np.ndarray],
    result: ClusteringResult,
) -> None:
    """
    Save clustering outputs to output_dir:
      - embeddings.npz
      - clustering.json
      - report.html
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. embeddings.npz
    npz_path = output_dir / "embeddings.npz"
    np.savez(npz_path, **{str(k): v for k, v in embeddings.items()})
    console.print(f"[dim]Saved embeddings  → {npz_path}[/]")

    # 2. clustering.json
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
                    str(label): paths
                    for label, paths in result.clusters.items()
                },
            },
            indent=2,
        )
    )
    console.print(f"[dim]Saved clustering  → {cluster_path}[/]")

    # 3. report.html
    html_path = output_dir / "report.html"
    console.save_html(str(html_path))
    console.print(f"[dim]Saved report      → {html_path}[/]")


def main() -> None:
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Cluster speaker embeddings from WAV files using agglomerative hierarchical clustering."
    )
    parser.add_argument(
        "speakers",
        nargs="+",
        type=str,
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
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine distance threshold for merging clusters (default: 0.5). Ignored if --n-clusters is set.",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="average",
        choices=["average", "complete", "single"],
        help="Linkage criterion for AHC (default: average).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Force exact number of clusters. Overrides --threshold.",
    )
    args = parser.parse_args()

    # Resolve input paths (recursively scan directories)
    speaker_paths = resolve_audio_paths(args.speakers, recursive=True)

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
            distance_threshold=args.threshold,
            linkage=args.linkage,
            n_clusters=args.n_clusters,
        )

    display_clustering(result)
    save_results(args.output_dir, embeddings, result)


if __name__ == "__main__":
    main()
