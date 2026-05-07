# compare_speakers.py

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from audio_utils import resolve_audio_paths
from pyannote.audio import Inference, Model
from rich.console import Console
from rich.table import Table
from scipy.spatial.distance import cdist

console = Console(record=True)

THRESHOLD_SAME: float = 0.3
THRESHOLD_POSSIBLE: float = 0.15
THRESHOLD_MATRIX_WARN: float = 0.5


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


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file into waveform tensor.
    Returns:
        waveform: Tensor shape (channels, time)
        sample_rate: int
    """
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate


def compute_embedding(
    inference: Inference,
    waveform: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """
    Compute embedding from preloaded waveform.
    Ensures output is always 2D: (1, D)
    """
    embedding = inference(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
    )
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    if embedding.ndim == 1:
        embedding = embedding[None, :]
    return embedding


def log_audio_info(name: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Log useful debug info for a loaded waveform.
    """
    channels, num_samples = waveform.shape
    duration_sec = num_samples / sample_rate
    console.print(f"\n[bold cyan]{name} info:[/]")
    console.print(f"  Path: [white]{name}[/white]")
    console.print(f"  Sample rate: [white]{sample_rate} Hz[/white]")
    console.print(f"  Channels: [white]{channels}[/white]")
    console.print(f"  Samples: [white]{num_samples}[/white]")
    console.print(f"  Duration: [white]{duration_sec:.2f} sec[/white]")
    console.print(f"  Dtype: [white]{waveform.dtype}[/white]")
    console.print(f"  Min: [white]{waveform.min().item():.4f}[/white]")
    console.print(f"  Max: [white]{waveform.max().item():.4f}[/white]")
    console.print(f"  Mean: [white]{waveform.mean().item():.4f}[/white]")
    console.print(f"  Std: [white]{waveform.std().item():.4f}[/white]")


def interpret_similarity(
    similarity: float,
    threshold_same: float = THRESHOLD_SAME,
    threshold_possible: float = THRESHOLD_POSSIBLE,
) -> str:
    """
    Provide human-readable interpretation of similarity score.
    """
    if similarity >= threshold_same:
        return "[green]SAME speaker (high confidence)[/]"
    elif similarity >= threshold_possible:
        return "[yellow]POSSIBLY same speaker[/]"
    else:
        return "[red]DIFFERENT speakers[/]"


def create_similarity_matrix(
    speaker_paths: list,
    embeddings: Dict[str, torch.Tensor],
    labels: Dict[str, str],
    threshold_same: float = THRESHOLD_SAME,
    threshold_matrix_warn: float = THRESHOLD_MATRIX_WARN,
) -> None:
    """
    Create and display a similarity matrix for multiple speakers.
    """
    n = len(speaker_paths)

    table = Table(title="Speaker Similarity Matrix")
    table.add_column("Speaker", style="cyan")

    for i in range(n):
        table.add_column(f"Spk {i + 1}", justify="right")

    for i in range(n):
        row = [labels[speaker_paths[i]]]
        for j in range(n):
            if i == j:
                row.append("[green]1.0000[/]")
            else:
                distance = float(
                    cdist(
                        embeddings[speaker_paths[i]],
                        embeddings[speaker_paths[j]],
                        metric="cosine",
                    )[0, 0]
                )
                similarity = 1.0 - distance

                if similarity >= threshold_same:
                    color = "green"
                elif similarity >= threshold_matrix_warn:
                    color = "yellow"
                else:
                    color = "red"

                row.append(f"[{color}]{similarity:.4f}[/]")
        table.add_row(*row)

    console.print("\n")
    console.print(table)


def display_pairwise_analysis(
    speaker_paths: list,
    embeddings: Dict[str, torch.Tensor],
    labels: Dict[str, str],
    threshold_same: float = THRESHOLD_SAME,
    threshold_possible: float = THRESHOLD_POSSIBLE,
) -> None:
    """
    Display detailed pairwise analysis for all speaker combinations.
    """
    console.print("\n[bold]Pairwise Analysis:[/]\n")

    for i in range(len(speaker_paths)):
        for j in range(i + 1, len(speaker_paths)):
            path1 = speaker_paths[i]
            path2 = speaker_paths[j]

            distance = float(
                cdist(embeddings[path1], embeddings[path2], metric="cosine")[0, 0]
            )
            similarity = 1.0 - distance
            interpretation = interpret_similarity(
                similarity=similarity,
                threshold_same=threshold_same,
                threshold_possible=threshold_possible,
            )

            console.print(f"[bold]{labels[path1]}[/] vs [bold]{labels[path2]}[/]:")
            console.print(
                f"  Cosine similarity: [white]{similarity:.4f}[/] "
                f"(distance: [white]{distance:.4f}[/])"
            )
            console.print(f"  Result: {interpretation}\n")


def derive_speaker_labels(speaker_paths: List[str]) -> Dict[str, str]:
    """
    Derive a display label for each speaker path.
    Falls back to 'Speaker N' if stems collide or are ambiguous.
    """
    labels = {}
    stems = [Path(p).stem for p in speaker_paths]

    if len(set(stems)) == len(stems):
        for path, stem in zip(speaker_paths, stems):
            labels[path] = stem
    else:
        for i, path in enumerate(speaker_paths, start=1):
            labels[path] = f"Speaker {i}"

    return labels


def build_speaker_groups(
    speaker_paths: List[str],
    labels: Dict[str, str],
) -> Dict[str, dict]:
    """
    Group file paths under an incremental speaker key, distinct from display labels.

    Returns:
        {
            "speaker_0": {
                "label": "alice",
                "files": ["path/to/alice.wav"]
            },
            "speaker_1": {
                "label": "bob",
                "files": ["path/to/bob_session1.wav", "path/to/bob_session2.wav"]
            },
        }
    """
    label_to_files: Dict[str, List[str]] = {}
    for path in speaker_paths:
        label = labels[path]
        label_to_files.setdefault(label, []).append(str(path))

    groups = {}
    for i, (label, files) in enumerate(label_to_files.items()):
        groups[f"speaker_{i}"] = {
            "label": label,
            "files": files,
        }

    return groups


def cluster_speakers(
    speaker_paths: List[str],
    embeddings: Dict[str, torch.Tensor],
    threshold_same: float = THRESHOLD_SAME,
) -> Dict[str, str]:
    """
    Cluster speaker paths by embedding similarity.
    Returns a labels dict: path -> "Speaker N" where N is the cluster index.
    Uses greedy single-linkage: a file joins an existing cluster if its
    similarity to ALL members exceeds threshold_same.
    """
    clusters: List[List[str]] = []  # list of groups of paths

    for path in speaker_paths:
        placed = False
        for cluster in clusters:
            # Check similarity against all members of this cluster
            if all(
                1.0 - float(cdist(embeddings[path], embeddings[member], metric="cosine")[0, 0])
                >= threshold_same
                for member in cluster
            ):
                cluster.append(path)
                placed = True
                break
        if not placed:
            clusters.append([path])

    labels: Dict[str, str] = {}
    for i, cluster in enumerate(clusters, start=1):
        for path in cluster:
            labels[path] = f"Speaker {i}"

    return labels


def save_speakers_json(
    output_dir: Path,
    labels: Dict[str, str],
    groups: Dict[str, dict],
    console: Console,
) -> None:
    """
    Save speakers.json with:
      - 'labels': path -> display label mapping
      - 'groups': speaker_N -> { label, files } mapping
    """
    speakers_path = output_dir / "speakers.json"
    payload = {
        "labels": {str(k): v for k, v in labels.items()},
        "groups": groups,
    }
    speakers_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[dim]Saved speakers  → {speakers_path}[/]")


def save_results(
    output_dir: Path,
    speaker_paths: List[str],
    embeddings: Dict[str, torch.Tensor],
    labels: Dict[str, str],
    console: Console,
    threshold_same: float = THRESHOLD_SAME,
    threshold_possible: float = THRESHOLD_POSSIBLE,
) -> None:
    """
    Save similarity matrix (CSV), pairwise analysis (JSON),
    embeddings (NPZ), speakers (JSON), and rich console report (HTML)
    to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. speakers.json
    groups = build_speaker_groups(speaker_paths=speaker_paths, labels=labels)
    save_speakers_json(
        output_dir=output_dir,
        labels=labels,
        groups=groups,
        console=console,
    )

    # 2. embeddings.npz — keyed by label
    npz_path = output_dir / "embeddings.npz"
    np.savez(npz_path, **{labels[p]: v for p, v in embeddings.items()})
    console.print(f"[dim]Saved embeddings → {npz_path}[/]")

    # 3. similarity_matrix.csv
    csv_path = output_dir / "similarity_matrix.csv"
    n = len(speaker_paths)
    with csv_path.open("w") as f:
        header = "speaker," + ",".join(labels[p] for p in speaker_paths)
        f.write(header + "\n")
        for p1 in speaker_paths:
            row = [labels[p1]]
            for p2 in speaker_paths:
                if p1 == p2:
                    row.append("1.0000")
                else:
                    dist = float(
                        cdist(embeddings[p1], embeddings[p2], metric="cosine")[0, 0]
                    )
                    row.append(f"{1.0 - dist:.4f}")
            f.write(",".join(row) + "\n")
    console.print(f"[dim]Saved matrix    → {csv_path}[/]")

    # 4. pairwise_analysis.json
    json_path = output_dir / "pairwise_analysis.json"
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = speaker_paths[i], speaker_paths[j]
            dist = float(cdist(embeddings[p1], embeddings[p2], metric="cosine")[0, 0])
            sim = 1.0 - dist
            pairs.append(
                {
                    "speaker_1": labels[p1],
                    "speaker_2": labels[p2],
                    "source_1": str(p1),
                    "source_2": str(p2),
                    "cosine_distance": round(dist, 6),
                    "cosine_similarity": round(sim, 6),
                    "interpretation": (
                        "same_speaker"
                        if sim >= threshold_same
                        else "possibly_same"
                        if sim >= threshold_possible
                        else "different_speakers"
                    ),
                }
            )
    json_path.write_text(json.dumps({"pairs": pairs}, indent=2))
    console.print(f"[dim]Saved analysis  → {json_path}[/]")

    # 5. report.html (rich console capture)
    html_path = output_dir / "report.html"
    console.save_html(str(html_path))
    console.print(f"[dim]Saved report    → {html_path}[/]")


def main() -> None:
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Compare speaker embeddings from WAV files using cosine similarity."
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
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-ts",
        "--threshold-same",
        default=THRESHOLD_SAME,
        type=float,
        metavar="FLOAT",
        help=f"cosine similarity threshold to classify as same speaker (default: {THRESHOLD_SAME})",
    )
    parser.add_argument(
        "-tp",
        "--threshold-possible",
        default=THRESHOLD_POSSIBLE,
        type=float,
        metavar="FLOAT",
        help=f"cosine similarity threshold for 'possibly same speaker' (default: {THRESHOLD_POSSIBLE})",
    )
    parser.add_argument(
        "-tw",
        "--threshold-matrix-warn",
        default=THRESHOLD_MATRIX_WARN,
        type=float,
        metavar="FLOAT",
        help=f"similarity threshold for yellow warning in matrix display (default: {THRESHOLD_MATRIX_WARN})",
    )
    args = parser.parse_args()

    speaker_paths = resolve_audio_paths(args.speakers, recursive=True)

    if len(speaker_paths) < 2:
        console.print(
            "[red]Error: At least 2 speaker files are required for comparison.[/]"
        )
        sys.exit(1)

    with console.status("[bold green]Loading embedding model..."):
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")

    waveforms = {}
    sample_rates = {}
    with console.status("[bold green]Loading audio files..."):
        for path in speaker_paths:
            waveform, sr = load_audio(path=path)
            waveforms[path] = waveform
            sample_rates[path] = sr
            log_audio_info(name=path, waveform=waveform, sample_rate=sr)

    embeddings = {}
    with console.status("[bold green]Computing speaker embeddings..."):
        for path, waveform in waveforms.items():
            embeddings[path] = compute_embedding(
                inference=inference,
                waveform=waveform,
                sample_rate=sample_rates[path],
            )

    # Cluster AFTER embeddings are computed
    labels = cluster_speakers(
        speaker_paths=speaker_paths,
        embeddings=embeddings,
        threshold_same=args.threshold_same,
    )

    num_clusters = len(set(labels.values()))
    console.print(
        f"\n[bold green]Detected {num_clusters} unique speaker(s) "
        f"across {len(speaker_paths)} segment(s).[/]\n"
    )

    if len(speaker_paths) == 2:
        distance = float(
            cdist(
                embeddings[speaker_paths[0]],
                embeddings[speaker_paths[1]],
                metric="cosine",
            )[0, 0]
        )
        similarity = 1.0 - distance
        interpretation = interpret_similarity(
            similarity=similarity,
            threshold_same=args.threshold_same,
            threshold_possible=args.threshold_possible,
        )

        console.print(f"\n[bold blue]Speaker 1:[/] {labels[speaker_paths[0]]}")
        console.print(f"[bold yellow]Speaker 2:[/] {labels[speaker_paths[1]]}")
        console.print(
            f"[bold magenta]Cosine distance:[/] [white]{distance:.4f}[/white]"
        )
        console.print(
            f"[bold green]Cosine similarity:[/] [white]{similarity:.4f}[/white]"
        )
        console.print(f"\n[bold]Analysis:[/] {interpretation}\n")
    else:
        create_similarity_matrix(
            speaker_paths=speaker_paths,
            embeddings=embeddings,
            labels=labels,
            threshold_same=args.threshold_same,
            threshold_matrix_warn=args.threshold_matrix_warn,
        )
        display_pairwise_analysis(
            speaker_paths=speaker_paths,
            embeddings=embeddings,
            labels=labels,
            threshold_same=args.threshold_same,
            threshold_possible=args.threshold_possible,
        )

    save_results(
        output_dir=args.output_dir,
        speaker_paths=speaker_paths,
        embeddings=embeddings,
        labels=labels,
        console=console,
        threshold_same=args.threshold_same,
        threshold_possible=args.threshold_possible,
    )


if __name__ == "__main__":
    main()
