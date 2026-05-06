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


def interpret_similarity(similarity: float) -> str:
    """
    Provide human-readable interpretation of similarity score.
    """
    if similarity >= 0.7:
        return "[green]SAME speaker (high confidence)[/]"
    elif similarity >= 0.3:
        return "[yellow]POSSIBLY same speaker[/]"
    else:
        return "[red]DIFFERENT speakers[/]"


def create_similarity_matrix(
    speaker_paths: list, embeddings: Dict[str, torch.Tensor]
) -> None:
    """
    Create and display a similarity matrix for multiple speakers.
    """
    n = len(speaker_paths)

    # Create results table
    table = Table(title="Speaker Similarity Matrix")
    table.add_column("Speaker", style="cyan")

    # Add columns for each speaker
    for i in range(n):
        table.add_column(f"Spk {i + 1}", justify="right")

    # Compute and display all pairwise similarities
    for i in range(n):
        row = [f"Speaker {i + 1}"]
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

                # Color code based on similarity
                if similarity >= 0.7:
                    color = "green"
                elif similarity >= 0.5:
                    color = "yellow"
                else:
                    color = "red"

                row.append(f"[{color}]{similarity:.4f}[/]")
        table.add_row(*row)

    console.print("\n")
    console.print(table)


def display_pairwise_analysis(
    speaker_paths: list, embeddings: Dict[str, torch.Tensor]
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
            interpretation = interpret_similarity(similarity)

            console.print(f"[bold]Speaker {i + 1}[/] vs [bold]Speaker {j + 1}[/]:")
            console.print(
                f"  Cosine similarity: [white]{similarity:.4f}[/] "
                f"(distance: [white]{distance:.4f}[/])"
            )
            console.print(f"  Result: {interpretation}\n")


def save_results(
    output_dir: Path,
    speaker_paths: List[str],
    embeddings: Dict[str, torch.Tensor],
    console: Console,
) -> None:
    """
    Save similarity matrix (CSV), pairwise analysis (JSON),
    embeddings (NPZ), and rich console report (HTML) to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. embeddings.npz
    npz_path = output_dir / "embeddings.npz"
    np.savez(npz_path, **{str(k): v for k, v in embeddings.items()})
    console.print(f"[dim]Saved embeddings → {npz_path}[/]")

    # 2. similarity_matrix.csv
    csv_path = output_dir / "similarity_matrix.csv"
    n = len(speaker_paths)
    with csv_path.open("w") as f:
        header = "speaker," + ",".join(f"spk_{i + 1}" for i in range(n))
        f.write(header + "\n")
        for i, p1 in enumerate(speaker_paths):
            row = [f"spk_{i + 1}"]
            for j, p2 in enumerate(speaker_paths):
                if i == j:
                    row.append("1.0000")
                else:
                    dist = float(
                        cdist(embeddings[p1], embeddings[p2], metric="cosine")[0, 0]
                    )
                    row.append(f"{1.0 - dist:.4f}")
            f.write(",".join(row) + "\n")
    console.print(f"[dim]Saved matrix    → {csv_path}[/]")

    # 3. pairwise_analysis.json
    json_path = output_dir / "pairwise_analysis.json"
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = speaker_paths[i], speaker_paths[j]
            dist = float(cdist(embeddings[p1], embeddings[p2], metric="cosine")[0, 0])
            sim = 1.0 - dist
            pairs.append(
                {
                    "speaker_1": str(p1),
                    "speaker_2": str(p2),
                    "cosine_distance": round(dist, 6),
                    "cosine_similarity": round(sim, 6),
                    "interpretation": (
                        "same_speaker"
                        if sim >= 0.7
                        else "possibly_same"
                        if sim >= 0.5
                        else "different_speakers"
                    ),
                }
            )
    json_path.write_text(json.dumps({"pairs": pairs}, indent=2))
    console.print(f"[dim]Saved analysis  → {json_path}[/]")

    # 4. report.html  (rich console capture)
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
    args = parser.parse_args()

    # Resolve input paths (recursively scan directories)
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
            waveform, sr = load_audio(path)
            waveforms[path] = waveform
            sample_rates[path] = sr
            log_audio_info(path, waveform, sr)

    embeddings = {}
    with console.status("[bold green]Computing speaker embeddings..."):
        for path, waveform in waveforms.items():
            embeddings[path] = compute_embedding(
                inference, waveform, sample_rates[path]
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
        interpretation = interpret_similarity(similarity)

        console.print(f"\n[bold blue]Speaker 1:[/] {speaker_paths[0]}")
        console.print(f"[bold yellow]Speaker 2:[/] {speaker_paths[1]}")
        console.print(
            f"[bold magenta]Cosine distance:[/] [white]{distance:.4f}[/white]"
        )
        console.print(
            f"[bold green]Cosine similarity:[/] [white]{similarity:.4f}[/white]"
        )
        console.print(f"\n[bold]Analysis:[/] {interpretation}\n")
    else:
        create_similarity_matrix(speaker_paths, embeddings)
        display_pairwise_analysis(speaker_paths, embeddings)

    save_results(args.output_dir, speaker_paths, embeddings, console)


if __name__ == "__main__":
    main()
