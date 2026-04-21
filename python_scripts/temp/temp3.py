#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torchaudio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import (
    PretrainedSpeakerEmbedding,
)

# -----------------------------
# Types
# -----------------------------
BackendType = Literal[1, 2, 3, 4]

# -----------------------------
# Constants
# -----------------------------
BACKEND_MAP: dict[BackendType, str] = {
    1: "speechbrain/spkrec-ecapa-voxceleb",
    2: "pyannote/embedding",
    3: "nvidia/speakerverification_en_titanet_large",
    4: "hbredin/wespeaker-voxceleb-resnet34-LM",
}

DEFAULT_SAMPLE_RATE = 16000

console = Console()


# -----------------------------
# Utils
# -----------------------------
def load_audio(
    path: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> torch.Tensor:
    """Load audio file and return (1, 1, samples) tensor."""
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # (1, 1, samples)

    return waveform


def compute_embedding(
    model,
    waveform: torch.Tensor,
) -> np.ndarray:
    """Compute embedding."""
    return model(waveform)


def compute_distance(
    emb1: np.ndarray,
    emb2: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Compute distance between embeddings."""
    return float(cdist(emb1, emb2, metric=metric)[0, 0])


def interpret_result(distance: float, threshold: float) -> str:
    """Interpret similarity result."""
    return "SAME SPEAKER" if distance < threshold else "DIFFERENT SPEAKER"


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speaker verification CLI using multiple backends."
    )

    # Positional
    parser.add_argument("audio1", type=Path, help="Path to first audio file")
    parser.add_argument("audio2", type=Path, help="Path to second audio file")

    # Backend
    parser.add_argument(
        "-b",
        "--backend",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Backend: 1=SpeechBrain (default), 2=pyannote, 3=NeMo, 4=WeSpeaker",
    )

    # Device
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)",
    )

    # Threshold
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for cosine distance",
    )

    # Token
    parser.add_argument(
        "-k",
        "--token",
        type=str,
        default=None,
        help="HuggingFace token",
    )

    # Cache dir
    parser.add_argument(
        "-c",
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory",
    )

    return parser.parse_args()


# -----------------------------
# Main Logic
# -----------------------------
def main() -> None:
    args = parse_args()

    backend: BackendType = args.backend
    model_name = BACKEND_MAP[backend]

    device = torch.device(args.device)

    console.print(
        Panel.fit(
            f"[bold cyan]Speaker Verification[/bold cyan]\n"
            f"Backend: [yellow]{backend}[/yellow] → {model_name}\n"
            f"Device: [green]{device}[/green]"
        )
    )

    # Validate files
    for path in [args.audio1, args.audio2]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    # Load model
    console.print("[bold]Loading model...[/bold]")
    model = PretrainedSpeakerEmbedding(
        model_name,
        device=device,
        token=args.token,
        cache_dir=args.cache_dir,
    )

    # Process audio with progress
    waveforms: list[torch.Tensor] = []

    for path in tqdm([args.audio1, args.audio2], desc="Loading audio"):
        waveform = load_audio(path)
        waveforms.append(waveform)

    # Compute embeddings
    embeddings: list[np.ndarray] = []

    for waveform in tqdm(waveforms, desc="Computing embeddings"):
        emb = compute_embedding(model, waveform)
        embeddings.append(emb)

    emb1, emb2 = embeddings

    # Compute distance
    distance = compute_distance(emb1, emb2)
    decision = interpret_result(distance, args.threshold)

    # Display results
    table = Table(title="Results")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Distance (cosine)", f"{distance:.6f}")
    table.add_row("Threshold", f"{args.threshold:.3f}")
    table.add_row("Decision", decision)

    console.print(table)

    # Extra debug info
    console.print(
        Panel.fit(
            f"[bold]Embedding Details[/bold]\n"
            f"Shape: {emb1.shape}\n"
            f"Backend: {model_name}"
        )
    )


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    main()