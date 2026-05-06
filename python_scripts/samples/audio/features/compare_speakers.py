import argparse
from typing import Tuple

import torch
import torchaudio
from pyannote.audio import Inference, Model
from rich.console import Console
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Compatibility shim (unchanged)
# ---------------------------------------------------------------------------
import types
import sys


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
# ---------------------------------------------------------------------------


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file into waveform tensor.

    Returns:
        waveform: Tensor shape (channels, time)
        sample_rate: int
    """
    waveform, sample_rate = torchaudio.load(path)

    # Ensure mono (pyannote works best with mono)
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
    embedding = inference({
        "waveform": waveform,
        "sample_rate": sample_rate,
    })

    # Ensure numpy array
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()

    # Ensure 2D shape (1, D)
    if embedding.ndim == 1:
        embedding = embedding[None, :]

    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two speaker embeddings (wav files) using cosine similarity."
    )
    parser.add_argument("speaker1", type=str, help="Path to first speaker WAV file")
    parser.add_argument("speaker2", type=str, help="Path to second speaker WAV file")
    args = parser.parse_args()

    console = Console()

    model = Model.from_pretrained("pyannote/embedding")
    inference = Inference(model, window="whole")

    with console.status("[bold green]Loading audio..."):
        waveform1, sr1 = load_audio(args.speaker1)
        waveform2, sr2 = load_audio(args.speaker2)

    with console.status("[bold green]Computing embeddings..."):
        embedding1 = compute_embedding(inference, waveform1, sr1)
        embedding2 = compute_embedding(inference, waveform2, sr2)

    # Compute cosine distance & similarity
    distance = float(cdist(embedding1, embedding2, metric="cosine")[0, 0])
    similarity = 1.0 - distance

    console.print(f"\n[bold blue]Speaker 1:[/] {args.speaker1}")
    console.print(f"[bold yellow]Speaker 2:[/] {args.speaker2}")
    console.print(f"[bold magenta]Cosine distance:[/] [white]{distance:.4f}[/white]")
    console.print(f"[bold green]Cosine similarity:[/] [white]{similarity:.4f}[/white]\n")


if __name__ == "__main__":
    main()
