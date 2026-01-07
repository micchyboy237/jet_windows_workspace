"""
Standalone loudness normalization utility.

Provides a single, reusable function to normalize audio to a target LUFS value
using pyloudnorm (ITU-R BS.1770-4 compliant). Handles caching of meters for
performance, clipping prevention, and graceful fallback for silent/failed cases.
"""

from __future__ import annotations

import logging
from typing import Union, Optional
import os

import numpy as np
import numpy.typing as npt
import torch
import soundfile as sf
from io import BytesIO

import pyloudnorm as pyln

logger = logging.getLogger(__name__)

# Cache meters per sample rate to avoid expensive recreation
_METER_CACHE: dict[int, pyln.Meter] = {}

# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]


def normalize_loudness(
    audio: AudioInput,
    sample_rate: int,
    target_lufs: float = -14.0,
    min_lufs_threshold: float = -70.0,
    headroom_factor: float = 1.05,
) -> np.ndarray:
    """
    Normalize audio to a target integrated loudness (LUFS).

    Args:
        audio: Input audio – can be:
               - File path (str or os.PathLike)
               - Raw audio bytes
               - NumPy array (floating or integer dtype)
               - torch.Tensor
               When path/bytes provided, sample_rate is inferred from file.
        sample_rate: Sample rate of the audio in Hz.
        target_lufs: Desired integrated loudness in LUFS (default -14.0).
        min_lufs_threshold: If measured loudness is below this, skip normalization
                            to avoid amplifying pure noise/silence.
        headroom_factor: Multiplier applied after normalization to prevent clipping
                         (default 1.05 → ~0.87 peak).

    Returns:
        Normalized audio array with the same shape as input.

    Raises:
        ImportError: If pyloudnorm is not installed.
        ValueError: If input parameters are invalid.
        RuntimeError: If audio loading fails (for path/bytes inputs).
    """
    # Load and unify input to np.float32 ndarray
    if isinstance(audio, (str, os.PathLike)):
        # File path input
        audio_path = str(audio)
        loaded_audio, loaded_sr = sf.read(audio_path, always_2d=False)
        sample_rate = loaded_sr
    elif isinstance(audio, bytes):
        # Raw bytes input
        with BytesIO(audio) as buf:
            loaded_audio, loaded_sr = sf.read(buf, always_2d=False)
        sample_rate = loaded_sr
    elif isinstance(audio, torch.Tensor):
        loaded_audio = audio.cpu().numpy()
    elif isinstance(audio, np.ndarray):
        loaded_audio = audio
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # Convert to float32
    if not isinstance(loaded_audio, np.ndarray) or loaded_audio.dtype != np.float32:
        loaded_audio = np.asarray(loaded_audio, dtype=np.float32)

    audio = loaded_audio  # Now unified

    if audio.ndim == 1:
        pass  # mono – already good
    elif audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            logger.warning("Audio appears to have channels as first dimension – transposing")
            audio = audio.T
    else:
        raise ValueError("Audio must be 1D (mono) or 2D (samples, channels)")

    # Cache meter per sample rate
    meter = _METER_CACHE.get(sample_rate)
    if meter is None:
        meter = pyln.Meter(sample_rate)
        _METER_CACHE[sample_rate] = meter

    try:
        measured_lufs = meter.integrated_loudness(audio)
        logger.debug(f"Measured integrated loudness: {measured_lufs:.2f} LUFS")
    except Exception as exc:
        logger.warning(f"LUFS measurement failed ({exc}), returning original audio")
        return audio.copy()  # Return copy to avoid modifying input arrays/tensors

    if measured_lufs <= min_lufs_threshold:
        logger.debug("Audio too quiet – skipping loudness normalization")
        return audio.copy()

    try:
        normalized = pyln.normalize.loudness(audio, measured_lufs, target_lufs)
    except Exception as exc:
        logger.warning(f"LUFS normalization failed ({exc}), returning original audio")
        return audio.copy()

    # Prevent clipping with headroom
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized /= peak * headroom_factor
        logger.debug(f"Applied headroom – post-norm peak reduced to {np.max(np.abs(normalized)):.3f}")

    return normalized.copy()  # Safe return (original input unchanged)


if __name__ == "__main__":
    import argparse
    import sys
    import soundfile as sf
    import shutil
    from pathlib import Path
    from rich import print as rprint

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_5mins.wav"
    OUTPUT_AUDIO = OUTPUT_DIR / (Path(INPUT_AUDIO).stem + "_norm" + Path(INPUT_AUDIO).suffix)

    parser = argparse.ArgumentParser(
        description="Normalize loudness of a WAV file to target LUFS (ITU-R BS.1770-4)."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",  # Make positional input optional
        default=Path(INPUT_AUDIO),
        help="Input WAV file path"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",  # Make it optional
        default=OUTPUT_AUDIO,
        help="Output WAV file path (default: <input>_norm.wav)"
    )
    parser.add_argument(
        "-t",
        "--target",
        type=float,
        default=-14.0,
        help="Target integrated loudness in LUFS (default: -14.0)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.is_file():
        rprint(f"[red]Error: Input file not found: {input_path}[/red]")
        sys.exit(1)

    output_path = args.output or input_path.with_name(f"{input_path.stem}_norm{input_path.suffix}")

    rprint(f"[bold]Loading audio:[/bold] {input_path}")
    audio, sr = sf.read(input_path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    # Measure original loudness
    meter = pyln.Meter(sr)
    try:
        original_lufs = meter.integrated_loudness(audio)
    except Exception:
        original_lufs = float("-inf")

    rprint(f"[bold]Normalizing[/bold] to {args.target} LUFS...")
    normalized_audio = normalize_loudness(audio, sr, target_lufs=args.target)

    # Measure final loudness
    try:
        final_lufs = meter.integrated_loudness(normalized_audio)
    except Exception:
        final_lufs = float("-inf")

    rprint(f"[green]Original:[/green] {original_lufs:.2f} LUFS → [green]Normalized:[/green] {final_lufs:.2f} LUFS")
    rprint(f"[bold]Writing output:[/bold] {output_path}")
    sf.write(output_path, normalized_audio, sr)
    rprint("[bold green]Done![/bold green]")
