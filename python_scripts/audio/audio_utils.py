from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import numpy.typing as npt
import torch


# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]


def get_audio_duration(
    audio: AudioInput,
    sample_rate: Optional[int] = None,
) -> float:
    """Get duration of audio from various input types."""
    # Case 1: File path
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            return float(librosa.get_duration(path=str(path)))
        except Exception as e:
            raise ValueError(f"Failed to read audio file {path}: {e}") from e

    # Case 2: Raw bytes
    if isinstance(audio, bytes):
        try:
            buffer = io.BytesIO(audio)
            y, sr = librosa.load(buffer, sr=None, mono=False)
            return float(librosa.get_duration(y=y, sr=sr))
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes: {e}") from e

    # Case 3: numpy array
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy array input")
        return float(librosa.get_duration(y=audio, sr=sample_rate))

    # Case 4: torch tensor
    if torch is not None and isinstance(audio, torch.Tensor):
        if sample_rate is None:
            raise ValueError("sample_rate is required for torch.Tensor input")
        y = audio.detach().cpu().numpy()
        return float(librosa.get_duration(y=y, sr=sample_rate))

    raise TypeError(f"Unsupported audio input type: {type(audio)}")
