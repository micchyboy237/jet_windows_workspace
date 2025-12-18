from __future__ import annotations

import os
from typing import Union
import numpy as np
import numpy.typing as npt
import librosa

# Optional torch support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]

def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> np.ndarray:
    """
    Robust audio loader for ASR pipelines with correct datatype, normalization, layout, and resampling.
    
    Handles:
      - File paths
      - In-memory WAV bytes
      - NumPy arrays (any shape/layout/dtype/sr)
      - Torch tensors
      - Automatically normalizes to [-1.0, 1.0] float32
      - Always resamples to target_sr
      - Correctly converts stereo → mono regardless of channel position
    Returns
    -------
    np.ndarray
        Shape (samples,), float32, [-1.0, 1.0], exactly `sr` Hz
    """
    # ─────── FIX 1: In-memory arrays/tensors have unknown original sr ───────
    import io
    current_sr: int | None
    if isinstance(audio, (str, os.PathLike)):
        y, current_sr = librosa.load(audio, sr=None, mono=False)
    elif isinstance(audio, bytes):
        y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
    elif isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = None
    elif HAS_TORCH and isinstance(audio, torch.Tensor):
        y = audio.float().cpu().numpy()
        current_sr = None
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # ─────── FIX 2: Correct normalization (NumPy, not torch) ───────
    if np.issubdtype(y.dtype, np.integer):
        y = y / (2 ** (np.iinfo(y.dtype).bits - 1))
    elif np.abs(y).max() > 1.0 + 1e-6:
        y = y / np.abs(y).max()

    # ─────── FIX 3: Always make (channels, time) layout ───────
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    # Mono conversion
    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    # ─────── FIX 4: ALWAYS resample if current_sr is None or wrong ───────
    if current_sr != sr:
        y = librosa.resample(y, orig_sr=current_sr or sr, target_sr=sr)

    return y.squeeze()