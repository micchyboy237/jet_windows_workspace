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

def resample_audio(
    audio: npt.NDArray[np.float32],
    orig_sr: int,
    target_sr: int = 16000,
) -> npt.NDArray[np.float32]:
    """
    Resample audio array to the target sample rate using linear interpolation.

    This is a lightweight, dependency-free implementation suitable for real-time
    or batch processing where adding heavy dependencies (e.g., librosa, torchaudio)
    is undesirable.

    Args:
        audio: Input audio as float32 numpy array. Shape can be (samples,) or (channels, samples).
        orig_sr: Original sample rate of the input audio.
        target_sr: Desired sample rate (default: 16000 Hz, required by Whisper models).

    Returns:
        Resampled audio as float32 numpy array with the same number of channels.

    Raises:
        ValueError: If orig_sr or target_sr is <= 0, or if audio is empty.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers.")
    if audio.size == 0:
        raise ValueError("Input audio array is empty.")

    if orig_sr == target_sr:
        return audio.copy()

    # Compute the resampling ratio and new length
    ratio = target_sr / orig_sr
    old_length = audio.shape[-1]
    new_length = int(np.round(old_length * ratio))

    # Determine if mono or multi-channel
    if audio.ndim == 1:
        # Mono: (samples,)
        old_indices = np.linspace(0, old_length - 1, new_length)
        resampled = np.interp(old_indices, np.arange(old_length), audio)
    else:
        # Multi-channel: (channels, samples)
        resampled_channels = []
        for channel in audio:
            old_indices = np.linspace(0, old_length - 1, new_length)
            resampled_channels.append(np.interp(old_indices, np.arange(old_length), channel))
        resampled = np.stack(resampled_channels)

    return resampled.astype(np.float32)
