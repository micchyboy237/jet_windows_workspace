from __future__ import annotations

from typing import Union, Sequence, Generator, Tuple
from pathlib import Path

import os
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

# Supported audio extensions
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma",
    ".webm", ".mp4", ".mkv", ".avi"
}

AudioPathsInput = Union[str, Path, Sequence[Union[str, Path]]]

def resolve_audio_paths(audio_inputs: AudioPathsInput, recursive: bool = False) -> list[str]:
    """
    Resolve single file, list, or directory into a sorted list of absolute audio file paths as strings.
    """
    inputs = [audio_inputs] if isinstance(audio_inputs, (str, Path)) else audio_inputs
    resolved_paths: list[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for p in path.glob(pattern):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                    resolved_paths.append(p.resolve())
        elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            resolved_paths.append(path.resolve())
        elif path.exists():
            print(f"Skipping non-audio file: {path}")
        else:
            print(f"Path not found: {path}")

    if not resolved_paths:
        raise ValueError("No valid audio files found from provided inputs.")

    # Return sorted list of absolute path strings
    return sorted(str(p) for p in resolved_paths)

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


def split_audio(
    audio: np.ndarray,
    sr: int = 16000,
    chunk_duration_s: float = 15.0,
    overlap_s: float = 3.0,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Splits audio into overlapping chunks.

    Defaults are optimized for Whisper / faster-whisper:
    - sr=16000 (Whisper native sample rate)
    - chunk_duration_s=15.0 (balanced latency vs context)
    - overlap_s=3.0 (prevents word truncation at boundaries)

    Yields:
        (audio_chunk, chunk_start_time_seconds)
    """
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if overlap_s < 0:
        raise ValueError("overlap_s must be >= 0")
    if overlap_s >= chunk_duration_s:
        raise ValueError("overlap_s must be < chunk_duration_s")

    chunk_size = int(chunk_duration_s * sr)
    overlap_size = int(overlap_s * sr)
    step_size = chunk_size - overlap_size

    total_samples = len(audio)

    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = audio[start:end]

        chunk_start_time = start / sr
        yield chunk, chunk_start_time

        if end == total_samples:
            break

        start += step_size


def load_audio_bytes(
    audio_bytes: bytes,
    expected_sample_rate: int,
    channels: int = 1,
    dtype: npt.DTypeLike = np.float32,  # ← modern way, accepts dtype or type
) -> tuple[np.ndarray, int]:
    """
    Load raw PCM bytes from live capture / microphone stream
    
    Args:
        audio_bytes: Raw PCM bytes
        expected_sample_rate: Sample rate of the audio (16000, 44100, etc.)
        channels: Number of channels in the buffer (usually 1 for mono)
        dtype: Data type of samples (np.float32, np.int16, etc.)
    """
    # Get the actual item size (bytes per sample)
    itemsize = np.dtype(dtype).itemsize   # ← this is the key fix
    
    byte_count = len(audio_bytes)
    sample_count = byte_count // (channels * itemsize)
    
    if byte_count % (channels * itemsize) != 0:
        raise ValueError(
            f"Audio bytes length {byte_count} is not divisible by "
            f"(channels={channels} × itemsize={itemsize}) → incomplete frame?"
        )

    array = np.frombuffer(audio_bytes, dtype=dtype)
    
    # Reshape if multi-channel, then downmix to mono
    if channels > 1:
        array = array.reshape(-1, channels).mean(axis=1).astype(np.float32)
    
    return array, expected_sample_rate


def convert_audio_to_tensor(
    audio_data: np.ndarray | list[np.ndarray], sr: int = 16000
) -> torch.Tensor:
    """
    Convert numpy audio array or list of chunks to torch tensor suitable for Silero VAD.
    - Ensures mono
    - Converts to float32 in range [-1.0, 1.0]
    - Requires 16kHz input!
    """
    # Accept either a single np.ndarray or a list of chunks
    if isinstance(audio_data, list):
        audio = np.concatenate(audio_data, axis=0)
    else:
        audio = np.asarray(audio_data)

    # Normalize integer PCM to float32 in [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    # If already float, ensure [-1, 1]
    elif np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
    else:
        raise ValueError("Unsupported audio dtype")

    tensor = torch.from_numpy(audio)

    # Convert to mono if multi-channel (average channels)
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=1)

    # Sanity checks
    assert tensor.abs().max() <= 1.0 + 1e-5, "Audio not normalized!"
    assert sr == 16000, "Wrong sample rate for Silero VAD: must be 16000 Hz"

    return tensor  # shape: (N_samples,), float32, [-1, 1], 16kHz
