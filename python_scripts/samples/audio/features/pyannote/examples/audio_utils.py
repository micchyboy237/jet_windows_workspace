# servers/audio_server/python_scripts/server/utils/audio_utils.py
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO, Union, Sequence

import numpy as np
import numpy.typing as npt
import soundfile as sf  # Robust in-memory audio handling (supports WAV, MP3, etc.)
import torch
import wave

from logging import getLogger
from rich.logging import RichHandler
import logging

# Replace your existing logger setup with this configuration
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO in production if needed
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

logger = getLogger("audio_utils")  # Give it a meaningful name
logger.propagate = False  # Prevent duplicate logs if other handlers exist


AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

SAMPLE_RATE = 16000
DTYPE = 'int16'
CHANNELS = 2

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
    audio_source: AudioInput,
    target_sr: int = 16000,
) -> npt.NDArray[np.float32]:
    """
    Load audio from various input types supported by faster-whisper.

    Supported inputs:
        - str or os.PathLike: Path to audio file
        - bytes or bytearray: Raw audio data in memory (e.g., WAV/MP3 bytes)
        - np.ndarray: Pre-loaded waveform (float or integer types)
        - torch.Tensor: PyTorch tensor waveform

    Args:
        audio_source: The source of the audio data.
        target_sr: Desired sampling rate (default 16000 Hz – Whisper standard).

    Returns:
        Mono float32 numpy array normalized to [-1.0, 1.0] at target_sr.
    """
    # Case 1: Raw bytes in memory
    if isinstance(audio_source, (bytes, bytearray)):
        with sf.SoundFile(io.BytesIO(audio_source)) as f:
            source_sr = f.samplerate
            audio = f.read(dtype="float32")
            if f.channels > 1:
                audio = np.mean(audio, axis=1)  # Convert to mono

    # Case 2: File path
    elif isinstance(audio_source, (str, os.PathLike, Path)):
        import librosa

        audio, source_sr = librosa.load(audio_source, sr=None, mono=True)

    # Case 3: Pre-loaded numpy array
    elif isinstance(audio_source, np.ndarray):
        audio = audio_source
        source_sr = target_sr  # Assume already correct if no sr info available

    # Case 4: PyTorch tensor
    elif isinstance(audio_source, torch.Tensor):
        audio = audio_source.cpu().numpy()
        source_sr = target_sr

    else:
        raise ValueError(f"Unsupported audio_source type: {type(audio_source).__name__}")

    # Resample if necessary
    if source_sr != target_sr:
        audio = resample_audio(audio, orig_sr=source_sr, target_sr=target_sr)

    # Ensure mono (in case earlier steps missed it)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32)


def resample_audio(
    audio: npt.NDArray[np.float32],
    orig_sr: int,
    target_sr: int = 16000,
) -> npt.NDArray[np.float32]:
    """
    Resample an audio signal to the target sampling rate using high-quality librosa resampling.

    Args:
        audio: Input waveform (mono, float32).
        orig_sr: Original sampling rate.
        target_sr: Desired sampling rate (default 16000 Hz).

    Returns:
        Resampled waveform as float32 numpy array.
    """
    if orig_sr == target_sr:
        return audio

    import librosa

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.astype(np.float32)

def get_input_channels() -> int:
    device_info = sf.query_devices(sf.default.device[0], 'input')
    channels = device_info['max_input_channels']
    logger.debug(f"Detected {channels} input channels")
    return channels

def get_wav_bytes(audio_data: Union[np.ndarray, bytes]) -> bytes:
    """
    Generate WAV file bytes in memory without saving to disk.

    Accepts either np.ndarray or raw PCM bytes.
    """
    data_bytes = _validate_audio_data(audio_data)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data_bytes)

    buffer.seek(0)
    wav_bytes = buffer.read()
    logger.info(f"Generated {len(wav_bytes)} bytes of in-memory WAV audio")
    return wav_bytes

def get_wav_fileobj(audio_data: Union[np.ndarray, bytes]) -> BinaryIO:
    """
    Generate a file-like object containing WAV data in memory.

    Accepts either np.ndarray or raw PCM bytes.
    """
    data_bytes = _validate_audio_data(audio_data)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data_bytes)

    buffer.seek(0)
    logger.info("Generated in-memory WAV file-like object")
    return buffer

def _validate_audio_data(audio_data: Union[np.ndarray, bytes]) -> bytes:
    sample_width = np.dtype(DTYPE).itemsize
    frame_size = CHANNELS * sample_width

    if isinstance(audio_data, np.ndarray):
        if not np.issubdtype(audio_data.dtype, np.integer):
            raise ValueError(f"Array must have integer dtype, got {audio_data.dtype}")
        if audio_data.size == 0:
            raise ValueError("Empty audio array")
        # Reshape to 2D if mono vector
        if audio_data.ndim == 1:
            if CHANNELS != 1:
                raise ValueError(f"Mono vector supplied but CHANNELS={CHANNELS}")
            audio_data = audio_data.reshape(-1, 1)
        elif audio_data.ndim != 2 or audio_data.shape[1] != CHANNELS:
            raise ValueError(f"Array must have shape (frames, {CHANNELS}) or (frames,) for mono")
        return audio_data.tobytes()

    # Raw bytes case
    if not isinstance(audio_data, (bytes, bytearray)):
        raise TypeError("audio_data must be np.ndarray or bytes/bytearray")
    if len(audio_data) == 0:
        raise ValueError("Empty audio bytes")
    if len(audio_data) % frame_size != 0:
        raise ValueError(
            f"Bytes length {len(audio_data)} not divisible by frame size {frame_size} "
            f"(channels={CHANNELS}, sample_width={sample_width})"
        )
    return bytes(audio_data)
