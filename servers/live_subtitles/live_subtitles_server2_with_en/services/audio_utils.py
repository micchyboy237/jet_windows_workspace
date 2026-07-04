from __future__ import annotations

from typing import Union, Sequence, Generator, Tuple, Optional
from pathlib import Path

import io
import os
import numpy as np
import numpy.typing as npt
import librosa
import torch

try:
    from services.audio_config import SAMPLE_RATE
except ImportError:
    from audio_config import SAMPLE_RATE

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


def _collect_audio_files(
    root_path: Path, 
    recursive: bool = False,
    includes: Optional[list[str]] = None
) -> set[Path]:
    """
    Collect audio files from a directory using optional include patterns.
    
    Args:
        root_path: Root directory to search
        recursive: Whether to search recursively
        includes: Optional glob patterns to filter files
    
    Returns:
        Set of Path objects for matching audio files
    """
    if not includes:
        # No includes specified, use default glob behavior
        pattern = "**/*" if recursive else "*"
        return {p for p in root_path.glob(pattern) if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS}
    
    # Use include patterns - each pattern is treated as a relative glob from root_path
    matched_files = set()
    
    for include_pattern in includes:
        # Use rglob for recursive patterns, glob for non-recursive
        # rglob automatically handles **/ patterns correctly
        if recursive or '**' in include_pattern:
            for p in root_path.rglob(include_pattern):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                    matched_files.add(p)
        else:
            for p in root_path.glob(include_pattern):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                    matched_files.add(p)
    
    return matched_files


def resolve_audio_paths(
    audio_inputs: AudioPathsInput,
    recursive: bool = False,
    includes: Optional[list[str]] = None
) -> list[str]:
    """
    Resolve single file, list, or directory into a sorted list of absolute audio file paths as strings.
    
    When a direct file path is provided (not a directory), it will be accepted if it's a valid
    audio file, regardless of include patterns. Include patterns are primarily meant for
    directory scanning.
    
    Args:
        audio_inputs: Single file, list of files, or directory path
        recursive: Whether to recursively search directories (default: False)
        includes: Optional list of glob patterns to filter files (e.g., ['**/sound.wav', '*.mp3'])
                  Patterns are relative to each input directory.
                  Note: Direct file paths bypass include pattern filtering.
    Returns:
        Sorted list of absolute path strings to valid audio files
    Examples:
        # Get all WAV files recursively
        resolve_audio_paths("audio_dir/", recursive=True, includes=["**/*.wav"])
        # Get specific files from any subdirectory
        resolve_audio_paths("audio_dir/", includes=["**/sound.wav", "**/music.mp3"])
        # Get MP3 files only from top level
        resolve_audio_paths("audio_dir/", includes=["*.mp3"])
        # Direct file paths always work, regardless of include patterns
        resolve_audio_paths("specific_file.wav", includes=["**/sound.wav"])
    """
    inputs = [audio_inputs] if isinstance(audio_inputs, (str, Path)) else audio_inputs
    resolved_paths: list[Path] = []
    
    for item in inputs:
        path = Path(item)
        
        if path.is_dir():
            # For directories, apply include patterns
            matched_files = _collect_audio_files(path, recursive=recursive, includes=includes)
            resolved_paths.extend(p.resolve() for p in matched_files)
            if not matched_files:
                print(f"Warning: No matching audio files found in directory: {path}")
                if includes:
                    print(f"  Include patterns used: {includes}")
                    
        elif path.is_file():
            # For direct file paths, check if it's a valid audio file
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                # Direct file paths bypass include patterns - they were explicitly specified
                resolved_paths.append(path.resolve())
                if includes:
                    print(f"Note: Direct file path '{path.name}' accepted (bypasses include patterns)")
            else:
                print(f"Skipping non-audio file: {path}")
                
        elif path.exists():
            print(f"Skipping non-file, non-directory path: {path}")
        else:
            print(f"Path not found: {path}")
    
    if not resolved_paths:
        raise ValueError(
            "No valid audio files found from provided inputs.\n"
            f"  Inputs: {audio_inputs}\n"
            f"  Includes filter: {includes if includes else 'None'}\n"
            f"  Recursive: {recursive}"
        )
    
    # Return sorted list of absolute path strings
    return sorted(str(p) for p in resolved_paths)


def resolve_audio_paths_as_np_list(
    audio_inputs: AudioPathsInput,
    sr: int = SAMPLE_RATE,
    mono: bool = True,
    recursive: bool = False,
    includes: Optional[list[str]] = None,
) -> list[np.ndarray]:
    """
    Resolve audio files from paths and load them as numpy arrays.
    
    Args:
        audio_inputs: Single file, list of files, or directory path
        sr: Target sample rate for loaded audio (default: 16000 Hz)
        mono: Whether to convert to mono (default: True)
        recursive: Whether to recursively search directories (default: False)
        includes: Optional list of glob patterns to filter files (e.g., ['**/sound.wav', '*.mp3'])
    
    Returns:
        List of numpy arrays containing audio data for each file
        
    Raises:
        ValueError: If no valid audio files are found
        RuntimeError: If any audio file fails to load
    """
    # Get all audio file paths using resolve_audio_paths
    audio_paths = resolve_audio_paths(audio_inputs, recursive=recursive, includes=includes)
    
    # Load each audio file into a numpy array
    audio_data_list = []
    failed_files = []
    
    for audio_path in audio_paths:
        try:
            audio_array, actual_sr = load_audio(audio_path, sr=sr, mono=mono)
            
            # Resample if the loaded sample rate doesn't match target
            if actual_sr != sr:
                audio_array = resample_audio(audio_array, actual_sr, target_sr=sr)
            
            audio_data_list.append(audio_array)
            
        except Exception as e:
            failed_files.append((audio_path, str(e)))
    
    # Report any failures
    if failed_files:
        error_msg = f"Failed to load {len(failed_files)} audio file(s):\n"
        for file_path, error in failed_files:
            error_msg += f"  - {file_path}: {error}\n"
        raise RuntimeError(error_msg)
    
    if not audio_data_list:
        raise ValueError("No audio data could be loaded from the provided inputs.")
    
    return audio_data_list


def resolve_audio_paths_as_tensor_list(
    audio_inputs: AudioPathsInput,
    sr: int = SAMPLE_RATE,
    mono: bool = True,
    recursive: bool = False,
    includes: Optional[list[str]] = None,
    device: Union[str, torch.device] = "cpu",
) -> list["torch.Tensor"]:
    """
    Resolve audio files from paths and load them as torch tensors.
    
    Args:
        audio_inputs: Single file, list of files, or directory path
        sr: Target sample rate for loaded audio (default: 16000 Hz)
        mono: Whether to convert to mono (default: True)
        recursive: Whether to recursively search directories (default: False)
        includes: Optional list of glob patterns to filter files (e.g., ['**/sound.wav', '*.mp3'])
        device: Target device for the tensors (default: "cpu")
    
    Returns:
        List of torch tensors containing audio data for each file
        
    Raises:
        ImportError: If torch is not installed
        ValueError: If no valid audio files are found
        RuntimeError: If any audio file fails to load
    """
    # Get all audio file paths using resolve_audio_paths
    audio_paths = resolve_audio_paths(audio_inputs, recursive=recursive, includes=includes)
    
    # Load each audio file into a numpy array first, then convert to tensor
    audio_tensor_list = []
    failed_files = []
    
    for audio_path in audio_paths:
        try:
            # Load audio as numpy array
            audio_array, actual_sr = load_audio(audio_path, sr=sr, mono=mono)
            
            # Resample if the loaded sample rate doesn't match target
            if actual_sr != sr:
                audio_array = resample_audio(audio_array, actual_sr, target_sr=sr)
            
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio_array).to(device)
            audio_tensor_list.append(audio_tensor)
            
        except Exception as e:
            failed_files.append((audio_path, str(e)))
    
    # Report any failures
    if failed_files:
        error_msg = f"Failed to load {len(failed_files)} audio file(s):\n"
        for file_path, error in failed_files:
            error_msg += f"  - {file_path}: {error}\n"
        raise RuntimeError(error_msg)
    
    if not audio_tensor_list:
        raise ValueError("No audio data could be loaded from the provided inputs.")
    
    return audio_tensor_list


def load_audio(
    audio: AudioInput,
    sr: Optional[int] = SAMPLE_RATE,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Robust audio loader for ASR pipelines.
    Handles:
      - File paths
      - In-memory audio bytes (container OR raw PCM)
      - NumPy arrays
      - Torch tensors

    Args:
        sr: Target sample rate after loading. Pass None to keep the file's
            native sample rate (file/bytes inputs only). Arrays and tensors
            have no embedded rate and will use SAMPLE_RATE as fallback.

    Returns:
        (audio: np.ndarray [samples], sr: int)
    """
    def _decode_raw_pcm(
        data: bytes,
        expected_sr: int,
        channels: int = 1,
        dtype: npt.DTypeLike = np.int16,
    ) -> tuple[np.ndarray, int]:
        """Decode raw PCM bytes into numpy array."""
        itemsize = np.dtype(dtype).itemsize
        if len(data) % (channels * itemsize) != 0:
            raise ValueError(
                f"Invalid raw PCM buffer: {len(data)} bytes not divisible by "
                f"(channels={channels} × itemsize={itemsize})"
            )
        arr = np.frombuffer(data, dtype=dtype)
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)
        return arr, expected_sr

    current_sr: Optional[int] = None

    # ─────── Input handling ───────
    if isinstance(audio, (str, os.PathLike)):
        y, current_sr = librosa.load(audio, sr=None, mono=False)
    elif isinstance(audio, bytes):
        try:
            y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
        except Exception:
            fallback = sr if sr is not None else SAMPLE_RATE
            y, current_sr = _decode_raw_pcm(
                data=audio,
                expected_sr=fallback,
                channels=1,
                dtype=np.int16,
            )
    elif isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = None   # no embedded rate; resolved below
    elif isinstance(audio, torch.Tensor):
        y = audio.detach().float().cpu().numpy()
        current_sr = None   # no embedded rate; resolved below
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # ─────── Normalize (safety) ───────
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    if y.size > 0:
        max_val = np.abs(y).max()
        if max_val > 1.0 + 1e-6:
            y = y / max_val

    # ─────── Ensure (channels, time) ───────
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    # ─────── Mono conversion ───────
    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    # ─────── Sample rate resolution ───────
    # current_sr: rate detected from file/bytes (None for arrays/tensors)
    # sr:         caller's requested target (None = "keep native")
    # fallback:   SAMPLE_RATE for array/tensor inputs with no embedded rate
    fallback_sr = sr if sr is not None else SAMPLE_RATE
    effective_sr = current_sr if current_sr is not None else fallback_sr

    # Determine target: if caller passed None, keep effective_sr (no resample)
    target_sr = sr if sr is not None else effective_sr

    # ─────── Resample if needed ───────
    if effective_sr != target_sr:
        import logging
        logging.getLogger(__name__).debug(
            "load_audio: resampling %dHz → %dHz", effective_sr, target_sr
        )
        y = librosa.resample(y, orig_sr=effective_sr, target_sr=target_sr)
        effective_sr = target_sr

    return y.squeeze().astype(np.float32), effective_sr


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


def get_audio_duration(
    audio: AudioInput,
    sr: Optional[int] = None,
) -> float:
    """
    Get the duration of an audio input in seconds.
    
    Handles various input types efficiently:
    - File paths: Uses librosa to get duration without loading full audio
    - Bytes: Loads from buffer to determine duration
    - NumPy arrays: Calculates from shape and sample rate
    - Torch tensors: Calculates from shape and sample rate
    
    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        sr: Sample rate for array/tensor inputs. If None, defaults to SAMPLE_RATE (16000)
           For file inputs, sr is ignored and detected automatically
    
    Returns:
        Duration in seconds as a float
    
    Raises:
        TypeError: If audio input type is not supported
    
    Examples:
        # File duration
        duration = get_audio_duration("audio.wav")
        
        # Numpy array duration (uses default SAMPLE_RATE)
        duration = get_audio_duration(audio_array)
        
        # Numpy array with custom sample rate
        duration = get_audio_duration(audio_array, sr=44100)
        
        # Bytes duration
        duration = get_audio_duration(audio_bytes)
    """
    # Use default sample rate if not provided
    if sr is None:
        sr = SAMPLE_RATE
    
    if isinstance(audio, (str, os.PathLike)):
        # Use librosa to get duration without loading the entire file
        import librosa
        return librosa.get_duration(path=audio)
    
    elif isinstance(audio, bytes):
        import librosa
        # Load from bytes to get duration
        y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
        if y.ndim == 1:
            num_samples = len(y)
        else:
            num_samples = y.shape[-1]  # Take last dimension for multi-channel
        return num_samples / current_sr
    
    elif isinstance(audio, np.ndarray):
        # Use the last dimension as the time dimension
        num_samples = audio.shape[-1] if audio.ndim > 1 else len(audio)
        return num_samples / sr
    
    elif isinstance(audio, torch.Tensor):
        # Use the last dimension as the time dimension
        num_samples = audio.shape[-1] if audio.ndim > 1 else len(audio)
        return num_samples / sr
    
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")
