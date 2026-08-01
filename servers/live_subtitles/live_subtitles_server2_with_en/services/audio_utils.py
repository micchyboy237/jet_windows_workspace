from __future__ import annotations

from typing import Union, Sequence, Generator, Tuple, NamedTuple, Optional
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


class SegmentInfo(NamedTuple):
    """Metadata about a single audio segment in the combined output."""
    index: int
    start_sample: int
    end_sample: int       # exclusive
    start_time: float
    end_time: float
    duration: float
    source: str           # file path stem, or type name for non-file inputs


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
    
    Args:
        audio_inputs: Single file, list of files, or directory path
        recursive: Whether to recursively search directories (default: False)
        includes: Optional list of glob patterns to filter files (e.g., ['**/sound.wav', '*.mp3'])
                  Patterns are relative to each input directory
    
    Returns:
        Sorted list of absolute path strings to valid audio files
    
    Examples:
        # Get all WAV files recursively
        resolve_audio_paths("audio_dir/", recursive=True, includes=["**/*.wav"])
        
        # Get specific files from any subdirectory
        resolve_audio_paths("audio_dir/", includes=["**/sound.wav", "**/music.mp3"])
        
        # Get MP3 files only from top level
        resolve_audio_paths("audio_dir/", includes=["*.mp3"])
        
        # Complex patterns
        resolve_audio_paths("audio_dir/", includes=["recordings/**/voice*.wav"])
    """
    inputs = [audio_inputs] if isinstance(audio_inputs, (str, Path)) else audio_inputs
    resolved_paths: list[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            # Use the new collection method for directories
            matched_files = _collect_audio_files(path, recursive=recursive, includes=includes)
            resolved_paths.extend(p.resolve() for p in matched_files)
            
        elif path.is_file():
            # For individual files, check if they match includes patterns (if any)
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                if not includes or any(
                    path.match(pattern) or 
                    path.match(f"**/{pattern}")  # Allow **/ matching for individual files too
                    for pattern in includes
                ):
                    resolved_paths.append(path.resolve())
                else:
                    print(f"Skipping file not matching include patterns: {path}")
            else:
                print(f"Skipping non-audio file: {path}")
        elif path.exists():
            print(f"Skipping non-audio file: {path}")
        else:
            print(f"Path not found: {path}")

    if not resolved_paths:
        raise ValueError("No valid audio files found from provided inputs.")

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


def combine_audio_paths(
    audio_inputs: list[AudioInput],
    gap: float = 0.5,
    sr: Optional[int] = SAMPLE_RATE,
    mono: bool = True,
    return_segments: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list[SegmentInfo]]]:
    """
    Combine multiple audio inputs into a single numpy array with configurable silence gap.

    This function handles various audio input types (file paths, bytes, numpy arrays,
    torch tensors) and concatenates them with a silence gap between each segment.

    Args:
        audio_inputs: List of audio inputs to combine. Each element can be:
            - File path (str or PathLike)
            - Raw audio bytes
            - Numpy array (floating or integer)
            - Torch tensor
        gap: Duration of silence in seconds to insert between audio segments (default: 0.5)
        sr: Target sample rate for all audio segments (default: SAMPLE_RATE, typically 16000)
        mono: Whether to convert all audio to mono before combining (default: True)
        return_segments: If True, returns a tuple (combined_audio, segments_info).
            segments_info is a list of SegmentInfo namedtuples with start/end times.
            (default: False for backward compatibility)

    Returns:
        If return_segments is False:
            Single numpy array containing all audio segments concatenated with silence gaps.
            Shape: (total_samples,) for mono, (channels, total_samples) for multi-channel.
        If return_segments is True:
            Tuple of (combined_audio, segments_info) where segments_info is a list of
            SegmentInfo namedtuples.

    Raises:
        ValueError: If audio_inputs list is empty
        TypeError: If any audio input type is not supported
        RuntimeError: If any audio segment fails to load

    Examples:
        # Backward compatible (just audio)
        combined = combine_audio_paths(["intro.wav", "main.wav"], gap=0.5)

        # With segment info
        combined, segments = combine_audio_paths(
            ["intro.wav", "main.wav"], gap=0.5, return_segments=True
        )
        for seg in segments:
            print(f"{seg.source}: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")
    """
    if not audio_inputs:
        raise ValueError("audio_inputs list cannot be empty")

    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    effective_sr = sr if sr is not None else SAMPLE_RATE
    gap_samples = int(gap * effective_sr)

    loaded_segments: list[np.ndarray] = []
    failed_inputs: list[tuple[int, str, str]] = []
    sources: list[str] = []  # Track source identifiers

    # Load all audio inputs and resolve source names
    for idx, audio_input in enumerate(audio_inputs):
        # Determine source identifier
        if isinstance(audio_input, (str, os.PathLike)):
            source = Path(audio_input).stem
        elif isinstance(audio_input, bytes):
            source = f"bytes_input_{idx}"
        elif isinstance(audio_input, np.ndarray):
            source = f"numpy_array_{idx}"
        elif isinstance(audio_input, torch.Tensor):
            source = f"tensor_{idx}"
        else:
            source = f"unknown_{idx}"

        try:
            audio_array, actual_sr = load_audio(
                audio=audio_input,
                sr=sr,
                mono=mono,
                return_as_tensor=False,
            )
            loaded_segments.append(audio_array)
            sources.append(source)
        except Exception as e:
            failed_inputs.append((idx, type(audio_input).__name__, str(e)))

    if failed_inputs:
        error_msg = f"Failed to load {len(failed_inputs)} audio input(s):\n"
        for idx, input_type, error in failed_inputs:
            error_msg += f"  - Input {idx} ({input_type}): {error}\n"
        raise RuntimeError(error_msg)

    if not loaded_segments:
        raise RuntimeError("No audio segments were successfully loaded")

    is_multichannel = loaded_segments[0].ndim > 1

    # Pre-compute segment info while we know individual lengths
    segment_infos: list[SegmentInfo] = []
    current_sample = 0

    for i, segment in enumerate(loaded_segments):
        # Get sample count along time axis
        num_samples = segment.shape[-1] if is_multichannel else len(segment)
        duration = num_samples / effective_sr

        start_sample = current_sample
        end_sample = start_sample + num_samples

        segment_infos.append(SegmentInfo(
            index=i,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_sample / effective_sr,
            end_time=end_sample / effective_sr,
            duration=duration,
            source=sources[i],
        ))

        # Advance: segment + optional gap
        current_sample = end_sample + (gap_samples if i < len(loaded_segments) - 1 else 0)

    # Build combined audio
    if gap_samples > 0:
        if is_multichannel:
            num_channels = loaded_segments[0].shape[0]
            silence = np.zeros((num_channels, gap_samples), dtype=np.float32)
        else:
            silence = np.zeros(gap_samples, dtype=np.float32)

    combined_segments = []
    for i, segment in enumerate(loaded_segments):
        combined_segments.append(segment)
        if i < len(loaded_segments) - 1 and gap_samples > 0:
            combined_segments.append(silence.copy())

    if is_multichannel:
        combined_audio = np.concatenate(combined_segments, axis=1)
    else:
        combined_audio = np.concatenate(combined_segments, axis=0)

    result = combined_audio.astype(np.float32)

    if return_segments:
        return result, segment_infos
    return result


def load_audio(
    audio: AudioInput,
    sr: Optional[int] = SAMPLE_RATE,
    mono: bool = True,
    return_as_tensor: bool = False,
) -> Union[Tuple[np.ndarray, int], Tuple[torch.Tensor, int]]:
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
        mono: Whether to convert multi-channel audio to mono.
        return_as_tensor: If True, return audio as a torch.Tensor instead of
            numpy array.

    Returns:
        (audio: np.ndarray or torch.Tensor [samples], sr: int)
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

    # ─────── Final output format ───────
    audio_array = y.squeeze().astype(np.float32)
    
    if return_as_tensor:
        return torch.from_numpy(audio_array), effective_sr
    else:
        return audio_array, effective_sr


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
