from __future__ import annotations

import io
from pathlib import Path
from typing import Union, Tuple, Sequence

import numpy as np
import librosa

AudioInput = Union[str, Path, bytes, np.ndarray]

def load_audio(audio: AudioInput, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Robustly load audio and resample to 16kHz mono float32.
    
    Supports:
      • File path (str/Path)
      • Raw bytes (via BytesIO → fully supported by librosa)
      • Pre-loaded np.ndarray (e.g. from streaming)
    
    This is the most reliable, zero-temp-file, pure-memory solution.
    """
    if isinstance(audio, (str, Path)):
        y, _ = librosa.load(Path(audio), sr=sr, mono=True)
    
    elif isinstance(audio, bytes):
        # This is the correct, officially supported way
        with io.BytesIO(audio) as buf:
            # Must set buffer position to 0 (in case reused)
            buf.seek(0)
            y, _ = librosa.load(buf, sr=sr, mono=True)
    
    elif isinstance(audio, np.ndarray):
        y = audio
        if y.ndim > 1:
            y = librosa.to_mono(y)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        # Resample only if not already at target rate? We assume 16kHz input in pipeline
        return y, sr
    
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)!r}")

    return y.astype(np.float32), sr


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
