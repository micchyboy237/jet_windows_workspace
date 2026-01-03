from __future__ import annotations

import json

from typing import Union, Sequence, Any
from pathlib import Path

import os
import numpy as np
import numpy.typing as npt

# Optional torch support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


def save_file(data: Any, filepath: str | Path) -> None:
    """
    Saves arbitrary data as a JSON file in a generic, reusable way.
    
    - Creates parent directories if they don't exist
    - Uses UTF-8 encoding and ensures ASCII compatibility is off
    - Pretty-prints with consistent, readable formatting
    - Works with any JSON-serializable object (dict, list, dataclass instances via asdict, etc.)
    
    Args:
        data: The data to save (must be JSON-serializable)
        filepath: Target file path (str or pathlib.Path)
    
    Raises:
        TypeError: If data cannot be serialized to JSON
        OSError: If file cannot be written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,          # deterministic output for easier diffing/testing
            default=str              # fallback for objects that aren't natively serializable
        )

    print(f"Saved: {filepath}")


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
