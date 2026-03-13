from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf  # <-- fast, supports many formats, free & popular
import torch

try:
    import torch
except ImportError:
    torch = None

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
    # ─────────────────────────────
    # Case 1: File path
    # ─────────────────────────────
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            return float(librosa.get_duration(path=str(path)))
        except Exception as e:
            raise ValueError(f"Failed to read audio file {path}: {e}") from e

    # ─────────────────────────────
    # Case 2: Raw bytes
    # ─────────────────────────────
    if isinstance(audio, bytes):
        try:
            buffer = io.BytesIO(audio)
            y, sr = librosa.load(buffer, sr=None, mono=False)
            return float(librosa.get_duration(y=y, sr=sr))
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes: {e}") from e

    # ─────────────────────────────
    # Case 3: numpy array
    # ─────────────────────────────
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy array input")
        return float(librosa.get_duration(y=audio, sr=sample_rate))

    # ─────────────────────────────
    # Case 4: torch tensor
    # ─────────────────────────────
    if torch is not None and isinstance(audio, torch.Tensor):
        if sample_rate is None:
            raise ValueError("sample_rate is required for torch.Tensor input")

        y = audio.detach().cpu().numpy()
        return float(librosa.get_duration(y=y, sr=sample_rate))

    raise TypeError(f"Unsupported audio input type: {type(audio)}")


def extract_audio_segment(
    audio_path: Union[str, Path, bytes, np.ndarray],
    *,
    start: float = 0.0,
    end: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Extract a partial audio segment from a file path, raw bytes, or numpy array.

    For numpy input, sample_rate must be provided.
    """
    if start < 0:
        raise ValueError("start must be >= 0")

    # --- Resolve audio source ---
    if isinstance(audio_path, (str, Path)):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        data, sr = sf.read(audio_path, dtype="float32")

    elif isinstance(audio_path, bytes):
        data, sr = sf.read(io.BytesIO(audio_path), dtype="float32")

    elif isinstance(audio_path, np.ndarray):
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided when audio_path is np.ndarray"
            )
        data = audio_path.astype(np.float32, copy=False)
        sr = sample_rate

    else:
        raise TypeError("audio_path must be Path, bytes, or np.ndarray")

    total_frames = data.shape[0]
    start_frame = int(start * sr)

    if start_frame >= total_frames:
        raise ValueError("start is beyond audio duration")

    if end is None:
        end = get_audio_duration(audio_path)

    if end <= start:
        raise ValueError("end must be greater than start")
    end_frame = min(int(end * sr), total_frames)

    return data[start_frame:end_frame], sr


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    import soundfile as sf

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\0001_video_en_sub_mono_16k.wav"

    start = 0
    end = 60

    if end is None:
        end = get_audio_duration(INPUT_AUDIO)

    # Extract from raw input audio
    segment, sr = extract_audio_segment(INPUT_AUDIO, start=start, end=end)
    output_path = OUTPUT_DIR / "extracted_audio.wav"
    sf.write(output_path, segment, sr)
    print("Extracted from raw input. Saved at:")
    print(output_path)
