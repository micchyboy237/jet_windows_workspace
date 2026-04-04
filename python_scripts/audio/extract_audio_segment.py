from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
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


def extract_audio_segment(
    audio_path: Union[str, Path, bytes, np.ndarray],
    *,
    start: float = 0.0,
    end: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Extract an audio segment and **always resample to mono 16kHz**.
    
    Returns:
        Tuple[np.ndarray, int]: (audio_data as float32, 16000)
    """
    if start < 0:
        raise ValueError("start must be >= 0")

    target_sr = 16000

    # --- Resolve audio source ---
    if isinstance(audio_path, (str, Path)):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # Load with librosa → mono + 16kHz directly
        data, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

    elif isinstance(audio_path, bytes):
        buffer = io.BytesIO(audio_path)
        data, sr = librosa.load(buffer, sr=target_sr, mono=True)

    elif isinstance(audio_path, np.ndarray):
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided when input is np.ndarray"
            )
        # Resample + convert to mono if needed
        data = librosa.to_mono(audio_path.astype(np.float32, copy=False).T)
        if sample_rate != target_sr:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
        sr = target_sr

    elif torch is not None and isinstance(audio_path, torch.Tensor):
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided when input is torch.Tensor"
            )
        data = audio_path.detach().cpu().numpy()
        data = librosa.to_mono(data.astype(np.float32, copy=False).T)
        if sample_rate != target_sr:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
        sr = target_sr

    else:
        raise TypeError("audio_path must be str/Path, bytes, np.ndarray, or torch.Tensor")

    # Handle segment extraction
    total_frames = len(data)
    start_frame = int(start * sr)
    if start_frame >= total_frames:
        raise ValueError("start is beyond audio duration")

    if end is None:
        end = get_audio_duration(audio_path, sample_rate=sr)

    if end <= start:
        raise ValueError("end must be greater than start")

    end_frame = min(int(end * sr), total_frames)

    segment = data[start_frame:end_frame]

    return segment.astype(np.float32, copy=False), sr  # sr is always 16000


# ─────────────────────────────
# Test / CLI
# ─────────────────────────────
if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    DEFAULT_INPUT_AUDIO = r"C:\Users\druiv\.cache\video\0001_video_en_sub.mp4"

    parser = argparse.ArgumentParser(description="Extract mono 16kHz audio segment.")
    parser.add_argument(
        "input_audio",
        nargs="?",
        default=DEFAULT_INPUT_AUDIO,
        help=f"Path to input audio/video file (default: {DEFAULT_INPUT_AUDIO})",
    )
    parser.add_argument("-s", "--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("-e", "--end", type=float, default=None, help="End time in seconds")

    args = parser.parse_args()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = args.start
    end = args.end

    # Extract mono 16kHz segment
    segment, sr = extract_audio_segment(args.input_audio, start=start, end=end)

    output_path = OUTPUT_DIR / "extracted_audio_16k_mono.wav"
    sf.write(output_path, segment, sr, subtype="PCM_16")   # 16-bit PCM for compatibility

    print(f"Extracted mono 16kHz audio segment. Saved at:")
    print(output_path)
    print(f"Duration: {len(segment)/sr:.2f} seconds | Sample rate: {sr} Hz | Channels: 1")
