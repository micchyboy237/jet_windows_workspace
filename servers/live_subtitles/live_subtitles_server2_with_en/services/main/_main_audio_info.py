import argparse
import shutil
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
from rich.console import Console

from audio_info import display_audio_info
from norm_speech_loudness import normalize_audio_for_vad
from quant import quantize_audio
from dtype_conversion import convert_audio_dtype

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
)


def load_audio_from_file(
    file_path: Union[str, Path],
    sr: int = 16000,
    mono: bool = True,
) -> np.ndarray:
    """
    Load audio from file and return as float32 numpy array.

    Uses librosa for robust format support (WAV, MP3, FLAC, etc.).
    Returns mono float32 audio normalized to [-1, 1] range.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000 for VAD compatibility)
        mono: Convert to mono if True (recommended for VAD)

    Returns:
        Audio as float32 numpy array

    Raises:
        FileNotFoundError: If file_path doesn't exist
        ValueError: If audio loading fails
    """
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    console.print(f"Loading audio from: {file_path}")
    console.print(f"Target sr={sr}, mono={mono}")
    
    try:
        y, native_sr = librosa.load(
            str(file_path),
            sr=sr,
            mono=mono,
        )
        console.print(
            f"Loaded audio: shape={y.shape}, "
            f"native_sr={native_sr}, "
            f"duration={len(y)/sr:.2f}s"
        )
        console.print(
            f"Audio stats: min={y.min():.4f}, max={y.max():.4f}, "
            f"rms={np.sqrt(np.mean(y**2)):.4f}"
        )
        return y
        
    except Exception as e:
        console.print(f"[red]Failed to load audio file: {e}[/red]")
        raise ValueError(f"Failed to load audio from {file_path}: {e}") from e


def get_args():
    parser = argparse.ArgumentParser(
        description="Display detailed audio info with optional normalization and quantization"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Normalize audio loudness for VAD before displaying info",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Convert audio to int16 after (optional) normalization",
    )
    return parser.parse_args()


def main():
    args = get_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    
    console.print(f"Processing: {audio_path}")
    console.print(f"Normalize: {args.normalize}, Quantize: {args.quantize}")
    
    # Load audio
    audio_np = load_audio_from_file(audio_path)
    
    # Apply normalization if requested
    if args.normalize:
        console.print("Normalizing audio for VAD...")
        audio_np, norm_info = normalize_audio_for_vad(
            audio_np, 
            max_peak_db="standard",
        )
        console.print(
            f"Normalization complete: "
            f"original_rms={norm_info['original_rms_db']}dB, "
            f"final_rms={norm_info['final_rms_db']}dB, "
            f"gain={norm_info['applied_gain_db']}dB"
        )
    
    # Apply quantization if requested
    if args.quantize:
        console.print("Quantizing audio to int16...")
        audio_np = convert_audio_dtype(audio_np, "int16")
        # audio_np, quant_meta = quantize_audio(audio_np, "int16")
        console.print(
            f"Quantization complete: dtype={audio_np.dtype}, "
            f"range=[{audio_np.min()}, {audio_np.max()}]"
        )
    
    # Display detailed audio info
    display_audio_info(audio_np)


if __name__ == "__main__":
    main()
