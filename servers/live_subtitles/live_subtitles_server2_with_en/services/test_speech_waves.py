import numpy as np
from speech_waves import WaveShapeConfig, check_speech_waves
from services.audio_config import SAMPLE_RATE, HOP_SIZE
from audio_utils import load_audio
from vad_firered import extract_speech_timestamps
from quant import quantize_audio

import argparse
import shutil
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/sub_audio/start_32s_recording_3_speakers.wav").expanduser().resolve()
)

# Default parameters
SAMPLING_RATE: int = SAMPLE_RATE
HOP_SIZE: int = HOP_SIZE
VAD_THRESHOLD: float = 0.3
MIN_PROMINENCE: float = 0.05
MIN_EXCURSION: float = 0.04
MIN_PEAK_PROB: float = 0.55
MIN_FRAMES: int = 3
MIN_DURATION_SEC: float = 0.25
BASELINE_THRESHOLD: float = 0.5
MIN_SPEECH_DURATION_MS: int = 250
MIN_SILENCE_DURATION_MS: int = 100


def generate_waves(audio, **kwargs):
    """
    Generate speech waves from audio with customizable parameters.
    
    Args:
        audio: Input audio array
        **kwargs: Optional overrides for wave generation parameters
            sampling_rate, hop_size, vad_threshold, min_prominence,
            min_excursion, min_peak_prob, min_frames, min_duration_sec,
            baseline_threshold, min_speech_duration_ms, min_silence_duration_ms
    
    Returns:
        tuple: (all_waves, valid_waves)
    """
    # Extract parameters with defaults
    sampling_rate = kwargs.get('sampling_rate', SAMPLING_RATE)
    hop_size = kwargs.get('hop_size', HOP_SIZE)
    vad_threshold = kwargs.get('vad_threshold', VAD_THRESHOLD)
    min_prominence = kwargs.get('min_prominence', MIN_PROMINENCE)
    min_excursion = kwargs.get('min_excursion', MIN_EXCURSION)
    min_peak_prob = kwargs.get('min_peak_prob', MIN_PEAK_PROB)
    min_frames = kwargs.get('min_frames', MIN_FRAMES)
    min_duration_sec = kwargs.get('min_duration_sec', MIN_DURATION_SEC)
    baseline_threshold = kwargs.get('baseline_threshold', BASELINE_THRESHOLD)
    min_speech_duration_ms = kwargs.get('min_speech_duration_ms', MIN_SPEECH_DURATION_MS)
    min_silence_duration_ms = kwargs.get('min_silence_duration_ms', MIN_SILENCE_DURATION_MS)
    
    # Extract speech scores
    _, scores = extract_speech_timestamps(
        audio=audio,
        include_non_speech=False,
        threshold=vad_threshold,
        min_speech_duration_sec=min_speech_duration_ms / 1000.0,
        min_silence_duration_sec=min_silence_duration_ms / 1000.0,
        with_scores=True,
    )
    
    # Build WaveShapeConfig from parameters
    shape_cfg = WaveShapeConfig(
        min_prominence=min_prominence,
        min_excursion=min_excursion,
        min_peak_prob=min_peak_prob,
        min_frames=min_frames,
        min_duration_sec=min_duration_sec,
        baseline_threshold=baseline_threshold,
    )
    
    # Generate waves
    all_waves = check_speech_waves(
        speech_probs=scores,
        threshold=vad_threshold,
        sampling_rate=sampling_rate,
        shape_cfg=shape_cfg,
    )
    
    valid_waves = [wave for wave in all_waves if wave["is_valid"]]
    
    return all_waves, valid_waves


def save_waves(all_waves, valid_waves, output_dir, prefix=""):
    """Save wave results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{prefix}_" if prefix else ""
    
    all_waves_path = output_dir / f"{prefix}all_waves.json"
    with open(all_waves_path, 'w') as f:
        json.dump(all_waves, f, indent=2)
    
    valid_waves_path = output_dir / f"{prefix}valid_waves.json"
    with open(valid_waves_path, 'w') as f:
        json.dump(valid_waves, f, indent=2)
    
    print(f"Saved {prefix}all_waves to: {all_waves_path}")
    print(f"Saved {prefix}valid_waves to: {valid_waves_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD"
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
    
    args = parser.parse_args()
    
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE)
    
    # Generate waves for original audio
    print("Processing original audio...")
    all_waves_orig, valid_waves_orig = generate_waves(audio)
    save_waves(all_waves_orig, valid_waves_orig, output_dir, prefix="original")
    
    print(f"Original - All waves: {len(all_waves_orig)}, Valid waves: {len(valid_waves_orig)}")
    print()
    
    # Quantize and generate waves
    print("Processing quantized audio...")
    quantized_audio_np, _ = quantize_audio(
        audio, target_dtype="int16", sr=SAMPLE_RATE, verbose=True
    )
    all_waves_quant, valid_waves_quant = generate_waves(quantized_audio_np)
    save_waves(all_waves_quant, valid_waves_quant, output_dir, prefix="quantized")
    
    print(f"Quantized - All waves: {len(all_waves_quant)}, Valid waves: {len(valid_waves_quant)}")


if __name__ == "__main__":
    main()
