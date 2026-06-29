from __future__ import annotations
import argparse
import json
import shutil
import soundfile as sf
import numpy as np

from pathlib import Path
from rich.console import Console
from speech_waves import (
    extract_pure_speech_audio,
    get_valid_speech_waves,
    SAMPLE_RATE,
)
from audio_utils import load_audio

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
)

parser = argparse.ArgumentParser(
    description="Extract and analyse speech waves from audio using FireRedVAD.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ── Input / output ────────────────────────────────────────────────────────
parser.add_argument(
    "input",
    nargs="?",
    default=DEFAULT_AUDIO,
    help="Input audio file path.",
)
parser.add_argument(
    "-o",
    "--output-dir",
    default=OUTPUT_DIR,
    type=Path,
    help="Output results directory.",
)

args = parser.parse_args()

output_dir = args.output_dir
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Test extract_pure_speech_audio
waveform_np, sr = load_audio(args.input)
pure_speech = extract_pure_speech_audio(
    audio=waveform_np,
    sampling_rate=SAMPLE_RATE,  # assumed, could be parameterized
)
orig_s = len(waveform_np) / SAMPLE_RATE
filt_s = len(pure_speech) / SAMPLE_RATE
console.print(f"[info]🎯 Speech filtered: {filt_s:.2f}s from {orig_s:.2f}s[/info]")

# get_valid_speech_waves now accepts AudioInput (file path, bytes, or numpy array)
# Internally it calls load_audio() to handle any input type
speech_waves_tuples = get_valid_speech_waves(
    args.input,
    with_audio=True,
)

console.print(f"\n[bold]Found {len(speech_waves_tuples)} valid speech waves[/bold]\n")

if not speech_waves_tuples:
    console.print("[yellow]No valid speech waves found. Exiting.[/yellow]")
    exit(0)

waves_dir = output_dir / "waves"
waves_dir.mkdir(parents=True, exist_ok=True)

for idx, (speech_wave, wave_audio_np) in enumerate(speech_waves_tuples):
    wave_num = idx + 1
    wave_output_dir = waves_dir / f"wave_{wave_num:03d}"
    wave_output_dir.mkdir(parents=True, exist_ok=True)

    # Save speech wave metadata
    speech_wave_output = wave_output_dir / "speech_wave.json"
    with open(speech_wave_output, "w", encoding="utf-8") as f:
        json.dump(speech_wave, f, indent=2, ensure_ascii=False)

    # Save audio chunk as WAV
    wave_audio_output = wave_output_dir / "sound.wav"
    sf.write(
        str(wave_audio_output),
        wave_audio_np,
        SAMPLE_RATE,
    )

    d = speech_wave["details"]
    console.print(
        f"[green]✓ Wave {wave_num:03d}:[/green] "
        f"dur={d['duration_sec']:.2f}s, "
        f"frames=[{d['frame_start']}:{d['frame_end']}], "
        f"peak={d['max_prob']:.3f}, "
        f"composite={d['composite_score']:.4f}"
    )
    console.print(f"  [dim]metadata → {speech_wave_output}[/dim]")
    console.print(f"  [dim]audio    → {wave_audio_output}[/dim]")
    console.print()

waves_abs = waves_dir.resolve()
console.print(
    f"[bold green]✓ Done![/bold green] All waves saved to "
    f"[cyan][link=file://{waves_abs}]{waves_abs}[/link][/cyan]"
)
