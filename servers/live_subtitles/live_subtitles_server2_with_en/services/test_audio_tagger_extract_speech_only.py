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
from audio_tagger import AudioTagger
from audio_config import SAMPLE_RATE
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

audio_tagger = AudioTagger()

# Test extract_pure_speech_audio
waveform_np, sr = load_audio(args.input, sr=SAMPLE_RATE)
pure_speech = audio_tagger.extract_speech_only(
    waveform_np,
    sample_rate=SAMPLE_RATE,
    # edges_only=True,
)
orig_s = len(waveform_np) / SAMPLE_RATE
filt_s = len(pure_speech) / SAMPLE_RATE
console.print(f"[info]🎯 Speech filtered: {filt_s:.2f}s from {orig_s:.2f}s[/info]")
