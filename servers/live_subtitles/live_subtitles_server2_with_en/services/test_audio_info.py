import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_rich_traceback
from custom_logging import linkify
from audio_tagger import (
    AUDIO_TAGGING_MODEL,
    CLASS_LABELS_INDICES_CSV,
    AudioTagger,
)
from audio_config import SAMPLE_RATE
from serialization_utils import serialize
from norm_speech_loudness import normalize_audio_for_vad
from dtype_conversion import convert_audio_dtype
from main._main_audio_tagger import get_args, _format_predictions_with_emphasis, _get_probability_bar

install_rich_traceback(show_locals=True)
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

audio_path = r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav"
# audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_002\sound.wav"

speech_threshold = 0.1
sample_rate = 16000

args = get_args(audio_path, OUTPUT_DIR)
audio_path = args.audio_path
tagger = AudioTagger(
    speech_prob_threshold=speech_threshold,
)

console.print(f"\n[bold]Analyzing audio: {linkify(audio_path)}[/bold]\n")

output_dir_param = Path(args.output_dir)
shutil.rmtree(output_dir_param, ignore_errors=True)
output_dir_param.mkdir(parents=True, exist_ok=True)

# Load original audio once for duration calculations
original_audio, original_sr = sf.read(audio_path)
original_duration = len(original_audio) / original_sr

audio_np = convert_audio_dtype(original_audio, "int16")
