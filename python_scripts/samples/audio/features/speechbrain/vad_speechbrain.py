import os
import shutil
from pathlib import Path
import argparse
import warnings
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy

# ====================== ARGPARSE + AUDIO PATH ======================
DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\audio\generated\extract_audio_segment\extracted_audio_16k_mono.wav"

parser = argparse.ArgumentParser(
    description="Run SpeechBrain VAD and save detected speech segments."
)
parser.add_argument(
    "audio_path",
    type=str,
    nargs="?",
    default=DEFAULT_AUDIO_PATH,
    help=f"Path to input audio file (default: '{DEFAULT_AUDIO_PATH}')",
)
args = parser.parse_args()
audio_path = Path(args.audio_path)

# ====================== OUTPUT SETUP ======================
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/vad-crdnn-libriparty").expanduser().resolve()
)

# ====================== WARNINGS ======================
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=ImportWarning)


print("Loading VAD model...")

vad_model = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir=SAVE_DIR,
    local_strategy=LocalStrategy.COPY,
    run_opts={"device": "cpu"},
)

# Change working directory to avoid SpeechBrain absolute path bug
prev_cwd = Path.cwd()
os.chdir(audio_path.parent)

try:
    audio_file = audio_path.name

    print("Running VAD on audio...")
    boundaries = vad_model.get_speech_segments(audio_file)

finally:
    # Always restore original working directory
    os.chdir(prev_cwd)

# ====================== SAVE OUTPUT ======================
output_path = OUTPUT_DIR / "speech_segments.txt"
vad_model.save_boundaries(boundaries, str(output_path))

# ====================== PRINT RESULTS ======================
print("Detected speech segments:")
for i, segment in enumerate(boundaries):
    start = segment[0].item()
    end = segment[1].item()
    print(f"Segment {i+1}: {start:.2f}s to {end:.2f}s → SPEECH")

print(f"\nResults saved to: {output_path.resolve()}")