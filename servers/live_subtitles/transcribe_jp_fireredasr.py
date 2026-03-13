import argparse
import json
import shutil
from pathlib import Path

from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
from fireredasr2s.fireredasr2 import FireRedAsr2Config
from rich.console import Console
from rich.pretty import pprint

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cache_dir = Path("~/.cache/pretrained_models").expanduser().resolve()
asr_model_dir = cache_dir / "FireRedASR2-AED"
vad_model_dir = cache_dir / "FireRedVAD/VAD"
lid_model_dir = cache_dir / "FireRedLID"
punc_model_dir = cache_dir / "FireRedPunc"

asr_config = FireRedAsr2Config(
    return_timestamp=True,
)

asr_system_config = FireRedAsr2SystemConfig(
    asr_model_dir=asr_model_dir,
    asr_config=asr_config,
    vad_model_dir=vad_model_dir,
    lid_model_dir=lid_model_dir,
    punc_model_dir=punc_model_dir,
    enable_vad=True,
    enable_lid=False,
    enable_punc=False,
)  # Use default config
asr_system = FireRedAsr2System(asr_system_config)


default_audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker_mono_16k.wav"
parser = argparse.ArgumentParser(description="Run FireRedASR2 on a wav file.")
parser.add_argument(
    "audio_path",
    nargs="?",
    default=default_audio_path,
    help="Path to input audio file (wav).",
)
args = parser.parse_args()

audio_path = args.audio_path
result = asr_system.process(audio_path)
print("Result:")
pprint(result)

result_json_path = OUTPUT_DIR / "transcription_result.json"
with open(result_json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

console.print(f"[bold green]Saved JSON result to:[/bold green] {result_json_path}")
