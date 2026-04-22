import json
from transformers import pipeline
import argparse

DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"

parser = argparse.ArgumentParser(description="Run speech separation model.")
parser.add_argument("audio_path",
                    nargs="?",
                    default=DEFAULT_AUDIO,
                    help="Path to input .wav audio file")
args = parser.parse_args()

AUDIO_PATH = args.audio_path

REPO_ID = "litagin/anime_speech_emotion_classification"
pipe = pipeline(
    "audio-classification",
    model=REPO_ID,
    feature_extractor=REPO_ID,
    trust_remote_code=True,
    device="cpu",
)

top_k = 5
results = pipe(AUDIO_PATH, top_k=top_k)
print(f"Results ({len(results)})")
print(json.dumps(results, indent=2))
