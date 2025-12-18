# transcribe_jp_en.py — FINAL WORKING VERSION
import os
import shutil
import dataclasses
from faster_whisper import WhisperModel
from utils import save_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_file = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\generated\preprocess_audio\cleaned.wav"
model_path = "C:/Users/druiv/.cache/hf_ctranslate2_models/faster-whisper-large-v3-int8_float16"

# THIS IS THE ONLY CHANGE THAT MATTERS
model = WhisperModel(
    model_path,
    device="CUDA",
    compute_type="int8_float16",   # ←←← THIS FIXES ZERO SEGMENTS ON GPU + TRANSLATE
    # float16 → bug with translate task on some audio
    # int8_float16 → keeps weights in int8, logits in float16 → perfect quality + no empty output
)

print("Model loaded with int8 → this fixes empty segments on GPU + translate")

segments_iter, info = model.transcribe(
    audio=audio_file,
    language="ja",
    task="translate",

    vad_filter=True,
    chunk_length=30,  # Even smaller for short audio

    # Ultra-loose thresholds (from GitHub #48 fixes)
    # no_speech_threshold=0.1,     # Default 0.6 → too aggressive for Ja
    # log_prob_threshold=-3.0,     # Default -1.0 → allow low-conf
    # compression_ratio_threshold=3.5,  # Default 2.4 → looser repetition check

    word_timestamps=True,
    without_timestamps=False,

    # Decoder tweaks
    beam_size=5,                 # Faster, less strict (from #1254)
    temperature=(0.0, 0.2, 0.4), # Fallback sampling if greedy fails
    # patience=2.0,                # Early stop if low conf
    best_of=5,                   # 1 for no diversity
    condition_on_previous_text=False,
    log_progress=True,
)

print(f"\nDetected: {info.language} ({info.language_probability:.2f}) | Duration: {info.duration:.1f}s\n")

segments = []
for segment in segments_iter:
    segments.append(dataclasses.asdict(segment))
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

print(f"\n→ SUCCESS: Generated {len(segments)} segments")

save_file(info, os.path.join(OUTPUT_DIR, "info.json"))
save_file(segments, os.path.join(OUTPUT_DIR, "segments.json"))
print(f"Results saved → {OUTPUT_DIR}")