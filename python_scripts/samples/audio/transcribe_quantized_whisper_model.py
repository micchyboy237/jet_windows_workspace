import os
from typing import Literal
from pathlib import Path

from faster_whisper import WhisperModel

# ──────────────────────────────────────────────────────────────
# Configuration (identical to your original)
# ──────────────────────────────────────────────────────────────
ModelSizes = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]

BEST_CONFIG = {
    "tiny":            ("cuda", "int8_float16"),
    "base":            ("cuda", "int8_float16"),
    "small":           ("cuda", "int8_float16"),
    "medium":          ("cuda", "int8_float16"),
    "large":           ("cpu",  "int8"),
    "large-v2":        ("cpu",  "int8"),
    "large-v3":        ("cpu",  "int8"),
    "large-v3-turbo":  ("cpu",  "int8"),
    "distil-large-v3": ("cuda", "int8_float16"),
}

model_size: ModelSizes = "large-v2"
device, compute_type = BEST_CONFIG.get(model_size, ("cpu", "int8"))
quantized_model_path = f"C:/Users/druiv/.cache/hf_ctranslate2_models/whisper-{model_size}-ct2"

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\1.wav"

# ──────────────────────────────────────────────────────────────
# Load your existing local CTranslate2-converted model (zero download)
# ──────────────────────────────────────────────────────────────
if not os.path.isdir(quantized_model_path):
    raise FileNotFoundError(
        f"Local quantized model not found:\n{quantized_model_path}\n"
        "Make sure you ran ct2-transformers-converter for whisper-large-v3"
    )

model = WhisperModel(
    quantized_model_path,      # reuse your existing folder
    device=device,             # change to "cpu" if needed
    compute_type=compute_type,    # "int8_float16" or "int8" works great on GTX 1660
)

# ──────────────────────────────────────────────────────────────
# 1. Detect language + Transcription in original language
# ──────────────────────────────────────────────────────────────
segments, info = model.transcribe(
    audio_path,
    task="transcribe",
    beam_size=3,
    best_of=1,
    patience=1.0,
    temperature=(0.0, 0.2, 0.4),
    vad_filter=False,
    condition_on_previous_text=False,
)

print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

print("\nTranscription (original language):")
original_text = " ".join(segment.text for segment in segments).strip()
print(original_text)

# ──────────────────────────────────────────────────────────────
# 2. Translation to English (single call)
# ──────────────────────────────────────────────────────────────
segments_en, _ = model.transcribe(
    audio_path,
    task="translate",
    language="ja",
    beam_size=3,
    best_of=1,
    patience=1.0,
    temperature=(0.0, 0.2, 0.4),
    vad_filter=False,
    condition_on_previous_text=False,
)

print("\nEnglish translation:")
english_text = " ".join(segment.text for segment in segments_en).strip()
print(english_text)