import os
import ctranslate2
import librosa
import transformers
from typing import Literal

# Supported quantized Whisper model sizes
QuantizedModelSizes = Literal[
    "tiny", "base", "small", "medium", "large-v2", "large-v3"
]

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
model_size: QuantizedModelSizes = "large-v2"
quantized_model_path = f"C:/Users/druiv/.cache/hf_ctranslate2_models/whisper-{model_size}-ct2"

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\1.wav"

# ──────────────────────────────────────────────────────────────
# Load model & processor
# ──────────────────────────────────────────────────────────────
if not os.path.exists(quantized_model_path):
    raise FileNotFoundError(
        f"Quantized model not found at:\n{quantized_model_path}\n"
        "Run: ct2-transformers-converter --model openai/whisper-large-v3 --output_dir <path> --quantization int8_float16"
    )

processor = transformers.WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
model = ctranslate2.models.Whisper(quantized_model_path)

# ──────────────────────────────────────────────────────────────
# Load and preprocess audio
# ──────────────────────────────────────────────────────────────
audio, _ = librosa.load(audio_path, sr=16000, mono=True)
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# ──────────────────────────────────────────────────────────────
# 1. Detect language (optional but helpful)
# ──────────────────────────────────────────────────────────────
detected = model.detect_language(features)
language_token, language_prob = detected[0][0]
print(f"Detected language: {language_token} (confidence: {language_prob:.2f})")

# ──────────────────────────────────────────────────────────────
# 2. Transcription in original language
# ──────────────────────────────────────────────────────────────
transcribe_prompt = processor.tokenizer.convert_tokens_to_ids(
    ["<|startoftranscript|>", language_token, "<|transcribe|>", "<|notimestamps|>"]
)

transcription_result = model.generate(features, [transcribe_prompt])
transcription = processor.decode(
    transcription_result[0].sequences_ids[0],
    skip_special_tokens=True
)
print("\nTranscription (original language):")
print(transcription)

# ──────────────────────────────────────────────────────────────
# 3. Translation to English
# ──────────────────────────────────────────────────────────────
translate_prompt = processor.tokenizer.convert_tokens_to_ids(
    ["<|startoftranscript|>", "<|en|>", "<|translate|>", "<|notimestamps|>"]
)

translation_result = model.generate(features, [translate_prompt])
translation = processor.decode(
    translation_result[0].sequences_ids[0],
    skip_special_tokens=True
)
print("\nEnglish translation:")
print(translation)