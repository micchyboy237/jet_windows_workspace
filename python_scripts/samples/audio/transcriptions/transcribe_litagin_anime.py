import json
import torch
from transformers import pipeline

# Recommended kwargs for this model (from model card)
generate_kwargs = {
    "language": "ja",               # "ja" = Japanese (preferred over full name in some cases)
    "task": "transcribe",
    "no_repeat_ngram_size": 0,      # disables repetition penalty on n-grams
    "repetition_penalty": 1.0,      # no extra penalty
    # Optional: helps with some anime-style speech if you notice hallucinations
    # "condition_on_previous_text": False,
}

pipe = pipeline(
    "automatic-speech-recognition",
    model="litagin/anime-whisper",
    device="cuda",
    torch_dtype=torch.float32,
    chunk_length_s=30.0,           # good default for long audio
    batch_size=4,                 # lowered from 64 → more stable on 1660 6GB VRAM
    # return_timestamps=False,     # uncomment if you don't need word-level timestamps
)

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# Run transcription
result = pipe(audio_path, return_timestamps=True,generate_kwargs=generate_kwargs)

print("\nResult:")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\nTranscription:")
print(result["text"])