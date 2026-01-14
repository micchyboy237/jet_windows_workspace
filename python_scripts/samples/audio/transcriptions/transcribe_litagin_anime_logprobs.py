from typing import Sequence, Literal
import json
import soundfile as sf
import numpy as np
import torch
import math
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# =========================
# Config
# =========================
MODEL_ID = "litagin/anime-whisper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Short audio
AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# Long audio
# AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

GENERATE_KWARGS = {
    "language": "ja",
    "task": "transcribe",
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
}

# =========================
# Utilities for Confidence & Quality
# =========================
def calculate_confidence_score(
    token_logprobs: Sequence[float],
    *,
    clip_min: float = -5.0,
    clip_max: float = 0.0,
) -> float:
    if not token_logprobs:
        return 0.0

    clipped = np.clip(token_logprobs, clip_min, clip_max)
    probs = np.exp(clipped)
    mean_prob = float(np.mean(probs))
    return round(mean_prob * 100.0, 2)


QualityLabel = Literal[
    "very low",
    "low",
    "medium",
    "high",
    "very high",
]

def categorize_quality_label(confidence_score: float) -> QualityLabel:
    if confidence_score < 20.0:
        return "very low"
    if confidence_score < 40.0:
        return "low"
    if confidence_score < 60.0:
        return "medium"
    if confidence_score < 80.0:
        return "high"
    return "very high"

# =========================
# Load model + processor
# =========================
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
).to(DEVICE)

model.eval()

# =========================
# Load + resample audio
# =========================
audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")

# Convert to mono if needed
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Resample to 16kHz if needed
if sample_rate != 16000:
    audio = torch.from_numpy(audio)
    audio = torch.nn.functional.interpolate(
        audio.unsqueeze(0).unsqueeze(0),
        scale_factor=16000 / sample_rate,
        mode="linear",
        align_corners=False,
    ).squeeze().numpy()

# =========================
# Preprocess
# =========================
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
)

input_features = inputs.input_features.to(DEVICE)

# =========================
# Generate with scores
# =========================
with torch.no_grad():
    outputs = model.generate(
        input_features,
        return_dict_in_generate=True,
        output_scores=True,
        **GENERATE_KWARGS,
    )

# =========================
# Decode tokens
# =========================
sequences = outputs.sequences
tokens = sequences[0]

text = processor.tokenizer.decode(
    tokens,
    skip_special_tokens=True,
)

# =========================
# Compute token logprobs
# =========================
transition_scores = model.compute_transition_scores(
    sequences,
    outputs.scores,
    normalize_logits=True,
)

token_logprobs = transition_scores[0].tolist()
token_ids = tokens.tolist()

# Align tokens & logprobs (skip non-text/control tokens)
token_data = []
for token_id, logprob in zip(token_ids, token_logprobs):
    # Decode token ID directly (Whisper byte-BPE safe)
    text_piece = processor.tokenizer.decode(
        [token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if not text_piece.strip():
        continue

    token_data.append({
        "token": text_piece,
        "logprob": float(logprob),
    })

# =========================
# Aggregate confidence
# =========================
avg_logprob = (
    sum(t["logprob"] for t in token_data) / len(token_data)
    if token_data else float("nan")
)

token_logprobs = [t["logprob"] for t in token_data]
confidence_score = calculate_confidence_score(token_logprobs)
quality_label = categorize_quality_label(confidence_score)

# =========================
# Output
# =========================
result = {
    "text": text,
    "avg_logprob": avg_logprob,
    "confidence_score": confidence_score,
    "quality": quality_label,
    "tokens": token_data,
}


print("\n=== TOKEN LOGPROBS ===")
print(json.dumps(token_data, indent=2, ensure_ascii=False))

print("\n=== TRANSCRIPTION ===")
print(text)

print("\n=== AVERAGE LOGPROB ===")
print(avg_logprob)

print("\n=== CONFIDENCE ===")
print(f"Score: {confidence_score}")
print(f"Quality: {quality_label}")
