from typing import List, Dict, Any
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

REPO_ID = "litagin/anime_speech_emotion_classification"

# Load components manually (no pipeline)
feature_extractor = AutoFeatureExtractor.from_pretrained(REPO_ID, trust_remote_code=True)
model = AutoModelForAudioClassification.from_pretrained(REPO_ID, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load audio (your existing approach is fine)
audio, sr = librosa.load(
    r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav",
    sr=16000,
)

# Preprocess (NO torchcodec involved)
inputs = feature_extractor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probs = torch.nn.functional.softmax(logits, dim=-1)

# Top-k
top_k = 5
values, indices = torch.topk(probs, k=top_k)

id2label = model.config.id2label

result: List[Dict[str, Any]] = [
    {
        "label": id2label[idx.item()],
        "score": val.item(),
    }
    for val, idx in zip(values[0], indices[0])
]

print(result)