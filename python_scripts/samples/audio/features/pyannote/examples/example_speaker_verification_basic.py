# example_speaker_verification_basic.py
"""
Basic Speaker Verification — Are These Two Clips the Same Person?
=================================================================
Goal: Load two audio files → extract embeddings → compare with cosine distance.

This is the core use case for speaker_verification.py.
A small cosine distance (< ~0.5) usually means same speaker.
A large cosine distance (> ~1.0) usually means different speakers.
(The exact threshold depends on the model and your False Accept tolerance.)

Install requirements:
  pip install pyannote.audio scipy

You may need a HuggingFace token:
  https://huggingface.co/settings/tokens
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# ------------------------------------------------------------------
# 1. Build the embedding extractor
#    PretrainedSpeakerEmbedding() is the smart router — it picks the
#    right backend class automatically from the model name string.
# ------------------------------------------------------------------
get_embedding = PretrainedSpeakerEmbedding(
    "pyannote/embedding",
    # token="hf_your_token_here",   # needed for gated HuggingFace models
    # cache_dir="/tmp/pyannote",     # where to store downloaded model files
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_embedding.to(device)

print(f"Backend        : {type(get_embedding).__name__}")
print(f"Sample rate    : {get_embedding.sample_rate} Hz")
print(f"Embedding size : {get_embedding.dimension} dimensions")
print(f"Distance metric: {get_embedding.metric}")
print(f"Min audio len  : {get_embedding.min_num_samples} samples "
      f"({get_embedding.min_num_samples / get_embedding.sample_rate * 1000:.1f} ms)\n")

# ------------------------------------------------------------------
# 2. Load audio waveforms
#    The embedding model expects:
#      waveforms shape: (batch_size, num_channels, num_samples)
#      - batch_size  = number of clips processed at once
#      - num_channels = 1  (mono only)
#      - num_samples = length of the audio clip
# ------------------------------------------------------------------
import torchaudio

def load_mono(path: str, sample_rate: int) -> torch.Tensor:
    """Load an audio file and return a (1, 1, num_samples) tensor."""
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.unsqueeze(0)   # → (1, 1, num_samples)

sr = get_embedding.sample_rate

# Replace these with real audio files
clip_A1 = load_mono("speaker_A_clip1.wav", sr)   # Alice, recording 1
clip_A2 = load_mono("speaker_A_clip2.wav", sr)   # Alice, recording 2 (same person)
clip_B  = load_mono("speaker_B_clip1.wav", sr)   # Bob   (different person)

print(f"Clip A1 shape: {clip_A1.shape}  ({clip_A1.shape[-1] / sr:.2f}s)")
print(f"Clip A2 shape: {clip_A2.shape}  ({clip_A2.shape[-1] / sr:.2f}s)")
print(f"Clip B  shape: {clip_B.shape}   ({clip_B.shape[-1] / sr:.2f}s)\n")

# ------------------------------------------------------------------
# 3. Extract embeddings
#    Returns np.ndarray of shape (batch_size, dimension)
# ------------------------------------------------------------------
emb_A1 = get_embedding(clip_A1)   # shape: (1, 512)
emb_A2 = get_embedding(clip_A2)   # shape: (1, 512)
emb_B  = get_embedding(clip_B)    # shape: (1, 512)

print(f"Embedding shape: {emb_A1.shape}")
print(f"Embedding norm (A1): {np.linalg.norm(emb_A1):.4f}\n")

# ------------------------------------------------------------------
# 4. Compare embeddings using cosine distance
#    cdist returns a (1, 1) matrix — we take [0][0] for the scalar
# ------------------------------------------------------------------
dist_same    = cdist(emb_A1, emb_A2, metric="cosine")[0][0]
dist_diff    = cdist(emb_A1, emb_B,  metric="cosine")[0][0]

print("=== Cosine Distance Results ===")
print(f"  Alice vs Alice (same person) : {dist_same:.4f}  ← should be LOW")
print(f"  Alice vs Bob   (different)   : {dist_diff:.4f}  ← should be HIGH\n")

# ------------------------------------------------------------------
# 5. Make a verification decision
#    This threshold (0.5) is a starting point — tune it on your data.
# ------------------------------------------------------------------
THRESHOLD = 0.5

def verify(dist: float, threshold: float = THRESHOLD) -> str:
    return "✓ SAME speaker" if dist < threshold else "✗ DIFFERENT speaker"

print("=== Verification Decisions ===")
print(f"  Alice vs Alice : {verify(dist_same)}")
print(f"  Alice vs Bob   : {verify(dist_diff)}")

# ------------------------------------------------------------------
# 6. Batch processing — compare many clips at once
#    Stack clips along the batch dimension for GPU efficiency
# ------------------------------------------------------------------
print("\n=== Batch Embedding Extraction ===")

# Pad clips to the same length before stacking
max_len = max(clip_A1.shape[-1], clip_A2.shape[-1], clip_B.shape[-1])

def pad_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
    pad_size = length - tensor.shape[-1]
    return torch.nn.functional.pad(tensor, (0, pad_size))

batch = torch.cat([
    pad_to(clip_A1, max_len),
    pad_to(clip_A2, max_len),
    pad_to(clip_B,  max_len),
], dim=0)   # shape: (3, 1, max_len)

batch_embs = get_embedding(batch)   # shape: (3, 512)
print(f"Batch embeddings shape: {batch_embs.shape}")

dist_matrix = cdist(batch_embs, batch_embs, metric="cosine")
labels = ["Alice-1", "Alice-2", "Bob"]

print("\nPairwise cosine distance matrix:")
header = "          " + "  ".join(f"{l:>9}" for l in labels)
print(header)
for i, li in enumerate(labels):
    row = "  ".join(f"{dist_matrix[i,j]:9.4f}" for j in range(len(labels)))
    print(f"  {li:>8}: {row}")
