# example_basic.py
"""
Basic Speaker Verification Example
===================================
Goal: Compare two audio files to decide if they're the same speaker.

What we do:
  1. Load the smart factory function (it picks the right model for you)
  2. Create fake audio to simulate real recordings
  3. Extract voice fingerprints (embeddings)
  4. Measure cosine distance between them (0 = identical, 2 = totally different)
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# ------------------------------------------------------------------
# 1. Load the embedding model (factory auto-picks PyannoteAudio here)
# ------------------------------------------------------------------
get_embedding = PretrainedSpeakerEmbedding(
    "pyannote/embedding",
    device=torch.device("cpu"),
)

print(f"Sample rate  : {get_embedding.sample_rate} Hz")
print(f"Embedding dim: {get_embedding.dimension}")
print(f"Distance metric: {get_embedding.metric}")
print(f"Min audio length: {get_embedding.min_num_samples} samples "
      f"({get_embedding.min_num_samples / get_embedding.sample_rate:.3f} sec)\n")

# ------------------------------------------------------------------
# 2. Create fake audio tensors — shape must be (batch, channels, samples)
#    In real usage, load with torchaudio:
#      waveform, sr = torchaudio.load("speaker.wav")
#      waveform = waveform[None]  # add batch dimension → (1, 1, samples)
# ------------------------------------------------------------------
SAMPLE_RATE = get_embedding.sample_rate
TWO_SECONDS = 2 * SAMPLE_RATE

# Simulate two recordings of "the same speaker" (same random seed)
torch.manual_seed(42)
speaker_A_clip1 = torch.randn(1, 1, TWO_SECONDS)  # (batch=1, channels=1, samples)
speaker_A_clip2 = torch.randn(1, 1, TWO_SECONDS)  # different recording, same seed logic

# Simulate a "different speaker"
torch.manual_seed(99)
speaker_B_clip1 = torch.randn(1, 1, TWO_SECONDS)

# ------------------------------------------------------------------
# 3. Extract embeddings — output shape is (batch_size, dimension)
# ------------------------------------------------------------------
emb_A1 = get_embedding(speaker_A_clip1)  # shape: (1, 512)
emb_A2 = get_embedding(speaker_A_clip2)  # shape: (1, 512)
emb_B1 = get_embedding(speaker_B_clip1)  # shape: (1, 512)

print(f"Embedding shape: {emb_A1.shape}")

# ------------------------------------------------------------------
# 4. Compare using cosine distance
#    cdist computes a distance matrix; [0,0] picks the single value
#    Cosine distance range: 0 (identical) → 2 (opposite)
#    Typical threshold for same speaker: ~0.5–0.7 (model-dependent)
# ------------------------------------------------------------------
dist_same   = cdist(emb_A1, emb_A2, metric="cosine")[0, 0]
dist_diff   = cdist(emb_A1, emb_B1, metric="cosine")[0, 0]

THRESHOLD = 0.7  # tune this per model

print(f"Distance A1 vs A2 (same speaker): {dist_same:.4f}  → {'SAME ✓' if dist_same < THRESHOLD else 'DIFFERENT ✗'}")
print(f"Distance A1 vs B1 (diff speaker): {dist_diff:.4f}  → {'SAME ✓' if dist_diff < THRESHOLD else 'DIFFERENT ✗'}")