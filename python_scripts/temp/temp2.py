# example_pipeline.py
"""
SpeakerEmbedding High-Level Pipeline
======================================
Goal: Show the easiest way to extract embeddings from audio files
      using the full Pipeline class, with and without VAD (silence removal).

The Pipeline wraps everything — file loading, resampling, masking, model
inference — into a single .apply(file) call.

Two modes:
  1. No segmentation  → embed the whole file including silence
  2. With segmentation → use a VAD model to find speech, embed only that
"""

import os
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding

# ------------------------------------------------------------------
# MODE 1: Basic pipeline — no silence removal
# Best for: clean studio recordings, pre-segmented clips
# ------------------------------------------------------------------
print("Mode 1: Pipeline without Voice Activity Detection")
print("-" * 50)

pipeline_simple = SpeakerEmbedding(
    embedding="pyannote/embedding",
    segmentation=None,              # No VAD — embed the whole file
    # token="hf_your_token_here",   # for gated HuggingFace models
    # cache_dir="/tmp/pyannote_cache",
)

# AudioFile can be a path string, Path object, or a dict with "audio" key
# For demonstration, we use a dict with a waveform tensor directly
audio_file_1 = {"waveform": torch.randn(1, 16000 * 3), "sample_rate": 16000}
audio_file_2 = {"waveform": torch.randn(1, 16000 * 3), "sample_rate": 16000}

emb1 = pipeline_simple(audio_file_1)  # returns numpy array, shape (1, dim)
emb2 = pipeline_simple(audio_file_2)

distance = cdist(emb1, emb2, metric="cosine")[0, 0]
print(f"Embedding shape: {emb1.shape}")
print(f"Cosine distance between two clips: {distance:.4f}\n")


# ------------------------------------------------------------------
# MODE 2: Pipeline with VAD segmentation
# Best for: real-world recordings with pauses, noise, or multiple speakers
#
# How it works internally (from apply() method):
#   1. Load waveform from file
#   2. Run segmentation model → get speech probability per frame
#   3. Cube the probabilities (speech**3) to sharpen the mask
#      → This makes the mask more aggressive: 0.9→0.73, but 0.5→0.125
#   4. Feed waveform + mask into the embedding model
# ------------------------------------------------------------------
print("Mode 2: Pipeline with Voice Activity Detection (VAD)")
print("-" * 50)

pipeline_vad = SpeakerEmbedding(
    embedding="pyannote/embedding",
    segmentation="pyannote/segmentation",   # VAD model from HuggingFace
)

# In real usage:
#   emb = pipeline_vad("path/to/audio.wav")
#   emb = pipeline_vad({"audio": "recording.mp3"})
#   emb = pipeline_vad({"waveform": tensor, "sample_rate": 16000})
print("(Skipped actual inference — requires model download)")
print("Expected usage:")
print('  emb = pipeline_vad("speaker_recording.wav")')
print('  # or')
print('  emb = pipeline_vad({"audio": "recording.mp3"})\n')


# ------------------------------------------------------------------
# MODE 3: Comparing multiple speakers (real-world pattern)
# ------------------------------------------------------------------
print("Mode 3: Building a speaker database")
print("-" * 50)

# Simulate a library of known speakers
known_speakers = {
    "Alice": {"waveform": torch.randn(1, 16000 * 5), "sample_rate": 16000},
    "Bob":   {"waveform": torch.randn(1, 16000 * 5), "sample_rate": 16000},
    "Carol": {"waveform": torch.randn(1, 16000 * 5), "sample_rate": 16000},
}

# Extract and store embeddings
speaker_db = {}
for name, audio in known_speakers.items():
    speaker_db[name] = pipeline_simple(audio)
    print(f"  Enrolled {name}: embedding shape {speaker_db[name].shape}")

# Now identify an unknown speaker
unknown_audio = {"waveform": torch.randn(1, 16000 * 3), "sample_rate": 16000}
unknown_emb = pipeline_simple(unknown_audio)

print("\nIdentifying unknown speaker:")
for name, known_emb in speaker_db.items():
    dist = cdist(unknown_emb, known_emb, metric="cosine")[0, 0]
    print(f"  vs {name}: distance = {dist:.4f}")

best_match = min(speaker_db, key=lambda n: cdist(unknown_emb, speaker_db[n], metric="cosine")[0, 0])
print(f"\nBest match: {best_match}")