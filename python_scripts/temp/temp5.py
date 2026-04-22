# example_backends.py
"""
Using Each Embedding Backend Directly
======================================
Goal: Show how to instantiate each of the 4 backends explicitly,
      rather than using the auto-detecting factory function.

When to use each backend:
  - PyannoteAudio  → Best default; works out of the box with HuggingFace models
  - SpeechBrain    → Great accuracy; large model zoo on HuggingFace
  - NeMo           → NVIDIA's toolkit; good for production on GPU clusters
  - WeSpeaker/ONNX → Lightweight & fast; no GPU required; good for edge devices
"""

import numpy as np
import torch

# ------------------------------------------------------------------
# Shared test audio: (batch=2, channels=1, 2 seconds of audio)
# Two clips in one batch — demonstrating batch processing
# ------------------------------------------------------------------
SAMPLE_RATE = 16_000
audio_batch = torch.randn(2, 1, 2 * SAMPLE_RATE)

# Optional: a mask that says "only use the first 1.5 seconds of each clip"
# Shape: (batch_size, num_samples) — values between 0.0 and 1.0
# 1.0 = "this region has speech", 0.0 = "this is silence/noise"
mask = torch.zeros(2, 2 * SAMPLE_RATE)
mask[:, : int(1.5 * SAMPLE_RATE)] = 1.0  # first 1.5 sec is speech


# ==================================================================
# BACKEND 1: PyannoteAudio (default — recommended starting point)
# ==================================================================
print("=" * 60)
print("Backend 1: PyannoteAudio")
print("=" * 60)

from pyannote.audio.pipelines.speaker_verification import (
    PyannoteAudioPretrainedSpeakerEmbedding,
)

# Requires: pip install pyannote.audio
# Requires HuggingFace token for gated models — pass token="hf_..."
pyannote_model = PyannoteAudioPretrainedSpeakerEmbedding(
    embedding="pyannote/embedding",
    device=torch.device("cpu"),
    # token="hf_your_token_here",   # needed for gated HF models
    # cache_dir="/path/to/cache",   # optional local cache
)

embeddings = pyannote_model(audio_batch)
print(f"Output shape (no mask) : {embeddings.shape}")  # (2, 512)

embeddings_masked = pyannote_model(audio_batch, masks=mask)
print(f"Output shape (masked)  : {embeddings_masked.shape}")  # (2, 512)
print()


# ==================================================================
# BACKEND 2: SpeechBrain
# ==================================================================
print("=" * 60)
print("Backend 2: SpeechBrain")
print("=" * 60)

from pyannote.audio.pipelines.speaker_verification import (
    SpeechBrainPretrainedSpeakerEmbedding,
)

# Requires: pip install speechbrain
# Supports model versioning with "@": "model_name@main"
speechbrain_model = SpeechBrainPretrainedSpeakerEmbedding(
    embedding="speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"),
    # token="hf_your_token_here",
    cache_dir=r"C:\Users\druiv\.cache\pretrained_models\spkrec-ecapa-voxceleb",
)

embeddings = speechbrain_model(audio_batch)
print(f"Output shape (no mask): {embeddings.shape}")

# Key difference vs pyannote: wav_lens is normalized (0.0–1.0 range)
# Short clips get NaN embeddings so downstream code knows to skip them
embeddings_masked = speechbrain_model(audio_batch, masks=mask)
nan_mask = np.isnan(embeddings_masked[:, 0])
print(f"Any NaN embeddings (too-short clips)? {nan_mask.any()}")
print()


# ==================================================================
# BACKEND 3: NVIDIA NeMo
# ==================================================================
print("=" * 60)
print("Backend 3: NeMo (NVIDIA)")
print("=" * 60)

from pyannote.audio.pipelines.speaker_verification import (
    NeMoPretrainedSpeakerEmbedding,
)

# Requires: pip install nemo_toolkit[asr]
# Note: NeMo does NOT support cache_dir or token parameters
nemo_model = NeMoPretrainedSpeakerEmbedding(
    embedding="nvidia/speakerverification_en_titanet_large",
    device=torch.device("cpu"),
)

embeddings = nemo_model(audio_batch)
print(f"Output shape (no mask): {embeddings.shape}")

# NeMo uses absolute wav_lens (number of samples), not normalized
embeddings_masked = nemo_model(audio_batch, masks=mask)
print(f"Output shape (masked) : {embeddings_masked.shape}")
print()


# ==================================================================
# BACKEND 4: WeSpeaker via ONNX (lightweight, no GPU needed)
# ==================================================================
print("=" * 60)
print("Backend 4: WeSpeaker/ONNX")
print("=" * 60)

from pyannote.audio.pipelines.speaker_verification import (
    ONNXWeSpeakerPretrainedSpeakerEmbedding,
)

# Requires: pip install onnxruntime
# Downloads a .onnx file from HuggingFace Hub automatically
# You can also pass a local path to an .onnx file
wespeaker_model = ONNXWeSpeakerPretrainedSpeakerEmbedding(
    embedding="hbredin/wespeaker-voxceleb-resnet34-LM",
    device=torch.device("cpu"),
    # token="hf_your_token_here",
    # cache_dir="/tmp/onnx_cache",
)

# Key difference: WeSpeaker first converts audio → fbank features
# then feeds those features into the ONNX session
embeddings = wespeaker_model(audio_batch)
print(f"Output shape (no mask): {embeddings.shape}")

# With masks: processes each clip individually (not batched)
embeddings_masked = wespeaker_model(audio_batch, masks=mask)
print(f"Output shape (masked) : {embeddings_masked.shape}")

# WeSpeaker also exposes fbank features directly if needed
fbank = wespeaker_model.compute_fbank(audio_batch)
print(f"Fbank feature shape   : {fbank.shape}")  # (2, num_frames, 80)