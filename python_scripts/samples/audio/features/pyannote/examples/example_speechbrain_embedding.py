# File: example_speechbrain_embedding.py (updated)
import torch
import os
from pathlib import Path
from custom_speaker_embedding_classes import SpeechBrainPretrainedSpeakerEmbedding

# Optional: Specify a cache directory
# cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "pyannote")
cache_dir = str(
    Path("~/.cache/pretrained_models/spkrec-ecapa-voxceleb").expanduser().resolve()
)

# Create the embedding model with Windows-compatible settings
embedding_model = SpeechBrainPretrainedSpeakerEmbedding(
    embedding="speechbrain/spkrec-ecapa-voxceleb",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_dir=cache_dir,  # Explicit cache directory
)

# Test with random audio
audio_batch = torch.randn(3, 1, 32000)
embeddings = embedding_model(audio_batch)
print(f"Embedding shape: {embeddings.shape}")

# Compare two speakers
speaker1_emb = embedding_model(torch.randn(1, 1, 16000))
speaker2_emb = embedding_model(torch.randn(1, 1, 16000))

from scipy.spatial.distance import cosine
similarity = 1 - cosine(speaker1_emb[0], speaker2_emb[0])
print(f"Speaker similarity: {similarity:.3f}")