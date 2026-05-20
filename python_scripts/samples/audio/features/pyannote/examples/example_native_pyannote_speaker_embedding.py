# Working Example
import torch
from pyannote.audio.pipelines.speaker_verification import PyannoteAudioPretrainedSpeakerEmbedding

# Initialize (requires: pip install pyannote.audio)
embedding_model = PyannoteAudioPretrainedSpeakerEmbedding(
    embedding="pyannote/embedding",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # token="your_huggingface_token",  # Required for gated models
    # cache_dir="./pyannote_models"
)

# Basic embedding extraction
audio = torch.randn(1, 1, 16000)
embedding = embedding_model(audio)
print(f"Embedding shape: {embedding.shape}")

# With voice activity weights (soft masks)
weights = torch.rand(1, 16000)  # Continuous weights between 0-1
weighted_embedding = embedding_model(audio, masks=weights)
print(f"Weighted embedding shape: {weighted_embedding.shape}")
