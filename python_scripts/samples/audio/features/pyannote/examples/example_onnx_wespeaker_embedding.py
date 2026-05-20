# Working Example
import torch
from pyannote.audio.pipelines.speaker_verification import ONNXWeSpeakerPretrainedSpeakerEmbedding

# Initialize (requires: pip install onnxruntime-gpu)
embedding_model = ONNXWeSpeakerPretrainedSpeakerEmbedding(
    embedding="hbredin/wespeaker-voxceleb-resnet34-LM",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # token=None,  # Optional HuggingFace token for private models
    # cache_dir="./onnx_models"
)

# Single speaker embedding
audio = torch.randn(1, 1, 16000)  # 1 second of audio
embedding = embedding_model(audio)
print(f"Embedding shape: {embedding.shape}")  # (1, 256)

# Batch processing with masks
batch_audio = torch.randn(4, 1, 32000)  # 4 speakers, 2 seconds each
masks = torch.ones(4, 32000)  # All active speech
embeddings = embedding_model(batch_audio, masks=masks)
print(f"Batch embeddings shape: {embeddings.shape}")  # (4, 256)

# FBANK feature visualization (optional)
fbank_features = embedding_model.compute_fbank(batch_audio)
print(f"FBANK features shape: {fbank_features.shape}")  # (4, num_frames, 80)
