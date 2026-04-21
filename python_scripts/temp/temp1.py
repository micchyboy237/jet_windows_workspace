import torch
from pyannote.audio import Inference
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

pretrained_speaker_model_dir = r"C:\Users\druiv\.cache\pretrained_models\spkrec-ecapa-voxceleb"
model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"),
    cache_dir=pretrained_speaker_model_dir,
)
inference = Inference(model, window="whole")

full_audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

def extract_embedding(start_s: float, end_s: float):
    excerpt = Segment(start_s, end_s)
    embedding = inference.crop(full_audio_path, excerpt)
    # `embedding` is (1 x D) numpy array extracted from the file excerpt.
    return embedding

# compare embeddings using "cosine" distance
from scipy.spatial.distance import cdist

embedding1 = extract_embedding(3.99, 5.24)
embedding2 = extract_embedding(7.23, 9.29)
distance = cdist(embedding1, embedding2, metric="cosine")[0][0]

# Compute similarity score (0.0 = identical, 1.0 = completely different)
similarity_score = 1 - distance

print(f"Cosine Distance : {distance:.4f}")
print(f"Similarity Score: {similarity_score:.4f}")
print(f"Same Speaker    : {'Yes' if similarity_score > 0.85 else 'No'} (threshold = 0.85)")