from pyannote.audio import Model, Inference
import torch.nn.functional as F
import torch

model = Model.from_pretrained("pyannote/embedding")  # HF token may be required
inference = Inference(model, window="whole")

audio_path1 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_002\sound.wav"
audio_path2 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_005\sound.wav"

# Extract embeddings
emb1 = inference(audio_path1)  # numpy array
emb2 = inference(audio_path2)

# Convert to torch tensors and normalize
emb1 = torch.tensor(emb1).unsqueeze(0)
emb2 = torch.tensor(emb2).unsqueeze(0)
emb1 = F.normalize(emb1, p=2, dim=1)
emb2 = F.normalize(emb2, p=2, dim=1)

# Cosine similarity
sim = F.cosine_similarity(emb1, emb2, dim=1).mean().item()
print(f"Cosine Similarity: {sim:.4f}")