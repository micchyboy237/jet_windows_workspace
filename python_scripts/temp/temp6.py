import nemo.collections.asr as nemo_asr
import torch

# Load model
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    model_name="titanet_large"
)

audio_path1 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_002\sound.wav"
audio_path2 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_005\sound.wav"

# Extract embeddings
emb1 = speaker_model.get_embedding(audio_path1)
emb2 = speaker_model.get_embedding(audio_path2)

# Compute cosine similarity
def cosine_similarity(a, b):
    a = a.squeeze()
    b = b.squeeze()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

sim = cosine_similarity(emb1, emb2).item()
print(f"Cosine Similarity: {sim:.4f}")