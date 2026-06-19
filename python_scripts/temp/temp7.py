import librosa
import torch
from speechbrain.inference.speaker import EncoderClassifier

# Load model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb"
)

audio_path1 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_002\sound.wav"
audio_path2 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_005\sound.wav"

# Load two audio files with librosa
signal1, fs1 = librosa.load(audio_path1, sr=None)  # sr=None preserves original sample rate
signal2, fs2 = librosa.load(audio_path2, sr=None)

# Convert numpy arrays to torch tensors and add batch dimension
# librosa returns shape (samples,) so we need to reshape to (1, samples)
signal1_tensor = torch.from_numpy(signal1).unsqueeze(0).float()
signal2_tensor = torch.from_numpy(signal2).unsqueeze(0).float()

# Extract embeddings
emb1 = classifier.encode_batch(signal1_tensor)  # [1, 1, 192]
emb2 = classifier.encode_batch(signal2_tensor)

# Cosine similarity
cos_sim = torch.nn.functional.cosine_similarity(
    emb1.squeeze(0).squeeze(0), 
    emb2.squeeze(0).squeeze(0), 
    dim=0
).item()

print(f"Cosine Similarity: {cos_sim:.4f}")