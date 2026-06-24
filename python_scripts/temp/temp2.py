import librosa
import torch
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

def get_embedding(audio_path):
    # Load audio with librosa (always returns float32, can specify target sr)
    signal, fs = librosa.load(audio_path, sr=None, mono=True)  # sr=None keeps original sr
    
    # Convert to torch tensor
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # Add batch dim [1, samples]
    
    # Optional: resample to 16kHz if needed
    if fs != 16000:
        # librosa resampling
        signal_resampled = librosa.resample(signal, orig_sr=fs, target_sr=16000)
        signal_tensor = torch.from_numpy(signal_resampled).float().unsqueeze(0)
        fs = 16000
    
    emb = classifier.encode_batch(signal_tensor)  # [1, 1, 192]
    return emb.squeeze(0).squeeze(0)

audio_path1 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_002\sound.wav"
audio_path2 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_005\sound.wav"
audio_path3 = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_007\sound.wav"

# Example: Multiple segments for one speaker
segments = [audio_path1, audio_path2, audio_path3]  # each ~2-3s
embeddings = [get_embedding(f) for f in segments]

# Compute centroid
centroid = torch.mean(torch.stack(embeddings), dim=0)

# Check similarities to centroid (for health)
similarities = []
for emb in embeddings:
    sim = torch.nn.functional.cosine_similarity(emb, centroid, dim=0).item()
    similarities.append(sim)

avg_sim = np.mean(similarities)
print(f"Average similarity to centroid: {avg_sim:.4f}")
print(f"Total estimated speech duration: ~{len(segments) * 2.5:.1f} seconds")  # rough

# Threshold example: Filter poor segments
good_embeddings = [emb for emb, sim in zip(embeddings, similarities) if sim > 0.65]
if len(good_embeddings) >= 5:  # minimum count
    final_centroid = torch.mean(torch.stack(good_embeddings), dim=0)
    print("Centroid is healthy ✅")
else:
    print("Not enough good segments → unhealthy centroid")
