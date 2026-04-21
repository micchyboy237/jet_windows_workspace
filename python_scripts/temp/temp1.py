import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\last_10_segments\segments_20260411_021222\sound.wav"

signal, fs = torchaudio.load(audio_path)
embeddings = classifier.encode_batch(signal)
