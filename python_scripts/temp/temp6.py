import torch
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
import numpy as np
from scipy.spatial.distance import cdist

# Load models
embedding_model = Model.from_pretrained("pyannote/embedding", 
                                        use_auth_token="YOUR_HF_TOKEN")
embedding_inference = Inference(embedding_model, window="whole")

vad_model = Model.from_pretrained("pyannote/segmentation-3.0", 
                                  use_auth_token="YOUR_HF_TOKEN")
vad = VoiceActivityDetection(segmentation=vad_model)
vad.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

# Run VAD
audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
vad_annotation = vad(audio_path)  # pyannote.core.Annotation

# Extract clean speech segments (filter to 3s+)
embeddings = []
segments = []

for segment, _, _ in vad_annotation.itertracks(yield_label=True):
    if segment.duration >= 3.0:  # Best practice: 3–15s+
        waveform, sr = embedding_inference.audio.crop(audio_path, segment)
        emb = embedding_inference.crop(audio_path, segment)  # or use waveform
        embeddings.append(emb.squeeze())  # (D,)
        segments.append(segment)

embeddings = np.array(embeddings)  # (N, D)

print(f"Extracted {len(embeddings)} clean speech segments")