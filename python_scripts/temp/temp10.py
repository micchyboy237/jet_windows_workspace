from pyannote.audio import Model, Inference

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model, window="sliding", duration=3.0, step=1.0)

audio_path = r"C:\Users\druiv\.cache\files\audio\sub_audio\start_32s_recording_3_speakers.wav"

embeddings = inference(audio_path)
# embeddings: (N x D) SlidingWindowFeature
# embeddings[i] corresponds to [i*step, i*step + duration]
data, window = embeddings.data, embeddings.sliding_window
print(data.shape, window.start, window.duration, window.step)
