from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="YOUR_HF_TOKEN")
pipeline.to(device)

diarization = pipeline("meeting.wav")

inference = Inference(model, window="whole")
inference.to(device)

speaker_embeddings = {}

for turn, _, speaker in diarization.itertracks(yield_label=True):
    if turn.duration < 2.0:          # skip very short segments
        continue
    seg = Segment(turn.start, turn.end)
    emb = inference.crop("meeting.wav", seg)
    emb = emb.reshape(1, -1)
    
    if speaker not in speaker_embeddings:
        speaker_embeddings[speaker] = []
    speaker_embeddings[speaker].append(emb)

# Average embeddings per speaker
for spk, embs in speaker_embeddings.items():
    avg_emb = np.mean(np.vstack(embs), axis=0, keepdims=True)
    print(f"Speaker {spk}: averaged embedding shape {avg_emb.shape}")