from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# Using default models
pipeline = SpeakerDiarization()

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_018\sound.wav"

output = pipeline(audio_path)

# Print standard diarization with overlaps
print("Standard Diarization:")
for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{speaker}: [{turn.start:.1f}s -> {turn.end:.1f}s]")

# The output also provides speaker embeddings
for i, speaker_label in enumerate(output.speaker_diarization.labels()):
    embedding_vector = output.speaker_embeddings[i]
    print(f"Embedding for {speaker_label}: shape {embedding_vector.shape}")



print("\nExclusive Diarization (No Overlaps):")
for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True):
    print(f"{speaker}: [{turn.start:.1f}s -> {turn.end:.1f}s]")

output_dict = output.serialize()
# Structure:
# {
#   'diarization': [{'start': 0.0, 'end': 2.5, 'speaker': 'SPEAKER_00'}, ...],
#   'exclusive_diarization': [{'start': 0.0, 'end': 2.5, 'speaker': 'SPEAKER_00'}, ...]
# }
import json
print("Serialized Output")
print(json.dumps(output_dict, indent=2))

# Force exactly 4 speakers
# output_exact = pipeline(audio_path, num_speakers=4)

# Force between 2 and 5 speakers
# output_range = pipeline(audio_path, min_speakers=2, max_speakers=5)
