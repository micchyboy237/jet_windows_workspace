from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
)
audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_018\sound.wav"
output = pipeline(audio_path)
# print(output.speaker_diarization)            # regular speaker diarization
# print(output.exclusive_speaker_diarization)  # exclusive speaker diarization
print(f"Output ({type(output)}):")
print(output)