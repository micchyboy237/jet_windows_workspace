# instantiate the pipeline
from pyannote.audio import Pipeline

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

pipeline = Pipeline.from_pretrained(
  "pyannote/speech-separation-ami-1.0",
)

# run the pipeline on an audio file
diarization, sources = pipeline(audio_path)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

# dump sources to disk as SPEAKER_XX.wav files
import scipy.io.wavfile
for s, speaker in enumerate(diarization.labels()):
    scipy.io.wavfile.write(f'{speaker}.wav', 16000, sources.data[:,s])
