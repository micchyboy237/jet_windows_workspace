# instantiate the pipeline
from pyannote.audio import Pipeline
import argparse
import os

DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"

parser = argparse.ArgumentParser(description="Speaker diarization with pyannote.audio")
parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO,
                    help="Path to the audio file (default: use hardcoded path)")
args = parser.parse_args()

audio_path = args.audio_path

pipeline = Pipeline.from_pretrained(
  "pyannote/speech-separation-ami-1.0",
)

# run the pipeline on an audio file
diarization, sources = pipeline(audio_path)

# dump the diarization output to disk using RTTM format
rttm_output_path = os.path.abspath("audio.rttm")
with open(rttm_output_path, "w") as rttm:
    diarization.write_rttm(rttm)
print(f"RTTM file saved to: {rttm_output_path}")

# dump sources to disk as SPEAKER_XX.wav files
import scipy.io.wavfile
for s, speaker in enumerate(diarization.labels()):
    wav_output_path = os.path.abspath(f'{speaker}.wav')
    scipy.io.wavfile.write(wav_output_path, 16000, sources.data[:,s])
    print(f"WAV for {speaker} saved to: {wav_output_path}")
