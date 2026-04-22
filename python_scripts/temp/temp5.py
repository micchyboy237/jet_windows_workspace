import os
import torch
import soundfile as sf  # <-- ADD THIS
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import argparse

DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"

parser = argparse.ArgumentParser(description="Speaker diarization with pyannote.audio")
parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO,
                    help="Path to the audio file (default: use hardcoded path)")
args = parser.parse_args()

audio_path = args.audio_path

# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"))

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# === PRELOAD AUDIO (this is the fix) ===
waveform_np, sample_rate = sf.read(
    audio_path,
    always_2d=True,      # ensures (time, channels) shape even for mono
    dtype="float32"
)
waveform = torch.from_numpy(waveform_np.T)  # convert to (channels, time) torch.Tensor

preloaded_audio = {
    "waveform": waveform,      # must be (channel, time) tensor
    "sample_rate": sample_rate
}

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    output = pipeline(preloaded_audio, hook=hook)  # <-- pass dict instead of str path

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
