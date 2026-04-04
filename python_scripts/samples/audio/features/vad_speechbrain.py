import os
from pathlib import Path
import warnings
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy

# Suppress some noisy warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=ImportWarning)

print("Loading VAD model...")

vad_model = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir=r"C:\Users\druiv\.cache\pretrained_models\vad-crdnn-libriparty",  # Use absolute path
    local_strategy=LocalStrategy.COPY,   # Keeps Windows happy
    run_opts={"device": "cpu"}           # Force CPU for stability on Windows
)



audio_path = Path(r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav")

# Change working directory to the file's folder
os.chdir(audio_path.parent)

# Pass only the filename (NOT full path)
audio_file = audio_path.name

print("Running VAD on audio...")
boundaries = vad_model.get_speech_segments(audio_file)

vad_model.save_boundaries(boundaries, "speech_segments.txt")

print("Detected speech segments:")
for i, segment in enumerate(boundaries):
    start = segment[0].item()
    end = segment[1].item()
    print(f"Segment {i+1}: {start:.2f}s to {end:.2f}s → SPEECH")

print("\nResults saved to speech_segments.txt")