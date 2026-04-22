# run_separation_pipeline.py

import torch
import torchaudio
import scipy.io.wavfile
import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pathlib import Path
import shutil

# ── CONFIG ──────────────────────────────────────────────────────────────────
AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"
HF_TOKEN   = os.getenv("HF_TOKEN")
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
# ────────────────────────────────────────────────────────────────────────────

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── STEP 1: Load the full pipeline ───────────────────────────────────────────
print("Loading speech-separation-ami-1.0 pipeline …")
pipeline = Pipeline.from_pretrained(
    "pyannote/speech-separation-ami-1.0",
    token=HF_TOKEN
)

# ── STEP 2: Move to GPU if available ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Running on: {device}")
pipeline.to(device)

# ── STEP 3: Pre-load audio into memory (faster than reading from disk twice) ─
print("Pre-loading audio into memory …")
waveform, sample_rate = torchaudio.load(AUDIO_PATH)
audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

# ── STEP 4: Run the pipeline with a progress hook ────────────────────────────
print("Running pipeline (this may take a while for long files) …")
with ProgressHook() as hook:
    diarization, sources = pipeline(audio_in_memory, hook=hook)

# diarization : pyannote Annotation  – who spoke when
# sources     : SlidingWindowFeature – separated audio per speaker

# ── STEP 5: Save diarization as RTTM ─────────────────────────────────────────
rttm_path = OUTPUT_DIR / "diarization.rttm"
with open(rttm_path, "w") as f:
    diarization.write_rttm(f)
print(f"  Saved diarization → {rttm_path}")

# ── STEP 6: Save per-speaker separated audio ─────────────────────────────────
print("Saving per-speaker WAV files …")
labels = diarization.labels()   # e.g. ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']

for s, speaker in enumerate(labels):
    # sources.data shape: (total_samples, num_speakers)
    spk_audio = sources.data[:, s]   # (total_samples,) as float32

    # Normalise to avoid clipping, then convert to 16-bit PCM
    max_val = abs(spk_audio).max()
    if max_val > 0:
        spk_audio = spk_audio / max_val   # normalise to [-1, 1]
    spk_audio_int16 = (spk_audio * 32767).astype("int16")

    out_path = OUTPUT_DIR / f"{speaker}.wav"
    scipy.io.wavfile.write(out_path, 16_000, spk_audio_int16)
    print(f"  Saved {out_path}")

# ── STEP 7: Print a human-readable diarization summary ───────────────────────
print("\n── Diarization Timeline ──────────────────────────────────")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"  [{turn.start:6.1f}s → {turn.end:6.1f}s]  {speaker}")

print("\nDone! Check the rttm file")
print(rttm_path)
