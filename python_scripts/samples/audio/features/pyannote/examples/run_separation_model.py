# run_separation_model.py

import os
import torch
import torchaudio
import scipy.io.wavfile
import numpy as np
from pyannote.audio import Model

# ── CONFIG ──────────────────────────────────────────────────────────────────
AUDIO_PATH    = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"
HF_TOKEN      = os.getenv("HF_TOKEN")
SAMPLE_RATE   = 16_000
CHUNK_SECONDS = 5.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)   # 80 000 samples per chunk
MAX_SPEAKERS  = 3
OUTPUT_DIR    = "output_model"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── STEP 1: Load audio ───────────────────────────────────────────────────────
print("Loading audio …")
waveform, sr = torchaudio.load(AUDIO_PATH)
# waveform shape: (channels, samples)

# ── STEP 2: Ensure mono 16 kHz ───────────────────────────────────────────────
if waveform.shape[0] > 1:
    print(f"  Converting {waveform.shape[0]}-channel audio to mono …")
    waveform = waveform.mean(dim=0, keepdim=True)   # average channels → (1, samples)

if sr != SAMPLE_RATE:
    print(f"  Resampling from {sr} Hz to {SAMPLE_RATE} Hz …")
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform  = resampler(waveform)

total_samples = waveform.shape[1]
print(f"  Audio: {total_samples} samples  ({total_samples / SAMPLE_RATE:.1f} s)")

# ── STEP 3: Load the chunk model ─────────────────────────────────────────────
print("Loading separation-ami-1.0 model …")
model = Model.from_pretrained("pyannote/separation-ami-1.0", token=HF_TOKEN)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Running on: {device}")
model = model.to(device)

# ── STEP 4: Slice into 5-second chunks ───────────────────────────────────────
# Pad the end so the last chunk is always exactly 5 s long
remainder = total_samples % CHUNK_SAMPLES
if remainder != 0:
    pad_size = CHUNK_SAMPLES - remainder
    waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    print(f"  Padded {pad_size} samples to make full chunks")

num_chunks = waveform.shape[1] // CHUNK_SAMPLES
print(f"  Processing {num_chunks} chunk(s) of {CHUNK_SECONDS} s each …")

# Accumulators – we'll concatenate results along the time axis
all_sources     = []   # list of (80000, 3) numpy arrays  (separated audio)
all_diarization = []   # list of (624, 3)  numpy arrays  (speaker activity)

# ── STEP 5: Run model chunk-by-chunk ─────────────────────────────────────────
with torch.inference_mode():
    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        end   = start + CHUNK_SAMPLES

        # Shape the model expects: (batch=1, channels=1, samples=80000)
        chunk = waveform[:, start:end].unsqueeze(0).to(device)  # (1,1,80000)

        diarization, sources = model(chunk)
        # diarization : (1, num_frames≈624, 3)  – values 0..1 (speaker active probability)
        # sources     : (1, 80000, 3)            – separated waveforms per speaker

        all_diarization.append(diarization[0].cpu().numpy())   # (624, 3)
        all_sources.append(sources[0].cpu().numpy())            # (80000, 3)

        if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            print(f"  Chunk {i+1}/{num_chunks} done")

# ── STEP 6: Stitch chunks together ───────────────────────────────────────────
print("Stitching chunks …")
# sources shape after stitch: (total_samples_padded, 3)
sources_full     = np.concatenate(all_sources,     axis=0)   # (N*80000, 3)
diarization_full = np.concatenate(all_diarization, axis=0)   # (N*624,   3)

# Trim back to original length (remove the padding we added)
sources_full = sources_full[:total_samples, :]

# ── STEP 7: Save per-speaker WAV files ───────────────────────────────────────
print("Saving speaker WAVs …")
for spk in range(MAX_SPEAKERS):
    spk_audio = sources_full[:, spk]                   # (total_samples,)
    spk_audio = (spk_audio * 32767).astype(np.int16)   # float → 16-bit PCM
    out_path  = os.path.join(OUTPUT_DIR, f"SPEAKER_{spk:02d}.wav")
    scipy.io.wavfile.write(out_path, SAMPLE_RATE, spk_audio)
    print(f"  Saved {out_path}")

# ── STEP 8: Print a quick activity summary ───────────────────────────────────
print("\nSpeaker activity summary (fraction of frames active):")
for spk in range(MAX_SPEAKERS):
    activity = diarization_full[:, spk].mean()
    print(f"  SPEAKER_{spk:02d}: {activity:.1%} active")

print("\nDone! Check the 'output_model/' folder.")
