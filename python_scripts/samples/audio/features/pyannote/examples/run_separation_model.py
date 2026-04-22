# run_separation_model.py

import os
import json
import torch
import torchaudio
import scipy.io.wavfile
import numpy as np
from pyannote.audio import Model
from pathlib import Path
import shutil

# ── CONFIG ──────────────────────────────────────────────────────────────────
AUDIO_PATH    = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"
HF_TOKEN      = os.getenv("HF_TOKEN")
SAMPLE_RATE   = 16_000
CHUNK_SECONDS = 5.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)   # 80 000 samples per chunk
FRAMES_PER_CHUNK   = 624                            # model always outputs 624 frames per 5 s chunk
SECONDS_PER_FRAME  = CHUNK_SECONDS / FRAMES_PER_CHUNK  # ≈ 0.00801 s per frame
ACTIVITY_THRESHOLD = 0.5                            # probability cutoff: above = speaker active
MAX_SPEAKERS  = 3
OUTPUT_DIR    = Path(__file__).parent / "generated" / Path(__file__).stem
SEGMENTS_DIR  = OUTPUT_DIR / "segments"
# ────────────────────────────────────────────────────────────────────────────

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ── HELPER: convert a boolean activity array into (start_s, end_s) runs ─────
def frames_to_segments(activity: np.ndarray, chunk_offset_s: float) -> list[tuple[float, float]]:
    """
    activity       : 1-D boolean array of length FRAMES_PER_CHUNK
    chunk_offset_s : time (seconds) at which this chunk starts in the full recording

    Returns a list of (abs_start_s, abs_end_s) tuples for every contiguous
    run of True values.  Empty list when the speaker is silent the whole chunk.
    """
    if not activity.any():
        return []

    segments = []
    # Pad with False at both ends so np.diff catches edges cleanly
    padded = np.concatenate(([False], activity, [False]))
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]   # frame indices where speaker turns ON
    ends   = np.where(diff == -1)[0]   # frame indices where speaker turns OFF

    for s, e in zip(starts, ends):
        abs_start = chunk_offset_s + s * SECONDS_PER_FRAME
        abs_end   = chunk_offset_s + e * SECONDS_PER_FRAME
        if abs_end > abs_start:          # skip zero-length artefacts
            segments.append((abs_start, abs_end))

    return segments

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

# ── STEP 7: Save per-segment WAVs + metadata ─────────────────────────────────
# Each folder:  segments/speaker_<spk>_<start_ms>_<end_ms>/
#   sound.wav   – the isolated speaker audio for that segment
#   meta.json   – timing, speaker id, chunk index, activity stats
print("Saving segments …")
seg_count = 0
for chunk_idx, chunk_diар in enumerate(all_diarization):
    # chunk_diар shape: (624, 3)  – raw probability floats
    chunk_offset_s = chunk_idx * CHUNK_SECONDS          # where this chunk starts in the full file
    chunk_start_sample = chunk_idx * CHUNK_SAMPLES

    for spk in range(MAX_SPEAKERS):
        activity_probs  = chunk_diар[:, spk]            # (624,) floats 0..1
        activity_bool   = activity_probs >= ACTIVITY_THRESHOLD

        segs = frames_to_segments(activity_bool, chunk_offset_s)
        for abs_start_s, abs_end_s in segs:
            # Convert absolute times → sample indices in sources_full
            seg_start_sample = int(abs_start_s * SAMPLE_RATE)
            seg_end_sample   = min(int(abs_end_s   * SAMPLE_RATE), total_samples)

            if seg_end_sample <= seg_start_sample:      # skip empty slices
                continue

            # ── folder name uses milliseconds for readability ──────────────
            start_ms = int(abs_start_s * 1000)
            end_ms   = int(abs_end_s   * 1000)
            seg_name = f"speaker_{spk}_{start_ms}_{end_ms}"
            seg_dir  = SEGMENTS_DIR / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)

            # ── sound.wav ──────────────────────────────────────────────────
            seg_audio = sources_full[seg_start_sample:seg_end_sample, spk]
            # Normalise to [-1, 1] then convert to 16-bit PCM
            max_val = np.abs(seg_audio).max()
            if max_val > 1e-6:                          # avoid divide-by-zero on silence
                seg_audio = seg_audio / max_val
            seg_audio_i16 = (seg_audio * 32767).astype(np.int16)
            scipy.io.wavfile.write(seg_dir / "sound.wav", SAMPLE_RATE, seg_audio_i16)

            # ── meta.json ──────────────────────────────────────────────────
            mean_prob = float(activity_probs.mean())
            peak_prob = float(activity_probs.max())
            meta = {
                "speaker_id":          spk,
                "start_s":             round(abs_start_s, 4),
                "end_s":               round(abs_end_s,   4),
                "duration_s":          round(abs_end_s - abs_start_s, 4),
                "start_ms":            start_ms,
                "end_ms":              end_ms,
                "chunk_index":         chunk_idx,
                "chunk_offset_s":      chunk_offset_s,
                "sample_rate":         SAMPLE_RATE,
                "num_samples":         int(seg_end_sample - seg_start_sample),
                "activity_threshold":  ACTIVITY_THRESHOLD,
                "mean_activity_prob":  round(mean_prob, 4),
                "peak_activity_prob":  round(peak_prob, 4),
                "segment_dir":         str(seg_dir),
            }
            with open(seg_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            seg_count += 1

print(f"  Saved {seg_count} segment(s) → {SEGMENTS_DIR}")

# ── STEP 8: Save full per-speaker WAV files ───────────────────────────────────
print("Saving speaker WAVs …")
for spk in range(MAX_SPEAKERS):
    spk_audio = sources_full[:, spk]                   # (total_samples,)
    spk_audio = (spk_audio * 32767).astype(np.int16)   # float → 16-bit PCM
    out_path  = OUTPUT_DIR / f"SPEAKER_{spk:02d}.wav"
    scipy.io.wavfile.write(out_path, SAMPLE_RATE, spk_audio)
    print(f"  Saved {out_path}")

# ── STEP 9: Print a quick activity summary ────────────────────────────────────
print("\nSpeaker activity summary (fraction of frames active):")
for spk in range(MAX_SPEAKERS):
    activity = diarization_full[:, spk].mean()
    print(f"  SPEAKER_{spk:02d}: {activity:.1%} active")

print("\nDone! Check the results here")
print(OUTPUT_DIR)
