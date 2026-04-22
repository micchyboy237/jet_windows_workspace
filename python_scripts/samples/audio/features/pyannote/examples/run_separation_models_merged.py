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
MIN_SEGMENT_DURATION_S = 0.3   # drop any segment shorter than this (seconds)
MIN_SILENCE_MERGE_S    = 0.2   # bridge same-speaker gaps shorter than this (seconds)
OUTPUT_DIR    = Path(__file__).parent / "generated" / Path(__file__).stem
SEGMENTS_DIR  = OUTPUT_DIR / "segments"
# ────────────────────────────────────────────────────────────────────────────

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ── HELPER A: extract raw (start_s, end_s) runs from one chunk's activity ────
def frames_to_segments(activity: np.ndarray, chunk_offset_s: float) -> list[tuple[float, float]]:
    """
    Converts one speaker's boolean frame array into a list of (abs_start_s, abs_end_s)
    pairs using the chunk's absolute time offset.  No merging here — kept simple
    so the caller can do cross-chunk merging correctly later.
    """
    if not activity.any():
        return []

    padded = np.concatenate(([False], activity, [False]))
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    result = []
    for s, e in zip(starts, ends):
        abs_start = chunk_offset_s + s * SECONDS_PER_FRAME
        abs_end   = chunk_offset_s + e * SECONDS_PER_FRAME
        if abs_end > abs_start:
            result.append((abs_start, abs_end))
    return result


# ── HELPER B: merge segments belonging to the SAME speaker across ALL chunks ──
def merge_segments_by_speaker(
    segments: list[tuple[float, float, int, int, np.ndarray]],
    merge_gap_s: float,
) -> list[tuple[float, float, int, int, np.ndarray]]:
    """
    segments   : list of (start_s, end_s, spk, chunk_idx, activity_probs)
                 already sorted by start_s
    merge_gap_s: bridge gaps ≤ this value — but ONLY between segments of the
                 same speaker label.  Different speakers are never merged.

    Returns the same tuple format, sorted by start_s.
    The chunk_idx and activity_probs of the EARLIER segment are kept when two
    segments are merged (they're metadata, not audio data).
    """
    if not segments:
        return []

    merged = [list(segments[0])]          # work with lists so we can mutate
    for cur in segments[1:]:
        cur_start, cur_end, cur_spk, cur_chunk, cur_probs = cur
        prev = merged[-1]
        prev_start, prev_end, prev_spk, prev_chunk, prev_probs = prev

        gap = cur_start - prev_end
        same_speaker = (cur_spk == prev_spk)   # ← the key guard

        if same_speaker and gap <= merge_gap_s:
            prev[1] = cur_end              # extend end time only
            # keep prev chunk_idx / activity_probs as representative metadata
        else:
            merged.append(list(cur))

    return [tuple(m) for m in merged]


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
print("Saving segments …")

# ── 7a: collect raw segments from every (chunk × speaker) pair ───────────────
all_segments_flat = []
for chunk_idx, chunk_diar in enumerate(all_diarization):
    chunk_offset_s = chunk_idx * CHUNK_SECONDS

    for spk in range(MAX_SPEAKERS):
        activity_probs = chunk_diar[:, spk]
        activity_bool  = activity_probs >= ACTIVITY_THRESHOLD

        for abs_start_s, abs_end_s in frames_to_segments(activity_bool, chunk_offset_s):
            all_segments_flat.append(
                (abs_start_s, abs_end_s, spk, chunk_idx, activity_probs)
            )

# ── 7b: sort by start time, then merge gaps — speaker-aware, cross-chunk ──────
all_segments_flat.sort(key=lambda x: x[0])
all_segments_flat = merge_segments_by_speaker(all_segments_flat, MIN_SILENCE_MERGE_S)

# ── 7c: drop segments that are too short AFTER merging ───────────────────────
# (filtering after merge means a pair of 150ms bursts bridged into 350ms is kept)
all_segments_flat = [
    seg for seg in all_segments_flat
    if (seg[1] - seg[0]) >= MIN_SEGMENT_DURATION_S
]

# ── 7d: write each segment with a leading sequential index ───────────────────
seg_count = len(all_segments_flat)
index_width = len(str(seg_count))   # e.g. 4 digits if ≤9999 segments

for seg_idx, (abs_start_s, abs_end_s, spk, chunk_idx, activity_probs) in \
        enumerate(all_segments_flat):

    seg_start_sample = int(abs_start_s * SAMPLE_RATE)
    seg_end_sample   = min(int(abs_end_s * SAMPLE_RATE), total_samples)

    start_ms = int(abs_start_s * 1000)
    end_ms   = int(abs_end_s   * 1000)

    # Zero-padded index so filenames sort correctly in any file explorer
    seg_name = f"seg_{seg_idx:0{index_width}d}_speaker_{spk}_{start_ms}_{end_ms}"
    seg_dir  = SEGMENTS_DIR / seg_name
    seg_dir.mkdir(parents=True, exist_ok=True)

    # ── sound.wav ──────────────────────────────────────────────────────────
    seg_audio = sources_full[seg_start_sample:seg_end_sample, spk]
    max_val = np.abs(seg_audio).max()
    if max_val > 1e-6:
        seg_audio = seg_audio / max_val
    seg_audio_i16 = (seg_audio * 32767).astype(np.int16)
    scipy.io.wavfile.write(seg_dir / "sound.wav", SAMPLE_RATE, seg_audio_i16)

    # ── meta.json ──────────────────────────────────────────────────────────
    meta = {
        "segment_index":        seg_idx,
        "speaker_id":           spk,
        "start_s":              round(abs_start_s, 4),
        "end_s":                round(abs_end_s,   4),
        "duration_s":           round(abs_end_s - abs_start_s, 4),
        "start_ms":             start_ms,
        "end_ms":               end_ms,
        "chunk_index":          chunk_idx,
        "chunk_offset_s":       chunk_idx * CHUNK_SECONDS,
        "sample_rate":          SAMPLE_RATE,
        "num_samples":          int(seg_end_sample - seg_start_sample),
        "activity_threshold":   ACTIVITY_THRESHOLD,
        "min_segment_duration": MIN_SEGMENT_DURATION_S,
        "min_silence_merge":    MIN_SILENCE_MERGE_S,
        "mean_activity_prob":   round(float(activity_probs.mean()), 4),
        "peak_activity_prob":   round(float(activity_probs.max()),  4),
        "segment_dir":          str(seg_dir),
    }
    with open(seg_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

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
