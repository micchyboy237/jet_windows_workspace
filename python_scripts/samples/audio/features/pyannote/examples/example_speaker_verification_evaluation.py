# example_speaker_verification_evaluation.py
"""
Speaker Verification Evaluation — EER, DET Curve, Batch Trials
================================================================
Goal: Measure how accurately an embedding model verifies speakers.

The standard protocol:
  1. You have a list of "trials": pairs of audio clips with a label
     (1 = same speaker, 0 = different speaker)
  2. For each pair: extract embeddings → compute cosine distance
  3. At every possible threshold: count false accepts and false rejects
  4. Find the EER — the threshold where both error types are equal

Covers:
  1. Building and evaluating a trial list
  2. The DET curve and EER
  3. Precision / Recall at a fixed threshold
  4. Embedding caching for efficient evaluation
  5. min_num_samples guard — handling very short clips
  6. NaN embedding detection and filtering
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
get_embedding = PretrainedSpeakerEmbedding(
    "pyannote/embedding",
    # token="hf_your_token_here",
)
SR = get_embedding.sample_rate
MIN_SAMPLES = get_embedding.min_num_samples

print(f"Model           : pyannote/embedding")
print(f"Dimension       : {get_embedding.dimension}")
print(f"Min audio length: {MIN_SAMPLES} samples ({MIN_SAMPLES/SR*1000:.1f} ms)\n")


def load_clip(path: str) -> torch.Tensor:
    """Load audio → (1, 1, num_samples) tensor at model's sample rate."""
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.unsqueeze(0)   # (1, 1, N)


# ==================================================================
# FEATURE 1: Building and evaluating a trial list
# ==================================================================
print("=== Feature 1: Trial List Evaluation ===\n")

# A trial list is a list of dicts, each with:
#   "file1"     → path to clip 1
#   "file2"     → path to clip 2
#   "reference" → 1 if same speaker, 0 if different

# Replace with your actual trial list (e.g. from VoxCeleb or AMI)
trials = [
    {"file1": "alice_1.wav", "file2": "alice_2.wav", "reference": 1},
    {"file1": "alice_1.wav", "file2": "bob_1.wav",   "reference": 0},
    {"file1": "bob_1.wav",   "file2": "bob_2.wav",   "reference": 1},
    {"file1": "carol_1.wav", "file2": "alice_1.wav", "reference": 0},
    {"file1": "carol_1.wav", "file2": "carol_2.wav", "reference": 1},
]

# Embed unique files only (avoid re-computing the same file twice)
embedding_cache = {}

def get_cached_embedding(path: str) -> np.ndarray:
    if path not in embedding_cache:
        clip = load_clip(path)
        embedding_cache[path] = get_embedding(clip)   # (1, dim)
    return embedding_cache[path]

y_true, y_pred = [], []

for trial in trials:
    emb1 = get_cached_embedding(trial["file1"])
    emb2 = get_cached_embedding(trial["file2"])
    distance = cdist(emb1, emb2, metric="cosine")[0][0]
    y_pred.append(distance)
    y_true.append(trial["reference"])
    same = "SAME" if trial["reference"] == 1 else "DIFF"
    print(f"  [{same}]  {Path(trial['file1']).stem:15s} vs {Path(trial['file2']).stem:15s}"
          f"  dist={distance:.4f}")

print(f"\n  Cached {len(embedding_cache)} unique embeddings for {len(trials)} trials\n")


# ==================================================================
# FEATURE 2: EER and DET curve
# ==================================================================
print("=== Feature 2: EER (Equal Error Rate) ===\n")
print("""
EER is the threshold where:
  False Accept Rate (FAR) = False Reject Rate (FRR)

  FAR = (different-speaker pairs accepted as same) / (all different pairs)
  FRR = (same-speaker pairs rejected as different) / (all same pairs)

Lower EER = better model.
""")

try:
    from pyannote.metrics.binary_classification import det_curve
    _, _, _, eer = det_curve(y_true, np.array(y_pred), distances=True)
    print(f"  EER = {eer * 100:.3f}%")
    print(f"  (On VoxCeleb1, state-of-the-art models achieve < 1% EER)\n")
except ImportError:
    print("  (pyannote.metrics not installed — install with: pip install pyannote.metrics)")

    # Manual EER approximation without pyannote.metrics
    thresholds = np.linspace(0, 2, 1000)
    same_mask = np.array(y_true) == 1
    diff_mask = ~same_mask
    dists = np.array(y_pred)

    best_eer, best_threshold = float("inf"), 0.5
    for t in thresholds:
        far = np.mean(dists[diff_mask] < t) if diff_mask.any() else 0
        frr = np.mean(dists[same_mask] >= t) if same_mask.any() else 0
        eer_approx = abs(far - frr)
        if eer_approx < best_eer:
            best_eer = eer_approx
            best_threshold = t

    print(f"  Approximate EER threshold : {best_threshold:.4f}")
    print(f"  (Install pyannote.metrics for exact EER)\n")


# ==================================================================
# FEATURE 3: Precision and Recall at a fixed threshold
# ==================================================================
print("=== Feature 3: Precision / Recall at Fixed Threshold ===\n")

THRESHOLD = 0.5   # tune on a development set
dists  = np.array(y_pred)
labels = np.array(y_true)
preds  = (dists < THRESHOLD).astype(int)   # 1 = predicted same, 0 = predicted different

tp = np.sum((preds == 1) & (labels == 1))
fp = np.sum((preds == 1) & (labels == 0))
tn = np.sum((preds == 0) & (labels == 0))
fn = np.sum((preds == 0) & (labels == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

print(f"  Threshold  : {THRESHOLD}")
print(f"  Precision  : {precision:.4f}  (of predicted SAME, fraction truly SAME)")
print(f"  Recall     : {recall:.4f}  (of all SAME pairs, fraction correctly found)")
print(f"  F1-score   : {f1:.4f}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")


# ==================================================================
# FEATURE 4: min_num_samples guard — short clip handling
# ==================================================================
print("=== Feature 4: min_num_samples Guard ===\n")
print(f"""
If a clip is shorter than {MIN_SAMPLES} samples ({MIN_SAMPLES/SR*1000:.1f} ms),
the model cannot produce a valid embedding.

All four backends handle this the same way:
  - The output row for that clip is filled with NaN
  - No exception is raised — the batch continues processing
  - You must check for NaN before using the embeddings

The exact minimum is found at init time by binary search:
  try increasingly small inputs until the model raises RuntimeError,
  then take the smallest input that succeeded.
""")

SR = get_embedding.sample_rate
MIN = get_embedding.min_num_samples

short_clip = torch.randn(1, 1, max(1, MIN - 100))    # intentionally too short
valid_clip = torch.randn(1, 1, MIN + 1000)           # safely above minimum

emb_short = get_embedding(short_clip)
emb_valid = get_embedding(valid_clip)

print(f"  Short clip  ({short_clip.shape[-1]:6d} samples) → NaN? {np.any(np.isnan(emb_short))}")
print(f"  Valid clip  ({valid_clip.shape[-1]:6d} samples) → NaN? {np.any(np.isnan(emb_valid))}\n")


# ==================================================================
# FEATURE 5: NaN detection and filtering in batch evaluation
# ==================================================================
print("=== Feature 5: Filtering NaN Embeddings ===\n")
print("""
When processing many clips in a batch, some may be too short or
contain no speech (after masking). Their rows will be NaN.
You must filter these out before computing distances.
""")

# Simulate a batch where one embedding is NaN (too-short clip)
batch_embs = np.random.randn(5, get_embedding.dimension)
batch_embs[2, :] = np.nan   # simulate a too-short clip

valid_mask = ~np.any(np.isnan(batch_embs), axis=1)
valid_embs = batch_embs[valid_mask]
valid_idxs = np.where(valid_mask)[0]

print(f"  Batch size          : {len(batch_embs)}")
print(f"  NaN rows            : {(~valid_mask).sum()}  (indices: {np.where(~valid_mask)[0].tolist()})")
print(f"  Valid rows kept     : {valid_mask.sum()}  (indices: {valid_idxs.tolist()})")
print(f"  Valid embedding shape: {valid_embs.shape}\n")

# Safe distance computation
if len(valid_embs) >= 2:
    dist_mat = cdist(valid_embs, valid_embs, metric="cosine")
    print(f"  Distance matrix shape (valid only): {dist_mat.shape}")
else:
    print("  Not enough valid embeddings to compute distances.")


# ==================================================================
# FEATURE 6: EER threshold sweep — find the best operating point
# ==================================================================
print("\n=== Feature 6: Threshold Sweep ===\n")
print("""
In production you need to pick a fixed operating threshold.
Sweep over thresholds on a development set to find the best trade-off.
""")

# Use the y_true / y_pred from Feature 1
dists  = np.array(y_pred)
labels = np.array(y_true)

print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'|FAR-FRR|':>10}")
print("  " + "-" * 44)

same_mask = labels == 1
diff_mask = ~same_mask

for t in np.arange(0.2, 1.4, 0.2):
    far = float(np.mean(dists[diff_mask] < t)) if diff_mask.any() else 0.0
    frr = float(np.mean(dists[same_mask] >= t)) if same_mask.any() else 0.0
    gap = abs(far - frr)
    marker = " ← EER point" if gap < 0.1 else ""
    print(f"  {t:10.2f}  {far:8.3f}  {frr:8.3f}  {gap:10.4f}{marker}")
