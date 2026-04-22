# example_speaker_verification_pipeline.py
"""
SpeakerEmbedding Pipeline — Audio File → One Embedding
=========================================================
Goal: Use the high-level SpeakerEmbedding Pipeline class, which
      accepts an audio file dict instead of raw waveform tensors.

This is more convenient than using the raw backend classes directly
when you're working with files rather than pre-loaded tensors.

It also supports optional VAD (voice activity detection) weighting:
  - Without segmentation: embed the entire audio uniformly
  - With    segmentation: focus the embedding on speech frames only
    (speech probability is cubed → strongly down-weights near-silence)

Covers:
  1. Basic usage — file path → embedding
  2. With segmentation model — VAD-weighted embedding
  3. AudioFile dict formats accepted
  4. Comparing two speakers with the pipeline
  5. The VAD weight formula (scores**3)
  6. Running the CLI evaluator (main())
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding


# ==================================================================
# FEATURE 1: Basic usage — file path → one embedding
# ==================================================================
print("=== Feature 1: Basic SpeakerEmbedding Pipeline ===\n")

pipeline = SpeakerEmbedding(
    embedding="pyannote/embedding",
    segmentation=None,    # no VAD — embed the whole file uniformly
    # token="hf_your_token_here",
)

# SpeakerEmbedding.apply() returns np.ndarray of shape (1, dimension)
emb_alice = pipeline("alice.wav")    # ← replace with your file
emb_bob   = pipeline("bob.wav")

print(f"Embedding shape : {emb_alice.shape}   (1 × {emb_alice.shape[-1]} dims)")

dist = cdist(emb_alice, emb_bob, metric="cosine")[0][0]
print(f"Cosine distance : {dist:.4f}")
print(f"Decision        : {'same speaker' if dist < 0.5 else 'different speaker'}\n")


# ==================================================================
# FEATURE 2: With segmentation — VAD-weighted embedding
# ==================================================================
print("=== Feature 2: VAD-Weighted Embedding ===\n")
print("""
When you add a segmentation model, the pipeline:
  1. Runs the segmentation model on the file
  2. Takes per-frame speech probability scores
  3. Cubes them: weights = scores ** 3
     → prob=0.9 → weight=0.73  (keep)
     → prob=0.5 → weight=0.13  (down-weight heavily)
     → prob=0.1 → weight=0.001 (nearly ignore)
  4. Passes these weights to the embedding model

Effect: the embedding becomes less polluted by silence and background noise.
This typically improves verification accuracy on noisy recordings.
""")

pipeline_vad = SpeakerEmbedding(
    embedding="pyannote/embedding",
    segmentation="pyannote/segmentation",   # adds VAD weighting
    # token="hf_your_token_here",
)

emb_alice_vad = pipeline_vad("alice.wav")
emb_alice_plain = pipeline("alice.wav")    # no VAD (from Feature 1)

# Both are (1, 512) — same shape, different content
diff = cdist(emb_alice_vad, emb_alice_plain, metric="cosine")[0][0]
print(f"Without VAD weights embedding norm : {np.linalg.norm(emb_alice_plain):.4f}")
print(f"With    VAD weights embedding norm : {np.linalg.norm(emb_alice_vad):.4f}")
print(f"Distance between the two           : {diff:.4f}  (small = similar)\n")

# Visualise the weight formula
print("VAD weight formula  scores**3:")
for prob in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]:
    weight = prob ** 3
    bar = "█" * int(weight * 30)
    print(f"  prob={prob:.2f}  weight={weight:.4f}  {bar}")
print()


# ==================================================================
# FEATURE 3: AudioFile dict formats
# ==================================================================
print("=== Feature 3: AudioFile Dict Formats ===\n")
print("""
SpeakerEmbedding.apply() accepts several AudioFile formats.
All are equivalent — use whichever is most convenient.
""")

# Format A: plain file path string
file_a = "alice.wav"

# Format B: Path object
from pathlib import Path
file_b = Path("alice.wav")

# Format C: dict with audio path (allows adding metadata like uri)
file_c = {
    "uri": "alice_recording_01",   # optional identifier
    "audio": "alice.wav",
}

# Format D: dict with pre-loaded waveform tensor
import torchaudio
waveform, sr = torchaudio.load("alice.wav")
if waveform.shape[0] > 1:
    waveform = waveform.mean(0, keepdim=True)
file_d = {
    "waveform": waveform,
    "sample_rate": sr,
}

for fmt, f in [("string path", file_a), ("Path object", file_b),
               ("dict + audio", file_c), ("dict + waveform", file_d)]:
    emb = pipeline(f)
    print(f"  {fmt:20s}  → embedding shape {emb.shape}")
print()


# ==================================================================
# FEATURE 4: Comparing multiple speakers with the pipeline
# ==================================================================
print("=== Feature 4: Multi-Speaker Comparison ===\n")

audio_files = {
    "Alice" : "alice.wav",
    "Bob"   : "bob.wav",
    "Carol" : "carol.wav",
}

# Extract embeddings (cache them to avoid re-running the model)
embeddings = {name: pipeline(path) for name, path in audio_files.items()}

# Build pairwise distance matrix
names = list(embeddings.keys())
emb_matrix = np.vstack([embeddings[n] for n in names])   # (3, 512)
dist_matrix = cdist(emb_matrix, emb_matrix, metric="cosine")

print("Pairwise cosine distances:")
header = "         " + "  ".join(f"{n:>7}" for n in names)
print(header)
for i, ni in enumerate(names):
    row = "  ".join(f"{dist_matrix[i,j]:7.4f}" for j in range(len(names)))
    print(f"  {ni:>6}: {row}")
print()


# ==================================================================
# FEATURE 5: The VAD weight formula in detail
# ==================================================================
print("=== Feature 5: VAD Weight Formula (scores**3) ===\n")
print("""
Inside SpeakerEmbedding.apply():

  1. self._segmentation(file).data
     → SlidingWindowFeature.data  shape: (num_frames, 1)
        Values are speech probabilities in [0, 1]

  2. weights[np.isnan(weights)] = 0.0
     → Replace NaN frames (no prediction) with 0 weight

  3. weights = torch.from_numpy(weights ** 3)[None, :, 0]
     → Cube the probabilities: this aggressively penalises
        uncertain frames without completely ignoring them.
     → Shape becomes (1, num_frames) — ready for the embedding model

  4. self.embedding_model_(waveform, weights=weights)
     → The model uses these weights to produce a weighted mean
        embedding across frames (only confident speech contributes)
""")

# Demonstrate the cubing effect numerically
import numpy as np
probs = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.0])
weights_cubed = probs ** 3

print(f"  {'prob':>6}  {'weight (prob**3)':>18}  {'relative contribution':>22}")
for p, w in zip(probs, weights_cubed):
    bar = "▉" * int(w * 25)
    print(f"  {p:6.2f}  {w:18.4f}  {bar}")
print()


# ==================================================================
# FEATURE 6: Running the CLI evaluator
# ==================================================================
print("=== Feature 6: CLI Evaluator (main()) ===\n")
print("""
The module includes a main() function that computes EER on VoxCeleb trial lists.
Run it from the command line:

  python speaker_verification.py \\
    --protocol VoxCeleb.SpeakerVerification.VoxCeleb1 \\
    --subset test \\
    --embedding pyannote/embedding \\
    --segmentation pyannote/segmentation

What it does:
  1. Loads all trial pairs (two audio files + same/different label)
  2. Extracts embeddings for each unique audio file (with caching)
  3. Computes cosine distance for each trial pair
  4. Runs det_curve() to find the EER threshold
  5. Prints: protocol | subset | embedding | segmentation | EER = X.XXX%

EER (Equal Error Rate) — the operating point where:
  False Accept Rate  (wrong "same") == False Reject Rate (wrong "different")
  Lower EER = better verification system.
  State-of-the-art systems achieve < 1% EER on VoxCeleb1.
""")
