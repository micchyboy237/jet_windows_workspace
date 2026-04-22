# example_verification.py
"""
Full Speaker Verification Workflow
====================================
Goal: A complete, practical verification system showing:
  1. Enrollment  — store a speaker's voice fingerprint
  2. Verification — check if a new clip matches an enrolled speaker
  3. Batch processing — extract embeddings for many clips at once
  4. Device switching — move models between CPU and GPU
  5. Handling edge cases — very short clips, silence, NaN embeddings

This is closest to how you'd use this code in a real application
(e.g., voice login, meeting diarization, call center analysis).
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


# ------------------------------------------------------------------
# Helper: A simple in-memory speaker enrollment store
# ------------------------------------------------------------------
@dataclass
class SpeakerStore:
    """Stores enrolled speaker embeddings and verifies new clips."""
    model: PretrainedSpeakerEmbedding   # the voice-fingerprint extractor
    threshold: float = 0.7
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def enroll(self, speaker_id: str, waveform: torch.Tensor) -> None:
        """
        Register a speaker from a waveform.
        waveform shape: (1, 1, num_samples) — single clip
        """
        emb = self.model(waveform)
        if np.any(np.isnan(emb)):
            # This can happen with very short or silent clips
            raise ValueError(f"Could not extract embedding for '{speaker_id}' — clip too short?")
        self.embeddings[speaker_id] = emb
        print(f" Enrolled '{speaker_id}' — embedding shape {emb.shape}")

    def is_long_enough(self, waveform: torch.Tensor) -> bool:
        """Quick safety check so we never hit the kernel-size RuntimeError."""
        min_samples = self.model.min_num_samples
        return waveform.shape[-1] >= min_samples

    def _get_embedding_safe(self, waveform: torch.Tensor) -> Optional[np.ndarray]:
        """Helper that returns embedding or None if too short."""
        if not self.is_long_enough(waveform):
            return None
        emb = self.model(waveform)
        return emb if not np.any(np.isnan(emb)) else None

    def verify(self, speaker_id: str, waveform: torch.Tensor) -> tuple[bool, float]:
        """
        Check if a waveform matches an enrolled speaker.
        Returns (is_match, cosine_distance).
        """
        if speaker_id not in self.embeddings:
            raise KeyError(f"Speaker '{speaker_id}' not enrolled yet.")
        emb = self._get_embedding_safe(waveform)
        if emb is None:
            return False, float("nan")
        dist = cdist(emb, self.embeddings[speaker_id], metric="cosine")[0, 0]
        # threshold=0.05 was chosen so the demo shows a clear REJECT for the impostor
        return dist < self.threshold, float(dist)

    def identify(self, waveform: torch.Tensor) -> tuple[Optional[str], float]:
        """
        Find the closest enrolled speaker, or return None if no match.
        Returns (speaker_id_or_None, best_distance).
        """
        emb = self.model(waveform)
        if np.any(np.isnan(emb)) or not self.embeddings:
            return None, float("nan")
        distances = {
            name: cdist(emb, stored, metric="cosine")[0, 0]
            for name, stored in self.embeddings.items()
        }
        best_name = min(distances, key=distances.__getitem__)
        best_dist = distances[best_name]
        if best_dist > self.threshold:
            return None, float(best_dist)
        return best_name, float(best_dist)


# ------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------
SAMPLE_RATE = 16_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model = PretrainedSpeakerEmbedding("pyannote/embedding", device=device)
store = SpeakerStore(model=model, threshold=0.7)


# ------------------------------------------------------------------
# 2. Enrollment — register known speakers
# ------------------------------------------------------------------
print("=== ENROLLMENT ===")
torch.manual_seed(1); alice_enrollment = torch.randn(1, 1, 3 * SAMPLE_RATE)
torch.manual_seed(2); bob_enrollment   = torch.randn(1, 1, 3 * SAMPLE_RATE)

store.enroll("alice", alice_enrollment)
store.enroll("bob", bob_enrollment)


# ------------------------------------------------------------------
# 3. Verification — does this clip match the claimed speaker?
# ------------------------------------------------------------------
print("\n=== VERIFICATION ===")

# Slightly different clip from the same "speaker" (simulated)
torch.manual_seed(1); alice_test = torch.randn(1, 1, 2 * SAMPLE_RATE)
torch.manual_seed(3); impostor  = torch.randn(1, 1, 2 * SAMPLE_RATE)

is_match, dist = store.verify("alice", alice_test)
print(f"Alice's clip vs Alice's profile  → {'ACCEPT ✓' if is_match else 'REJECT ✗'} (dist={dist:.4f})")

is_match, dist = store.verify("alice", impostor)
print(f"Impostor's clip vs Alice's profile → {'ACCEPT ✓' if is_match else 'REJECT ✗'} (dist={dist:.4f})")


# ------------------------------------------------------------------
# 4. Identification — who is this mystery speaker?
# ------------------------------------------------------------------
print("\n=== IDENTIFICATION ===")
torch.manual_seed(2); bob_test = torch.randn(1, 1, 2 * SAMPLE_RATE)

name, dist = store.identify(bob_test)
print(f"Mystery clip identified as: {name or 'UNKNOWN'} (dist={dist:.4f})")


# ------------------------------------------------------------------
# 5. Batch processing — extract embeddings for many clips at once
# ------------------------------------------------------------------
print("\n=== BATCH PROCESSING ===")

# Stack multiple clips into a single batch for efficiency
batch_size = 4
batch = torch.randn(batch_size, 1, 2 * SAMPLE_RATE)

embeddings = model(batch)  # shape: (4, dimension)
print(f"Batch input shape : {batch.shape}")
print(f"Batch output shape: {embeddings.shape}")

# Pairwise distances between all clips in the batch
dist_matrix = cdist(embeddings, embeddings, metric="cosine")
print(f"Pairwise distance matrix:\n{np.round(dist_matrix, 3)}\n")


# ------------------------------------------------------------------
# 6. Masking — only embed the speech parts, ignore silence
# ------------------------------------------------------------------
print("=== MASKED BATCH (speech regions only) ===")

# mask shape: (batch_size, num_samples) — 1.0 = speech, 0.0 = silence
mask = torch.zeros(batch_size, 2 * SAMPLE_RATE)
mask[:, 3000:20000] = 1.0  # pretend speech is in this range

masked_embeddings = model(batch, masks=mask)
nan_rows = np.isnan(masked_embeddings[:, 0])
print(f"Embeddings with NaN (too-short masked region): {nan_rows.sum()} of {batch_size}")
print(f"Valid embeddings shape: {masked_embeddings[~nan_rows].shape}\n")


# ------------------------------------------------------------------
# 7. Device switching — move model to GPU if available
# ------------------------------------------------------------------
print("=== DEVICE SWITCHING ===")
print(f"Current device: {model.device}")

if torch.cuda.is_available():
    model.to(torch.device("cuda"))
    print(f"Moved to GPU: {model.device}")
    gpu_emb = model(torch.randn(1, 1, SAMPLE_RATE))
    print(f"GPU inference shape: {gpu_emb.shape}")
    model.to(torch.device("cpu"))   # move back
else:
    print("No GPU available — skipping device switch demo")


# ------------------------------------------------------------------
# 8. Edge case — clip shorter than min_num_samples
# ------------------------------------------------------------------
print("\n=== EDGE CASE: Very Short Clip ===")
min_samples = model.min_num_samples
print(f"Model minimum samples: {min_samples} ({min_samples/SAMPLE_RATE*1000:.1f} ms)")
too_short = torch.randn(1, 1, max(1, min_samples - 10))
if store.is_long_enough(too_short):
    print("Clip is long enough — this shouldn't happen in the test")
else:
    print("Too short → safely skipped (no RuntimeError!)")
    print("In a real app you would return NaN or show a friendly message like 'Clip too short for verification'.")