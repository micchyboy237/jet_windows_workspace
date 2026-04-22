"""
speaker_verification_utils.py
==============================
Reusable utility functions for speaker verification using pyannote.audio.

Responsibilities are split into focused, single-purpose functions:
  - Model loading
  - Audio I/O
  - Embedding extraction
  - Distance computation
  - Verification decision
  - High-level composed verification

Usage example:
    from speaker_verification_utils import load_embedding_model, load_audio_tensor, verify_speakers

    model = load_embedding_model()
    wav1, _ = load_audio_tensor("speaker_a.wav")
    wav2, _ = load_audio_tensor("speaker_b.wav")
    same, distance = verify_speakers(model, wav1, wav2, threshold=0.7)
    print(f"Same speaker: {same}  (distance={distance:.4f})")
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A voice embedding: numpy array of shape (1, D) where D is the model dimension.
Embedding = np.ndarray

# Accepted path types for audio files.
AudioPath = Union[str, Path]


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str = "pyannote/embedding",
    device: str = "cpu",
) -> PretrainedSpeakerEmbedding:
    """
    Load and return a PretrainedSpeakerEmbedding model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier or local path.
        Defaults to ``"pyannote/embedding"``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``, etc.).
        Defaults to ``"cpu"``.

    Returns
    -------
    PretrainedSpeakerEmbedding
        Initialised embedding model ready for inference.

    Example
    -------
    >>> model = load_embedding_model("pyannote/embedding", device="cuda")
    """
    return PretrainedSpeakerEmbedding(
        model_name,
        device=torch.device(device),
    )


# ---------------------------------------------------------------------------
# 2. Audio I/O
# ---------------------------------------------------------------------------

def load_audio_tensor(
    path: AudioPath,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file from disk and return a batch-ready tensor.

    The returned tensor shape is ``(1, channels, samples)`` — the leading
    dimension is the batch axis expected by pyannote embedding models.

    Parameters
    ----------
    path:
        Path to the audio file (WAV, FLAC, MP3, etc.).
    target_sample_rate:
        If provided, the waveform is resampled to this rate after loading.
        Pass ``model.sample_rate`` to ensure compatibility with your model.

    Returns
    -------
    waveform : torch.Tensor
        Shape ``(1, C, T)`` — batch × channels × samples.
    sample_rate : int
        Sample rate of the returned waveform (after optional resampling).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.

    Example
    -------
    >>> waveform, sr = load_audio_tensor("recording.wav", target_sample_rate=16000)
    >>> waveform.shape
    torch.Size([1, 1, 32000])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = torchaudio.load(str(path))  # shape: (C, T)

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    waveform = waveform.unsqueeze(0)  # (C, T) → (1, C, T)
    return waveform, sample_rate


def make_dummy_audio(
    duration_seconds: float,
    sample_rate: int,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate a random (white-noise) audio tensor for testing and development.

    Parameters
    ----------
    duration_seconds:
        Length of the generated audio clip in seconds.
    sample_rate:
        Sample rate in Hz (should match your model's expected rate).
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 1, samples)`` — batch × mono-channel × samples.

    Example
    -------
    >>> audio = make_dummy_audio(2.0, sample_rate=16000, seed=42)
    >>> audio.shape
    torch.Size([1, 1, 32000])
    """
    if seed is not None:
        torch.manual_seed(seed)
    num_samples = int(duration_seconds * sample_rate)
    return torch.randn(1, 1, num_samples)


# ---------------------------------------------------------------------------
# 3. Embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding(
    model: PretrainedSpeakerEmbedding,
    waveform: torch.Tensor,
) -> Embedding:
    """
    Extract a speaker embedding (voice fingerprint) from an audio tensor.

    Parameters
    ----------
    model:
        A loaded ``PretrainedSpeakerEmbedding`` instance.
    waveform:
        Audio tensor of shape ``(batch, channels, samples)``.
        Batch size is typically 1 for single-clip verification.

    Returns
    -------
    Embedding
        Numpy array of shape ``(1, D)`` where *D* is the model's embedding
        dimension (e.g. 512 for ``pyannote/embedding``).

    Example
    -------
    >>> emb = extract_embedding(model, waveform)
    >>> emb.shape
    (1, 512)
    """
    return model(waveform)


def extract_embeddings_batch(
    model: PretrainedSpeakerEmbedding,
    waveforms: list[torch.Tensor],
) -> list[Embedding]:
    """
    Extract embeddings for a list of waveforms, one at a time.

    This is a convenience wrapper around :func:`extract_embedding` for
    processing multiple clips without manual looping at the call site.

    Parameters
    ----------
    model:
        A loaded ``PretrainedSpeakerEmbedding`` instance.
    waveforms:
        List of audio tensors, each shaped ``(1, channels, samples)``.

    Returns
    -------
    list[Embedding]
        List of numpy arrays, each shaped ``(1, D)``, in the same order
        as the input waveforms.

    Example
    -------
    >>> embeddings = extract_embeddings_batch(model, [wav_a, wav_b, wav_c])
    """
    return [extract_embedding(model, waveform) for waveform in waveforms]


# ---------------------------------------------------------------------------
# 4. Distance computation
# ---------------------------------------------------------------------------

def compute_cosine_distance(emb1: Embedding, emb2: Embedding) -> float:
    """
    Compute the cosine distance between two speaker embeddings.

    Cosine distance ranges from **0** (identical direction) to **2**
    (opposite direction). A typical same-speaker threshold is 0.5–0.7,
    but the optimal value depends on the model and use-case.

    Parameters
    ----------
    emb1:
        First embedding, shape ``(1, D)``.
    emb2:
        Second embedding, shape ``(1, D)``.

    Returns
    -------
    float
        Scalar cosine distance in ``[0, 2]``.

    Example
    -------
    >>> dist = compute_cosine_distance(emb_a, emb_b)
    >>> print(f"{dist:.4f}")
    0.3271
    """
    return float(cdist(emb1, emb2, metric="cosine")[0, 0])


def compute_distance_matrix(
    embeddings_a: list[Embedding],
    embeddings_b: list[Embedding],
) -> np.ndarray:
    """
    Compute an all-pairs cosine distance matrix between two sets of embeddings.

    Useful for comparing one set of enrolment clips against one set of
    query clips in a batch evaluation scenario.

    Parameters
    ----------
    embeddings_a:
        List of *N* embeddings, each shaped ``(1, D)``.
    embeddings_b:
        List of *M* embeddings, each shaped ``(1, D)``.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(N, M)`` where entry ``[i, j]`` is the cosine
        distance between ``embeddings_a[i]`` and ``embeddings_b[j]``.

    Example
    -------
    >>> matrix = compute_distance_matrix([emb_a1, emb_a2], [emb_b1, emb_b2])
    >>> matrix.shape
    (2, 2)
    """
    stacked_a = np.vstack(embeddings_a)  # (N, D)
    stacked_b = np.vstack(embeddings_b)  # (M, D)
    return cdist(stacked_a, stacked_b, metric="cosine")


# ---------------------------------------------------------------------------
# 5. Verification decision
# ---------------------------------------------------------------------------

def is_same_speaker(
    emb1: Embedding,
    emb2: Embedding,
    threshold: float = 0.7,
) -> bool:
    """
    Decide whether two embeddings belong to the same speaker.

    Parameters
    ----------
    emb1:
        First speaker embedding, shape ``(1, D)``.
    emb2:
        Second speaker embedding, shape ``(1, D)``.
    threshold:
        Cosine distance threshold below which speakers are considered the same.
        Defaults to ``0.7``; tune this per model and operating point.

    Returns
    -------
    bool
        ``True`` if the cosine distance is **strictly less than** ``threshold``.

    Example
    -------
    >>> same = is_same_speaker(emb_a, emb_b, threshold=0.65)
    """
    return compute_cosine_distance(emb1, emb2) < threshold


# ---------------------------------------------------------------------------
# 6. High-level composed verification
# ---------------------------------------------------------------------------

def verify_speakers(
    model: PretrainedSpeakerEmbedding,
    waveform1: torch.Tensor,
    waveform2: torch.Tensor,
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """
    End-to-end speaker verification: extract embeddings and compare.

    This is a convenience function that composes :func:`extract_embedding`,
    :func:`compute_cosine_distance`, and :func:`is_same_speaker` into a
    single call.

    Parameters
    ----------
    model:
        A loaded ``PretrainedSpeakerEmbedding`` instance.
    waveform1:
        Audio tensor for the first speaker, shape ``(1, C, T)``.
    waveform2:
        Audio tensor for the second speaker, shape ``(1, C, T)``.
    threshold:
        Cosine distance threshold for the same-speaker decision.
        Defaults to ``0.7``.

    Returns
    -------
    same_speaker : bool
        ``True`` if the two clips are judged to be the same speaker.
    distance : float
        Raw cosine distance between the two embeddings.

    Example
    -------
    >>> model = load_embedding_model()
    >>> wav1, _ = load_audio_tensor("enrol.wav", target_sample_rate=model.sample_rate)
    >>> wav2, _ = load_audio_tensor("query.wav", target_sample_rate=model.sample_rate)
    >>> same, dist = verify_speakers(model, wav1, wav2, threshold=0.65)
    >>> print(f"Same: {same}, Distance: {dist:.4f}")
    """
    emb1 = extract_embedding(model, waveform1)
    emb2 = extract_embedding(model, waveform2)
    distance = compute_cosine_distance(emb1, emb2)
    return distance < threshold, distance
