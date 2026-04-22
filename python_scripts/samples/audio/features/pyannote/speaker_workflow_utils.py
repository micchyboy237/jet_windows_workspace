"""
speaker_workflow_utils.py
==========================
Reusable utility functions and the ``SpeakerStore`` dataclass for a
full speaker-verification workflow.

Complements the two earlier modules:
  - ``speaker_verification_utils.py``  — low-level model + embedding helpers
  - ``speaker_pipeline_utils.py``      — high-level SpeakerEmbedding pipeline

This module adds:
  1. Validation / safety checks     — length guard, NaN guard
  2. Safe & batch embedding helpers — masked inference, NaN filtering
  3. Pairwise distance matrix       — for batch analysis / diarisation
  4. Device management              — flexible CPU ↔ GPU switching
  5. SpeakerStore                   — enrolment, verification, identification

Typical usage
-------------
    from speaker_workflow_utils import SpeakerStore, move_model_to_device
    from speaker_verification_utils import load_embedding_model

    model = load_embedding_model(device="cuda")
    store = SpeakerStore(model=model, threshold=0.7)

    store.enroll("alice", alice_waveform)
    match, dist = store.verify("alice", test_waveform)
    name,  dist = store.identify(unknown_waveform)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch

from scipy.spatial.distance import cdist
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A voice embedding numpy array of shape (batch, D).
Embedding = np.ndarray

# Accepted device specifications.
DeviceLike = Union[str, torch.device]


# ---------------------------------------------------------------------------
# 1. Validation / safety checks
# ---------------------------------------------------------------------------

def is_long_enough(
    waveform: torch.Tensor,
    model: PretrainedSpeakerEmbedding,
) -> bool:
    """
    Return ``True`` if ``waveform`` meets the model's minimum sample requirement.

    Passing a clip shorter than ``model.min_num_samples`` to the model raises
    a ``RuntimeError`` (kernel size exceeds tensor size). Call this guard
    before any inference to avoid that crash.

    Parameters
    ----------
    waveform:
        Audio tensor of any shape; the last dimension is treated as samples.
    model:
        Loaded ``PretrainedSpeakerEmbedding`` whose ``min_num_samples``
        attribute defines the lower bound.

    Returns
    -------
    bool
        ``True`` when ``waveform.shape[-1] >= model.min_num_samples``.

    Example
    -------
    >>> if not is_long_enough(waveform, model):
    ...     print("Clip too short — skip or pad before embedding.")
    """
    return waveform.shape[-1] >= model.min_num_samples


def is_valid_embedding(embedding: Embedding) -> bool:
    """
    Return ``True`` if ``embedding`` contains no NaN values.

    The model produces NaN embeddings when the (optionally masked) audio
    region is too short or completely silent. Checking this before storing
    or comparing embeddings prevents silent propagation of invalid data.

    Parameters
    ----------
    embedding:
        Numpy array of any shape.

    Returns
    -------
    bool
        ``True`` when no element of ``embedding`` is NaN.

    Example
    -------
    >>> emb = model(waveform)
    >>> if not is_valid_embedding(emb):
    ...     raise ValueError("Embedding is invalid — clip may be silent or too short.")
    """
    return not bool(np.any(np.isnan(embedding)))


# ---------------------------------------------------------------------------
# 2. Safe & batch embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding_safe(
    model: PretrainedSpeakerEmbedding,
    waveform: torch.Tensor,
) -> Optional[Embedding]:
    """
    Extract a speaker embedding with length and NaN guards.

    Combines :func:`is_long_enough` and :func:`is_valid_embedding` into a
    single safe extraction call. Returns ``None`` instead of raising an
    exception when the clip is too short or produces a NaN embedding.

    Parameters
    ----------
    model:
        Loaded ``PretrainedSpeakerEmbedding`` instance.
    waveform:
        Audio tensor of shape ``(1, channels, samples)``.

    Returns
    -------
    Embedding or None
        Numpy array of shape ``(1, D)`` on success, ``None`` on failure.

    Example
    -------
    >>> emb = extract_embedding_safe(model, waveform)
    >>> if emb is None:
    ...     print("Could not embed clip — too short or silent.")
    """
    if not is_long_enough(waveform, model):
        return None
    emb = model(waveform)
    return emb if is_valid_embedding(emb) else None


def extract_batch_embeddings(
    model: PretrainedSpeakerEmbedding,
    batch: torch.Tensor,
) -> Embedding:
    """
    Extract embeddings for a stacked batch of audio clips in one forward pass.

    Batching clips together is more efficient than calling the model
    individually for each clip, especially on GPU.

    Parameters
    ----------
    model:
        Loaded ``PretrainedSpeakerEmbedding`` instance.
    batch:
        Stacked audio tensor of shape ``(N, channels, samples)`` where *N*
        is the number of clips.

    Returns
    -------
    Embedding
        Numpy array of shape ``(N, D)`` — one row per input clip.

    Example
    -------
    >>> batch = torch.stack([clip1, clip2, clip3], dim=0)  # (3, 1, T)
    >>> embeddings = extract_batch_embeddings(model, batch)
    >>> embeddings.shape
    (3, 512)
    """
    return model(batch)


def extract_masked_embeddings(
    model: PretrainedSpeakerEmbedding,
    batch: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[Embedding, np.ndarray]:
    """
    Extract embeddings for speech-only regions using a binary mask.

    The mask tells the model which samples contain speech (1.0) and which
    are silence or noise (0.0). Rows that produce NaN embeddings (masked
    region too short) are identified and returned as a boolean index so
    callers can filter them out.

    Parameters
    ----------
    model:
        Loaded ``PretrainedSpeakerEmbedding`` instance.
    batch:
        Audio tensor of shape ``(N, channels, samples)``.
    mask:
        Binary float tensor of shape ``(N, samples)`` where ``1.0`` marks
        speech frames and ``0.0`` marks non-speech frames.

    Returns
    -------
    embeddings : Embedding
        Full output array of shape ``(N, D)``, may contain NaN rows.
    valid_mask : np.ndarray
        Boolean array of shape ``(N,)`` — ``True`` where the embedding is
        valid (no NaN), ``False`` where it is NaN and should be discarded.

    Example
    -------
    >>> mask = torch.zeros(4, T); mask[:, 3000:20000] = 1.0
    >>> embeddings, valid = extract_masked_embeddings(model, batch, mask)
    >>> clean_embeddings = embeddings[valid]  # discard NaN rows
    """
    embeddings = model(batch, masks=mask)
    valid_mask = ~np.isnan(embeddings[:, 0])
    return embeddings, valid_mask


# ---------------------------------------------------------------------------
# 3. Pairwise distance matrix
# ---------------------------------------------------------------------------

def compute_pairwise_distances(embeddings: Embedding) -> np.ndarray:
    """
    Compute an N×N cosine distance matrix for a batch of embeddings.

    Useful for batch analysis, clustering, or visualising speaker similarity
    across many clips at once.

    Parameters
    ----------
    embeddings:
        Numpy array of shape ``(N, D)`` — typically the output of
        :func:`extract_batch_embeddings`.

    Returns
    -------
    np.ndarray
        Square distance matrix of shape ``(N, N)`` where entry ``[i, j]``
        is the cosine distance between clips *i* and *j*. The diagonal is
        always 0.

    Example
    -------
    >>> matrix = compute_pairwise_distances(embeddings)
    >>> print(np.round(matrix, 3))
    [[0.    0.312 0.891]
     [0.312 0.    0.754]
     [0.891 0.754 0.   ]]
    """
    return cdist(embeddings, embeddings, metric="cosine")


# ---------------------------------------------------------------------------
# 4. Device management
# ---------------------------------------------------------------------------

def move_model_to_device(
    model: PretrainedSpeakerEmbedding,
    device: DeviceLike,
) -> PretrainedSpeakerEmbedding:
    """
    Move a ``PretrainedSpeakerEmbedding`` model to the specified device.

    Accepts either a plain string (``"cpu"``, ``"cuda"``, ``"mps"``) or a
    ``torch.device`` object, normalising the input before calling
    ``model.to()``.

    Parameters
    ----------
    model:
        The embedding model to move.
    device:
        Target device — either a ``torch.device`` or a device string such
        as ``"cpu"``, ``"cuda"``, or ``"cuda:1"``.

    Returns
    -------
    PretrainedSpeakerEmbedding
        The same model instance after being moved (allows chaining).

    Example
    -------
    >>> model = move_model_to_device(model, "cuda")
    >>> emb = model(waveform.to("cuda"))

    >>> # Move back to CPU after GPU inference
    >>> model = move_model_to_device(model, "cpu")
    """
    target = device if isinstance(device, torch.device) else torch.device(device)
    model.to(target)
    return model


def get_best_available_device() -> torch.device:
    """
    Return the best available PyTorch device on the current machine.

    Priority order: CUDA GPU → Apple MPS → CPU.

    Returns
    -------
    torch.device
        ``torch.device("cuda")``, ``torch.device("mps")``, or
        ``torch.device("cpu")``.

    Example
    -------
    >>> device = get_best_available_device()
    >>> model = load_embedding_model(device=str(device))
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 5. SpeakerStore — enrolment, verification, identification
# ---------------------------------------------------------------------------

@dataclass
class SpeakerStore:
    """
    In-memory store for speaker enrolment, verification, and identification.

    Wraps a ``PretrainedSpeakerEmbedding`` model and a dict of enrolled
    speaker embeddings. All safety checks (length guard, NaN guard) are
    applied internally so callers do not need to repeat them.

    Attributes
    ----------
    model:
        Loaded embedding model used for all inference calls.
    threshold:
        Cosine distance below which a clip is accepted as a match.
        Defaults to ``0.7``; tune per model and operating point.
    embeddings:
        Internal store of ``{speaker_id: embedding}`` populated by
        :meth:`enroll`.

    Example
    -------
    >>> store = SpeakerStore(model=model, threshold=0.65)
    >>> store.enroll("alice", alice_waveform)
    >>> match, dist = store.verify("alice", test_waveform)
    >>> name,  dist = store.identify(unknown_waveform)
    """

    model: PretrainedSpeakerEmbedding
    threshold: float = 0.7
    embeddings: dict[str, Embedding] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enrolled_speakers(self) -> list[str]:
        """Return the list of currently enrolled speaker IDs."""
        return list(self.embeddings.keys())

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when no speakers have been enrolled yet."""
        return len(self.embeddings) == 0

    # ------------------------------------------------------------------
    # Enrolment
    # ------------------------------------------------------------------

    def enroll(self, speaker_id: str, waveform: torch.Tensor) -> None:
        """
        Extract and store an embedding for a known speaker.

        Parameters
        ----------
        speaker_id:
            Unique identifier for the speaker (e.g. ``"alice"``).
        waveform:
            Audio tensor of shape ``(1, 1, samples)``.

        Raises
        ------
        ValueError
            If the waveform is too short or produces a NaN embedding,
            indicating the clip cannot be used for enrolment.

        Example
        -------
        >>> store.enroll("alice", alice_waveform)
        """
        emb = extract_embedding_safe(self.model, waveform)
        if emb is None:
            raise ValueError(
                f"Could not enrol '{speaker_id}' — clip may be too short or silent. "
                f"Minimum length: {self.model.min_num_samples} samples "
                f"({self.model.min_num_samples / self.model.sample_rate:.3f} s)."
            )
        self.embeddings[speaker_id] = emb

    def remove(self, speaker_id: str) -> None:
        """
        Remove an enrolled speaker from the store.

        Parameters
        ----------
        speaker_id:
            ID of the speaker to remove.

        Raises
        ------
        KeyError
            If ``speaker_id`` is not in the store.

        Example
        -------
        >>> store.remove("alice")
        """
        if speaker_id not in self.embeddings:
            raise KeyError(f"Speaker '{speaker_id}' is not enrolled.")
        del self.embeddings[speaker_id]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        speaker_id: str,
        waveform: torch.Tensor,
    ) -> tuple[bool, float]:
        """
        Check whether a waveform matches an enrolled speaker.

        Parameters
        ----------
        speaker_id:
            ID of the enrolled speaker to verify against.
        waveform:
            Audio tensor of shape ``(1, 1, samples)``.

        Returns
        -------
        is_match : bool
            ``True`` if the cosine distance is below ``self.threshold``.
            Always ``False`` when the clip is too short or silent.
        distance : float
            Raw cosine distance. Returns ``float("nan")`` when the clip
            cannot be embedded.

        Raises
        ------
        KeyError
            If ``speaker_id`` has not been enrolled.

        Example
        -------
        >>> match, dist = store.verify("alice", test_waveform)
        >>> print(f"{'ACCEPT' if match else 'REJECT'}  dist={dist:.4f}")
        """
        if speaker_id not in self.embeddings:
            raise KeyError(f"Speaker '{speaker_id}' is not enrolled.")

        emb = extract_embedding_safe(self.model, waveform)
        if emb is None:
            return False, float("nan")

        dist = float(cdist(emb, self.embeddings[speaker_id], metric="cosine")[0, 0])
        return dist < self.threshold, dist

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(
        self,
        waveform: torch.Tensor,
    ) -> tuple[Optional[str], float]:
        """
        Find the closest enrolled speaker for an unknown clip.

        Applies ``self.threshold`` as a rejection gate: if the closest
        match exceeds the threshold, ``None`` is returned instead of a
        name, indicating an unknown speaker.

        Parameters
        ----------
        waveform:
            Audio tensor of shape ``(1, 1, samples)``.

        Returns
        -------
        speaker_id : str or None
            Name of the best-matching enrolled speaker, or ``None`` if the
            store is empty, the clip is invalid, or no match passes the
            threshold.
        distance : float
            Cosine distance to the best match. Returns ``float("nan")``
            when the clip cannot be embedded or the store is empty.

        Example
        -------
        >>> name, dist = store.identify(unknown_waveform)
        >>> print(name or "UNKNOWN", f"dist={dist:.4f}")
        """
        if self.is_empty:
            return None, float("nan")

        emb = extract_embedding_safe(self.model, waveform)
        if emb is None:
            return None, float("nan")

        distances = {
            name: float(cdist(emb, stored, metric="cosine")[0, 0])
            for name, stored in self.embeddings.items()
        }
        best_name = min(distances, key=distances.__getitem__)
        best_dist = distances[best_name]

        if best_dist > self.threshold:
            return None, best_dist
        return best_name, best_dist

    def rank(
        self,
        waveform: torch.Tensor,
    ) -> list[tuple[str, float]]:
        """
        Rank all enrolled speakers by cosine distance to a waveform.

        Unlike :meth:`identify`, no threshold is applied — every enrolled
        speaker is returned, sorted closest-first. Useful for top-N
        retrieval or auditing enrolment quality.

        Parameters
        ----------
        waveform:
            Audio tensor of shape ``(1, 1, samples)``.

        Returns
        -------
        list[tuple[str, float]]
            List of ``(speaker_id, distance)`` sorted ascending by distance.
            Returns an empty list when the store is empty or the clip is
            invalid.

        Example
        -------
        >>> for name, dist in store.rank(unknown_waveform):
        ...     print(f"  {name}: {dist:.4f}")
        """
        if self.is_empty:
            return []

        emb = extract_embedding_safe(self.model, waveform)
        if emb is None:
            return []

        distances = [
            (name, float(cdist(emb, stored, metric="cosine")[0, 0]))
            for name, stored in self.embeddings.items()
        ]
        return sorted(distances, key=lambda pair: pair[1])
