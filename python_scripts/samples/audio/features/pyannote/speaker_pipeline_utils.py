"""
speaker_pipeline_utils.py
==========================
Reusable utility functions for the high-level SpeakerEmbedding pipeline
from pyannote.audio.

Complements ``speaker_verification_utils.py`` (low-level model + embedding
helpers). This module focuses on the Pipeline abstraction, which handles
file loading, resampling, and optional VAD internally.

Three responsibility layers:
  1. Pipeline construction  — ``create_pipeline``
  2. Audio preparation      — ``make_audio_file``
  3. Embedding extraction   — ``extract_pipeline_embedding``
  4. Speaker enrolment      — ``enroll_speakers``
  5. Speaker identification — ``identify_speaker``, ``rank_speakers``

Typical usage
-------------
    from speaker_pipeline_utils import (
        create_pipeline,
        make_audio_file,
        enroll_speakers,
        identify_speaker,
    )

    pipeline = create_pipeline("pyannote/embedding")

    known = {
        "Alice": make_audio_file(alice_waveform, 16000),
        "Bob":   make_audio_file(bob_waveform,   16000),
    }
    db = enroll_speakers(pipeline, known)

    unknown_audio = make_audio_file(unknown_waveform, 16000)
    name, dist = identify_speaker(pipeline, unknown_audio, db)
    print(f"Best match: {name}  (distance={dist:.4f})")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A voice embedding: numpy array of shape (1, D).
Embedding = np.ndarray

# The dict format accepted by pyannote pipelines as an "audio file".
AudioFile = dict[str, torch.Tensor | int]

# A mapping of speaker name → stored embedding (the "speaker database").
SpeakerDatabase = dict[str, Embedding]


# ---------------------------------------------------------------------------
# 1. Pipeline construction
# ---------------------------------------------------------------------------

def create_pipeline(
    embedding: str = "pyannote/embedding",
    segmentation: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> SpeakerEmbedding:
    """
    Construct and return a ``SpeakerEmbedding`` pipeline.

    Parameters
    ----------
    embedding:
        Hugging Face model ID for the speaker embedding model.
        Defaults to ``"pyannote/embedding"``.
    segmentation:
        Hugging Face model ID for the VAD / segmentation model.
        Pass ``None`` (default) to embed the whole audio without silence removal.
        Pass ``"pyannote/segmentation"`` to activate Voice Activity Detection —
        recommended for real-world recordings with pauses or background noise.
    token:
        Hugging Face access token for gated models. When ``None``, the
        environment variable ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) is
        used automatically if present.
    cache_dir:
        Local directory for caching downloaded model weights.
        When ``None``, the default Hugging Face cache location is used.

    Returns
    -------
    SpeakerEmbedding
        Configured pipeline instance ready to call with an audio file.

    Examples
    --------
    Simple pipeline (no VAD):

    >>> pipeline = create_pipeline("pyannote/embedding")

    Pipeline with Voice Activity Detection:

    >>> pipeline = create_pipeline(
    ...     "pyannote/embedding",
    ...     segmentation="pyannote/segmentation",
    ...     token="hf_your_token_here",
    ... )
    """
    kwargs: dict = {"embedding": embedding, "segmentation": segmentation}

    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    return SpeakerEmbedding(**kwargs)


# ---------------------------------------------------------------------------
# 2. Audio preparation
# ---------------------------------------------------------------------------

def make_audio_file(
    waveform: torch.Tensor,
    sample_rate: int,
) -> AudioFile:
    """
    Build the ``AudioFile`` dict expected by pyannote pipelines.

    pyannote pipelines accept several input formats (file path, Path object,
    or a dict with a pre-loaded waveform). This function always produces the
    in-memory dict form, which avoids redundant disk I/O when the waveform
    is already in memory.

    Parameters
    ----------
    waveform:
        Audio tensor.  Accepted shapes:

        * ``(samples,)``         — mono, no batch/channel dims
        * ``(channels, samples)``— multi-channel, no batch dim
        * ``(1, channels, samples)`` — batched (batch dim is squeezed out)

        The tensor is stored as-is; pyannote handles shape normalisation
        internally.
    sample_rate:
        Sample rate in Hz (e.g. ``16000``).

    Returns
    -------
    AudioFile
        ``{"waveform": waveform, "sample_rate": sample_rate}``

    Example
    -------
    >>> audio = make_audio_file(torch.randn(1, 16000 * 3), sample_rate=16000)
    >>> emb = pipeline(audio)
    """
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# 3. Embedding extraction
# ---------------------------------------------------------------------------

def extract_pipeline_embedding(
    pipeline: SpeakerEmbedding,
    audio_file: AudioFile | str,
) -> Embedding:
    """
    Extract a speaker embedding from an audio file using the pipeline.

    The pipeline internally handles file loading, resampling, optional VAD
    masking, and model inference.

    Parameters
    ----------
    pipeline:
        A ``SpeakerEmbedding`` pipeline created with :func:`create_pipeline`.
    audio_file:
        Either a path string / ``Path`` pointing to an audio file on disk,
        or an ``AudioFile`` dict built with :func:`make_audio_file`.

    Returns
    -------
    Embedding
        Numpy array of shape ``(1, D)`` where *D* is the model's embedding
        dimension.

    Example
    -------
    >>> audio = make_audio_file(waveform, sample_rate=16000)
    >>> emb = extract_pipeline_embedding(pipeline, audio)
    >>> emb.shape
    (1, 512)
    """
    return pipeline(audio_file)


# ---------------------------------------------------------------------------
# 4. Speaker enrolment
# ---------------------------------------------------------------------------

def enroll_speakers(
    pipeline: SpeakerEmbedding,
    speaker_audios: dict[str, AudioFile | str],
) -> SpeakerDatabase:
    """
    Extract and store embeddings for a collection of known speakers.

    Iterates over a name → audio mapping, runs the pipeline on each clip,
    and returns a name → embedding mapping that can be used as a speaker
    database for identification.

    Parameters
    ----------
    pipeline:
        A ``SpeakerEmbedding`` pipeline created with :func:`create_pipeline`.
    speaker_audios:
        Mapping of speaker name to audio input.  Each value may be a path
        string or an ``AudioFile`` dict (from :func:`make_audio_file`).

    Returns
    -------
    SpeakerDatabase
        ``{speaker_name: embedding}`` where each embedding is shaped ``(1, D)``.

    Example
    -------
    >>> known = {
    ...     "Alice": make_audio_file(alice_wav, 16000),
    ...     "Bob":   make_audio_file(bob_wav,   16000),
    ... }
    >>> db = enroll_speakers(pipeline, known)
    >>> list(db.keys())
    ['Alice', 'Bob']
    """
    return {
        name: extract_pipeline_embedding(pipeline, audio)
        for name, audio in speaker_audios.items()
    }


# ---------------------------------------------------------------------------
# 5. Speaker identification
# ---------------------------------------------------------------------------

def identify_speaker(
    pipeline: SpeakerEmbedding,
    audio_file: AudioFile | str,
    speaker_db: SpeakerDatabase,
) -> tuple[str, float]:
    """
    Identify the closest matching speaker in the database for an unknown clip.

    Extracts an embedding from ``audio_file``, then finds the enrolled speaker
    whose embedding has the smallest cosine distance to it.

    Parameters
    ----------
    pipeline:
        A ``SpeakerEmbedding`` pipeline created with :func:`create_pipeline`.
    audio_file:
        The unknown speaker's audio — path string or ``AudioFile`` dict.
    speaker_db:
        A speaker database produced by :func:`enroll_speakers`.

    Returns
    -------
    best_name : str
        Name of the closest-matching enrolled speaker.
    best_distance : float
        Cosine distance to that speaker (lower = more similar; range 0–2).

    Raises
    ------
    ValueError
        If ``speaker_db`` is empty.

    Example
    -------
    >>> name, dist = identify_speaker(pipeline, unknown_audio, db)
    >>> print(f"Best match: {name}  (distance={dist:.4f})")
    Best match: Alice  (distance=0.3142)
    """
    if not speaker_db:
        raise ValueError("speaker_db is empty — enrol at least one speaker first.")

    unknown_emb = extract_pipeline_embedding(pipeline, audio_file)

    best_name = min(
        speaker_db,
        key=lambda name: float(cdist(unknown_emb, speaker_db[name], metric="cosine")[0, 0]),
    )
    best_distance = float(cdist(unknown_emb, speaker_db[best_name], metric="cosine")[0, 0])
    return best_name, best_distance


def rank_speakers(
    pipeline: SpeakerEmbedding,
    audio_file: AudioFile | str,
    speaker_db: SpeakerDatabase,
) -> list[tuple[str, float]]:
    """
    Rank all enrolled speakers by cosine distance to an unknown audio clip.

    Unlike :func:`identify_speaker`, which returns only the top-1 match,
    this function returns the full ranked list — useful for top-N retrieval,
    thresholding with rejection, or debugging enrolment quality.

    Parameters
    ----------
    pipeline:
        A ``SpeakerEmbedding`` pipeline created with :func:`create_pipeline`.
    audio_file:
        The unknown speaker's audio — path string or ``AudioFile`` dict.
    speaker_db:
        A speaker database produced by :func:`enroll_speakers`.

    Returns
    -------
    list[tuple[str, float]]
        List of ``(speaker_name, cosine_distance)`` tuples sorted by distance
        in ascending order (closest match first).

    Raises
    ------
    ValueError
        If ``speaker_db`` is empty.

    Example
    -------
    >>> rankings = rank_speakers(pipeline, unknown_audio, db)
    >>> for name, dist in rankings:
    ...     print(f"  {name}: {dist:.4f}")
    Alice: 0.3142
    Carol: 0.6891
    Bob:   0.9204
    """
    if not speaker_db:
        raise ValueError("speaker_db is empty — enrol at least one speaker first.")

    unknown_emb = extract_pipeline_embedding(pipeline, audio_file)

    distances = [
        (name, float(cdist(unknown_emb, emb, metric="cosine")[0, 0]))
        for name, emb in speaker_db.items()
    ]
    return sorted(distances, key=lambda pair: pair[1])
