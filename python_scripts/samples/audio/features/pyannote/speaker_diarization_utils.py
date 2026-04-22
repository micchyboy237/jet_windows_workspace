"""
speaker_diarization_utils.py
=============================
Reusable utility functions for speaker diarisation using pyannote.audio.

Complements the three earlier modules:
  - ``speaker_verification_utils.py``  — low-level embedding + distance helpers
  - ``speaker_pipeline_utils.py``      — high-level SpeakerEmbedding pipeline
  - ``speaker_workflow_utils.py``      — SpeakerStore, safe extraction, device utils

This module focuses exclusively on the diarisation pipeline:
  1. Pipeline loading     — ``load_diarization_pipeline``
  2. Audio I/O            — ``load_audio_as_tensor``
  3. Audio preparation    — ``make_preloaded_audio``  (thin re-export pattern)
  4. Inference            — ``run_diarization``
  5. Result parsing       — ``parse_diarization_output``, ``print_diarization_output``

Typical usage
-------------
    from speaker_diarization_utils import (
        load_diarization_pipeline,
        load_audio_as_tensor,
        make_preloaded_audio,
        run_diarization,
        parse_diarization_output,
        print_diarization_output,
    )

    pipeline = load_diarization_pipeline(
        model_name="pyannote/speaker-diarization-community-1",
        device="cuda",
    )
    waveform, sr = load_audio_as_tensor("recording.wav")
    audio        = make_preloaded_audio(waveform, sr)
    diarization  = run_diarization(pipeline, audio)
    turns        = parse_diarization_output(diarization)
    print_diarization_output(turns)
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional, Union

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Accepted device specifications.
DeviceLike = Union[str, torch.device]

# The dict format accepted by pyannote pipelines as a pre-loaded audio file.
AudioFile = dict[str, Union[torch.Tensor, int]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DiarizationTurn(NamedTuple):
    """
    A single speaker turn from a diarisation result.

    Attributes
    ----------
    start:
        Turn start time in seconds.
    end:
        Turn end time in seconds.
    speaker:
        Speaker label as returned by the pipeline (e.g. ``"SPEAKER_00"``).
    """

    start: float
    end: float
    speaker: str


# ---------------------------------------------------------------------------
# 1. Pipeline loading
# ---------------------------------------------------------------------------

def load_diarization_pipeline(
    model_name: str = "pyannote/speaker-diarization-community-1",
    token: Optional[str] = None,
    device: DeviceLike = "cpu",
) -> Pipeline:
    """
    Load a pyannote diarisation pipeline and move it to the target device.

    The Hugging Face access token is resolved in this priority order:

    1. The ``token`` parameter (if explicitly provided).
    2. The ``HF_TOKEN`` environment variable (fallback).
    3. ``None`` — works only for public / already-cached models.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier for the diarisation pipeline.
        Defaults to ``"pyannote/speaker-diarization-community-1"``.
    token:
        Hugging Face access token. When ``None``, the environment variable
        ``HF_TOKEN`` is read automatically.
    device:
        Target device for inference — a ``torch.device`` or a device string
        such as ``"cpu"``, ``"cuda"``, or ``"mps"``.
        Defaults to ``"cpu"``.

    Returns
    -------
    Pipeline
        Loaded and device-placed pyannote ``Pipeline`` ready for inference.

    Raises
    ------
    EnvironmentError
        If the model is gated and no valid token is available.

    Example
    -------
    >>> pipeline = load_diarization_pipeline(device="cuda")
    >>> pipeline = load_diarization_pipeline(
    ...     "pyannote/speaker-diarization-3.1",
    ...     token="hf_your_token",
    ...     device="cuda",
    ... )
    """
    resolved_token: Optional[str] = token or os.getenv("HF_TOKEN")

    pipeline = Pipeline.from_pretrained(
        model_name,
        token=resolved_token,
    )

    target_device = device if isinstance(device, torch.device) else torch.device(device)
    pipeline.to(target_device)

    return pipeline


# ---------------------------------------------------------------------------
# 2. Audio I/O
# ---------------------------------------------------------------------------

def load_audio_as_tensor(
    path: Union[str, os.PathLike],
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file from disk and return a channel-first ``torch.Tensor``.

    Uses ``soundfile`` for reading, which supports WAV, FLAC, OGG, and more.
    The resulting tensor is shaped ``(channels, time)`` — the format expected
    by pyannote pipelines.

    Parameters
    ----------
    path:
        Path to the audio file on disk.

    Returns
    -------
    waveform : torch.Tensor
        Float32 tensor of shape ``(channels, time)``.
    sample_rate : int
        Sample rate of the audio file in Hz.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    RuntimeError
        If ``soundfile`` cannot read the file (unsupported format, corrupt
        file, etc.).

    Example
    -------
    >>> waveform, sr = load_audio_as_tensor("recording.wav")
    >>> waveform.shape        # e.g. torch.Size([1, 480000]) for 30 s mono
    torch.Size([1, 480000])
    >>> sr
    16000
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # sf.read with always_2d=True → shape (time, channels)
    waveform_np, sample_rate = sf.read(
        path,
        always_2d=True,
        dtype="float32",
    )

    # Transpose to (channels, time) as expected by pyannote
    waveform = torch.from_numpy(waveform_np.T)
    return waveform, int(sample_rate)


# ---------------------------------------------------------------------------
# 3. Audio preparation
# ---------------------------------------------------------------------------

def make_preloaded_audio(
    waveform: torch.Tensor,
    sample_rate: int,
) -> AudioFile:
    """
    Build the ``AudioFile`` dict accepted by pyannote pipelines.

    Passing a pre-loaded waveform dict to the pipeline avoids repeated disk
    reads when the same audio is processed multiple times (e.g. diarisation
    followed by per-speaker embedding extraction).

    Parameters
    ----------
    waveform:
        Float tensor of shape ``(channels, time)``.
    sample_rate:
        Sample rate in Hz.

    Returns
    -------
    AudioFile
        ``{"waveform": waveform, "sample_rate": sample_rate}``

    Example
    -------
    >>> waveform, sr = load_audio_as_tensor("recording.wav")
    >>> audio = make_preloaded_audio(waveform, sr)
    >>> diarization = pipeline(audio)
    """
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

def run_diarization(
    pipeline: Pipeline,
    audio: Union[AudioFile, str, os.PathLike],
    show_progress: bool = True,
) -> object:
    """
    Run a pyannote diarisation pipeline on an audio input.

    Wraps the ``ProgressHook`` context manager so callers do not need to
    manage it manually. Progress display can be toggled off for batch jobs
    or unit tests.

    Parameters
    ----------
    pipeline:
        A loaded pyannote ``Pipeline`` (from :func:`load_diarization_pipeline`).
    audio:
        Audio input accepted by pyannote:

        * An ``AudioFile`` dict built with :func:`make_preloaded_audio`.
        * A file path string or ``Path`` pointing to an audio file on disk.
    show_progress:
        When ``True`` (default), attach a ``ProgressHook`` that prints a
        progress bar to stdout. Set to ``False`` for silent / batch execution.

    Returns
    -------
    Annotation
        pyannote ``Annotation`` object whose ``speaker_diarization`` iterator
        yields ``(segment, speaker_label)`` pairs.

    Example
    -------
    >>> diarization = run_diarization(pipeline, audio)
    >>> diarization = run_diarization(pipeline, "recording.wav", show_progress=False)
    """
    if show_progress:
        with ProgressHook() as hook:
            return pipeline(audio, hook=hook)
    return pipeline(audio)


# ---------------------------------------------------------------------------
# 5. Result parsing
# ---------------------------------------------------------------------------

def parse_diarization_output(
    diarization: object,
) -> list[DiarizationTurn]:
    """
    Convert a pyannote diarisation ``Annotation`` into a plain list of turns.

    Decouples the structured result from display logic: the returned list of
    :class:`DiarizationTurn` named tuples can be iterated, filtered, serialised
    to JSON, or passed into downstream processing without touching pyannote
    internals.

    Parameters
    ----------
    diarization:
        The ``Annotation`` object returned by :func:`run_diarization`.

    Returns
    -------
    list[DiarizationTurn]
        Chronologically ordered list of ``(start, end, speaker)`` named tuples.

    Example
    -------
    >>> turns = parse_diarization_output(diarization)
    >>> turns[0]
    DiarizationTurn(start=0.5, end=3.2, speaker='SPEAKER_00')
    >>> # Filter to a single speaker
    >>> alice_turns = [t for t in turns if t.speaker == "SPEAKER_00"]
    """
    return [
        DiarizationTurn(
            start=float(turn.start),
            end=float(turn.end),
            speaker=str(speaker),
        )
        for turn, speaker in diarization.speaker_diarization
    ]


def print_diarization_output(
    turns: list[DiarizationTurn],
    time_decimals: int = 1,
) -> None:
    """
    Print a formatted diarisation turn list to stdout.

    Kept separate from :func:`parse_diarization_output` so that parsing
    and display remain independently reusable — callers can parse once and
    display, log, or serialise as needed.

    Parameters
    ----------
    turns:
        List of :class:`DiarizationTurn` named tuples, typically from
        :func:`parse_diarization_output`.
    time_decimals:
        Number of decimal places for start/end timestamps.
        Defaults to ``1``.

    Example
    -------
    >>> print_diarization_output(turns)
    start=0.5s  stop=3.2s   speaker_SPEAKER_00
    start=3.4s  stop=7.1s   speaker_SPEAKER_01
    start=7.3s  stop=10.8s  speaker_SPEAKER_00
    """
    fmt = f".{time_decimals}f"
    for turn in turns:
        print(
            f"start={turn.start:{fmt}}s  "
            f"stop={turn.end:{fmt}}s   "
            f"speaker_{turn.speaker}"
        )
