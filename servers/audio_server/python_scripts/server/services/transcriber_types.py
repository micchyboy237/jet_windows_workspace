# servers/audio_server/python_scripts/server/services/transcriber_types.py
from __future__ import annotations

from typing import TypedDict, Literal, Optional


class TranscriptionResult(TypedDict):
    """Standard result format returned by single-audio transcription services."""
    text: str
    language: Optional[str]
    language_prob: Optional[float]
    # Optional fields that may be added by specific transcribers
    # e.g., segments, timestamps, etc. – kept flexible with total=False
    segments: Optional[list[dict]]  # type: ignore


class BatchResult(TypedDict):
    """
    Result structure used by batch byte-based transcription endpoints.
    Preserves original audio bytes for duration calculation in routers.
    """
    audio_bytes: bytes
    language: Optional[str]
    language_prob: Optional[float]
    text: str
    translation: Optional[str]