import numpy as np

from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

AudioInput = Union[np.ndarray, bytes, bytearray, str, Path]


class SpeechWaveMeta(TypedDict):
    has_risen: bool
    has_multi_passed: bool
    has_fallen: bool
    is_valid: bool


class SpeechWaveDetails(TypedDict):
    """Detailed insights including frame boundaries and probability statistics for a speech wave."""

    frame_start: int
    frame_end: int
    frame_len: int
    duration_sec: float
    min_prob: float
    max_prob: float
    avg_prob: float
    std_prob: float


class SpeechWave(SpeechWaveMeta):
    start_sec: float
    end_sec: float
    details: SpeechWaveDetails


class SpeechSegment(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]
    segment_probs: List[float]


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]
