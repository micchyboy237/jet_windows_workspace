# jet_python_modules/jet/audio/audio_waveform/speech_events.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .speech_types import SpeechFrame


@dataclass
class SpeechSegmentStartEvent:
    segment_id: int
    start_frame: int
    start_time_sec: float
    started_at: str
    segment_dir: Path | None = None  # to be set by a handler if desired


@dataclass
class SpeechSegmentEndEvent:
    segment_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    audio: np.ndarray
    prob_frames: list[SpeechFrame]
    forced_split: bool
    trigger_reason: str
    started_at: str
    segment_rms: float
    loudness: str
    has_sound: bool
    segment_dir: Path | None = None
