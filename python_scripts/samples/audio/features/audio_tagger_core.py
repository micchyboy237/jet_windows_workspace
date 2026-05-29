"""
audio_tagger_core.py
====================
Shared dataclasses, constants, and logging/console setup for sherpa-onnx audio tagging.
This module imports NOTHING from audio_tagger_base or audio_tagger_utils,
breaking circular dependencies.

Aligned with FireRed VAD:
    - Uses FRAME_SHIFT_SAMPLE (160 samples) as the fundamental unit
    - Windows are multiples of FireRed frames (100 frames = 1s window)
    - Supports per-segment tagging with absolute UTC timestamps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fireredvad.core.constants import (
    FRAME_SHIFT_SAMPLE,
)
from fireredvad.core.constants import (
    SAMPLE_RATE as FIRERED_SAMPLE_RATE,
)
from rich.console import Console

# ── Shared console & logger ────────────────────────────────────────────────
console = Console()
log = logging.getLogger(__name__)

# ── Shared constants ───────────────────────────────────────────────────────
BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
SAMPLE_RATE = FIRERED_SAMPLE_RATE
HOP_LENGTH = FRAME_SHIFT_SAMPLE


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class TaggingEvent:
    """Normalised event produced by either model backend.
    Now supports absolute UTC timestamps when processing speech segments.
    """

    name: Optional[str]
    class_index: Optional[int]
    prob: float
    max_prob: float = 0.0
    occurrences: int = 1
    chunk_start: float = 0.0
    chunk_end: float = 0.0
    chunk_index: int = 0
    time_utc_start: Optional[datetime] = None
    time_utc_end: Optional[datetime] = None


@dataclass
class TaggingResult:
    """Full result from one audio file or speech segment tagging run.
    Now supports both file-based and segment-based processing.
    """

    audio_path: str
    sample_rate: int
    duration: float
    elapsed_time: float
    events: List[TaggingEvent]
    chunk_count: int
    backend_name: str
    model_variant: str
    top_k: int
    is_speech_segment: bool = False
    segment_start_utc: Optional[datetime] = None
    segment_end_utc: Optional[datetime] = None

    @property
    def real_time_factor(self) -> float:
        return self.elapsed_time / self.duration if self.duration > 0 else 0.0
