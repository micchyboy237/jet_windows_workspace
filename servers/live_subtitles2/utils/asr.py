"""Japanese ASR utility using faster-whisper."""

from __future__ import annotations

from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # for tests


class TranscriptionSegment(TypedDict):
    start: float
    end: float
    text: str


class ASRTranscriber:
    """Reusable class for Japanese transcription."""

    def __init__(
        self,
        model_name: str = "kotoba-tech/kotoba-whisper-v2.0-faster",
        device: str = "cuda",
        compute_type: str = "float32",
    ):
        """Initialize the Whisper model."""
        self.device = device
        self.model_name = model_name
        if WhisperModel is None:
            self.model = None
        else:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )

    def transcribe_japanese_asr(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[str, List[TranscriptionSegment]]:
        """Transcribe Japanese audio to text. Core utility function."""
        if self.model is None or len(audio) < sample_rate * 0.5:
            return "", []
        segments, info = self.model.transcribe(
            audio,
            language="ja",
            beam_size=5,
            best_of=5,
            vad_filter=False,
        )
        full_text = "".join(seg.text for seg in segments).strip()
        seg_list: List[TranscriptionSegment] = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
        ]
        return full_text, seg_list
