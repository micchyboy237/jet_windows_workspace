# servers\audio_server\python_scripts\server\services\transcribe_kotoba_service.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, Dict, TypedDict

from faster_whisper import WhisperModel

from ..utils.audio_utils import load_audio
from ..utils.logger import get_logger

log = get_logger("kotoba_transcriber")

class KotobaTranscriptionResult(TypedDict):
    language: str
    language_prob: float
    text: str

_MODEL: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        log.info("[bold yellow]Loading Kotoba Whisper model[/] [cyan]kotoba-tech/kotoba-whisper-v2.0-faster[/] [blue]device=cuda[/] [green]compute_type=float32[/]")
        _MODEL = WhisperModel(
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            device="cuda",
            compute_type="float32",
        )
        log.info("[bold green]Kotoba Whisper model loaded and cached[/]")
    else:
        log.info(f"[dim]Kotoba model cache hit[/]")
    return _MODEL

def transcribe_kotoba_audio(
    audio: Any,
) -> KotobaTranscriptionResult:
    """
    Transcribe Japanese audio using the Kotoba Whisper faster-whisper model.
    
    Args:
        audio: Audio input compatible with load_audio (bytes, path, np.ndarray, etc.)
    
    Returns:
        Dict containing detected language, probability, and transcribed text.
    """
    model = _get_model()
    
    waveform: np.ndarray = load_audio(audio)  # Already float32, mono, 16kHz
    
    log.info("[bold cyan]Kotoba transcribe[/] processing → {:.2f}s audio".format(len(waveform) / 16000))
    
    segments, info = model.transcribe(
        waveform,
        language="ja",
        chunk_length=30,  # Better context than 15s
        condition_on_previous_text=False,  # Reduces repetitions/hallucinations
        temperature=0.0,  # More deterministic
    )
    
    text = " ".join(segment.text for segment in segments).strip()
    
    log.info(f"[bold green]Kotoba transcribed[/] → {text}")
    
    return {
        "language": info.language,
        "language_prob": info.language_probability,
        "text": text,
    }