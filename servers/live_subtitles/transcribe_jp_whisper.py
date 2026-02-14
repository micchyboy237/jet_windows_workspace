# servers/live_subtitles/transcribe_jp_whisper.py
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from logger import logger

# Global model (loaded once)
logger.info("Loading Whisper model kotoba-tech/kotoba-whisper-v2.0-faster ...")
WHISPER_MODEL = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float32",
)
logger.info("Whisper model loaded.")


def transcription_quality_label(avg_logprob: float) -> str:
    if not math.isfinite(avg_logprob):
        return "N/A"
    if avg_logprob > -0.3:
        return "Very High"
    if avg_logprob > -0.7:
        return "High"
    if avg_logprob > -1.2:
        return "Medium"
    if avg_logprob > -2.0:
        return "Low"
    return "Very Low"


def compute_transcription_confidence(
    segments: list[Segment],
) -> tuple[float, float, str]:  # avg_logprob, confidence, quality_label
    if not segments:
        return float("-inf"), 0.0, "N/A"

    logprob_sum = 0.0
    token_count = 0

    for seg in segments:
        if seg.avg_logprob is None:
            continue
        # More accurate token count if available
        seg_token_count = len(seg.tokens) if hasattr(seg, "tokens") and seg.tokens else max(1, len(seg.text) // 2)
        logprob_sum += seg.avg_logprob * seg_token_count
        token_count += seg_token_count

    if token_count == 0:
        return float("-inf"), 0.0, "N/A"

    avg_logprob = logprob_sum / token_count
    confidence = float(np.exp(avg_logprob)) if math.isfinite(avg_logprob) else 0.0
    quality = transcription_quality_label(avg_logprob)

    return avg_logprob, confidence, quality


@dataclass
class TranscriptionResult:
    text_ja: str
    confidence: float
    quality_label: str
    avg_logprob: Optional[float]
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]


def transcribe_japanese_whisper(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    context_prompt: str | None = None,
    save_temp_wav: Path | None = None,
    client_id: str = "unknown",
    utterance_id: str = "unknown",
    segment_num: int = 0,
) -> TranscriptionResult:
    """
    Core Whisper transcription function — can be called from server or tests.
    Returns structured result with text, confidence, segments, etc.
    """
    processing_started = datetime.now(timezone.utc)

    # ── Prepare audio file ───────────────────────────────────────
    if save_temp_wav:
        audio_path = save_temp_wav
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = processing_started.strftime("%Y%m%d_%H%M%S")
        audio_path = Path(f"temp_utt_{client_id}_{segment_num}_{timestamp}.wav")

    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sample_rate, arr)

    # ── Run transcription ─────────────────────────────────────────
    segments_iter, info = WHISPER_MODEL.transcribe(
        str(audio_path),
        language="ja",
        beam_size=5,
        vad_filter=False,
        initial_prompt=context_prompt,
        condition_on_previous_text=False,
    )

    segments = list(segments_iter)

    # ── Post-process ──────────────────────────────────────────────
    ja_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
    ja_text = " ".join(ja_parts).strip()

    avg_logprob, confidence, quality_label = compute_transcription_confidence(segments)

    # Clean segment dicts (remove tokens to reduce size)
    clean_segments = [
        {k: v for k, v in asdict(seg).items() if k != "tokens"} for seg in segments
    ]

    # Optional: clean up temp file
    if not save_temp_wav:
        try:
            audio_path.unlink()
        except:
            pass

    processing_duration = (datetime.now(timezone.utc) - processing_started).total_seconds()

    meta = {
        "client_id": client_id,
        "utterance_id": utterance_id,
        "segment_num": segment_num,
        "processing_duration_seconds": round(processing_duration, 3),
        "audio_duration_seconds": round(len(audio_bytes) / (2 * sample_rate), 3),
        "sample_rate": sample_rate,
        "context_prompt": context_prompt,
    }

    return TranscriptionResult(
        text_ja=ja_text,
        confidence=confidence,
        quality_label=quality_label,
        avg_logprob=avg_logprob if math.isfinite(avg_logprob) else None,
        segments=clean_segments,
        metadata=meta,
    )