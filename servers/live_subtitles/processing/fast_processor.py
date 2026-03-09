"""
Fast path: LLM-based Japanese transcription (SenseVoiceSmall) + English translation (llama.cpp)
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from transcribe_jp_llm import transcribe_japanese, TranscriptionResult
from translate_jp_en_llm import translate_japanese_to_english

logger = logging.getLogger(__name__)


def process_fast_llm(
    audio_bytes: bytes,
    sample_rate: int,
    client_id: str,
    segment_num: int,
    end_of_utterance_received_at: datetime,
    rms: float,
    hotwords: Optional[list[str]] = None,
    received_at: Optional[datetime] = None,
    context_prompt: Optional[str] = None,
    last_ja: Optional[str] = None,
    last_en: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Blocking fast-path processing.
    Returns dict compatible with ws_server_subtitles_handlers payload expectations.
    """
    processing_started_at = datetime.now(timezone.utc)

    # Prepare temporary file if persistent storage enabled
    temp_path: Optional[Path] = None
    if out_dir:
        ts = processing_started_at.strftime("%Y%m%d_%H%M%S")
        temp_path = out_dir / f"utterance_{client_id}_{segment_num:04d}_{ts}.wav"

    # 1. Transcription (SenseVoiceSmall / FunASR)
    trans_result: TranscriptionResult = transcribe_japanese(
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        save_temp_wav=temp_path,
        client_id=client_id,
        segment_num=segment_num,
        # hotwords=hotwords,          # ← pass when implemented
        # context_prompt=context_prompt,
    )

    ja_text = trans_result.get("text_ja", "").strip()

    # 2. Translation (llama.cpp)
    translation_result = translate_japanese_to_english(
        ja_text=ja_text,
        enable_scoring=False,  # set True when you want logprobs (slower)
        history=None,          # can pass conversation history later
    )

    en_text = translation_result.get("text", "").strip()

    processing_finished_at = datetime.now(timezone.utc)
    queue_wait_s = (processing_started_at - end_of_utterance_received_at).total_seconds()
    process_s = (processing_finished_at - processing_started_at).total_seconds()

    meta: Dict[str, Any] = {
        **trans_result.get("metadata", {}),
        "timestamp_iso": processing_started_at.isoformat(),
        "received_at": received_at.isoformat() if received_at else None,
        "end_of_utterance_received_at": end_of_utterance_received_at.isoformat(),
        "processing_started_at": processing_started_at.isoformat(),
        "processing_finished_at": processing_finished_at.isoformat(),
        "queue_wait_seconds": round(queue_wait_s, 3),
        "processing_duration_seconds": round(process_s, 3),
        "transcription": {
            "text_ja": ja_text,
            **{k: v for k, v in trans_result.items() if k != "text_ja" and k != "metadata"},
        },
        "translation": translation_result,
        "rms": rms,
        # "context": {"last_ja": last_ja, "last_en": last_en, "hotwords": hotwords},
    }

    result = {
        "transcription_ja": ja_text,
        "translation_en": en_text,
        "meta": meta,
        "transcription_confidence": trans_result.get("confidence"),
        "transcription_quality": trans_result.get("quality_label", "N/A"),
        "translation_confidence": translation_result.get("confidence"),
        "translation_quality": translation_result.get("quality", "N/A"),
    }

    preview_ja = ja_text[:60] + "..." if len(ja_text) > 60 else ja_text
    preview_en = en_text[:60] + "..." if len(en_text) > 60 else en_text

    logger.info(
        "[fast-llm] %s | ja=%s | en=%s | q=%.2fs p=%.2fs",
        client_id, preview_ja, preview_en, queue_wait_s, process_s
    )

    return result