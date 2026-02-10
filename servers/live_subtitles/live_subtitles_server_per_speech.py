# live_subtitles_server_per_speech.py
"""
Modern WebSocket live subtitles server (compatible with websockets ≥ 12.0 / 14.0+)
Receives Japanese audio chunks, buffers until end-of-utterance, transcribes & translates.
"""

import asyncio
import base64
import dataclasses
import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import isfinite  # for safe float rounding
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from rich.logging import RichHandler

# from segment_speaker_labeler import SegmentSpeakerLabeler
# from segment_emotion_classifier import SegmentEmotionClassifier


TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER_NAME = "Helsinki-NLP/opus-mt-ja-en"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("live-sub-server")

# ───────────────────────────────────────────────
# Quality/certainty labelers (see context section)
# ───────────────────────────────────────────────


def transcription_quality_label(avg_logprob: float) -> str:
    """
    Quality label for transcription confidence based on average log-probability.
    Higher (less negative) = better.
    """
    if not isfinite(avg_logprob):
        return "N/A"
    if avg_logprob > -0.3:
        return "Very High"
    elif avg_logprob > -0.7:
        return "High"
    elif avg_logprob > -1.2:
        return "Medium"
    elif avg_logprob > -2.0:
        return "Low"
    else:
        return "Very Low"


def translation_quality_label(log_prob: float | None) -> str:
    """
    Quality label for translation confidence based on cumulative log-probability.
    Higher (less negative) = better. Negative values are normal.
    """
    if log_prob is None or not isfinite(log_prob):
        return "N/A"
    if log_prob > -0.4:
        return "High"
    elif log_prob > -1.0:
        return "Good"
    elif log_prob > -2.0:
        return "Medium"
    else:
        return "Low"


def translation_confidence_score(
    log_prob: float | None,
    num_tokens: int | None = None,
    min_tokens: int = 1,
    fallback: float = 0.0,
) -> float:
    """
    Convert cumulative translation log-prob to a normalized confidence score [0.0, 1.0].

    Uses length-normalized per-token probability (geometric mean).
    """
    if (
        log_prob is None
        or not isinstance(log_prob, (int, float))
        or num_tokens is None
        or num_tokens <= 0
    ):
        return fallback

    # Avoid division by unrealistically small numbers
    effective_tokens = max(min_tokens, num_tokens)

    per_token_prob = math.exp(log_prob / effective_tokens)

    # Soft clip to [0, 1] (models rarely reach exactly 1.0)
    return float(min(1.0, max(0.0, per_token_prob)))


# ───────────────────────────────────────────────
# Load models once at startup
# ───────────────────────────────────────────────

logger.info("Loading Whisper model kotoba-tech/kotoba-whisper-v2.0-faster ...")
whisper_model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float32",
)

logger.info("Loading OPUS-MT ja→en tokenizer & translator ...")
from translate_jp_en import (
    translate_japanese_to_english,
)

# logger.info("Loading speaker labeler pyannote model & clustering strategy...")
# labeler = SegmentSpeakerLabeler()

# logger.info("Loading emotion classifier model...")
# emotion_classifier = SegmentEmotionClassifier(device=-1)

logger.info("Models loaded.")

# ───────────────────────────────────────────────
# Per-connection state (with timestamps)
# ───────────────────────────────────────────────


class ConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.current_utterance_id: str | None = None
        self.audio_buffer: bytearray = bytearray()
        self.chunk_count: int = 0
        self.last_context_prompt: str | None = None
        self.utterance_start: datetime | None = None

    def reset_utterance(self):
        self.audio_buffer = bytearray()
        self.chunk_count = 0

    def append_chunk(self, pcm_bytes: bytes, sample_rate: int):
        # Not used currently; present for compatibility.
        pass

    def clear_buffer(self):
        pass

    def get_duration_sec(self) -> float:
        return 0.0


# ───────────────────────────────────────────────
# Global state and output directory
# ───────────────────────────────────────────────
connected_states: dict[asyncio.StreamWriter, ConnectionState] = {}
executor = ThreadPoolExecutor(max_workers=1)  # conservative — GTX 1660

DEFAULT_OUT_DIR: Path | None = (
    None  # ← change to Path("utterances") if you want default permanent storage
)

# ───────────────────────────────────────────────
# Transcribe and translate with quality/confidence
# ───────────────────────────────────────────────


def transcribe_and_translate(
    audio_bytes: bytes,
    sr: int,
    client_id: str,
    utterance_id: str,
    segment_num: int,
    end_of_utterance_received_at: datetime,  # when end marker was received (UTC)
    received_at: datetime | None = None,  # when first chunk received (UTC, optional)
    context_prompt: str | None = None,
    out_dir: Path | None = None,
) -> tuple[str, str, float, dict]:
    """Blocking function — run in executor."""
    processing_started_at = datetime.now(timezone.utc)

    timestamp = processing_started_at.strftime("%Y%m%d_%H%M%S")
    stem = f"utterance_{client_id}_{segment_num:04d}_{timestamp}"

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / f"{stem}.wav"
        meta_path = out_dir / f"{stem}.json"
    else:
        audio_path = Path(f"temp_{stem}.wav")

    # Write audio
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sr, arr)

    # ─── Transcription ─────────────────────────────────────────────
    segments, info = whisper_model.transcribe(
        str(audio_path),
        language="ja",
        beam_size=5,
        vad_filter=False,
        initial_prompt=context_prompt,
        condition_on_previous_text=False,
    )

    ja_text_parts = []
    logprob_sum = 0.0
    token_count_proxy = 0

    segments_list = []
    for segment in segments:
        # Exclude tokens from metadata to keep JSON clean
        segment_dict = {
            k: v for k, v in dataclasses.asdict(segment).items() if k != "tokens"
        }
        segments_list.append(segment_dict)

        text = segment.text.strip()
        if text:
            ja_text_parts.append(text)

        if hasattr(segment, "avg_logprob") and segment.avg_logprob is not None:
            # Prefer exact token count if available
            if hasattr(segment, "tokens") and isinstance(segment.tokens, list):
                segment_token_count = len(segment.tokens)
            else:
                # Fallback to original approximation for Japanese
                segment_token_count = max(1, len(text) // 2)

            logprob_sum += segment.avg_logprob * segment_token_count
            token_count_proxy += segment_token_count

    ja_text = " ".join(ja_text_parts).strip()

    if token_count_proxy > 0:
        avg_logprob = logprob_sum / token_count_proxy
        transcription_confidence = float(np.exp(avg_logprob))
    else:
        avg_logprob = float("-inf")
        transcription_confidence = 0.0

    transcription_quality = transcription_quality_label(avg_logprob)

    # ─── Translation (updated: use standalone translation util) ─────
    # Replaces old batch translation logic — see @Untitled-7 (16-43)
    en_text, translation_logprob, translation_confidence, translation_quality = (
        translate_japanese_to_english(
            ja_text, beam_size=4, max_decoding_length=512, min_tokens_for_confidence=3
        )
    )

    # ─── Logging ──────────────────────────────────────────────────
    logger.info(
        "[transcription] avg_logprob=%.4f → conf=%.3f | quality=%s | ja=%s",
        avg_logprob,
        transcription_confidence,
        transcription_quality,
        ja_text[:70] + "..." if len(ja_text) > 70 else ja_text,
    )
    logger.info(
        "[translation] log_prob=%s → quality=%s | en=%s",
        f"{translation_logprob:.4f}" if translation_logprob is not None else "N/A",
        translation_quality,
        en_text[:70] + "..." if len(en_text) > 70 else en_text,
    )

    processing_finished_at = datetime.now(timezone.utc)
    processing_duration = (
        processing_finished_at - processing_started_at
    ).total_seconds()
    queue_wait_duration = (
        processing_started_at - end_of_utterance_received_at
    ).total_seconds()

    # ─── Build metadata ───────────────────────────────────────────
    meta = {
        "client_id": client_id,
        "utterance_id": utterance_id,
        "timestamp_iso": processing_started_at.isoformat(),
        "received_at": received_at.isoformat() if received_at else None,
        "end_of_utterance_received_at": end_of_utterance_received_at.isoformat(),
        "processing_started_at": processing_started_at.isoformat(),
        "processing_finished_at": processing_finished_at.isoformat(),
        "queue_wait_seconds": round(queue_wait_duration, 3),
        "processing_duration_seconds": round(processing_duration, 3),
        "audio_duration_seconds": round(len(audio_bytes) / 2 / sr, 3),
        "sample_rate": sr,
        "transcription": {
            "text_ja": ja_text,
            "avg_logprob": round(avg_logprob, 4) if isfinite(avg_logprob) else None,
            "confidence": round(transcription_confidence, 4),
            "quality_label": transcription_quality,
        },
        "translation": {
            "text_en": en_text,
            "log_prob": round(translation_logprob, 4)
            if translation_logprob is not None
            else None,
            "confidence": (
                round(translation_confidence, 4)
                if translation_confidence is not None
                else None
            ),
            "quality_label": translation_quality,
        },
        "context": {
            "prompt_used": context_prompt,
        },
        "segments": segments_list,
    }

    # Logging preview
    preview_ja = ja_text[:60] + ("..." if len(ja_text) > 60 else "")
    preview_en = en_text[:60] + ("..." if len(en_text) > 60 else "")
    logger.info(
        "[process] %s | thread=%s | ja: %s | en: %s | queue-wait: %.2fs | dur: %.2fs",
        audio_path.name,
        threading.current_thread().name,
        preview_ja,
        preview_en,
        queue_wait_duration,
        processing_duration,
    )

    # Save metadata if permanent storage
    if out_dir:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved metadata → {meta_path}")

    # Clean up temp file if temporary
    if not out_dir:
        audio_path.unlink(missing_ok=True)

    return ja_text, en_text, transcription_confidence, meta


# ───────────────────────────────────────────────
# Handler (with quality & confidence in payload)
# ───────────────────────────────────────────────


async def handler(websocket):
    """
    Modern websockets handler (websockets.asyncio.server style)
    Receives messages from one client.
    """
    client_id = f"{id(websocket):x}"[:8]
    state = ConnectionState(client_id)
    connected_states[websocket] = state
    logger.info(f"New client connected: {client_id}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type in ("speech_chunk", "complete_utterance"):
                    pcm_b64 = data.get("pcm")
                    if not pcm_b64:
                        logger.warning(f"[{client_id}] Missing pcm")
                        continue
                    pcm_bytes = base64.b64decode(pcm_b64)
                    sr = data.get("sample_rate", 16000)
                    utterance_id = data.get("utterance_id")
                    segment_num = data.get("segment_num")
                    chunk_index = data.get("chunk_index", 0)
                    is_final = data.get("is_final", False)
                    context_prompt = data.get("context_prompt")

                    if utterance_id != state.current_utterance_id:
                        state.current_utterance_id = utterance_id
                        state.reset_utterance()
                        state.utterance_start = datetime.now(timezone.utc)

                    state.audio_buffer.extend(pcm_bytes)
                    state.chunk_count += 1
                    state.last_context_prompt = context_prompt

                    # Transcribe EVERY chunk progressively
                    logger.info(f"[{client_id}] Processing chunk {chunk_index} for utt {utterance_id} {'(final)' if is_final else '(partial)'}")
                    await process_utterance(
                        websocket,
                        state,
                        sr,
                        utterance_id,
                        segment_num,
                        context_prompt,
                        data,
                        is_final=is_final,
                    )

                    if is_final:
                        logger.info(f"[{client_id}] Final chunk received for utt {utterance_id}")
                        state.reset_utterance()

                else:
                    logger.warning(f"[{client_id}] Unknown message type: {msg_type}")
            except json.JSONDecodeError:
                logger.warning(f"[{client_id}] Invalid JSON received")
            except Exception as e:
                logger.exception(f"[{client_id}] Message handler error: {e}")
    except Exception as e:
        logger.exception(f"[{client_id}] Connection error: {e}")
    finally:
        connected_states.pop(websocket, None)
        logger.info(f"[{client_id}] Disconnected")


async def process_utterance(
    websocket,
    state: ConnectionState,
    sample_rate: int,
    utterance_id: str,
    segment_num: int,
    context_prompt: str | None,
    data: dict,
    is_final: bool = False,      # <-- Added: is_final to signature
) -> None:
    if len(state.audio_buffer) < 1000:
        logger.warning(f"[{state.client_id}] Empty or too small utterance {utterance_id}")
        return

    # Calculate fallback duration if client didn't send it
    audio_duration_sec = len(state.audio_buffer) / (2 * sample_rate)

    loop = asyncio.get_running_loop()
    ja, en, conf, meta = await loop.run_in_executor(
        executor,
        transcribe_and_translate,
        bytes(state.audio_buffer),
        sample_rate,
        state.client_id,
        utterance_id,
        segment_num,
        datetime.now(timezone.utc),
        None,
        context_prompt,
        DEFAULT_OUT_DIR,
    )

    # Extract values from client message with safe defaults
    segment_idx = data.get("segment_idx")
    segment_num = data.get("segment_num")
    segment_type = data.get("segment_type", "speech")
    client_duration = data.get("duration_sec")
    avg_vad_confidence = data.get("avg_vad_confidence", 0.0)

    if segment_idx is None:
        segment_idx = 0  # or calculate from chunk_index if you want

    # Prefer client-provided duration, fallback to calculated
    duration_sec = client_duration if client_duration is not None else audio_duration_sec

    payload = {
        "type": "final_subtitle" if is_final else "partial_subtitle",
        "utterance_id": utterance_id,
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "segment_type": segment_type,
        "avg_vad_confidence": avg_vad_confidence,
        "transcription_ja": meta["transcription"]["text_ja"],
        "translation_en": meta["translation"]["text_en"],
        "duration_sec": round(duration_sec, 3),
        "transcription_confidence": meta["transcription"]["confidence"],
        "transcription_quality": meta["transcription"]["quality_label"],
        "translation_confidence": meta["translation"].get("confidence"),
        "translation_quality": meta["translation"]["quality_label"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
    }

    await websocket.send(json.dumps(payload, ensure_ascii=False))
    logger.info(f"[{state.client_id}] Sent {'final' if is_final else 'partial'} subtitle for utt {utterance_id}")


def pcm_bytes_to_waveform(pcm: bytes | bytearray, *, dtype=np.int16) -> np.ndarray:
    if len(pcm) == 0:
        raise ValueError("Empty PCM buffer")
    waveform = np.frombuffer(pcm, dtype=dtype)
    if waveform.size == 0:
        raise ValueError("Empty PCM buffer after conversion")
    waveform = waveform.astype(np.float32) / 32768.0
    return waveform


async def main():
    from websockets.asyncio.server import serve

    async with serve(
        handler,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=None,  # Allow arbitrarily large messages (long utterances can exceed default 1 MiB limit)
    ) as server:
        logger.info("WebSocket server listening on ws://0.0.0.0:8765")
        await server.serve_forever()


if __name__ == "__main__":
    import os
    import shutil

    out_dir_str = os.getenv("UTTERANCE_OUT_DIR")
    if out_dir_str:
        shutil.rmtree(out_dir_str, ignore_errors=True)
        DEFAULT_OUT_DIR = Path(out_dir_str).resolve()
        logger.info(f"Permanent utterance storage enabled: {DEFAULT_OUT_DIR}")
    else:
        logger.info(
            "Using temporary files for utterances (set UTTERANCE_OUT_DIR env var to enable permanent storage)"
        )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
