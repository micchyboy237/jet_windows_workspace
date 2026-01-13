# live_subtitles_server.py
"""
Modern WebSocket live subtitles server (compatible with websockets ≥ 12.0 / 14.0+)
Receives Japanese audio chunks, buffers until end-of-utterance, transcribes & translates.
"""

import asyncio
import base64
import json
import logging
import time
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone

import numpy as np
import scipy.io.wavfile as wavfile
from asr_wav2vec2 import JapaneseASR
from rich.logging import RichHandler
from transformers import AutoTokenizer

from translator_types import Translator  # adjust import if needed
from utils import split_sentences_ja
import threading

TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER_NAME = "Helsinki-NLP/opus-mt-ja-en"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("live-sub-server")

# ───────────────────────────────────────────────
# Quality/certainty labelers (see context section)
# ───────────────────────────────────────────────

import math
from math import isfinite  # for safe float rounding

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
    fallback: float = 0.0
) -> float:
    """
    Convert cumulative translation log-prob to a normalized confidence score [0.0, 1.0].
    
    Uses length-normalized per-token probability (geometric mean).
    """
    if log_prob is None or not isinstance(log_prob, (int, float)) or num_tokens is None or num_tokens <= 0:
        return fallback
    
    # Avoid division by unrealistically small numbers
    effective_tokens = max(min_tokens, num_tokens)
    
    per_token_prob = math.exp(log_prob / effective_tokens)
    
    # Soft clip to [0, 1] (models rarely reach exactly 1.0)
    return float(min(1.0, max(0.0, per_token_prob)))

# ───────────────────────────────────────────────
# Load models once at startup
# ───────────────────────────────────────────────

logger.info("Loading Japanese wav2vec2-large-xlsr-53 ...")
asr = JapaneseASR(device="cuda")   # or "cpu"

logger.info("Loading OPUS-MT ja→en tokenizer & translator ...")
tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER_NAME)
translator = Translator(
    TRANSLATOR_MODEL_PATH,
    device="cpu",
    compute_type="int8",
    inter_threads=4,  # tune to your cores
)

logger.info("Models loaded.")

# ───────────────────────────────────────────────
# Per-connection state (with timestamps)
# ───────────────────────────────────────────────

class ConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.buffer = bytearray()
        self.sample_rate: Optional[int] = None
        self.utterance_count = 0
        self.last_chunk_time = time.monotonic()
        self.first_chunk_received_at: Optional[datetime] = None
        self.end_of_utterance_received_at: Optional[datetime] = None

    def append_chunk(self, pcm_bytes: bytes, sample_rate: int):
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif self.sample_rate != sample_rate:
            logger.warning(f"Sample rate changed for {self.client_id} — keeping first")
        self.buffer.extend(pcm_bytes)
        now = datetime.now(timezone.utc)
        if self.first_chunk_received_at is None:
            self.first_chunk_received_at = now
        self.last_chunk_time = time.monotonic()

    def clear_buffer(self):
        self.buffer.clear()
        self.first_chunk_received_at = None
        self.end_of_utterance_received_at = None

    def get_duration_sec(self) -> float:
        if not self.buffer or self.sample_rate is None:
            return 0.0
        return len(self.buffer) / 2 / self.sample_rate  # int16

# ───────────────────────────────────────────────
# Global state and output directory
# ───────────────────────────────────────────────
connected_states: dict[asyncio.StreamWriter, ConnectionState] = {}
executor = ThreadPoolExecutor(max_workers=3)  # conservative — GTX 1660

DEFAULT_OUT_DIR: Optional[Path] = None  # ← change to Path("utterances") if you want default permanent storage

# ───────────────────────────────────────────────
# Transcribe and translate with quality/confidence
# ───────────────────────────────────────────────

def transcribe_and_translate(
    audio_bytes: bytes,
    sr: int,
    client_id: str,
    utterance_idx: int,
    end_of_utterance_received_at: datetime,        # when end marker was received (UTC)
    received_at: Optional[datetime] = None,        # when first chunk received (UTC, optional)
    out_dir: Optional[Path] = None,
) -> tuple[str, str, float, dict]:
    """Blocking function — run in executor."""
    processing_started_at = datetime.now(timezone.utc)

    timestamp = processing_started_at.strftime("%Y%m%d_%H%M%S")
    stem = f"utterance_{client_id}_{utterance_idx:04d}_{timestamp}"

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / f"{stem}.wav"
        meta_path = out_dir / f"{stem}.json"
        # Save input audio for audit/debugging
        arr = np.frombuffer(audio_bytes, dtype=np.int16)
        wavfile.write(str(audio_path), sr, arr)
    else:
        audio_path = Path(f"temp_{stem}.wav")

    # ─── Transcription with wav2vec2 ──────────────────────────────
    result = asr.transcribe(
        audio_bytes,
        input_sample_rate=sr,
        return_confidence=True,
        return_logprobs=False,
        num_beams=1,
    )

    ja_text = result["text"].strip()
    duration_sec = result["duration_sec"]

    avg_logprob = result.get("avg_logprob")
    if avg_logprob is not None:
        transcription_confidence = float(np.exp(avg_logprob))  # similar to old scale, but will be lower
        transcription_quality = result.get("quality_avg_logprob", "Unknown")
    else:
        transcription_confidence = 0.0
        transcription_quality = "N/A"

    # --- Translation (unchanged) ---
    sentences_ja: List[str] = split_sentences_ja(ja_text)
    en_sentences: List[str] = []
    translation_logprob: float | None = None
    translation_confidence: float | None = None

    if sentences_ja:
        batch_src_tokens = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(sent.strip()))
            for sent in sentences_ja if sent.strip()
        ]

        if batch_src_tokens:
            results = translator.translate_batch(
                batch_src_tokens,
                return_scores=True,          # Enable score extraction
                beam_size=4,
                max_decoding_length=512,
            )

            for result_item in results:
                hyp = result_item.hypotheses[0]
                en_sent = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(hyp),
                    skip_special_tokens=True
                ).strip()
                if en_sent:
                    en_sentences.append(en_sent)

                # Take score from first/best hypothesis
                if hasattr(result_item, "scores") and result_item.scores:
                    translation_logprob = result_item.scores[0]
                    num_output_tokens = len(hyp)
                    translation_confidence = translation_confidence_score(
                        translation_logprob, num_output_tokens, min_tokens=3
                    )

    en_text = "\n".join(en_sentences)

    translation_quality = translation_quality_label(translation_logprob)

    # --- Logging ---
    logger.info(
        "[transcription] avg_logprob=%s → conf=%.3f | quality=%s | ja=%s",
        f"{avg_logprob:.4f}" if avg_logprob is not None else "N/A",
        transcription_confidence,
        transcription_quality,
        ja_text[:70] + "..." if len(ja_text) > 70 else ja_text
    )
    logger.info(
        "[translation]   log_prob=%s → quality=%s | en=%s",
        f"{translation_logprob:.4f}" if translation_logprob is not None else "N/A",
        translation_quality,
        en_text[:70] + "..." if len(en_text) > 70 else en_text
    )

    processing_finished_at = datetime.now(timezone.utc)
    processing_duration = (processing_finished_at - processing_started_at).total_seconds()
    queue_wait_duration = (processing_started_at - end_of_utterance_received_at).total_seconds()

    # ─── Build metadata ───────────────────────────────────────────
    meta = {
        "client_id": client_id,
        "utterance_index": utterance_idx,
        "timestamp_iso": processing_started_at.isoformat(),
        "received_at": received_at.isoformat() if received_at else None,
        "end_of_utterance_received_at": end_of_utterance_received_at.isoformat(),
        "processing_started_at": processing_started_at.isoformat(),
        "processing_finished_at": processing_finished_at.isoformat(),
        "queue_wait_seconds": round(queue_wait_duration, 3),
        "processing_duration_seconds": round(processing_duration, 3),
        "audio_duration_seconds": round(duration_sec, 3),
        "sample_rate": sr,
        "transcription": {
            "text_ja": ja_text,
            "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
            "raw_avg_logprob": avg_logprob,
            "confidence": round(transcription_confidence, 4),
            "quality_label": transcription_quality,
            "chunks_info": result.get("chunks_info"),
        },
        "translation": {
            "text_en": en_text,
            "log_prob": round(translation_logprob, 4) if translation_logprob is not None else None,
            "confidence": round(translation_confidence, 4) if translation_confidence is not None else None,
            "quality_label": translation_quality,
        },
        "segments": [],  # optionally [{"text": ja_text, "start":0, "end":duration_sec}]
    }

    # Logging preview (still show names for legacy/debug, but temp.wav not saved)
    preview_ja = ja_text[:60] + ("..." if len(ja_text) > 60 else "")
    preview_en = en_text[:60] + ("..." if len(en_text) > 60 else "")
    logger.info(
        "[process] %s | thread=%s | ja: %s | en: %s | queue-wait: %.2fs | dur: %.2fs",
        audio_path.name,
        threading.current_thread().name,
        preview_ja, preview_en,
        queue_wait_duration, processing_duration
    )

    # Save metadata if permanent storage
    if out_dir:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved metadata → {meta_path}")

    # Clean up temp file if temporary
    if not out_dir and audio_path.exists():
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

                if msg_type == "audio":
                    pcm_b64 = data["pcm"]
                    sr = data.get("sample_rate", 16000)
                    pcm_bytes = base64.b64decode(pcm_b64)
                    state.append_chunk(pcm_bytes, sr)
                    logger.debug(f"[{client_id}] added {len(pcm_bytes)} bytes")

                elif msg_type == "end_of_utterance":
                    if not state.buffer:
                        logger.debug(f"[{client_id}] end-of-utterance but empty buffer")
                        continue

                    duration = state.get_duration_sec()
                    logger.info(f"[{client_id}] End of utterance — {duration:.2f}s")

                    # Capture the moment we received the end marker (UTC)
                    state.end_of_utterance_received_at = datetime.now(timezone.utc)

                    # Offload heavy work to thread, pass timestamps
                    loop = asyncio.get_running_loop()
                    ja, en, transcription_confidence, meta = await loop.run_in_executor(
                        executor,
                        transcribe_and_translate,
                        bytes(state.buffer),
                        state.sample_rate,
                        state.client_id,
                        state.utterance_count,
                        state.end_of_utterance_received_at,
                        state.first_chunk_received_at,   # received_at
                        DEFAULT_OUT_DIR,
                    )

                    # Payload follows updated context (includes quality/conf/logprob)
                    payload = {
                        "type": "subtitle",
                        "transcription_ja": meta["transcription"]["text_ja"],
                        "translation_en": meta["translation"]["text_en"],
                        "utterance_id": state.utterance_count,
                        "duration_sec": round(duration, 3),
                        "transcription_confidence": meta["transcription"]["confidence"],
                        "transcription_quality": meta["transcription"]["quality_label"],
                        "translation_logprob": meta["translation"]["log_prob"],
                        "translation_confidence": meta["translation"].get("confidence"),
                        "translation_quality": meta["translation"]["quality_label"],
                        "meta": meta,
                    }
                    await websocket.send(json.dumps(payload, ensure_ascii=False))

                    state.utterance_count += 1
                    state.clear_buffer()

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


async def main():
    from websockets.asyncio.server import serve
    async with serve(
        handler,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
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
        logger.info("Using temporary files for utterances (set UTTERANCE_OUT_DIR env var to enable permanent storage)")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")