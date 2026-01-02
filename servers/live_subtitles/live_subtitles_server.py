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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone  # ← UPDATED

import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from rich.logging import RichHandler
from transformers import AutoTokenizer

from translator_types import Translator  # adjust import if needed
import threading  # <-- Added for thread info in logging

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
# Load models once at startup
# ───────────────────────────────────────────────

logger.info("Loading Whisper model kotoba-tech/kotoba-whisper-v2.0-faster ...")
whisper_model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float32",
)

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
# Per-connection state (extended with timestamps)
# ───────────────────────────────────────────────

class ConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.buffer = bytearray()
        self.sample_rate: Optional[int] = None
        self.utterance_count = 0
        self.last_chunk_time = time.monotonic()
        self.first_chunk_received_at: Optional[datetime] = None  # ← NEW
        self.end_of_utterance_received_at: Optional[datetime] = None  # ← NEW

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
# Updated transcribe_and_translate — extra timing params
# ───────────────────────────────────────────────
def transcribe_and_translate(
    audio_bytes: bytes,
    sr: int,
    client_id: str,
    utterance_idx: int,
    end_of_utterance_received_at: datetime,        # NEW: when end marker was received (UTC)
    received_at: Optional[datetime] = None,        # NEW: when first chunk received (UTC, optional)
    out_dir: Optional[Path] = None,
) -> tuple[str, str]:
    """Blocking function — run in executor."""
    processing_started_at = datetime.now(timezone.utc)

    timestamp = processing_started_at.strftime("%Y%m%d_%H%M%S")
    stem = f"utterance_{client_id}_{utterance_idx:04d}_{timestamp}"

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / f"{stem}.wav"
        meta_path = out_dir / f"{stem}.json"
    else:
        audio_path = Path(f"temp_{stem}.wav")

    # Write audio
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sr, arr)

    # ─── Heavy work ─────────────────────────────────────────────
    segments, info = whisper_model.transcribe(
        str(audio_path),
        language="ja",
        beam_size=5,
        vad_filter=False,
    )
    ja_text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()

    if not ja_text:
        en_text = ""
    else:
        src_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(ja_text))
        results = translator.translate_batch([src_tokens])
        en_tokens = results[0].hypotheses[0]
        en_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(en_tokens),
            skip_special_tokens=True
        )
    # ───────────────────────────────────────────────────────────

    processing_finished_at = datetime.now(timezone.utc)
    processing_duration = (processing_finished_at - processing_started_at).total_seconds()
    queue_wait_duration = (processing_started_at - end_of_utterance_received_at).total_seconds()

    # ─── Build metadata ────────────────────────────────────────
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
        "audio_duration_seconds": round(len(audio_bytes) / 2 / sr, 3),
        "sample_rate": sr,
        "transcription_ja": ja_text,
        "translation_en": en_text,
    }

    # Logging preview
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
    if not out_dir:
        audio_path.unlink(missing_ok=True)

    return ja_text, en_text

# ───────────────────────────────────────────────
# Updated handler — new timestamp logic for queue and processing
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
                    ja, en = await loop.run_in_executor(
                        executor,
                        transcribe_and_translate,
                        bytes(state.buffer),
                        state.sample_rate,
                        state.client_id,
                        state.utterance_count,
                        state.end_of_utterance_received_at,
                        state.first_chunk_received_at,  # may be None if not set
                        DEFAULT_OUT_DIR,
                    )

                    # Send result back
                    payload = {
                        "type": "subtitle",
                        "transcription": ja,
                        "translation": en,
                        "utterance_id": state.utterance_count,
                        "duration_sec": round(duration, 3),
                    }
                    await websocket.send(json.dumps(payload))

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
        # Optional: add ping/pong keep-alive
        ping_interval=20,
        ping_timeout=60,
    ) as server:
        logger.info("WebSocket server listening on ws://0.0.0.0:8765")
        await server.serve_forever()


if __name__ == "__main__":
    import os
    import shutil
    out_dir_str = os.getenv(
        "UTTERANCE_OUT_DIR",
        Path(__file__).parent / "generated" / Path(__file__).stem
    )
    shutil.rmtree(out_dir_str)
    if out_dir_str:
        DEFAULT_OUT_DIR = Path(out_dir_str).resolve()
        logger.info(f"Permanent utterance storage enabled: {DEFAULT_OUT_DIR}")
    else:
        logger.info("Using temporary files for utterances (set UTTERANCE_OUT_DIR env var to enable permanent storage)")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")