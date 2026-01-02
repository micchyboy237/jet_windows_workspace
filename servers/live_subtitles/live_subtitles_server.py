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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from rich.logging import RichHandler
from transformers import AutoTokenizer

# Reuse your existing translator setup
from translator_types import Translator  # adjust import if needed

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
# Per-connection state
# ───────────────────────────────────────────────

class ConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.buffer = bytearray()
        self.sample_rate: Optional[int] = None
        self.utterance_count = 0
        self.last_chunk_time = time.monotonic()

    def append_chunk(self, pcm_bytes: bytes, sample_rate: int):
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif self.sample_rate != sample_rate:
            logger.warning(f"Sample rate changed for {self.client_id} — keeping first")
        self.buffer.extend(pcm_bytes)
        self.last_chunk_time = time.monotonic()

    def clear_buffer(self):
        self.buffer.clear()

    def get_duration_sec(self) -> float:
        if not self.buffer or self.sample_rate is None:
            return 0.0
        return len(self.buffer) / 2 / self.sample_rate  # int16


# Global state
connected_states: dict[asyncio.StreamWriter, ConnectionState] = {}
executor = ThreadPoolExecutor(max_workers=3)  # conservative — GTX 1660


def transcribe_and_translate(audio_bytes: bytes, sr: int) -> tuple[str, str]:
    """Blocking function — run in executor."""
    start = time.monotonic()

    # Optional: save temp file for easier debugging
    temp_path = Path(f"temp_utterance_{int(time.time()*1000)}.wav")
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(temp_path), sr, arr)

    # Transcribe
    segments, info = whisper_model.transcribe(
        str(temp_path),
        language="ja",
        beam_size=5,
        vad_filter=False,
    )
    ja_text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()

    # Translate
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

    duration = time.monotonic() - start
    preview_ja = ja_text[:60] + ("..." if len(ja_text) > 60 else "")
    preview_en = en_text[:60] + ("..." if len(en_text) > 60 else "")
    logger.info(
        "[process] %s → ja: %s | en: %s | took %.2fs",
        temp_path.name, preview_ja, preview_en, duration
    )

    temp_path.unlink(missing_ok=True)
    return ja_text, en_text


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

                    # Offload heavy work to thread
                    loop = asyncio.get_running_loop()
                    ja, en = await loop.run_in_executor(
                        executor,
                        transcribe_and_translate,
                        bytes(state.buffer),
                        state.sample_rate
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
        logger.info(f"WebSocket server listening on ws://0.0.0.0:8765")
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")