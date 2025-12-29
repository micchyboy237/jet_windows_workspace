# live_subtitles_server.py

import asyncio
import time
import base64
import json
from typing import List

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol
from faster_whisper import WhisperModel

from rich.logging import RichHandler

import logging

# =============================
# Logging setup
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("asr-server")

# =============================
# Configuration
# =============================

HOST = "0.0.0.0"
PORT = 8765

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # PCM16
MIN_FLUSH_SECONDS = 0.5  # Reduce to 0.5 s (~16 chunks) – faster cycle, still enough context for reliable transcription

# =============================
# Load ASR model (production)
# =============================

asr_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8",
)
log.info("[ASR] Whisper model 'large-v3' loaded (device=cuda, compute_type=int8)")

# =============================
# Utilities
# =============================

def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert PCM16 bytes → float32 numpy array"""
    audio = np.frombuffer(pcm, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


# =============================
# Client handler
# =============================

# =============================
# Server improvements (live_subtitles_server.py)
# =============================

async def handle_client(ws: WebSocketServerProtocol) -> None:
    """
    Receives PCM16 audio frames (speech-only),
    runs streaming ASR,
    sends partial/final subtitles.
    """

    log.info("[client:%s] Connected", ws.remote_address)

    audio_buffer: List[bytes] = []
    buffered_samples = 0
    chunks_received = 0
    asr_calls = 0
    last_flush_time = time.monotonic()

    try:
        async for message in ws:
            data = json.loads(message)

            if data.get("type") != "audio":
                continue

            pcm_chunk = base64.b64decode(data["pcm"])
            chunk_samples = len(pcm_chunk) // BYTES_PER_SAMPLE
            audio_buffer.append(pcm_chunk)
            buffered_samples += chunk_samples
            chunks_received += 1
            log.debug("[client:%s] Received chunk #%d (%d samples → total %.3f s)",
                      ws.remote_address, chunks_received, chunk_samples, buffered_samples / SAMPLE_RATE)

            buffered_seconds = buffered_samples / SAMPLE_RATE

            log.debug(
                "[buffer] Buffered %.3f seconds (%d chunks)",
                buffered_seconds, len(audio_buffer)
            )

            # ---- ASR flush condition ----
            if buffered_seconds < MIN_FLUSH_SECONDS:
                continue

            flush_start = time.monotonic()
            log.info("[asr:%s] Starting transcription on %.3f s buffer (%d chunks)",
                     ws.remote_address, buffered_seconds, len(audio_buffer))

            # ---- Run ASR ----
            pcm_all = b"".join(audio_buffer)
            audio_f32 = pcm16_bytes_to_float32(pcm_all)

            segments, info = asr_model.transcribe(
                audio_f32,
                language="ja",
                vad_filter=False,                  # client already did VAD
                condition_on_previous_text=True,   # Keep context across flushes
                word_timestamps=False,             # Keep simple for now (no timestamps needed)
                beam_size=5,
                without_timestamps=True,
                temperature=0.0,
                log_prob_threshold=-0.5,           # Slightly stricter than -1.0 to reduce empty results
                no_speech_threshold=0.6,           # Default is 0.6 – keep or lower if too silent
            )

            text = "".join(seg.text for seg in segments).strip()
            asr_calls += 1

            processing_time = time.monotonic() - flush_start
            log.info("[asr:%s] Completed in %.2f s → Raw text: %r", ws.remote_address, processing_time, text)

            if text:
                log.info("[client:%s] Transcription: %s", ws.remote_address, text)
                await ws.send(json.dumps({
                    "type": "subtitle",
                    "transcription": text,
                }))

            # Buffer cleared only after successful transcription
            audio_buffer.clear()
            buffered_samples = 0
            last_flush_time = time.monotonic()
            log.debug("[buffer:%s] Cleared after flush", ws.remote_address)
    except websockets.exceptions.ConnectionClosedOK:
        log.info("[client:%s] Disconnected normally", ws.remote_address)
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning("[client:%s] Disconnected abnormally: %s", ws.remote_address, e)
    except Exception:
        log.exception("[client:%s] Unexpected error", ws.remote_address)
    finally:
        uptime = time.monotonic() - last_flush_time if 'last_flush_time' in locals() else 0
        log.info(
            "[client:%s] Session ended – chunks: %d, ASR calls: %d, uptime: %.1f s",
            ws.remote_address, chunks_received, asr_calls, uptime
        )


# =============================
# Server entrypoint
# =============================

async def main() -> None:
    async with websockets.serve(
        handle_client,
        host=HOST,
        port=PORT,
        max_size=None,
        compression=None,
        ping_interval=30,  # Match client for high traffic
        ping_timeout=30,
        close_timeout=10,
    ):
        print(f"🚀 WebSocket server running at ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
