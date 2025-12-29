# live_subtitles_client.py

import asyncio
import base64
import json
import logging
import time
from collections import deque
from typing import Deque

import sounddevice as sd
import websockets
from pysilero_vad import SileroVoiceActivityDetector

from rich.logging import RichHandler

# =============================
# Logging setup
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("mic-streamer")

# =============================
# Configuration
# =============================

WS_URL = "ws://192.168.68.150:8765"

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

VAD_THRESHOLD = 0.5  # speech probability threshold

# =============================
# Initialize Silero VAD
# =============================

vad = SileroVoiceActivityDetector()  # model_path uses default

VAD_CHUNK_SAMPLES = vad.chunk_samples()
VAD_CHUNK_BYTES = vad.chunk_bytes()

log.info("[VAD] chunk_samples=%s, chunk_bytes=%s", VAD_CHUNK_SAMPLES, VAD_CHUNK_BYTES)

# =============================
# RTT tracking for batched responses
# =============================

# Keep a short history of recent processing times (in seconds)
ProcessingTime = float
recent_rtts: Deque[ProcessingTime] = deque(maxlen=10)

# Queue of pending send timestamps – one entry per chunk sent
# Server flushes in batches, so we pop the oldest when a subtitle arrives
pending_sends: deque[float] = deque()

last_send_start = None

# =============================
# Audio capture + streaming
# =============================

async def stream_microphone(ws) -> None:
    """
    Capture mic audio, apply VAD, send speech-only PCM frames, track send times.
    """
    pcm_buffer = bytearray()
    chunks_sent = 0
    chunks_detected = 0

    def audio_callback(indata, frames: int, time_, status) -> None:
        nonlocal pcm_buffer
        if status:
            log.warning("Audio callback status: %s", status)
        pcm_buffer.extend(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=VAD_CHUNK_SAMPLES,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
    ):
        log.info("[microphone] Microphone streaming started")

        try:
            while True:
                await asyncio.sleep(0.01)

                processed = 0
                while len(pcm_buffer) >= VAD_CHUNK_BYTES:
                    chunk = bytes(pcm_buffer[:VAD_CHUNK_BYTES])
                    del pcm_buffer[:VAD_CHUNK_BYTES]

                    speech_prob: float = vad(chunk)

                    if speech_prob >= VAD_THRESHOLD:
                        chunks_detected += 1
                        processed += 1

                        # Record send start time for RTT measurement
                        send_start = time.monotonic()
                        global last_send_start
                        last_send_start = send_start
                        pending_sends.append(send_start)

                        payload = {
                            "type": "audio",
                            "sample_rate": SAMPLE_RATE,
                            "pcm": base64.b64encode(chunk).decode("ascii"),
                        }

                        try:
                            await ws.send(json.dumps(payload))
                            chunks_sent += 1

                            # Calculate round-trip processing time (client send → subtitle received)
                            # This is updated in receive_subtitles() when a subtitle arrives
                            if recent_rtts:
                                avg_rtt = sum(recent_rtts) / len(recent_rtts)
                                log.info(
                                    "[speech → server] Detected & sent chunk #%d (prob=%.2f) | "
                                    "Avg server processing: %.2f s",
                                    chunks_sent, speech_prob, avg_rtt
                                )
                            else:
                                log.info(
                                    "[speech → server] Detected & sent chunk #%d (prob=%.2f) | "
                                    "Awaiting first response...",
                                    chunks_sent, speech_prob
                                )

                        except websockets.ConnectionClosed:
                            log.warning("Connection closed while sending audio chunk")
                            return
                        pending_sends.append(send_start)  # in case the above log branch was taken

                    # Silence chunks are skipped silently (debug if needed)

                if processed > 0:
                    log.debug("Processed %d speech chunk(s) this cycle", processed)

        except asyncio.CancelledError:
            log.info("[task] Streaming task cancelled")
            raise
        finally:
            log.info(
                "[microphone] Microphone streaming stopped | "
                "Speech chunks detected: %d | Sent to server: %d",
                chunks_detected, chunks_sent
            )
            if recent_rtts:
                log.info("Average server processing time: %.2f s", sum(recent_rtts) / len(recent_rtts))


async def receive_subtitles(ws) -> None:
    """
    Receive partial/final subtitles from server and measure processing latency.
    """
    async for msg in ws:
        data = json.loads(msg)

        if data.get("type") == "subtitle":
            ja = data.get("transcription", "").strip()
            en = data.get("translation", "").strip()

            # Record server processing time (time from last chunk send to subtitle receipt)
            if pending_sends:
                rtt = time.monotonic() - pending_sends.popleft()
                recent_rtts.append(rtt)
                # last_send_start = None

            if ja or en:
                log.info("[subtitle] JA: %s", ja)
                if en:
                    log.info("[subtitle] EN: %s", en)
            else:
                log.debug("[subtitle] Empty transcription received")


# =============================
# Main entrypoint
# =============================

async def main() -> None:
    try:
        async with websockets.connect(
            WS_URL,
            max_size=None,
            compression=None,
            ping_interval=30,      # Increase to 30s for high traffic
            ping_timeout=30,       # Wait 30s for pong
            close_timeout=10,      # Reduce close timeout
        ) as ws:
            log.info("Connected to %s", WS_URL)
            await asyncio.gather(
                stream_microphone(ws),
                receive_subtitles(ws),
            )
    except websockets.ConnectionClosedOK:
        log.info("Connection closed normally by server")
    except websockets.ConnectionClosedError as e:
        log.warning("Connection closed abnormally: %s", e)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.exception("Unexpected error: %s", e)
    finally:
        log.info("Client shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
