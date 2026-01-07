"""Live Subtitles WebSocket Server for Japanese Streaming Videos

Uses faster-whisper with optimized low-latency settings.
Client streams raw PCM16 audio chunks (16kHz, mono) over WebSocket.
Server uses VAD-triggered buffering: accumulates speech audio until silence,
then transcribes the segment and sends back timed subtitle segments.

Suitable for real-time captions on Japanese adult streams (noisy, breathy audio).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from rich.console import Console
from rich.logging import RichHandler
from websockets.legacy.server import serve
from websockets.server import WebSocketServerProtocol

# ----------------------------------------------------------------------
# Logging setup with rich
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(stderr=True), show_path=False)],
)
log = logging.getLogger("live_subtitles_server")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8765
SAMPLE_RATE = 16000  # Client must send 16kHz mono PCM16
SILENCE_DURATION = 0.8  # seconds of silence to trigger transcription
MIN_SPEECH_DURATION = 0.3  # ignore very short speech bursts

# Model choice: on GTX 1660 (6GB VRAM) use "medium" or "large" with int8
MODEL_SIZE = "medium"          # or "large" / "distil-large-v3" for speed
COMPUTE_TYPE = "int8_float16"  # fast on GTX 1660, good accuracy for Japanese
DEVICE = "cuda"

# ----------------------------------------------------------------------
# Load model once (shared across connections)
# ----------------------------------------------------------------------
log.info("Loading faster-whisper model %s (%s)...", MODEL_SIZE, COMPUTE_TYPE)
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=".cache",  # optional persistent cache
)

@dataclass
class ClientState:
    """Per-connection buffer and VAD state."""
    audio_buffer: bytearray = bytearray()
    speech_start: Optional[float] = None  # timestamp in seconds

    def add_chunk(self, chunk: bytes) -> None:
        self.audio_buffer.extend(chunk)

    def clear(self) -> None:
        self.audio_buffer.clear()
        self.speech_start = None

# Active connections
clients: Dict[WebSocketServerProtocol, ClientState] = {}

# ----------------------------------------------------------------------
# Transcription function (called when silence detected)
# ----------------------------------------------------------------------
async def transcribe_and_send(websocket: WebSocketServerProtocol, state: ClientState) -> None:
    if len(state.audio_buffer) < SAMPLE_RATE * 0.5:  # too short
        state.clear()
        return

    # Convert bytes to float32 numpy array (PCM16 -> float32)
    audio_np = (
        np.frombuffer(state.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    )

    # Run VAD to get speech timestamps (helps with noisy adult content)
    speech_ts = get_speech_timestamps(
        audio_np,
        VadOptions(
            min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(SILENCE_DURATION * 1000),
            window_size_samples=512,
            threshold=0.5,
        ),
    )

    if not speech_ts:
        state.clear()
        return

    # Transcribe with optimized real-time settings
    segments, _ = model.transcribe(
        audio_np,
        language="ja",
        task="transcribe",
        beam_size=1,
        temperature=0.0,
        best_of=1,
        vad_filter=True,
        chunk_length=15,  # ~15s windows for lower latency
        condition_on_previous_text=True,
        word_timestamps=False,
        without_timestamps=False,
        log_progress=False,
    )

    # Send each segment as JSON
    for seg in segments:
        await websocket.send(
            f'{{"start": {seg.start:.2f}, "end": {seg.end:.2f}, "text": "{seg.text.strip()}"}}'
        )

    log.info("Sent %d subtitle segments", len(list(segments)))
    state.clear()

# ----------------------------------------------------------------------
# WebSocket handler
# ----------------------------------------------------------------------
async def handler(websocket: WebSocketServerProtocol) -> None:
    log.info("Client connected: %s", websocket.remote_address)
    state = ClientState()
    clients[websocket] = state

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue  # ignore text/control messages

            chunk_size = len(message)
            duration = chunk_size / (2 * SAMPLE_RATE)  # 2 bytes per sample

            # Convert chunk to numpy for VAD
            chunk_np = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0

            # Quick inline VAD check on this chunk
            speech_ts = get_speech_timestamps(
                chunk_np,
                VadOptions(min_silence_duration_ms=int(SILENCE_DURATION * 1000)),
            )

            state.add_chunk(message)

            has_speech = bool(speech_ts)
            if has_speech:
                state.speech_start = None  # reset silence timer
            else:
                # Silence detected in this chunk
                if state.speech_start is None:
                    state.speech_start = asyncio.get_event_loop().time()

                elapsed_silence = asyncio.get_event_loop().time() - state.speech_start
                if elapsed_silence >= SILENCE_DURATION:
                    await transcribe_and_send(websocket, state)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.exception("Error in connection: %s", e)
    finally:
        if websocket in clients:
            del clients[websocket]
        log.info("Client disconnected: %s", websocket.remote_address)

# ----------------------------------------------------------------------
# Server startup
# ----------------------------------------------------------------------
async def main() -> None:
    log.info("Starting WebSocket server on %s:%d", HOST, PORT)
    log.info("Expected client audio: 16kHz mono PCM16 raw bytes")
    async with serve(handler, HOST, PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped")