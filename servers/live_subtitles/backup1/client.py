"""Live Subtitles WebSocket Client

Connects to the faster-whisper server, captures audio from microphone (or system loopback),
resamples to 16kHz mono int16 PCM, streams raw bytes in real-time.

Designed for providing live Japanese captions on adult streaming videos played locally.
Uses sounddevice for capture and pyrubberband or sounddevice resampling if needed.
"""

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box

# ----------------------------------------------------------------------
# Logging setup with rich
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(stderr=True), show_path=False)],
)
log = logging.getLogger("live_subtitles_client")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SERVER_URL = "ws://localhost:8765"          # Change if server is remote
TARGET_SR = 16000                           # Must match server expectation
CHANNELS = 1                                # Mono
DTYPE = "int16"                             # Raw PCM16
BLOCKSIZE = 4096                            # Audio chunk size in frames (~0.25s at 16kHz)
DEVICE: Optional[int] = None                # None = default input; set index for specific mic/loopback

# Display settings
MAX_SUBTITLES = 10                          # How many recent lines to show

# ----------------------------------------------------------------------
# Global state for live display
# ----------------------------------------------------------------------
@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str

subtitle_queue: queue.Queue[Optional[SubtitleLine]] = queue.Queue()
recent_subtitles: list[SubtitleLine] = []

def make_display_table() -> Table:
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 1), expand=True)
    table.add_column("Time", style="cyan", justify="right")
    table.add_column("Text", style="white")
    
    for sub in recent_subtitles[-MAX_SUBTITLES:]:
        time_str = f"{sub.start:.1f}s → {sub.end:.1f}s"
        table.add_row(time_str, sub.text.strip())
    
    return table

# ----------------------------------------------------------------------
# WebSocket receiver task
# ----------------------------------------------------------------------
async def subtitle_receiver(websocket: websockets.WebSocketClientProtocol) -> None:
    """Receive JSON subtitle segments from server and update display."""
    try:
        async for message in websocket:
            try:
                import json
                data = json.loads(message)
                line = SubtitleLine(
                    start=float(data["start"]),
                    end=float(data["end"]),
                    text=data["text"],
                )
                subtitle_queue.put(line)
            except Exception as e:
                log.error("Failed to parse subtitle message: %s", e)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.error("WebSocket receive error: %s", e)
    finally:
        subtitle_queue.put(None)  # signal end

# ----------------------------------------------------------------------
# Audio capture and streaming task
# ----------------------------------------------------------------------
async def audio_streamer(websocket: websockets.WebSocketClientProtocol) -> None:
    """Capture audio from microphone/system, resample if needed, stream raw int16 bytes."""
    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        
        # Convert to int16 bytes (already int16 if input matches)
        if indata.dtype != np.int16:
            # Clip and convert safely
            audio_int16 = np.int16(indata * 32767)
        else:
            audio_int16 = indata.copy()
        
        # To mono if stereo
        if audio_int16.shape[1] > 1:
            audio_int16 = np.mean(audio_int16, axis=1, keepdims=True).astype(np.int16)
        
        raw_bytes = audio_int16.tobytes()
        
        # Put in queue for async sending
        loop.call_soon_threadsafe(send_queue.put_nowait, raw_bytes)

    send_queue: queue.Queue[bytes] = queue.Queue()
    loop = asyncio.get_running_loop()

    try:
        with sd.InputStream(
            samplerate=TARGET_SR,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            device=DEVICE,
            callback=callback,
        ):
            log.info(
                Panel(
                    f"[bold green]Streaming audio → {SERVER_URL}[/]\n"
                    f"Device: {sd.query_devices(DEVICE)['name'] if DEVICE is not None else 'default'}\n"
                    f"Rate: {TARGET_SR}Hz, Mono, PCM16",
                    title="Live Subtitles Client",
                )
            )

            while True:
                chunk = await loop.run_in_executor(None, send_queue.get)
                await websocket.send(chunk)

    except Exception as e:
        log.error("Audio capture/streaming error: %s", e)
    finally:
        log.info("Audio streaming stopped")

# ----------------------------------------------------------------------
# Main client with live display
# ----------------------------------------------------------------------
async def main() -> None:
    global recent_subtitles

    try:
        async with websockets.connect(SERVER_URL, ping_interval=20, ping_timeout=60) as websocket:
            log.info("Connected to server")

            # Start tasks
            receiver_task = asyncio.create_task(subtitle_receiver(websocket))
            streamer_task = asyncio.create_task(audio_streamer(websocket))

            with Live(make_display_table(), refresh_per_second=4, console=Console()) as live:
                while receiver_task.done() is False and streamer_task.done() is False:
                    try:
                        line = subtitle_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue

                    if line is None:  # end signal
                        break

                    recent_subtitles.append(line)
                    live.update(make_display_table())

            # Cancel tasks gracefully
            receiver_task.cancel()
            streamer_task.cancel()
            await asyncio.gather(receiver_task, streamer_task, return_exceptions=True)

    except websockets.ConnectionClosed as e:
        log.error(f"Connection closed: {e.reason or e.code}")
    except Exception as e:
        log.exception("Client error: %s", e)
    finally:
        log.info("Client stopped")

if __name__ == "__main__":
    # List devices on first run if needed
    if DEVICE is None:
        log.info("Available input devices:")
        print(sd.query_devices())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Client stopped by user")