# live_subtitles_client_with_overlay.py

import os
import shutil
import asyncio
import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import sounddevice as sd
import websockets
from pysilero_vad import SileroVoiceActivityDetector
from rich.logging import RichHandler
import scipy.io.wavfile as wavfile  # Add this import at the top with other imports
import numpy as np  # needed for saving wav with frombuffer

import datetime

import sys
from threading import Thread
from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Configuration (now flexible)
# =============================

@dataclass(frozen=True)
class Config:
    ws_url: str = os.getenv("WS_URL", "ws://192.168.68.150:8765")
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    vad_threshold: float = 0.5
    vad_model_path: str | None = None  # allow custom model if needed
    max_rtt_history: int = 10
    reconnect_attempts: int = 5
    reconnect_delay: float = 3.0

config = Config()

# =============================
# SRT global state for subtitle syncing
# =============================

# Global SRT sequence counter (1-based)
srt_sequence = 1

# Track when each segment actually started (wall-clock time)
segment_start_wallclock: dict[int, float] = {}  # segment_num → time.time()

# =============================
# Initialize Silero VAD - STATIC INFO ONLY
# =============================

# Use static constants - safe without instance
VAD_CHUNK_SAMPLES = SileroVoiceActivityDetector.chunk_samples()
VAD_CHUNK_BYTES = SileroVoiceActivityDetector.chunk_bytes()
log.info("[VAD] chunk_samples=%s, chunk_bytes=%s", VAD_CHUNK_SAMPLES, VAD_CHUNK_BYTES)

# =============================
# RTT tracking – fixed & improved
# =============================

ProcessingTime = float
recent_rtts: Deque[ProcessingTime] = deque(maxlen=config.max_rtt_history)
pending_sends: deque[float] = deque()  # One entry per sent chunk

# =============================
# Audio capture + streaming (with segment & silence tracking)
# =============================

async def stream_microphone(ws) -> None:
    # Lazy instantiate VAD only when streaming starts
    model_path = config.vad_model_path
    if model_path is None:
        # Use bundled default as per library design
        from pysilero_vad import _DEFAULT_MODEL_PATH
        model_path = _DEFAULT_MODEL_PATH
    vad = SileroVoiceActivityDetector(model_path)

    # Speech segment tracking
    speech_start_time: float | None = None
    current_speech_seconds: float = 0.0
    silence_start_time: float | None = None
    max_silence_seconds: float = 1.5  # Stop segment after 1.5s silence

    segments_dir = os.path.join(OUTPUT_DIR, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    current_segment_num: int | None = None
    current_segment_buffer: bytearray | None = None
    speech_chunks_in_segment: int = 0

    pcm_buffer = bytearray()
    chunks_sent = 0
    chunks_detected = 0
    total_chunks_processed = 0

    def audio_callback(indata, frames: int, time_, status) -> None:
        nonlocal pcm_buffer
        if status:
            log.warning("Audio callback status: %s", status)
        pcm_buffer.extend(bytes(indata))

    with sd.RawInputStream(
        samplerate=config.sample_rate,
        blocksize=VAD_CHUNK_SAMPLES,
        channels=config.channels,
        dtype=config.dtype,
        callback=audio_callback,
    ) as stream:
        log.info("[microphone] Microphone streaming started")
        try:
            while True:
                await asyncio.sleep(0.01)
                processed = 0
                while len(pcm_buffer) >= VAD_CHUNK_BYTES:
                    chunk = bytes(pcm_buffer[:VAD_CHUNK_BYTES])
                    del pcm_buffer[:VAD_CHUNK_BYTES]
                    total_chunks_processed += 1

                    speech_prob: float = vad(chunk)
                    if speech_prob >= config.vad_threshold:
                        chunks_detected += 1
                        processed += 1
                        # Reset silence timer
                        silence_start_time = None
                        if speech_start_time is None:
                            speech_start_time = time.monotonic()
                            # Start new segment
                            current_segment_num = len([d for d in os.listdir(segments_dir) if d.startswith("segment_")]) + 1
                            segment_start_wallclock[current_segment_num] = time.time()
                            current_segment_buffer = bytearray()
                            speech_chunks_in_segment = 0  # reset for new segment
                            log.info("[speech] Speech started | segment_%04d", current_segment_num)
                        # Append chunk to current segment buffer
                        if current_segment_buffer is not None:
                            current_segment_buffer.extend(chunk)
                            speech_chunks_in_segment += 1  # count chunk in segment
                        send_start = time.monotonic()
                        pending_sends.append(send_start)
                        payload = {
                            "type": "audio",
                            "sample_rate": config.sample_rate,
                            "pcm": base64.b64encode(chunk).decode("ascii"),
                        }
                        try:
                            await ws.send(json.dumps(payload))
                            chunks_sent += 1
                        except websockets.ConnectionClosed:
                            log.warning("WebSocket closed during send")
                            return
                    else:
                        # Silence chunk
                        if speech_start_time is not None and silence_start_time is None:
                            silence_start_time = time.monotonic()
                        # Check if silence too long → end segment
                        if (speech_start_time is not None and silence_start_time is not None
                            and time.monotonic() - silence_start_time > max_silence_seconds):
                            # Compute precise duration using number of samples in the segment buffer
                            if current_segment_buffer is not None:
                                num_samples = len(current_segment_buffer) // 2  # int16 = 2 bytes
                                duration = num_samples / config.sample_rate
                            else:
                                num_samples = 0
                                duration = 0.0

                            log.info(
                                "[speech] Speech segment ended | duration: %.2fs | chunks: %d",
                                duration, speech_chunks_in_segment
                            )

                            # Save segment audio + metadata
                            if current_segment_num is not None and current_segment_buffer is not None:
                                segment_dir = os.path.join(segments_dir, f"segment_{current_segment_num:04d}")
                                os.makedirs(segment_dir, exist_ok=True)

                                # Compute precise duration from audio length
                                num_samples = len(current_segment_buffer) // 2  # int16 = 2 bytes per sample
                                duration = num_samples / config.sample_rate

                                wav_path = os.path.join(segment_dir, "sound.wav")
                                wavfile.write(
                                    wav_path,
                                    config.sample_rate,
                                    np.frombuffer(current_segment_buffer, dtype=np.int16)
                                )

                                metadata = {
                                    "segment_id": current_segment_num,
                                    "duration_seconds": round(duration, 3),
                                    "approx_time_duration_seconds": round(time.monotonic() - speech_start_time, 3),
                                    "num_chunks": speech_chunks_in_segment,
                                    "num_samples": num_samples,
                                    "start_time_monotonic": speech_start_time,
                                    "end_time_monotonic": time.monotonic(),
                                    "sample_rate": config.sample_rate,
                                    "channels": config.channels,
                                }
                                meta_path = os.path.join(segment_dir, "metadata.json")
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                                # log.info("[save] Segment saved → %s | precise_duration: %.3fs", segment_dir, duration)

                            # Signal end of utterance to server
                            try:
                                await ws.send(json.dumps({"type": "end_of_utterance"}))
                                log.info("[speech → server] Sent end_of_utterance marker")
                            except websockets.ConnectionClosed:
                                log.warning("WebSocket closed while sending end marker")
                                return

                            speech_start_time = None
                            silence_start_time = None
                            current_segment_num = None
                            current_segment_buffer = None
                            speech_chunks_in_segment = 0  # reset counter

                    # if speech_prob >= config.vad_threshold:
                    #     avg_rtt = sum(recent_rtts) / len(recent_rtts) if recent_rtts else None
                    #     log.info(
                    #         "[speech → server] Sent chunk %d | prob: %.3f | segment: %.2fs%s",
                    #         chunks_sent,
                    #         speech_prob,
                    #         time.monotonic() - speech_start_time if speech_start_time else current_speech_seconds,
                    #         f" | avg_rtt: {avg_rtt:.3f}s" if avg_rtt is not None else ""
                    #     )
                if processed > 0:
                    log.debug("Processed %d speech chunk(s) this cycle", processed)
                # Periodic status update
                if total_chunks_processed % 100 == 0:
                    status = "SPEAKING" if speech_start_time else "SILENCE"
                    seg_dur = time.monotonic() - speech_start_time if speech_start_time else 0.0
                    log.info(
                        "[status] chunks_processed: %d | sent: %d | detected: %d | state: %s | seg: %.2fs",
                        total_chunks_processed, chunks_sent, chunks_detected, status, seg_dur
                    )
        except asyncio.CancelledError:
            log.info("[task] Streaming task cancelled")
            raise
        finally:
            log.info(
                "[microphone] Stopped | Processed: %d | Detected: %d | Sent: %d",
                total_chunks_processed, chunks_detected, chunks_sent
            )
            if recent_rtts:
                log.info("Final avg server processing time: %.3fs", sum(recent_rtts) / len(recent_rtts))

# =============================
# Subtitle SRT writing helpers
# =============================

def write_srt_block(
    sequence: int,
    start_sec: float,
    duration_sec: float,
    ja: str,
    en: str,
    file_path: str | os.PathLike
) -> None:
    """Append one subtitle block to an SRT file."""
    start_dt = datetime.datetime.fromtimestamp(start_sec)
    end_dt   = datetime.datetime.fromtimestamp(start_sec + duration_sec)

    start_str = start_dt.strftime("%H:%M:%S") + f",{int((start_dt.microsecond / 1000)) :03d}"
    end_str   = end_dt.strftime("%H:%M:%S")   + f",{int((end_dt.microsecond / 1000))   :03d}"

    block = (
        f"{sequence}\n"
        f"{start_str} --> {end_str}\n"
        f"{ja.strip()}\n"
        f"{en.strip()}\n"
        "\n"
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(block)

    # log.info("[SRT] Appended block #%d to %s", sequence, os.path.basename(file_path))

# =============================
# Subtitles/RTT receiver with SRT writing
# =============================

async def receive_subtitles(ws) -> None:
    global srt_sequence

    all_srt_path = os.path.join(OUTPUT_DIR, "all_subtitles.srt")

    async for msg in ws:
        try:
            data = json.loads(msg)
            if data.get("type") != "subtitle":
                continue

            ja = data.get("transcription", "").strip()
            en = data.get("translation", "").strip()
            utterance_id = data.get("utterance_id")
            duration_sec = data.get("duration_sec", 0.0)

            if pending_sends:
                rtt = time.monotonic() - pending_sends.popleft()
                recent_rtts.append(rtt)

            if not (ja or en):
                log.debug("[subtitle] Empty transcription received")
                continue

            log.info("[subtitle] JA: %s", ja)
            if en:
                log.info("[subtitle] EN: %s", en)

            # Find which segment this belongs to (utterance_id == segment_num)
            segment_num = utterance_id + 1   # usually 0-based from server → 1-based folder
            segment_dir = os.path.join(OUTPUT_DIR, "segments", f"segment_{segment_num:04d}")

            start_time = segment_start_wallclock.get(segment_num)
            if start_time is None:
                log.warning("[SRT] No start time recorded for segment_%04d — using current time", segment_num)
                start_time = time.time() - duration_sec

            # Write per-segment SRT
            per_seg_srt = os.path.join(segment_dir, "subtitles.srt")
            write_srt_block(
                sequence=srt_sequence,
                start_sec=start_time,
                duration_sec=duration_sec,
                ja=ja,
                en=en,
                file_path=per_seg_srt
            )

            # Append to global SRT
            write_srt_block(
                sequence=srt_sequence,
                start_sec=start_time,
                duration_sec=duration_sec,
                ja=ja,
                en=en,
                file_path=all_srt_path
            )

            srt_sequence += 1

            # Display the segment on the live overlay
            overlay.add_message(
                translated_text=en,
                source_text=ja,
                start_sec=start_time,
                end_sec=start_time + duration_sec,
                duration_sec=duration_sec,
            )

        except Exception as e:
            log.exception("[subtitle receive] Error processing message: %s", e)

# =============================
# Main with reconnection logic
# =============================

async def main() -> None:
    attempt = 0
    while True:
        try:
            async with websockets.connect(
                config.ws_url,
                max_size=None,
                compression=None,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                attempt = 0  # reset on success
                log.info("Connected to %s", config.ws_url)
                await asyncio.gather(
                    stream_microphone(ws),
                    receive_subtitles(ws),
                )
                break  # normal exit

        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
            log.warning("Connection closed")
        except OSError as e:
            log.error("Network error: %s", e)
        except Exception as e:
            log.exception("Unexpected error: %s", e)

        attempt += 1
        if attempt >= config.reconnect_attempts:
            log.error("Max reconnection attempts reached. Exiting.")
            break

        delay = config.reconnect_delay * (2 ** (attempt - 1))  # exponential backoff
        log.info("Reconnecting in %.1fs (attempt %d/%d)...", delay, attempt, config.reconnect_attempts)
        await asyncio.sleep(delay)

    log.info("Client shutdown complete")

if __name__ == "__main__":
    app = QApplication([])  # Re-uses existing instance if any
    overlay = LiveSubtitlesOverlay.create(app=app, title="Live Japanese Subtitles")
    
    def recording_thread() -> None:
        asyncio.run(main())
    
    Thread(target=recording_thread, daemon=True).start()
    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())