# live_subtitles_client_with_overlay.py

import os
import shutil
import asyncio
import base64
import json
import logging
import time
import queue
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

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
ALL_SPEECH_META_PATH = os.path.join(OUTPUT_DIR, "all_speech_meta.json")
ALL_TRANSLATION_META_PATH = os.path.join(OUTPUT_DIR, "all_translation_meta.json")
ALL_SPEECH_PROBS_INDEX_PATH = os.path.join(OUTPUT_DIR, "all_speech_probs.json")
ALL_PROBS_PATH = os.path.join(OUTPUT_DIR, "all_probs.json")

CONTINUOUS_AUDIO_MAX_SECONDS = 320.0  # a bit more than 5 min
audio_buffer: deque[tuple[float, bytes]] = deque()   # (monotonic_time, chunk)
audio_total_samples: int = 0
LAST_5MIN_WAV = os.path.join(OUTPUT_DIR, "last_5_mins.wav")

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
    min_speech_duration: float = 0.25          # seconds; ignore shorter speech bursts
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    vad_threshold: float = 0.3
    max_silence_seconds: float = 0.5   # seconds of silence before ending segment
    vad_model_path: str | None = None  # allow custom model if needed
    max_rtt_history: int = 10
    reconnect_attempts: int = 5
    reconnect_delay: float = 3.0

config = Config()

# =============================
# SRT global state for subtitle syncing
# =============================

# Global: when the recording stream actually began (wall-clock)
stream_start_time: float | None = None

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

    # Declare globals that are modified inside this function
    global audio_total_samples, audio_buffer

    # Speech segment tracking
    speech_start_time: float | None = None
    current_speech_seconds: float = 0.0
    silence_start_time: float | None = None
    max_vad_confidence: float = 0.0           # ← new
    last_vad_confidence: float = 0.0          # ← new
    speech_duration_sec: float = 0.0
    first_vad_confidence: float = 0.0         # ← new: confidence of the first speech chunk
    min_vad_confidence: float = 1.0           # ← new: lowest confidence during speech
    vad_confidence_sum: float = 0.0           # ← new: for average calculation
    speech_chunk_count: int = 0               # ← new: number of speech chunks (for avg)

    # ← NEW: audio energy tracking
    speech_energy_sum: float = 0.0          # sum of RMS values for speech chunks
    speech_energy_sum_squares: float = 0.0  # sum of squares for variance / std dev
    max_energy: float = 0.0                 # peak RMS in segment
    min_energy: float = float("inf")        # lowest RMS in speech chunk

    segments_dir = os.path.join(OUTPUT_DIR, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    current_segment_num: int | None = None
    current_segment_buffer: bytearray | None = None
    speech_chunks_in_segment: int = 0

    # Thread-safe queue for audio chunks from callback → asyncio task
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
    speech_prob_history: list[float] = []           # ← NEW: per-segment VAD probs
    all_prob_history: list[float] = []
    chunk_type: Literal["speech", "non_speech", "silent"] = "silent"
    chunks_sent = 0
    chunks_detected = 0
    total_chunks_processed = 0

    def audio_callback(indata, frames: int, time_, status) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        # indata is a read-only bytes-like object; we copy it to owned bytes
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            # Rare: consumer too slow → drop frame to avoid blocking callback
            log.warning("[audio] Queue full – dropping audio frame to prevent callback blockage")

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
                while True:
                    try:
                        chunk = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        total_chunks_processed += 1

                    # chunk is exactly VAD_CHUNK_BYTES long (sounddevice blocksize)
                    # but we double-check for safety
                    if len(chunk) != VAD_CHUNK_BYTES:
                        log.warning("[audio] Unexpected chunk size: %d (expected %d)", len(chunk), VAD_CHUNK_BYTES)
                        continue

                    # ← NEW: compute RMS energy for this chunk
                    chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(chunk_np ** 2))

                    has_sound = rms > 50.0

                    # Skip processing if there is no sufficiently loud sound in the audio chunk
                    # if not has_sound:
                    #     continue

                    speech_prob: float = vad(chunk) if has_sound else 0.0

                    all_prob_history.append(speech_prob)

                    if not has_sound:
                        chunk_type = "silent"
                    elif speech_prob >= config.vad_threshold:
                        chunk_type = "speech"
                    else:
                        chunk_type = "non_speech"

                    if has_sound:
                        # Always add to continuous buffer (audible speech or non_speech)
                        chunk_time = time.monotonic()
                        audio_buffer.append((chunk_time, chunk, speech_prob, rms, chunk_type))
                        audio_total_samples += len(chunk) // 2   # int16

                    # Trim old data
                    while audio_total_samples / config.sample_rate > CONTINUOUS_AUDIO_MAX_SECONDS:
                        if audio_buffer:
                            _, old_chunk, _, _, _ = audio_buffer.popleft()
                            audio_total_samples -= len(old_chunk) // 2

                    # Write file every ~60 chunks
                    if total_chunks_processed % 60 == 0 and total_chunks_processed > 0:
                        _write_last_5min_wav()


                    if len(all_prob_history) % 100 == 0:
                        # Save all probs
                        _append_to_global_all_probs(all_prob_history)
                        all_prob_history = all_prob_history[-20:]   # optional: keep small overlap

                    if chunk_type == "speech":
                        chunks_detected += 1
                        processed += 1
                        # Track best & latest VAD score for this segment
                        last_vad_confidence = speech_prob
                        if speech_prob > max_vad_confidence:
                            max_vad_confidence = speech_prob
                        if speech_prob < min_vad_confidence:
                            min_vad_confidence = speech_prob
                        # Reset silence timer
                        silence_start_time = None
                        if speech_start_time is None:
                            speech_start_time = time.monotonic()
                            # Start new segment
                            current_segment_num = len([d for d in os.listdir(segments_dir) if d.startswith("segment_")]) + 1
                            segment_start_wallclock[current_segment_num] = time.time()
                            current_segment_buffer = bytearray()
                            speech_chunks_in_segment = 0  # reset for new segment
                            max_vad_confidence = 0.0                # ← reset
                            last_vad_confidence = 0.0               # ← reset
                            first_vad_confidence = speech_prob      # ← capture first speech prob
                            min_vad_confidence = speech_prob       # ← initialize min with first value
                            vad_confidence_sum = 0.0               # ← reset
                            speech_prob_history = []                # ← reset for new segment
                            speech_chunk_count = 0                 # ← reset
                            # ← NEW resets
                            speech_energy_sum = 0.0
                            speech_energy_sum_squares = 0.0
                            max_energy = 0.0
                            min_energy = float("inf")
                            log.info("[speech] Speech started | segment_%04d", current_segment_num)
                            speech_duration_sec = 0.0
                        # Append chunk to current segment buffer
                        if current_segment_buffer is not None:
                            current_segment_buffer.extend(chunk)
                            speech_chunks_in_segment += 1  # count chunk in segment
                        # Accumulate for average
                        vad_confidence_sum += speech_prob
                        speech_prob_history.append(speech_prob)         # ← NEW: store every accepted prob
                        speech_chunk_count += 1

                        if rms > max_energy:
                            max_energy = rms
                        if rms < min_energy:
                            min_energy = rms
                        speech_energy_sum += rms
                        speech_energy_sum_squares += rms ** 2

                        send_start = time.monotonic()
                        pending_sends.append(send_start)
                        speech_duration_sec += VAD_CHUNK_SAMPLES / config.sample_rate
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
                            and time.monotonic() - silence_start_time > config.max_silence_seconds):
                            # Compute precise duration using number of samples in the segment buffer
                            if current_segment_buffer is not None:
                                num_samples = len(current_segment_buffer) // 2  # int16 = 2 bytes
                                duration = num_samples / config.sample_rate
                            else:
                                num_samples = 0
                                duration = 0.0

                            if speech_duration_sec < config.min_speech_duration:
                                log.info(
                                    "[speech] Segment too short (%.3fs < %.3fs) — discarded",
                                    speech_duration_sec, config.min_speech_duration
                                )
                                # Reset without sending or saving
                                speech_start_time = None
                                silence_start_time = None
                                current_segment_num = None
                                current_segment_buffer = None
                                speech_chunks_in_segment = 0
                                continue

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
                                base_time = stream_start_time or segment_start_wallclock[current_segment_num]

                                # Use relative seconds from stream start (fallback to current segment start time if stream_start_time isn't set)
                                metadata = {
                                    "segment_id": current_segment_num,
                                    "duration_sec": round(duration, 3),
                                    "num_chunks": speech_chunks_in_segment,
                                    "num_samples": num_samples,
                                    "start_sec": round(segment_start_wallclock[current_segment_num] - base_time, 3),
                                    "end_sec": round(time.time() - base_time, 3),
                                    "sample_rate": config.sample_rate,
                                    "channels": config.channels,
                                    "vad_confidence": {
                                        "first": round(first_vad_confidence, 4),
                                        "last": round(last_vad_confidence, 4),
                                        "min": round(min_vad_confidence, 4),
                                        "max": round(max_vad_confidence, 4),
                                        "ave": round(vad_confidence_sum / speech_chunk_count, 4)
                                            if speech_chunk_count > 0
                                            else 0.0,
                                    },
                                    "audio_energy": {
                                        # ← CHANGED: explicitly convert numpy float32 → Python float
                                        "rms_min": float(round(min_energy, 4)) if min_energy != float("inf") else 0.0,
                                        "rms_max": float(round(max_energy, 4)),
                                        "rms_ave": float(round(speech_energy_sum / speech_chunk_count, 4))
                                            if speech_chunk_count > 0
                                            else 0.0,
                                        "rms_std": float(round(
                                            np.sqrt(
                                                (speech_energy_sum_squares / speech_chunk_count)
                                                - (speech_energy_sum / speech_chunk_count) ** 2
                                            ),
                                            4,
                                        ))
                                        if speech_chunk_count > 0
                                        else 0.0,
                                    },
                                }

                                # Define and write speech_probs.json **first**
                                probs_path = os.path.join(segment_dir, "speech_probs.json")
                                with open(probs_path, "w", encoding="utf-8") as f:
                                    json.dump({
                                        "segment_id": current_segment_num,
                                        "vad_threshold": config.vad_threshold,
                                        "chunk_count": len(speech_prob_history),
                                        "probs": [round(p, 4) for p in speech_prob_history],
                                    }, f, indent=2, ensure_ascii=False)

                                # Now we can safely reference probs_path in metadata
                                metadata["speech_probs_path"] = str(probs_path)
                                metadata["speech_prob_count"] = len(speech_prob_history)

                                speech_meta_path = os.path.join(segment_dir, "speech_meta.json")
                                with open(speech_meta_path, "w", encoding="utf-8") as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                                # Append to central all_speech_meta.json
                                all_speech_meta = []
                                if os.path.exists(ALL_SPEECH_META_PATH):
                                    try:
                                        with open(ALL_SPEECH_META_PATH, "r", encoding="utf-8") as f:
                                            all_speech_meta = json.load(f)
                                    except Exception:
                                        pass  # start fresh if corrupted
                                relative_start = segment_start_wallclock[current_segment_num] - base_time
                                all_speech_meta.append({
                                    "segment_id": current_segment_num,
                                    **metadata,
                                    "start_sec": round(relative_start, 3),
                                    "end_sec": round(relative_start + duration, 3),
                                    "duration_sec": round(duration, 3),
                                    "segment_dir": f"segment_{current_segment_num:04d}",
                                    "wav_path": str(wav_path),
                                    "meta_path": str(speech_meta_path),
                                })
                                with open(ALL_SPEECH_META_PATH, "w", encoding="utf-8") as f:
                                    json.dump(all_speech_meta, f, indent=2, ensure_ascii=False)

                                # NEW: Append to global all_speech_probs index
                                _append_to_global_speech_probs_index(
                                    current_segment_num, duration, segment_start_wallclock, base_time, probs_path, speech_prob_history)

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
                            speech_prob_history = [] # clear after save

                    if speech_prob >= config.vad_threshold:
                        avg_rtt = sum(recent_rtts) / len(recent_rtts) if recent_rtts else None
                        
                        energy_info = ""
                        if speech_chunk_count > 0:
                            avg_rms   = speech_energy_sum / speech_chunk_count
                            rms_std   = np.sqrt(
                                (speech_energy_sum_squares / speech_chunk_count) -
                                (speech_energy_sum / speech_chunk_count) ** 2
                            )
                            energy_info = (
                                f" | rms: avg={avg_rms:.4f} ±{rms_std:.4f} "
                                f"[min={min_energy:.4f} max={max_energy:.4f}]"
                            )

                        log.info(
                            "[speech → server] Sent chunk %d | rms=%.4f | speech_prob=%.3f | dur=%.2fs%s%s",
                            chunks_sent,
                            rms,
                            speech_prob,
                            time.monotonic() - speech_start_time if speech_start_time else current_speech_seconds,
                            f" | RTT={avg_rtt:.3f}s" if avg_rtt is not None else "",
                            energy_info
                        )
                    else:
                        # Silence chunk – compute energy and always log speech_prob for VAD debugging
                        if has_sound:  # audible low-level sound (breath, noise, etc.)
                            log.debug(
                                "[no speech] Chunk has audible energy | rms=%.4f | speech_prob=%.3f | samples=%d",
                                rms, speech_prob, len(chunk_np)
                            )
                        else:
                            # log.debug(
                            #     "[silence] True silence chunk | speech_prob=%.3f | rms=%.4f",
                            #     speech_prob, rms
                            # )
                            pass
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

            # ── Flush remaining global probabilities ───────────────────────────────
            if all_prob_history:
                _append_to_global_all_probs(all_prob_history)
                log.info("Flushed %d remaining probabilities on stream shutdown", len(all_prob_history))

            # Final save of continuous audio
            if audio_buffer:
                log.info("[continuous] Final save of last_5_mins.wav on shutdown")
                _write_last_5min_wav()

            # Optional: also flush speech_prob_history if segment is active
            if speech_start_time is not None and speech_prob_history:
                log.warning("Active speech segment was interrupted — saving partial prob history")
                # You could save it to a special "interrupted_segment_XXX" folder
                # or just append to a global interrupted_probs.json

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
    global stream_start_time

    all_srt_path = os.path.join(OUTPUT_DIR, "all_subtitles.srt")

    async for msg in ws:
        try:
            data = json.loads(msg)
            if data.get("type") != "subtitle":
                continue

            ja = data.get("transcription_ja", "").strip()
            en = data.get("translation_en", "").strip()
            utterance_id = data.get("utterance_id")
            duration_sec = data.get("duration_sec", 0.0)

            # New fields from server
            trans_conf = data.get("transcription_confidence")
            trans_quality = data.get("transcription_quality")
            transl_conf = data.get("translation_confidence")    # normalized 0.0–1.0
            transl_quality = data.get("translation_quality")

            server_meta = data.get("meta", {})

            if pending_sends:
                rtt = time.monotonic() - pending_sends.popleft()
                recent_rtts.append(rtt)

            if not (ja or en):
                log.debug("[subtitle] Empty result received")
                continue

            log.info("[subtitle] JA: %s", ja)
            if en:
                log.info("[subtitle] EN: %s", en)

            # Log quality & confidence info
            log.info(
                "[quality] Transc: conf=%.3f | %s | Transl: conf=%.3f | %s",
                trans_conf if trans_conf is not None else 0.0,
                trans_quality or "N/A",
                transl_conf if transl_conf is not None else 0.0,
                transl_quality or "N/A"
            )

            if trans_conf is not None and trans_conf < 0.50:
                log.warning("[low-transc-conf] utt %d | conf=%.3f | %s", utterance_id, trans_conf, ja[:60])

            if transl_conf is not None and transl_conf < 0.70:
                log.warning("[low-transl-conf] utt %d | conf=%.3f | %s", utterance_id, transl_conf, en[:60])

            # Find segment
            segment_num = utterance_id + 1
            segment_dir = os.path.join(OUTPUT_DIR, "segments", f"segment_{segment_num:04d}")
            start_time = segment_start_wallclock.get(segment_num)
            if start_time is None:
                log.warning("[SRT] No start time for segment_%04d", segment_num)
                start_time = time.time() - duration_sec

            # Relative timing
            if stream_start_time is None:
                stream_start_time = start_time
                relative_start = 0.0
            else:
                relative_start = start_time - stream_start_time
            relative_end = relative_start + duration_sec

            # Prepare per-segment paths
            per_seg_srt = os.path.join(segment_dir, "subtitles.srt")
            speech_meta_path = os.path.join(segment_dir, "speech_meta.json")
            translation_meta_path = os.path.join(segment_dir, "translation_meta.json")

            # ── Build translation metadata ──────────────────────────────────────
            translation_meta = {
                "segment_id": segment_num,
                "utterance_id": utterance_id,
                "start_sec": round(relative_start, 3),
                "end_sec": round(relative_end, 3),
                "duration_sec": round(duration_sec, 3),
                "transcription": {
                    "text_ja": ja,
                    "confidence": round(trans_conf, 4) if trans_conf is not None else None,
                    "quality_label": trans_quality,
                },
                "translation": {
                    "text_en": en,
                    "confidence": round(transl_conf, 4) if transl_conf is not None else None,
                    "quality_label": transl_quality,
                },
                "server_meta": server_meta,
            }

            # Merge with existing file if present
            if os.path.exists(translation_meta_path):
                try:
                    with open(translation_meta_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    def merge_dicts(target, source):
                        for k, v in source.items():
                            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                                merge_dicts(target[k], v)
                            else:
                                target[k] = v
                    merge_dicts(existing, translation_meta)
                    translation_meta = existing
                except Exception as e:
                    log.warning("Failed to load existing translation_meta: %s", e)

            # Write per-segment translation meta
            with open(translation_meta_path, "w", encoding="utf-8") as f:
                json.dump(translation_meta, f, indent=2, ensure_ascii=False)

            # Append to global all_translation_meta.json
            all_translation = []
            if os.path.exists(ALL_TRANSLATION_META_PATH):
                try:
                    with open(ALL_TRANSLATION_META_PATH, "r", encoding="utf-8") as f:
                        all_translation = json.load(f)
                except Exception:
                    pass
            all_translation.append({
                **translation_meta,
                "segment_dir": f"segment_{segment_num:04d}",
                "translation_meta_path": str(translation_meta_path),
            })
            with open(ALL_TRANSLATION_META_PATH, "w", encoding="utf-8") as f:
                json.dump(all_translation, f, indent=2, ensure_ascii=False)

            # ── SRT output ──────────────────────────────────────────────────────
            write_srt_block(
                sequence=srt_sequence,
                start_sec=start_time,
                duration_sec=duration_sec,
                ja=ja,
                en=en,
                file_path=per_seg_srt
            )
            write_srt_block(
                sequence=srt_sequence,
                start_sec=start_time,
                duration_sec=duration_sec,
                ja=ja,
                en=en,
                file_path=all_srt_path
            )
            srt_sequence += 1

            # ── Overlay ─────────────────────────────────────────────────────────
            avg_vad_conf = 0.0
            if os.path.exists(speech_meta_path):
                try:
                    with open(speech_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    avg_vad_conf = meta.get("vad_confidence", {}).get("ave", 0.0)
                except Exception:
                    pass

            display_start = round(relative_start, 2)
            display_end = round(relative_end, 2)
            display_duration = round(duration_sec, 2)

            overlay.add_message(
                source_text=ja,
                translated_text=en,
                start_sec=display_start,
                end_sec=display_end,
                duration_sec=display_duration,
                segment_number=segment_num,
                avg_vad_confidence=round(avg_vad_conf, 3),
                transcription_confidence=round(trans_conf, 3) if trans_conf is not None else None,
                transcription_quality=trans_quality,
                translation_confidence=round(transl_conf, 3) if transl_conf is not None else None,
                translation_quality=transl_quality,
            )

        except json.JSONDecodeError:
            log.warning("[receive] Invalid JSON received")
        except Exception as e:
            log.exception("[receive] Error processing subtitle message: %s", e)

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


def _append_to_global_speech_probs_index(
    segment_num: int,
    duration: float,
    segment_start_wallclock: dict[int, float],
    base_time: float,
    probs_path: str,
    speech_prob_history: list[float],
) -> None:
    """Append segment info to the global all_speech_probs.json index."""
    all_probs_index: list[dict] = []
    if os.path.exists(ALL_SPEECH_PROBS_INDEX_PATH):
        try:
            with open(ALL_SPEECH_PROBS_INDEX_PATH, "r", encoding="utf-8") as f:
                all_probs_index = json.load(f)
        except Exception as e:
            log.warning("Could not load all_speech_probs.json: %s", e)

    relative_start = segment_start_wallclock.get(segment_num, time.time()) - base_time

    entry = {
        "segment_id": segment_num,
        "start_sec": round(relative_start, 3),
        "end_sec": round(relative_start + duration, 3),
        "duration_sec": round(duration, 3),
        "prob_count": len(speech_prob_history),  # ← fix: use actual count
        "probs_path": str(probs_path),
        "segment_dir": f"segment_{segment_num:04d}",
    }

    all_probs_index.append(entry)

    with open(ALL_SPEECH_PROBS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_probs_index, f, indent=2, ensure_ascii=False)


def _append_to_global_all_probs(
    new_probs: list[float]
) -> None:
    """Append segment info to the global all_speech_probs.json index."""
    all_probs: list[float] = []
    if os.path.exists(ALL_PROBS_PATH):
        try:
            with open(ALL_PROBS_PATH, "r", encoding="utf-8") as f:
                all_probs = json.load(f)
        except Exception as e:
            log.warning("Could not load all_probs.json: %s", e)

    all_probs.extend(new_probs)

    with open(ALL_PROBS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_probs, f, indent=2, ensure_ascii=False)


def _write_last_5min_wav():
   if not audio_buffer:
         return
   all_bytes = bytearray()
   for _, chunk, _, _, _ in audio_buffer:
         all_bytes.extend(chunk)
   arr = np.frombuffer(all_bytes, dtype=np.int16)
   wavfile.write(LAST_5MIN_WAV, config.sample_rate, arr)
   log.debug("[continuous] Updated last_5_mins.wav — %.1f seconds", 
               len(arr) / config.sample_rate)


if __name__ == "__main__":
    app = QApplication([])  # Re-uses existing instance if any
    overlay = LiveSubtitlesOverlay.create(app=app, title="Live Japanese Subtitles")
    
    def recording_thread() -> None:
        asyncio.run(main())
    
    Thread(target=recording_thread, daemon=True).start()
    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())