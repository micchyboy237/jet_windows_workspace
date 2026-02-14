# live_subtitles_client_with_overlay.py

import asyncio
import base64
import contextlib
import datetime
import json
import os
import queue
import shutil
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Literal

import numpy as np  # needed for saving wav with frombuffer
import scipy.io.wavfile as wavfile  # Add this import at the top with other imports
import sounddevice as sd
import websockets
from jet.audio.helpers.base import audio_buffer_duration
from jet.audio.helpers.energy import compute_rms, rms_to_loudness_label
from jet.audio.norm.norm_speech_loudness import normalize_speech_loudness

# from rich.logging import RichHandler
from jet.logger import logger as log
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay
from PyQt6.QtWidgets import QApplication
from pysilero_vad import SileroVoiceActivityDetector

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "segments")

pending_subtitles: dict[int, dict] = {}  # utterance_id → partial/complete data

# For safety — clean up entries older than ~30 utterances
MAX_PENDING = 50

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)

ALL_SPEECH_META_PATH = os.path.join(OUTPUT_DIR, "all_speech_meta.json")
ALL_TRANSLATION_META_PATH = os.path.join(OUTPUT_DIR, "all_translation_meta.json")
ALL_SPEECH_PROBS_INDEX_PATH = os.path.join(OUTPUT_DIR, "all_speech_probs.json")
ALL_PROBS_PATH = os.path.join(OUTPUT_DIR, "all_probs.json")

CONTINUOUS_AUDIO_MAX_SECONDS = 320.0

MAX_SPEECH_DURATION_SEC = 90.0
CHUNK_DURATION_SEC = 6.0
CHUNK_OVERLAP_SEC = 2.0
CONTEXT_PROMPT_MAX_WORDS = 40  # max tokens for context prompt to send to server

audio_total_samples: int = 0
LAST_5MIN_WAV = os.path.join(OUTPUT_DIR, "last_5_mins.wav")

_last_written_total_bytes: int = 0
_audio_buffer_is_dirty: bool = False

# =============================
# Configuration (now flexible)
# =============================


@dataclass(frozen=True)
class Config:
    ws_url: str = os.getenv("WS_URL", "ws://192.168.68.150:8765")
    min_speech_duration: float = 0.25  # JP backchannels: 「はい」「え？」
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    vad_start_threshold: float = 0.15  # hysteresis start
    vad_end_threshold: float = 0.05  # hysteresis end
    pre_roll_seconds: float = 0.35  # capture mora onsets
    max_silence_seconds: float = 0.9  # JP clause pauses
    vad_model_path: str | None = None  # allow custom model if needed
    max_rtt_history: int = 10
    reconnect_attempts: int = 5
    reconnect_delay: float = 3.0
    max_speech_duration_sec: float = MAX_SPEECH_DURATION_SEC

    # Make sure defaults are applied
    if max_speech_duration_sec <= 0:
        max_speech_duration_sec = 90.0


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

audio_buffer: deque[
    tuple[float, bytes, float, float, Literal["speech", "non_speech", "silent"]]
] = deque()  # (time, pcm, vad_prob, rms, chunk_type)

# NEW: pre-roll buffer for safe speech onset
pre_roll_buffer: deque[bytes] = deque(
    maxlen=int(config.pre_roll_seconds * config.sample_rate / VAD_CHUNK_SAMPLES)
)

# =============================
# RTT tracking – fixed & improved
# =============================

ProcessingTime = float
recent_rtts: deque[ProcessingTime] = deque(maxlen=config.max_rtt_history)
pending_sends: deque[float] = deque()  # One entry per sent chunk

latest_transcription_text: str = ""
utterance_id_generator = uuid.uuid4  # callable
current_utterance_id: int | None = None


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

    # Ensure globals are in a known state at function start
    global audio_total_samples, audio_buffer
    global latest_transcription_text
    global utterance_id_counter, current_utterance_id
    global chunk_index, utterance_start_time  # used & assigned

    # Remove dangling utterance_id warning/reset, now handled elsewhere.

    # Speech segment tracking
    speech_start_time: float | None = None
    current_speech_seconds: float = 0.0
    silence_start_time: float | None = None
    max_vad_confidence: float = 0.0  # ← new
    last_vad_confidence: float = 0.0  # ← new
    speech_duration_sec: float = 0.0
    first_vad_confidence: float = 0.0  # ← new: confidence of the first speech chunk
    min_vad_confidence: float = 1.0  # ← new: lowest confidence during speech
    vad_confidence_sum: float = 0.0  # ← new: for average calculation
    speech_chunk_count: int = 0  # ← new: number of speech chunks (for avg)

    # ← NEW: audio energy tracking
    speech_energy_sum: float = 0.0  # sum of RMS values for speech chunks
    speech_energy_sum_squares: float = 0.0  # sum of squares for variance / std dev
    max_energy: float = 0.0  # peak RMS in segment
    min_energy: float = float("inf")  # lowest RMS in speech chunk
    utterance_normalized_rms: float | None = None

    current_segment_num: int | None = None
    current_segment_buffer: bytearray | None = None
    speech_chunks_in_segment: int = 0

    # Increased buffer to absorb transient stalls (network / disk / logging)
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=300)

    # Utterance-level/chunking state for live chunked sending
    utterance_audio_buffer = bytearray()
    last_chunk_sent_time = None
    utterance_start_time = None
    last_overlap_samples = int(CHUNK_OVERLAP_SEC * config.sample_rate)

    global chunk_index
    chunk_index = 0

    # NEW: decouple websocket sending from audio processing
    send_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=300)

    speech_prob_history: list[float] = []  # ← NEW: per-segment VAD probs
    all_prob_history: list[float] = []
    chunk_type: Literal["speech", "non_speech", "silent"] = "silent"
    chunks_sent = 0
    chunks_detected = 0
    total_chunks_processed = 0
    speech_start_time = None
    segment_type: Literal["speech", "non_speech"] = "non_speech"

    async def ws_sender() -> None:
        """Dedicated websocket sender to avoid blocking audio processing."""
        try:
            while True:
                payload = await send_queue.get()
                try:
                    await ws.send(json.dumps(payload))
                except websockets.ConnectionClosed:
                    log.warning("[ws] Connection closed while sending")
                    return
        except asyncio.CancelledError:
            pass

    def audio_callback(indata, frames: int, time_, status) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        # indata is a read-only bytes-like object; we copy it to owned bytes
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            # Rare: consumer too slow → drop frame to avoid blocking callback
            # Throttle log spam
            if total_chunks_processed % 50 == 0:
                log.error("[audio] Queue full – dropping audio frame")

    with sd.RawInputStream(
        samplerate=config.sample_rate,
        blocksize=VAD_CHUNK_SAMPLES,
        channels=config.channels,
        dtype=config.dtype,
        callback=audio_callback,
    ) as stream:
        log.info("[microphone] Microphone streaming started")
        try:
            # Start websocket sender task
            sender_task = asyncio.create_task(ws_sender())

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
                        log.warning(
                            "[audio] Unexpected chunk size: %d (expected %d)",
                            len(chunk),
                            VAD_CHUNK_BYTES,
                        )
                        continue

                    # NEW: always keep pre-roll audio
                    pre_roll_buffer.append(chunk)

                    is_speech_ongoing = speech_start_time is not None
                    is_utterance_ongoing = current_utterance_id is not None

                    # ← NEW: compute RMS energy for this chunk
                    chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(chunk_np**2))

                    has_sound = rms > 0.0  # diagnostic only (do NOT gate speech)

                    # Start temp code
                    normalized_chunk_np = chunk_np / 32767.0
                    normalized_chunk_rms = compute_rms(normalized_chunk_np)
                    chunk_energy_label = rms_to_loudness_label(normalized_chunk_rms)
                    # End temp code

                    # Skip processing if there is no sufficiently loud sound in the audio chunk
                    # if not has_sound:
                    #     continue

                    speech_prob: float = vad(chunk)

                    all_prob_history.append(speech_prob)

                    # NEW: hysteresis-based speech decision
                    if not is_speech_ongoing:
                        is_speech_chunk = speech_prob >= config.vad_start_threshold
                    else:
                        is_speech_chunk = speech_prob >= config.vad_end_threshold

                    chunk_type = "speech" if is_speech_chunk else "non_speech"

                    # ────────────────────────────────────────────────
                    # Temporary VAD debug logging ────────────────────

                    # Option A: log EVERY audible chunk (good for short test runs)
                    if rms:
                        log.debug(
                            "[vad every] rms=%s | prob=%.3f | decision=%-9s | qsize=%3d | ongoing=%s",
                            chunk_energy_label,
                            speech_prob,
                            "SPEECH" if is_speech_chunk else "non-speech",
                            audio_queue.qsize(),
                            "yes" if is_speech_ongoing else "no",
                        )

                    # Option B: only log near-misses when NOT already speaking
                    # (less spam during long silence periods)
                    if (
                        not is_speech_ongoing
                        and 0.05 < speech_prob < config.vad_start_threshold
                    ):
                        log.info(
                            "[near-miss] rms=%.4f | prob=%.3f | threshold=%.2f | would start? %s",
                            rms,
                            speech_prob,
                            config.vad_start_threshold,
                            "YES"
                            if speech_prob >= config.vad_start_threshold
                            else "no",
                        )
                    # ────────────────────────────────────────────────

                    if has_sound:
                        # Always add to continuous buffer (audible speech or non_speech)
                        chunk_time = time.monotonic()
                        global _audio_buffer_is_dirty
                        audio_buffer.append(
                            (chunk_time, chunk, speech_prob, rms, chunk_type)
                        )
                        audio_total_samples += len(chunk) // 2  # int16
                        _audio_buffer_is_dirty = True

                    # Trim old data
                    while (
                        audio_total_samples / config.sample_rate
                        > CONTINUOUS_AUDIO_MAX_SECONDS
                    ):
                        if audio_buffer:
                            _, old_chunk, _, _, _ = audio_buffer.popleft()
                            audio_total_samples -= len(old_chunk) // 2

                    # Write file every ~60 chunks
                    if total_chunks_processed % 60 == 0 and total_chunks_processed > 0:
                        _write_last_5min_wav()

                    if len(all_prob_history) % 100 == 0:
                        # Save all probs
                        _append_to_global_all_probs(all_prob_history)
                        all_prob_history = all_prob_history[
                            -20:
                        ]  # optional: keep small overlap

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
                        if not is_speech_ongoing:
                            segment_type = "speech"
                            speech_start_time = time.monotonic()
                            # Start new segment
                            current_segment_num = (
                                len(
                                    [
                                        d
                                        for d in os.listdir(SEGMENTS_DIR)
                                        if d.startswith("segment_")
                                    ]
                                )
                                + 1
                            )
                            segment_start_wallclock[current_segment_num] = time.time()
                            current_segment_buffer = bytearray()

                            # NEW: prepend pre-roll audio (safe onset capture)
                            for pre_chunk in pre_roll_buffer:
                                current_segment_buffer.extend(pre_chunk)

                            speech_chunks_in_segment = 0  # reset for new segment
                            max_vad_confidence = 0.0  # ← reset
                            last_vad_confidence = 0.0  # ← reset
                            first_vad_confidence = (
                                speech_prob  # ← capture first speech prob
                            )
                            min_vad_confidence = (
                                speech_prob  # ← initialize min with first value
                            )
                            vad_confidence_sum = 0.0  # ← reset
                            speech_prob_history = []  # ← reset for new segment
                            speech_chunk_count = 0  # ← reset
                            # ← NEW resets
                            speech_energy_sum = 0.0
                            speech_energy_sum_squares = 0.0
                            max_energy = 0.0
                            min_energy = float("inf")
                            log.success(
                                "[speech] Speech started | segment_%04d",
                                current_segment_num,
                            )
                            start_new_utterance()  # ← crucial: start tracking utterance here
                            speech_duration_sec = 0.0

                        # Accumulate for average
                        vad_confidence_sum += speech_prob
                        speech_prob_history.append(
                            speech_prob
                        )  # ← NEW: store every accepted prob

                        speech_chunk_count += 1  # per segment

                        if rms > max_energy:
                            max_energy = rms
                        if rms < min_energy:
                            min_energy = rms
                        speech_energy_sum += rms
                        speech_energy_sum_squares += rms**2
                        speech_duration_sec += VAD_CHUNK_SAMPLES / config.sample_rate
                        if current_segment_buffer is not None:  # still per segment
                            speech_chunks_in_segment += 1

                        # --- utterance-level chunk sending ---
                        if is_utterance_ongoing:
                            utterance_audio_buffer.extend(chunk)
                            now = time.monotonic()

                            should_send_chunk = last_chunk_sent_time is None or (
                                now - last_chunk_sent_time >= CHUNK_DURATION_SEC
                            )
                            duration_exceeded = (
                                utterance_start_time is not None
                                and now - utterance_start_time
                                > config.max_speech_duration_sec
                            )

                            if should_send_chunk or duration_exceeded:
                                await send_audio_chunk(
                                    send_queue,
                                    utterance_audio_buffer,
                                    utterance_normalized_rms,
                                    segment_num=current_segment_num,
                                    avg_vad=round(
                                        vad_confidence_sum / speech_chunk_count, 4
                                    )
                                    if speech_chunk_count > 0
                                    else 0.0,
                                    is_final=False,
                                    chunk_index=chunk_index,
                                    speech_chunk_count=speech_chunk_count,
                                    context_prompt=build_context_prompt(),
                                )
                                last_chunk_sent_time = now
                                chunk_index += 1
                                # Keep overlap
                                overlap_start = (
                                    len(utterance_audio_buffer)
                                    - last_overlap_samples * 2
                                )
                                if overlap_start > 0:
                                    utterance_audio_buffer = utterance_audio_buffer[
                                        overlap_start:
                                    ]

                    else:
                        # Silence chunk
                        if is_speech_ongoing and silence_start_time is None:
                            silence_start_time = time.monotonic()
                        # Check if silence too long → end segment
                        if (
                            is_speech_ongoing
                            and silence_start_time is not None
                            and (
                                time.monotonic() - silence_start_time
                                > config.max_silence_seconds
                                or (
                                    utterance_start_time is not None
                                    and time.monotonic() - utterance_start_time
                                    > config.max_speech_duration_sec
                                )
                            )
                        ):
                            # Final chunk (if utterance ongoing)
                            if is_utterance_ongoing and len(utterance_audio_buffer) > 0:
                                await send_audio_chunk(
                                    send_queue,
                                    utterance_audio_buffer,
                                    utterance_normalized_rms,
                                    segment_num=current_segment_num,
                                    avg_vad=round(
                                        vad_confidence_sum / speech_chunk_count, 4
                                    )
                                    if speech_chunk_count > 0
                                    else 0.0,
                                    is_final=True,
                                    chunk_index=chunk_index,
                                    speech_chunk_count=speech_chunk_count,
                                    context_prompt=build_context_prompt(),
                                )

                                chunks_sent += 1
                            # Compute precise duration using number of samples in the segment buffer
                            duration = audio_buffer_duration(
                                current_segment_buffer, config.sample_rate
                            )

                            if speech_duration_sec < config.min_speech_duration:
                                log.warning(
                                    "[speech] Segment too short (%.3fs < %.3fs) — discarded",
                                    speech_duration_sec,
                                    config.min_speech_duration,
                                )

                                # Now reset local state
                                segment_type = "non_speech"
                                speech_start_time = None
                                silence_start_time = None
                                current_segment_num = 0
                                current_segment_buffer = None
                                speech_chunks_in_segment = 0
                                speech_prob_history = []
                                continue

                            log.success(
                                "[speech] Speech segment ended | duration: %.2fs | chunks: %d",
                                duration,
                                speech_chunks_in_segment,
                            )
                            # ------------------------------------------------------------------
                            # Loudness normalization (client-side) – applied to the full utterance
                            # (pre-roll + all chunks including short intra-utterance silences)
                            # This gives the server consistent-level speech for better transcription.
                            # ------------------------------------------------------------------
                            original_bytes = bytes(current_segment_buffer)
                            audio_int16 = np.frombuffer(original_bytes, dtype=np.int16)
                            audio_float = audio_int16.astype(np.float32) / 32768.0

                            normalized_float = normalize_speech_loudness(
                                audio_float,
                                sample_rate=config.sample_rate,
                            )

                            # Convert back to int16 with proper scaling and hard clipping
                            normalized_int16 = np.clip(
                                normalized_float * 32767.0, -32768, 32767
                            ).astype(np.int16)

                            # Replace the buffer with the normalized version
                            # → local WAV files and sent payload will both be normalized
                            current_segment_buffer = bytearray(
                                normalized_int16.tobytes()
                            )

                            # ── Compute stable utterance-level normalized RMS once ───────
                            utterance_normalized_rms = compute_rms(normalized_float)

                            log.info(
                                "[norm] Applied speech loudness normalization to segment_%04d "
                                "(%.2f s → target -13 LUFS)",
                                current_segment_num,
                                duration,
                            )

                            # Save segment audio + metadata
                            if (
                                current_segment_num is not None
                                and current_segment_buffer is not None
                            ):
                                segment_dir = os.path.join(
                                    SEGMENTS_DIR, f"segment_{current_segment_num:04d}"
                                )
                                os.makedirs(segment_dir, exist_ok=True)

                                # Compute precise duration from audio length
                                duration = audio_buffer_duration(
                                    current_segment_buffer, config.sample_rate
                                )
                                num_samples = (
                                    len(current_segment_buffer) // 2
                                )  # int16 = 2 bytes per sample

                                wav_path = os.path.join(segment_dir, "sound.wav")
                                wavfile.write(
                                    wav_path,
                                    config.sample_rate,
                                    normalized_int16,  # already normalized ndarray
                                )
                                base_time = (
                                    stream_start_time
                                    or segment_start_wallclock[current_segment_num]
                                )

                                # Use relative seconds from stream start (fallback to current segment start time if stream_start_time isn't set)
                                metadata = {
                                    "segment_id": current_segment_num,
                                    "duration_sec": round(duration, 3),
                                    "num_chunks": speech_chunks_in_segment,
                                    "num_samples": num_samples,
                                    "start_sec": round(
                                        segment_start_wallclock[current_segment_num]
                                        - base_time,
                                        3,
                                    ),
                                    "end_sec": round(time.time() - base_time, 3),
                                    "sample_rate": config.sample_rate,
                                    "channels": config.channels,
                                    "vad_confidence": {
                                        "first": round(first_vad_confidence, 4),
                                        "last": round(last_vad_confidence, 4),
                                        "min": round(min_vad_confidence, 4),
                                        "max": round(max_vad_confidence, 4),
                                        "ave": round(
                                            vad_confidence_sum / speech_chunk_count, 4
                                        )
                                        if speech_chunk_count > 0
                                        else 0.0,
                                    },
                                    "audio_energy": {
                                        # ← CHANGED: explicitly convert numpy float32 → Python float
                                        "rms_min": float(round(min_energy, 4))
                                        if min_energy != float("inf")
                                        else 0.0,
                                        "rms_max": float(round(max_energy, 4)),
                                        "rms_ave": float(
                                            round(
                                                speech_energy_sum / speech_chunk_count,
                                                4,
                                            )
                                        )
                                        if speech_chunk_count > 0
                                        else 0.0,
                                        "rms_std": float(
                                            round(
                                                np.sqrt(
                                                    (
                                                        speech_energy_sum_squares
                                                        / speech_chunk_count
                                                    )
                                                    - (
                                                        speech_energy_sum
                                                        / speech_chunk_count
                                                    )
                                                    ** 2
                                                ),
                                                4,
                                            )
                                        )
                                        if speech_chunk_count > 0
                                        else 0.0,
                                    },
                                }

                                # Define and write speech_probs.json **first**
                                probs_path = os.path.join(
                                    segment_dir, "speech_probs.json"
                                )
                                with open(probs_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {
                                            "segment_id": current_segment_num,
                                            "vad_threshold": config.vad_start_threshold,
                                            "chunk_count": len(speech_prob_history),
                                            "probs": [
                                                round(p, 4) for p in speech_prob_history
                                            ],
                                        },
                                        f,
                                        indent=2,
                                        ensure_ascii=False,
                                    )

                                # Now we can safely reference probs_path in metadata
                                metadata["speech_probs_path"] = str(probs_path)
                                metadata["speech_prob_count"] = len(speech_prob_history)

                                speech_meta_path = os.path.join(
                                    segment_dir, "speech_meta.json"
                                )
                                with open(speech_meta_path, "w", encoding="utf-8") as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                                # Append to central all_speech_meta.json
                                all_speech_meta = []
                                if os.path.exists(ALL_SPEECH_META_PATH):
                                    try:
                                        with open(
                                            ALL_SPEECH_META_PATH, encoding="utf-8"
                                        ) as f:
                                            all_speech_meta = json.load(f)
                                    except Exception:
                                        pass  # start fresh if corrupted
                                relative_start = (
                                    segment_start_wallclock[current_segment_num]
                                    - base_time
                                )
                                all_speech_meta.append(
                                    {
                                        "segment_id": current_segment_num,
                                        **metadata,
                                        "start_sec": round(relative_start, 3),
                                        "end_sec": round(relative_start + duration, 3),
                                        "duration_sec": round(duration, 3),
                                        "segment_dir": f"segment_{current_segment_num:04d}",
                                        "wav_path": str(wav_path),
                                        "meta_path": str(speech_meta_path),
                                    }
                                )
                                with open(
                                    ALL_SPEECH_META_PATH, "w", encoding="utf-8"
                                ) as f:
                                    json.dump(
                                        all_speech_meta, f, indent=2, ensure_ascii=False
                                    )

                                # NEW: Append to global all_speech_probs index
                                _append_to_global_speech_probs_index(
                                    current_segment_num,
                                    duration,
                                    segment_start_wallclock,
                                    base_time,
                                    probs_path,
                                    speech_prob_history,
                                )

                            # ── Compute average VAD confidence right here ────────────────────────────────
                            avg_vad = (
                                round(vad_confidence_sum / speech_chunk_count, 4)
                                if speech_chunk_count > 0
                                else 0.0
                            )

                            # Already sent via chunk — no need to re-send full here
                            # But we still log
                            log.success(
                                "[speech] Finalized segment_%04d | %.2fs | sent as chunks",
                                current_segment_num,
                                duration,
                            )
                            # Reset utterance state
                            current_utterance_id = None
                            utterance_audio_buffer = bytearray()
                            chunk_index = 0
                            utterance_normalized_rms = None

                            segment_type = "non_speech"
                            speech_start_time = None
                            silence_start_time = None
                            current_segment_num = None
                            current_segment_buffer = None
                            speech_chunks_in_segment = 0  # reset counter
                            speech_prob_history = []  # clear after save

                    # Add every chunk (speech or short silence) to the local buffer while utterance is ongoing
                    if (
                        speech_start_time is not None
                        and current_utterance_id is None  # new utterance
                        and current_segment_buffer is not None
                    ):
                        current_segment_buffer.extend(chunk)

                    if chunk_type == "speech":
                        avg_rtt = (
                            sum(recent_rtts) / len(recent_rtts) if recent_rtts else None
                        )

                        energy_info = ""
                        if speech_chunk_count > 0:
                            avg_rms = speech_energy_sum / speech_chunk_count
                            rms_std = np.sqrt(
                                (speech_energy_sum_squares / speech_chunk_count)
                                - (speech_energy_sum / speech_chunk_count) ** 2
                            )
                            energy_info = (
                                f" | rms: avg={avg_rms:.4f} ±{rms_std:.4f} "
                                f"[min={min_energy:.4f} max={max_energy:.4f}]"
                                f" | utterance dur: {(time.monotonic() - utterance_start_time):.1f}s"
                            )

                        log.orange(
                            "[speech] %d chunks | rms=%s | speech=%.3f | dur=%.2fs%s",
                            speech_chunk_count,
                            chunk_energy_label,
                            speech_prob,
                            time.monotonic() - speech_start_time
                            if speech_start_time
                            else 0.0,
                            f" | RTT={avg_rtt:.3f}s" if avg_rtt is not None else "",
                        )

                # if processed > 0:
                #     log.debug("\nProcessed %d speech chunk(s) this cycle\n", processed)

                # Periodic status update
                if total_chunks_processed % 100 == 0:
                    status = "SPEAKING" if speech_start_time else "SILENCE"
                    seg_dur = (
                        time.monotonic() - speech_start_time
                        if speech_start_time
                        else 0.0
                    )
                    log.info(
                        "[status] chunks_processed: %d | sent: %d | detected: %d | state: %s | seg: %.2fs",
                        total_chunks_processed,
                        speech_chunk_count,
                        chunks_detected,
                        status,
                        seg_dur,
                    )
        except asyncio.CancelledError:
            log.info("[task] Streaming task cancelled")
            raise
        finally:
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender_task

            log.info(
                "[microphone] Stopped | Processed: %d | Detected: %d | Sent: %d",
                total_chunks_processed,
                chunks_detected,
                chunks_sent,
            )
            if recent_rtts:
                log.info(
                    "Final avg server processing time: %.3fs",
                    sum(recent_rtts) / len(recent_rtts),
                )

            # ── Flush remaining global probabilities ───────────────────────────────
            if all_prob_history:
                _append_to_global_all_probs(all_prob_history)
                log.info(
                    "Flushed %d remaining probabilities on stream shutdown",
                    len(all_prob_history),
                )

            # Final save of continuous audio
            if audio_buffer:
                log.info("[continuous] Final save of last_5_mins.wav on shutdown")
                _write_last_5min_wav()

            # Optional: also flush speech_prob_history if segment is active
            if is_speech_ongoing and speech_prob_history:
                log.warning(
                    "Active speech segment was interrupted — saving partial prob history"
                )
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
    file_path: str | os.PathLike,
) -> None:
    """Append one subtitle block to an SRT file."""
    start_dt = datetime.datetime.fromtimestamp(start_sec)
    end_dt = datetime.datetime.fromtimestamp(start_sec + duration_sec)

    start_str = (
        start_dt.strftime("%H:%M:%S") + f",{int(start_dt.microsecond / 1000):03d}"
    )
    end_str = end_dt.strftime("%H:%M:%S") + f",{int(end_dt.microsecond / 1000):03d}"

    block = f"{sequence}\n{start_str} --> {end_str}\n{ja.strip()}\n{en.strip()}\n\n"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(block)

    # log.info("[SRT] Appended block #%d to %s", sequence, os.path.basename(file_path))


# =============================
# Thin message receiver — receives WebSocket messages and dispatches to handler
# =============================


async def receive_messages(ws) -> None:
    """Thin receiver: recv → parse → dispatch by type"""
    async for msg in ws:
        try:
            data = json.loads(msg)
            msg_type = data.get("type")

            if msg_type == "final_subtitle":
                await handle_final_subtitle(data)

            elif msg_type == "speaker_update":
                await handle_speaker_update(data)

            elif msg_type == "emotion_classification_update":
                await handle_emotion_classification_update(data)

            elif msg_type == "partial_subtitle":
                await handle_partial_subtitle(data)

            else:
                log.warning("[ws] Ignoring unknown message type: %s", msg_type)

        except websockets.ConnectionClosed:
            log.info("[receive] WebSocket connection closed cleanly")
            break
        except json.JSONDecodeError:
            log.error("[receive] Invalid JSON received")
        except Exception as e:
            log.error("[receive] Error in receive loop: %s", e)
            await asyncio.sleep(0.3)  # prevent tight loop on repeated errors


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
                    receive_messages(ws),
                    # handle_final_subtitle & handle_speaker_update are called from receive_messages
                )
                break  # normal exit

        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
            log.warning("Connection closed")
        except OSError as e:
            log.error("Network error: %s", e)
        except Exception as e:
            log.error("Unexpected error: %s", e)

        attempt += 1
        if attempt >= config.reconnect_attempts:
            log.error("Max reconnection attempts reached. Exiting.")
            break

        delay = config.reconnect_delay * (2 ** (attempt - 1))  # exponential backoff
        log.info(
            "Reconnecting in %.1fs (attempt %d/%d)...",
            delay,
            attempt,
            config.reconnect_attempts,
        )
        await asyncio.sleep(delay)

    log.info("Client shutdown complete")


async def handle_partial_subtitle(data: dict) -> None:
    """Handle incremental/partial transcription updates from server."""
    log.debug("[partial_subtitle] Received partial update")

    utterance_id = data.get("utterance_id")
    if utterance_id is None:
        log.warning("[partial_subtitle] Missing utterance_id")
        return

    ja = data.get("transcription_ja", "").strip()
    en = data.get("translation_en", "").strip()
    if not (ja or en):
        return

    # Reuse most of the same logic as final, but mark as partial
    segment_idx = data.get("segment_idx")
    segment_num = data.get("segment_num")
    segment_type = data.get("segment_type", "speech")
    duration_sec = data.get("duration_sec", 0.0)
    avg_vad_conf = data.get("avg_vad_confidence", 0.0)
    normalized_rms = data.get("normalized_rms", 0.0)
    trans_conf = data.get("transcription_confidence")
    trans_quality = data.get("transcription_quality")
    transl_conf = data.get("translation_confidence")
    transl_quality = data.get("translation_quality")
    server_meta = data.get("meta", {})

    # For partials, we update the overlay in real-time (no SRT write yet)
    start_time = segment_start_wallclock.get(segment_num, time.time())
    relative_start = (
        (start_time - (stream_start_time or start_time)) if stream_start_time else 0.0
    )
    relative_end = relative_start + duration_sec

    # Show partial result (can be overwritten by next partial or final)
    overlay.add_message(
        message_id=utterance_id,
        source_text=ja,
        translated_text=en,
        start_sec=round(relative_start, 2),
        end_sec=round(relative_end, 2),
        duration_sec=round(duration_sec, 2),
        segment_number=segment_num,
        avg_vad_confidence=round(avg_vad_conf, 3),
        normalized_rms=round(normalized_rms, 3),
        normalized_rms_label=rms_to_loudness_label(normalized_rms),
        transcription_confidence=round(trans_conf, 3)
        if trans_conf is not None
        else None,
        transcription_quality=trans_quality,
        translation_confidence=round(transl_conf, 3)
        if transl_conf is not None
        else None,
        translation_quality=transl_quality,
        is_partial=True,  # ← optional flag if your overlay supports it
    )

    log.info(
        "[partial] utt %s | JA: %s",
        utterance_id,
        ja[:60] + "..." if len(ja) > 60 else ja,
    )
    if en:
        log.debug("[partial] EN: %s", en[:60] + "..." if len(en) > 60 else en)


async def handle_final_subtitle(data: dict) -> None:
    log.info("[final_subtitle] Handling new final subtitle update...")
    global srt_sequence, stream_start_time, latest_transcription_text

    utterance_id = data.get("utterance_id")
    if utterance_id is None:
        log.warning("[final_subtitle] Missing utterance_id")
        return

    ja = data.get("transcription_ja", "").strip()
    en = data.get("translation_en", "").strip()
    if not (ja or en):
        log.debug("[final_subtitle] Empty text received for utt %s", utterance_id)
        return

    latest_transcription_text = ja

    # segment_num = utterance_id + 1
    segment_idx = data["segment_idx"]
    segment_num = data["segment_num"]
    segment_type = data["segment_type"]

    duration_sec = data.get("duration_sec", 0.0)
    avg_vad_conf = data.get("avg_vad_confidence")
    normalized_rms = data.get("normalized_rms")
    trans_conf = data.get("transcription_confidence")
    trans_quality = data.get("transcription_quality")
    transl_conf = data.get("translation_confidence")
    transl_quality = data.get("translation_quality")
    server_meta = data.get("meta", {})

    log.info("[final_subtitle] utt %s | JA: %s", utterance_id, ja[:80])
    if en:
        log.info("[final_subtitle] EN: %s", en[:80])
    log.info(
        "[quality] Transc: %.3f %s | Transl: %.3f %s",
        trans_conf or 0.0,
        trans_quality or "N/A",
        transl_conf or 0.0,
        transl_quality or "N/A",
    )

    start_time = segment_start_wallclock.get(segment_num)
    if start_time is None:
        log.warning("[timing] No start time found for segment_%04d", segment_num)
        start_time = time.time() - duration_sec

    if stream_start_time is None:
        stream_start_time = start_time
        relative_start = 0.0
    else:
        relative_start = start_time - stream_start_time
    relative_end = relative_start + duration_sec

    # Store partial data
    pending_subtitles[utterance_id] = {
        "ja": ja,
        "en": en,
        "duration_sec": duration_sec,
        "start_wallclock": start_time,
        "relative_start": relative_start,
        "relative_end": relative_end,
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "segment_type": segment_type,
        "avg_vad_conf": avg_vad_conf,
        "normalized_rms": normalized_rms,
        "trans_conf": trans_conf,
        "trans_quality": trans_quality,
        "transl_conf": transl_conf,
        "transl_quality": transl_quality,
        "server_meta": server_meta,
        "srt_written": False,
    }

    # Show text immediately (no speaker info yet)
    await _update_display_and_files(utterance_id)

    # Optional cleanup
    if len(pending_subtitles) > MAX_PENDING:
        oldest = min(pending_subtitles.keys())
        del pending_subtitles[oldest]


async def handle_speaker_update(data: dict) -> None:
    log.info("[speaker_update] Handling new speaker update...")
    utterance_id = data.get("utterance_id")
    if utterance_id is None:
        log.warning("[speaker_update] Missing utterance_id")
        return

    # segment_num = utterance_id + 1
    segment_idx = data["segment_idx"]
    segment_num = data["segment_num"]
    segment_type = data["segment_type"]
    segment_dir = os.path.join(OUTPUT_DIR, "segments", f"segment_{segment_num:04d}")

    speaker_clusters = data.get("cluster_speakers")
    speaker_is_same = data.get("is_same_speaker_as_prev")
    speaker_similarity = data.get("similarity_prev")
    speaker_meta = {
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "segment_type": segment_type,
        "is_same_speaker_as_prev": speaker_is_same,
        "similarity_prev": speaker_similarity,
    }

    # if utterance_id in pending_subtitles:
    #     pending_subtitles[utterance_id]["speaker_meta"] = speaker_meta
    #     log.info("[speaker_update] Enriched utt %s", utterance_id)
    #     await _update_display_and_files(utterance_id)
    # else:
    #     # Rare: speaker info arrived before text
    #     pending_subtitles[utterance_id] = {"speaker_meta": speaker_meta}
    #     log.debug("[speaker_update] Stored early speaker info for utt %s", utterance_id)

    # Write per-segment speakers info
    speaker_clusters_path = os.path.join(segment_dir, "speaker_cluster.json")
    speaker_meta_path = os.path.join(segment_dir, "speaker_meta.json")
    with open(speaker_clusters_path, "w", encoding="utf-8") as f:
        json.dump(speaker_clusters, f, indent=2, ensure_ascii=False)
    with open(speaker_meta_path, "w", encoding="utf-8") as f:
        json.dump(speaker_meta, f, indent=2, ensure_ascii=False)


async def handle_emotion_classification_update(data: dict) -> None:
    log.info(
        "[emotion_classification_update] Handling new emotion classification update..."
    )
    utterance_id = data.get("utterance_id")
    if utterance_id is None:
        log.warning("[emotion_classification_update] Missing utterance_id")
        return

    segment_idx = data["segment_idx"]
    segment_num = data["segment_num"]
    segment_type = data["segment_type"]
    subdir = "segments" if segment_type == "speech" else "segments_non_speech"
    segment_dir = os.path.join(OUTPUT_DIR, subdir, f"segment_{segment_num:04d}")
    Path(segment_dir).mkdir(parents=True, exist_ok=True)

    segment_type: Literal["speech", "non_speech"] = data["segment_type"]
    emotion_classification_all = data.get("emotion_all")
    emotion_top_label = data.get("emotion_top_label")
    emotion_top_score = data.get("emotion_top_score")
    emotion_top_label = data.get("emotion_top_label")
    emotion_classification_meta = {
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "segment_type": segment_type,
        "emotion_top_label": emotion_top_label,
        "emotion_top_score": emotion_top_score,
    }

    # Write per-segment emotion classification info
    emotion_classification_all_path = os.path.join(
        segment_dir, "emotion_classification_all.json"
    )
    emotion_classification_meta_path = os.path.join(
        segment_dir, "emotion_classification_meta.json"
    )
    with open(emotion_classification_all_path, "w", encoding="utf-8") as f:
        json.dump(emotion_classification_all, f, indent=2, ensure_ascii=False)
    with open(emotion_classification_meta_path, "w", encoding="utf-8") as f:
        json.dump(emotion_classification_meta, f, indent=2, ensure_ascii=False)


async def send_audio_chunk(
    send_queue: asyncio.Queue,
    buffer: bytearray,
    utterance_normalized_rms: float,
    segment_num: int = 0,
    avg_vad: float = 0.0,
    is_final: bool = False,
    chunk_index: int = 0,
    speech_chunk_count: int = 0,
    context_prompt: str = "",
) -> None:
    if len(buffer) == 0:
        return

    # ─── Normalize before sending ────────────────────────────────────────
    try:
        # Convert bytearray → int16 → float32 [-1,1]
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        if len(audio_int16) == 0:
            log.warning("[norm] Empty buffer before normalization — skipping send")
            return

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Apply speech-probability-weighted loudness normalization
        normalized_float = normalize_speech_loudness(
            audio_float,
            sample_rate=config.sample_rate,
        )

        # Back to int16
        normalized_int16 = np.clip(normalized_float * 32767.0, -32768, 32767).astype(
            np.int16
        )
        # Use stable utterance-level value (may be None → fallback)
        normalized_rms = (
            utterance_normalized_rms
            if utterance_normalized_rms is not None
            else compute_rms(normalized_float)  # fallback for safety
        )

        # Use normalized PCM for transmission
        pcm_to_send = normalized_int16.tobytes()

    except Exception as e:
        log.error("[norm] Normalization failed — sending original audio: %s", e)
        pcm_to_send = bytes(buffer)  # fallback

    # ─── Build payload with normalized audio ─────────────────────────────
    payload = {
        "type": "complete_utterance" if is_final else "speech_chunk",
        "sample_rate": config.sample_rate,
        "pcm": base64.b64encode(pcm_to_send).decode("ascii"),
        "chunk_index": chunk_index,
        "is_final": is_final,
        "context_prompt": context_prompt,
        "segment_num": segment_num,
        "utterance_id": current_utterance_id,
        "normalized_rms": normalized_rms,
        "avg_vad_confidence": avg_vad,
    }

    await send_queue.put(payload)

    log_fn = log.success if is_final else log.orange
    log_fn(
        "[%s] Sent chunk %d (%d so far) | rms=%s | vad=%.3f",
        "final" if is_final else "partial",
        chunk_index,
        speech_chunk_count,
        rms_to_loudness_label(normalized_rms),
        avg_vad,
        bright=True,
    )


def build_context_prompt() -> str:
    global latest_transcription_text
    if not latest_transcription_text:
        return ""
    words = latest_transcription_text.strip().split()
    if len(words) <= CONTEXT_PROMPT_MAX_WORDS:
        return " ".join(words)
    return " ".join(words[-CONTEXT_PROMPT_MAX_WORDS:])


def start_new_utterance():
    global current_utterance_id, utterance_start_time, chunk_index
    current_utterance_id = str(uuid.uuid4())
    utterance_start_time = time.monotonic()
    chunk_index = 0
    log.info("[utterance] Started new utterance %s", current_utterance_id)


async def _update_display_and_files(utt_id: int) -> None:
    if utt_id not in pending_subtitles:
        return

    entry = pending_subtitles[utt_id]

    # Require text to proceed with display & SRT
    if "ja" not in entry:
        return

    ja = entry["ja"]
    en = entry["en"]
    duration_sec = entry["duration_sec"]
    relative_start = entry["relative_start"]
    relative_end = entry["relative_end"]
    segment_idx = entry["segment_idx"]
    segment_num = entry["segment_num"]
    segment_type = entry["segment_type"]
    start_time = entry["start_wallclock"]

    avg_vad_conf = entry["avg_vad_conf"]
    normalized_rms = entry["normalized_rms"]

    trans_conf = entry.get("trans_conf")
    trans_quality = entry.get("trans_quality")
    transl_conf = entry.get("transl_conf")
    transl_quality = entry.get("transl_quality")

    subdir = "segments" if segment_type == "speech" else "segments_non_speech"
    segment_dir = os.path.join(OUTPUT_DIR, subdir, f"segment_{segment_num:04d}")
    Path(segment_dir).mkdir(parents=True, exist_ok=True)

    # ── Overlay ────────────────────────────────────────────────────────────────
    overlay.add_message(
        message_id=utt_id,
        source_text=ja,
        translated_text=en,
        start_sec=round(relative_start, 2),
        end_sec=round(relative_end, 2),
        duration_sec=round(duration_sec, 2),
        segment_number=segment_num,
        avg_vad_confidence=round(avg_vad_conf, 3),
        normalized_rms=round(normalized_rms, 3),
        normalized_rms_label=rms_to_loudness_label(normalized_rms),
        transcription_confidence=round(trans_conf, 3)
        if trans_conf is not None
        else None,
        transcription_quality=trans_quality,
        translation_confidence=round(transl_conf, 3)
        if transl_conf is not None
        else None,
        translation_quality=transl_quality,
        # New speaker fields (overlay can ignore them if not ready)
        # is_same_speaker_as_prev=is_same,
        # speaker_similarity=round(similarity, 3) if similarity is not None else None,
        # speaker_clusters=clusters,
    )

    # ── SRT (write only once) ─────────────────────────────────────────────────
    per_seg_srt = os.path.join(segment_dir, "subtitles.srt")
    all_srt_path = os.path.join(OUTPUT_DIR, "all_subtitles.srt")

    if not entry.get("srt_written", False):
        global srt_sequence
        write_srt_block(srt_sequence, start_time, duration_sec, ja, en, per_seg_srt)
        write_srt_block(srt_sequence, start_time, duration_sec, ja, en, all_srt_path)
        srt_sequence += 1
        entry["srt_written"] = True


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
            with open(ALL_SPEECH_PROBS_INDEX_PATH, encoding="utf-8") as f:
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


def _append_to_global_all_probs(new_probs: list[float]) -> None:
    """Append segment info to the global all_speech_probs.json index."""
    all_probs: list[float] = []
    if os.path.exists(ALL_PROBS_PATH):
        try:
            with open(ALL_PROBS_PATH, encoding="utf-8") as f:
                all_probs = json.load(f)
        except Exception as e:
            log.warning("Could not load all_probs.json: %s", e)

    all_probs.extend(new_probs)

    with open(ALL_PROBS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_probs, f, indent=2, ensure_ascii=False)


def _write_last_5min_wav():
    global _last_written_total_bytes, _audio_buffer_is_dirty

    if not audio_buffer:
        return

    current_total_bytes = sum(len(chunk) for _, chunk, _, _, _ in audio_buffer)

    if not _audio_buffer_is_dirty and current_total_bytes == _last_written_total_bytes:
        # log.debug("[continuous] Buffer unchanged — skipping wav write")
        return

    all_bytes = bytearray()
    for _, chunk, _, _, _ in audio_buffer:
        all_bytes.extend(chunk)

    arr = np.frombuffer(all_bytes, dtype=np.int16)
    wavfile.write(LAST_5MIN_WAV, config.sample_rate, arr)
    # log.debug(
    #     "[continuous] Updated last_5_mins.wav — %.1f seconds (%s)",
    #     len(arr) / config.sample_rate,
    #     format_bytes(len(all_bytes))
    # )

    _last_written_total_bytes = len(all_bytes)
    _audio_buffer_is_dirty = False


def format_bytes(size: int) -> str:
    """
    Convert a byte count to a human-readable string (e.g., '5.2 MB', '1.3 GB').

    Args:
        size: Number of bytes (integer)

    Returns:
        Formatted string with 1 decimal place
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    # Extremely large files (shouldn't happen here)
    return f"{size:.1f} PB"


def _on_clear():
    log.info("Clear operation completed successfully ✓")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(SEGMENTS_DIR, exist_ok=True)


if __name__ == "__main__":
    app = QApplication([])  # Re-uses existing instance if any
    overlay = LiveSubtitlesOverlay.create(
        app=app,
        title="Live Japanese Subtitles",
        on_clear=_on_clear,
    )

    def recording_thread() -> None:
        asyncio.run(main())

    Thread(target=recording_thread, daemon=True).start()
    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())
