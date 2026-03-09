# live_subtitles_client_per_speech.py

import asyncio
import base64
import contextlib
import json
import os
import queue
import shutil
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Thread
from typing import Literal

import numpy as np  # needed for saving wav with frombuffer
import scipy.io.wavfile as wavfile  # Add this import at the top with other imports
import sounddevice as sd
import websockets
from jet.audio.helpers.energy import compute_rms, rms_to_loudness_label
from jet.audio.speech.firered.speech_accumulator import LiveSpeechSegmentAccumulator
from jet.audio.speech.firered.vad import FireRedVAD

# from rich.logging import RichHandler
from jet.logger import logger as log
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay
from PyQt6.QtWidgets import QApplication
from ws_client_subtitles_handlers import WSClientLiveSubtitleHandlers
from ws_client_subtitles_utils import build_segment_metadata, find_segments_subdir

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "segments")

# For safety — clean up entries older than ~30 utterances
MAX_PENDING = 50

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)

ALL_SPEECH_META_PATH = os.path.join(OUTPUT_DIR, "all_speech_meta.json")
ALL_TRANSLATION_META_PATH = os.path.join(OUTPUT_DIR, "all_translation_meta.json")
ALL_SPEECH_PROBS_INDEX_PATH = os.path.join(OUTPUT_DIR, "all_speech_probs.json")
ALL_PROBS_PATH = os.path.join(OUTPUT_DIR, "all_probs.json")

CONTINUOUS_AUDIO_MAX_SECONDS = 320.0

SAMPLE_RATE = 16000
DTYPE = "int16"

CHUNK_DURATION_SEC = 6.0
CHUNK_OVERLAP_SEC = 0.0
MIN_SILENCE_DURATION_SEC = 0.4
MIN_SPEECH_DURATION_SEC = 0.5
MAX_SPEECH_DURATION_SEC = CHUNK_DURATION_SEC * 2
# MAX_SPEECH_OVERLAP_SEC = 10.0
CONTEXT_PROMPT_MAX_WORDS = 40  # max tokens for context prompt to send to server

SMALL_OVERLAP_SEC = 0.0

audio_total_samples: int = 0
LAST_5MIN_WAV = os.path.join(OUTPUT_DIR, "last_5_mins.wav")

_last_written_total_bytes: int = 0
_audio_buffer_is_dirty: bool = False

# =============================
# Configuration (now flexible)
# =============================


@dataclass(frozen=True)
class Config:
    ws_url: str = os.getenv("LOCAL_WS_LIVE_SUBTITLES_URL")
    sample_rate: int = SAMPLE_RATE
    dtype: str = DTYPE
    channels: int = 1
    min_silence_duration_sec: float = MIN_SILENCE_DURATION_SEC  # JP clause pauses
    min_speech_duration_sec: float = MIN_SPEECH_DURATION_SEC
    max_speech_duration_sec: float = MAX_SPEECH_DURATION_SEC
    # overlap_samples: int = int(MAX_SPEECH_OVERLAP_SEC * SAMPLE_RATE)
    vad_start_threshold: float = 0.50  # hysteresis start
    vad_end_threshold: float = 0.50  # hysteresis end
    pre_roll_seconds: float = (
        MIN_SILENCE_DURATION_SEC + MIN_SILENCE_DURATION_SEC
    )  # capture mora onsets
    vad_model_path: str | None = None  # allow custom model if needed
    max_rtt_history: int = 10
    reconnect_attempts: int = 5
    reconnect_delay: float = 3.0

    # Make sure defaults are applied
    if max_speech_duration_sec <= 0:
        max_speech_duration_sec = 90.0


config = Config()

# =============================
# SRT global state for subtitle syncing
# =============================

# Global: when the recording stream actually began (wall-clock)
stream_start_time: float | None = None


# =============================
# VAD configuration (now using SpeechBrain)
# =============================
VAD_CHUNK_SAMPLES = 512  # typical value used by many models
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample
# log.info("[VAD] Using chunk size of %d samples (%d bytes)", VAD_CHUNK_SAMPLES, VAD_CHUNK_BYTES)

audio_buffer: deque[
    tuple[float, bytes, float, float, Literal["speech", "non_speech", "silent"]]
] = deque()

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


utterance_id_generator = uuid.uuid4  # callable
current_utterance_id: int | None = None

# Track how much was already sent
last_sent_byte_length: int = 0


# =============================
# Audio capture + streaming (with segment & silence tracking)
# =============================


async def stream_microphone(ws) -> None:
    # Lazy instantiate VAD only when streaming starts
    model_path = config.vad_model_path
    if model_path is None:
        log.info("[VAD] Initializing SpeechBrain VAD (vad-crdnn-libriparty)")
    vad = FireRedVAD()

    # Ensure globals are in a known state at function start
    global audio_total_samples, audio_buffer
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

    all_speech_meta = []

    # ← NEW: audio energy tracking
    speech_energy_sum: float = 0.0  # sum of RMS values for speech chunks
    speech_energy_sum_squares: float = 0.0  # sum of squares for variance / std dev
    max_energy: float = 0.0  # peak RMS in segment
    min_energy: float = float("inf")  # lowest RMS in speech chunk
    utterance_rms: float | None = None

    # Increased buffer to absorb transient stalls (network / disk / logging)
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=400)

    current_segment: LiveSpeechSegmentAccumulator | None = None
    current_segment_num: int | None = None
    speech_chunks_in_segment: int = 0

    # Utterance-level/chunking state for live chunked sending
    last_chunk_sent_time = None
    utterance_start_time = None
    speech_duration_accumulated = 0.0

    chunk_index = 0

    # NEW: decouple websocket sending from audio processing
    send_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=400)

    speech_prob_history: list[float] = []  # ← NEW: per-segment VAD probs
    all_prob_history: list[float] = []
    chunk_type: Literal["speech", "non_speech", "silent"] = "silent"
    chunks_sent = 0
    chunks_detected = 0
    total_chunks_processed = 0
    speech_start_time = None
    segment_type: Literal["speech", "non_speech"] = "non_speech"

    fname = "sound.wav"

    global last_sent_byte_length

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

                    # ─── SpeechBrain VAD call ───────────────────────────────
                    speech_prob: float = vad.get_speech_prob(chunk_np)
                    all_prob_history.append(speech_prob)

                    # NEW: hysteresis-based speech decision
                    if not is_speech_ongoing:
                        is_speech_chunk = speech_prob >= config.vad_start_threshold
                    else:
                        is_speech_chunk = speech_prob >= config.vad_end_threshold

                    chunk_type = "speech" if is_speech_chunk else "non_speech"

                    # ────────────────────────────────────────────────
                    # Temporary VAD debug logging ────────────────────

                    # Only log near-misses when NOT already speaking
                    if (
                        not is_speech_ongoing
                        and config.vad_end_threshold
                        < speech_prob
                        < config.vad_start_threshold
                    ):
                        log.warning(
                            "[near-miss] rms=%.4f | prob=%.3f | threshold=%.2f | would start? %s",
                            rms,
                            speech_prob,
                            config.vad_start_threshold,
                            "YES"
                            if speech_prob >= config.vad_start_threshold
                            else "no",
                        )
                    # Log audible chunk (good for short test runs)
                    elif rms and not is_speech_chunk:
                        log.debug(
                            "[vad every] rms=%s | prob=%.3f | qsize=%3d | ongoing=%s",
                            chunk_energy_label,
                            speech_prob,
                            audio_queue.qsize(),
                            "yes" if is_speech_ongoing else "no",
                        )

                    # Write file every ~60 chunks
                    if total_chunks_processed % 60 == 0 and total_chunks_processed > 0:
                        _write_last_5min_wav()

                    if len(all_prob_history) % 100 == 0:
                        # Save all probs
                        _append_to_global_all_probs(all_prob_history)
                        all_prob_history = all_prob_history[
                            -20:
                        ]  # optional: keep small overlap

                    def reset_speech():
                        nonlocal current_segment
                        if current_segment is not None:
                            current_segment.reset()
                        log.success(
                            "[speech] Speech started | segment_%04d",
                            current_segment_num,
                        )
                        start_new_utterance()  # ← crucial: start tracking utterance here

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

                            # Update pre_roll_buffer with the last config.pre_roll_seconds from audio_buffer chunks
                            pre_roll_buffer.clear()
                            pre_roll_chunks = int(
                                config.pre_roll_seconds
                                * config.sample_rate
                                / VAD_CHUNK_SAMPLES
                            )
                            pre_roll_probs = []
                            for _, pre_chunk, speech_prob, *_ in list(audio_buffer)[
                                -pre_roll_chunks:
                            ]:
                                pre_roll_buffer.append(pre_chunk)
                                pre_roll_probs.append(speech_prob)
                            if pre_roll_buffer:
                                avg_pre_prob = sum(pre_roll_probs) / len(pre_roll_probs)
                                log.info(
                                    "[pre_roll] Added %d pre-roll chunks (avg prob %.3f)",
                                    len(pre_roll_buffer),
                                    avg_pre_prob,
                                )
                                # Adjust start_time backwards by number of pre-roll chunks if pre_roll_buffer has length
                                adjusted_start_time = speech_start_time - (
                                    len(pre_roll_buffer)
                                    * VAD_CHUNK_SAMPLES
                                    / config.sample_rate
                                )
                            else:
                                log.info("[pre_roll] No pre-roll chunks captured")
                                adjusted_start_time = speech_start_time

                            current_segment = LiveSpeechSegmentAccumulator(
                                sample_rate=config.sample_rate,
                                pre_roll_buffer=pre_roll_buffer,
                                max_pre_roll_duration_sec=config.pre_roll_seconds,
                                # Consider also passing last_sent_byte_length if needed later
                                start_time=adjusted_start_time,
                            )

                            reset_speech()

                        # elif not has_sound:
                        #     reset_speech()

                        # Accumulate for average
                        vad_confidence_sum += speech_prob
                        speech_prob_history.append(speech_prob)
                        speech_duration_accumulated += (
                            VAD_CHUNK_SAMPLES / config.sample_rate
                        )

                        speech_chunk_count += 1  # per segment

                        if rms > max_energy:
                            max_energy = rms
                        if rms < min_energy:
                            min_energy = rms
                        speech_energy_sum += rms
                        speech_energy_sum_squares += rms**2
                        speech_duration_sec += VAD_CHUNK_SAMPLES / config.sample_rate

                        # ──────────────── Duration safety (real audio length) ────────────────
                        current_duration_sec = len(current_segment.buffer) / (
                            config.sample_rate * 2
                        )

                        # --- utterance-level chunk sending ---
                        if is_utterance_ongoing:
                            now = time.monotonic()

                            # ────────────────────────────────────────────────
                            # Ensure first partial chunk is at least CHUNK_DURATION_SEC long
                            # ────────────────────────────────────────────────

                            if last_chunk_sent_time is None:
                                # First partial send → require enough accumulated speech duration
                                should_send_chunk = (
                                    speech_duration_accumulated >= CHUNK_DURATION_SEC
                                )
                            else:
                                # Subsequent partial sends → use time interval
                                should_send_chunk = (
                                    now - last_chunk_sent_time >= CHUNK_DURATION_SEC
                                )
                            duration_exceeded = (
                                current_duration_sec > config.max_speech_duration_sec
                            )
                            # ────────────────────────────────────────────────
                            # CHANGED: send mostly NEW audio only
                            #          use small overlap (0.8–1.5s) only at start of chunk
                            # ────────────────────────────────────────────────

                            small_overlap_bytes = int(
                                SMALL_OVERLAP_SEC * config.sample_rate * 2
                            )

                            if last_sent_byte_length == 0:
                                to_send = current_segment.buffer
                                sent_overlap_sec = 0.0
                            else:
                                # ensure pointer never exceeds buffer after trims
                                if last_sent_byte_length > len(current_segment.buffer):
                                    last_sent_byte_length = len(current_segment.buffer)

                                start_idx = last_sent_byte_length

                                # if no new data was appended skip sending
                                if start_idx >= len(current_segment.buffer):
                                    continue

                                if (
                                    len(current_segment.buffer) - last_sent_byte_length
                                    < small_overlap_bytes * 2
                                ):
                                    start_idx = max(
                                        0, last_sent_byte_length - small_overlap_bytes
                                    )

                                sent_overlap_sec = (
                                    last_sent_byte_length - start_idx
                                ) / (config.sample_rate * 2)
                                to_send = current_segment.buffer[start_idx:]

                            # ──────────────── NEW: per-segment subdirectory ────────────────
                            seg_subdir = find_segments_subdir(
                                segments_root=SEGMENTS_DIR,
                                utterance_id=current_utterance_id,
                                chunk_index=chunk_index,
                                create_if_missing=True,
                            )
                            # ────────────────────────────────────────────────────────────────

                            # Only send partial chunks once enough accumulated speech is available
                            if should_send_chunk or duration_exceeded:
                                # if (
                                #     speech_duration_accumulated
                                #     < config.min_speech_duration_sec
                                # ):
                                #     # Do not send partial chunks if still below min duration
                                #     continue

                                assert current_segment is not None

                                # ─── CRITICAL: Trim BEFORE sending if already too long ───
                                if duration_exceeded:
                                    log.warning(
                                        "[speech] Hard trimming – buffer reached %.1fs (max %.1fs)",
                                        current_duration_sec,
                                        config.max_speech_duration_sec,
                                    )
                                    # current_segment.trim_audio(
                                    #     config.max_speech_duration_sec // 2
                                    # )
                                    # Option A: trim old content (alternative approach)
                                    # current_segment.trim_audio(config.max_speech_duration_sec * 0.3)

                                    # Option B: send only new data (recommended for no duplicates)
                                    # Only send what hasn't been sent before in this segment
                                    # For hard trim: keep only recent part (already sliced above, but enforce)
                                    max_keep_bytes = int(
                                        config.max_speech_duration_sec
                                        * config.sample_rate
                                        * 2
                                    )
                                    to_send = current_segment.buffer[-max_keep_bytes:]

                                # buffer_segments = extract_and_display_buffered_segments(
                                #     # current_segment.buffer,
                                #     to_send,
                                #     is_partial=True,
                                #     chunk_duration=CHUNK_DURATION_SEC,
                                # )

                                stats = current_segment.get_stats()

                                await send_audio_chunk(
                                    send_queue,
                                    # current_segment.buffer,
                                    to_send,
                                    current_segment.start_time,
                                    duration_sec=stats["duration_sec"],
                                    overlap_sec=sent_overlap_sec,
                                    segment_num=current_segment_num,
                                    avg_vad=stats["vad_sum"]
                                    / stats["speech_chunk_count"]
                                    if stats["speech_chunk_count"] > 0
                                    else 0.0,
                                    is_final=False,
                                    chunk_index=chunk_index,
                                    speech_chunk_count=stats["speech_chunk_count"],
                                    # context_prompt=build_context_prompt(),
                                )
                                # Moved after send (both partial & final)
                                chunk_index += 1

                                # ─── NEW: Save partial audio after sending ───────────────────────
                                # Use new subdir for partial saves
                                partial_wav_path = os.path.join(seg_subdir, fname)

                                # Save the sent chunk (to_send), not the whole buffer
                                partial_audio_int16 = np.frombuffer(
                                    to_send, dtype=np.int16
                                ).copy()
                                wavfile.write(
                                    partial_wav_path,
                                    config.sample_rate,
                                    partial_audio_int16,
                                )

                                partial_audio_float = (
                                    partial_audio_int16.astype(np.float32) / 32768.0
                                )
                                sent_rms = compute_rms(partial_audio_float)
                                rms_label = rms_to_loudness_label(sent_rms)

                                # Save metadata.json in the same subdir as partial_wav_path (sound.wav)
                                partial_meta = build_segment_metadata(
                                    filename=fname,  # relative inside subdir
                                    utterance_id=current_utterance_id,
                                    chunk_index=chunk_index,
                                    is_partial=True,
                                    duration_sec=len(to_send)
                                    / (config.sample_rate * 2),
                                    start_sec=current_segment.start_time,
                                    sample_rate=config.sample_rate,
                                    channels=config.channels,
                                    sent_at=time.monotonic(),
                                    rms_label=rms_label,
                                    num_samples=len(to_send) // 2,
                                )
                                # Save as metadata.json, not <chunk>.json
                                meta_path = os.path.join(seg_subdir, "metadata.json")
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(partial_meta, f, indent=2)

                                # IMPORTANT: update to current absolute end, not len(to_send)
                                last_sent_byte_length = len(current_segment.buffer)
                                last_chunk_sent_time = now

                                # ────────────────────────────────────────────────
                                # (Moved trimming logic above, before sending)
                                # In this post-send block, only re-anchor timing if needed
                                # ────────────────────────────────────────────────

                                # Note: If duration_exceeded, trimming logic and anchor resets
                                # must happen BEFORE sending as above. This post-send soft check
                                # can remain as a warning or legacy, but buffer trim should
                                # not duplicate—just issue warning or skip.
                                if duration_exceeded:
                                    log.warning(
                                        "[speech] Max speech duration exceeded (%.1fs) — trimming segment buffer (legacy post-send warning, see above)",
                                        config.max_speech_duration_sec,
                                    )
                                    # Optional strong fix: actually shrink the accumulator buffer
                                    # (prevents memory explosion on very long speech)
                                    current_segment.trim_audio(
                                        config.max_speech_duration_sec * 0.5
                                    )

                                    # trimming invalidates previous byte pointer
                                    last_sent_byte_length = 0

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
                                > config.min_silence_duration_sec
                                or (
                                    utterance_start_time is not None
                                    and time.monotonic() - utterance_start_time
                                    > config.max_speech_duration_sec
                                )
                            )
                        ):
                            # ────────────────────────────────────────────────
                            # Final segment decision
                            # ────────────────────────────────────────────────
                            assert current_segment is not None
                            duration = current_segment.get_stats()["duration_sec"]

                            if (
                                speech_duration_accumulated
                                < config.min_speech_duration_sec
                            ):
                                log.warning(
                                    "[speech] Final segment too short (%.3fs < %.3fs) — discarded",
                                    speech_duration_accumulated,
                                    config.min_speech_duration_sec,
                                )
                                # Reset without saving or sending final chunk
                                segment_type = "non_speech"
                                speech_start_time = None
                                silence_start_time = None
                                current_segment_num = None
                                current_segment = None
                                pre_roll_buffer.clear()
                                speech_chunks_in_segment = 0
                                speech_prob_history = []
                                chunk_index = 0
                                speech_duration_accumulated = 0.0
                                current_utterance_id = None
                                continue

                            log.success(
                                "[speech] Speech segment ended | duration: %.2fs | chunks: %d",
                                duration,
                                speech_chunks_in_segment,
                            )

                            # Send final chunk if anything remains
                            if is_utterance_ongoing and len(current_segment.buffer) > 0:
                                # ─── Send only new data (same logic as partial) ───
                                small_overlap_bytes = int(
                                    SMALL_OVERLAP_SEC * config.sample_rate * 2
                                )
                                if last_sent_byte_length == 0:
                                    to_send = current_segment.buffer
                                else:
                                    if last_sent_byte_length > len(
                                        current_segment.buffer
                                    ):
                                        last_sent_byte_length = len(
                                            current_segment.buffer
                                        )

                                    start_idx = last_sent_byte_length

                                    if start_idx >= len(current_segment.buffer):
                                        continue

                                    if (
                                        len(current_segment.buffer)
                                        - last_sent_byte_length
                                        < small_overlap_bytes * 2
                                    ):
                                        start_idx = max(
                                            0,
                                            last_sent_byte_length - small_overlap_bytes,
                                        )
                                    to_send = current_segment.buffer[start_idx:]

                                # buffer_segments = extract_and_display_buffered_segments(
                                #     # current_segment.buffer,
                                #     to_send,
                                #     is_partial=False,
                                #     chunk_duration=CHUNK_DURATION_SEC,
                                # )

                                stats = current_segment.get_stats()

                                await send_audio_chunk(
                                    send_queue,
                                    to_send,
                                    # utterance_rms,
                                    current_segment.start_time,
                                    duration_sec=stats["duration_sec"],
                                    segment_num=current_segment_num,
                                    avg_vad=stats["vad_sum"]
                                    / stats["speech_chunk_count"]
                                    if stats["speech_chunk_count"] > 0
                                    else 0.0,
                                    is_final=True,
                                    chunk_index=chunk_index,
                                    speech_chunk_count=stats["speech_chunk_count"],
                                    # context_prompt=build_context_prompt(),
                                )
                                # Moved after send (both partial & final)
                                chunk_index += 1
                                chunks_sent += 1

                            # Save ORIGINAL audio (matches what server received)
                            original_bytes = bytes(current_segment.buffer)
                            audio_int16 = np.frombuffer(
                                original_bytes, dtype=np.int16
                            ).copy()

                            # Save segment audio + metadata
                            if (
                                current_segment_num is not None
                                and current_segment is not None
                            ):
                                # ──────────────── NEW: final also uses same subdir format ───────
                                seg_subdir = find_segments_subdir(
                                    segments_root=SEGMENTS_DIR,
                                    utterance_id=current_utterance_id,
                                    chunk_index=chunk_index,
                                    create_if_missing=True,
                                )

                                wav_path = os.path.join(seg_subdir, fname)

                                wavfile.write(
                                    wav_path,
                                    config.sample_rate,
                                    audio_int16,
                                )

                                duration = stats["duration_sec"]
                                num_samples = current_segment.duration_samples()
                                base_time = current_segment.start_time

                                stats = current_segment.get_stats()

                                # Centralized metadata building
                                metadata = {
                                    **build_segment_metadata(
                                        filename=fname,
                                        utterance_id=current_utterance_id,
                                        chunk_index=chunk_index,
                                        is_partial=False,
                                        duration_sec=duration,
                                        start_sec=current_segment.start_time,
                                        sample_rate=config.sample_rate,
                                        channels=config.channels,
                                        vad_stats={
                                            "first": stats.get(
                                                "vad_first", first_vad_confidence
                                            ),
                                            "last": stats.get(
                                                "vad_last", last_vad_confidence
                                            ),
                                            "min": stats["vad_min"],
                                            "max": stats["vad_max"],
                                            "avg": stats["vad_sum"]
                                            / stats["speech_chunk_count"]
                                            if stats["speech_chunk_count"] > 0
                                            else 0.0,
                                        },
                                        energy_stats={
                                            "rms_min": stats["energy_min"],
                                            "rms_max": stats["energy_max"],
                                            "rms_ave": stats["energy_sum"]
                                            / stats["speech_chunk_count"]
                                            if stats["speech_chunk_count"] > 0
                                            else 0.0,
                                            "rms_std": np.sqrt(
                                                (
                                                    stats["energy_sum_squares"]
                                                    / stats["speech_chunk_count"]
                                                )
                                                - (
                                                    stats["energy_sum"]
                                                    / stats["speech_chunk_count"]
                                                )
                                                ** 2
                                            )
                                            if stats["speech_chunk_count"] > 0
                                            else 0.0,
                                        },
                                        sent_at=time.monotonic(),
                                        num_samples=num_samples,
                                    )
                                }
                                metadata["num_chunks"] = stats["speech_chunk_count"]
                                metadata["end_sec"] = (
                                    time.monotonic()
                                )  # add end time separately

                                # Save metadata.json into segment subdirectory
                                meta_path = os.path.join(seg_subdir, "metadata.json")
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                                # Update all_speech_meta.json
                                with open(
                                    ALL_SPEECH_META_PATH, "w", encoding="utf-8"
                                ) as f:
                                    json.dump(
                                        all_speech_meta, f, indent=2, ensure_ascii=False
                                    )

                                # Optional: Append to global all_speech_probs index
                                _append_to_global_speech_probs_index(
                                    fname,
                                    duration,
                                    current_segment.start_time,
                                    base_time,
                                    meta_path,
                                    speech_prob_history,
                                )

                                # Load existing file only here (when we actually have something to append)
                                if not all_speech_meta and os.path.exists(
                                    ALL_SPEECH_META_PATH
                                ):
                                    try:
                                        with open(
                                            ALL_SPEECH_META_PATH, encoding="utf-8"
                                        ) as f:
                                            all_speech_meta = json.load(f)
                                    except Exception:
                                        all_speech_meta = []  # corrupted → start fresh

                                relative_start = current_segment.start_time - base_time
                                all_speech_meta.append(
                                    {
                                        "filename": fname,
                                        **metadata,
                                        "start_sec": relative_start,
                                        "end_sec": relative_start + duration,
                                        "duration_sec": duration,
                                        "wav_path": str(wav_path),
                                        "meta_path": str(meta_path),
                                    }
                                )
                                with open(
                                    ALL_SPEECH_META_PATH, "w", encoding="utf-8"
                                ) as f:
                                    json.dump(
                                        all_speech_meta, f, indent=2, ensure_ascii=False
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
                            last_sent_byte_length = 0
                            chunk_index = 0
                            speech_duration_accumulated = 0.0
                            utterance_rms = None

                            segment_type = "non_speech"
                            speech_start_time = None
                            silence_start_time = None
                            current_segment_num = None
                            current_segment = None
                            pre_roll_buffer.clear()
                            speech_chunks_in_segment = 0  # reset counter
                            speech_prob_history = []  # clear after save

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

                    # Add every chunk (speech or short silence) to the local buffer while utterance is ongoing
                    if speech_start_time is not None and current_segment is not None:
                        current_segment.append(chunk, speech_prob, rms)
                        speech_chunks_in_segment += 1

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
                                f" | utterance dur: {current_segment.get_duration_sec():.1f}s"
                            )

                        log.orange(
                            "[speech] dur=%.2fs | %d chunks | speech=%.3f | rms=%.4f - %s",
                            current_segment.get_duration_sec(),
                            speech_chunk_count,
                            speech_prob,
                            rms,
                            chunk_energy_label,
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


async def send_audio_chunk(
    send_queue: asyncio.Queue,
    buffer: bytearray,
    start_time: float,
    duration_sec: float = 0.0,
    overlap_sec: float = 0.0,
    segment_num: int = 0,
    segment_type: Literal["speech", "non-speech"] = "speech",
    avg_vad: float = 0.0,
    is_final: bool = False,
    chunk_index: int = 0,
    speech_chunk_count: int = 0,
    context_prompt: str = "",
) -> None:
    if len(buffer) == 0:
        return

    # Optional: also compute how much was already sent before this chunk
    previous_duration_sec = max(0.0, duration_sec - overlap_sec)

    # ─── Normalize before sending ────────────────────────────────────────
    try:
        # Convert bytearray → int16 → float32 [-1,1]
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        if len(audio_int16) == 0:
            log.warning("[norm] Empty buffer before normalization — skipping send")
            return

        # ─── Normalize to [-1, 1] BEFORE computing RMS ───────────────────────
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Optional: you can still apply your speech-weighted normalization here later
        # normalized_float = normalize_speech_loudness(
        #     audio_float, sample_rate=config.sample_rate
        # )
        # rms = compute_rms(normalized_float)

        rms = compute_rms(audio_float)  # now correct scale ~[0.0, ~1.0]

        # Use normalized PCM for transmission (but keep original int16 for now)
        pcm_to_send = audio_int16.tobytes()

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
        "segment_type": segment_type,
        "utterance_id": current_utterance_id,
        # "utterance_id": str(uuid.uuid4()),  # Temporarily create new id
        "rms": rms,
        "avg_vad_confidence": avg_vad,
        "start_time": start_time,
        "duration_sec": duration_sec,
        "overlap_sec": overlap_sec,  # ← added
        "previous_duration_sec": previous_duration_sec,  # optional
    }

    await send_queue.put(payload)

    log_fn = log.success if is_final else log.orange
    log_fn(
        "[%s] Sent chunk %d (%d so far) | rms=%s | vad=%.3f",
        "final" if is_final else "partial",
        chunk_index,
        speech_chunk_count,
        rms_to_loudness_label(rms),
        avg_vad,
        bright=True,
    )


def start_new_utterance():
    global current_utterance_id, utterance_start_time, chunk_index
    current_utterance_id = str(uuid.uuid4())
    utterance_start_time = time.monotonic()
    chunk_index = 0
    log.info("[utterance] Started new utterance %s", current_utterance_id)


def _append_to_global_speech_probs_index(
    filename: str,
    duration: float,
    start_time: float,
    base_time: float,
    meta_path: str,
    speech_prob_history: list[float],
) -> None:
    """Append audio file info to the global all_speech_probs.json index."""
    all_probs_index: list[dict] = []
    if os.path.exists(ALL_SPEECH_PROBS_INDEX_PATH):
        try:
            with open(ALL_SPEECH_PROBS_INDEX_PATH, encoding="utf-8") as f:
                all_probs_index = json.load(f)
        except Exception as e:
            log.warning("Could not load all_speech_probs.json: %s", e)

    relative_start = start_time - base_time

    entry = {
        "filename": filename,
        "start_sec": relative_start,
        "end_sec": relative_start + duration,
        "duration_sec": duration,
        "prob_count": len(speech_prob_history),
        "meta_path": str(meta_path),
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


# =============================
# Main with reconnection logic
# =============================


async def main(output_dir: str, overlay: LiveSubtitlesOverlay) -> None:
    ws_client_handlers = WSClientLiveSubtitleHandlers(output_dir, overlay)

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
                    ws_client_handlers.receive_messages(ws),
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


if __name__ == "__main__":
    app = QApplication([])  # Re-uses existing instance if any
    overlay = LiveSubtitlesOverlay.create(
        app=app,
        title="Live Japanese Subtitles",
        on_clear=_on_clear,
        hide_source_text=True,
    )

    def recording_thread() -> None:
        asyncio.run(main(OUTPUT_DIR, overlay))

    Thread(target=recording_thread, daemon=True).start()
    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())
