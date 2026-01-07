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
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

import datetime

import sys
from threading import Thread
from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay

from pathlib import Path
from typing import List, Tuple

from preprocessors import normalize_loudness

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
ALL_SPEECH_META_PATH = os.path.join(OUTPUT_DIR, "all_speech_meta.json")
ALL_TRANSLATION_META_PATH = os.path.join(OUTPUT_DIR, "all_translation_meta.json")

# ===================================================================
# Progressive full_sound.wav builder (standalone & reusable)
# ===================================================================

def write_full_sound_wav(output_dir: Path, sample_rate: int) -> None:
    """
    Rebuild <output_dir>/full_sound.wav by concatenating all saved audible segments
    (speech + audible non-speech) in chronological order based on start_sec.

    Args:
        output_dir: Base output directory containing 'segments' and 'non_speech_segments'
        sample_rate: Audio sample rate in Hz (e.g., 16000)
    """
    segments_dir = output_dir / "segments"
    non_speech_dir = output_dir / "non_speech_segments"
    full_wav_path = output_dir / "full_sound.wav"

    audio_segments: List[Tuple[float, np.ndarray]] = []

    def load_segment(base_dir: Path, seg_id: int) -> None:
        seg_dir = base_dir / f"segment_{seg_id:04d}"
        wav_path = seg_dir / "sound.wav"
        meta_path = seg_dir / "speech_meta.json" if base_dir == segments_dir else seg_dir / "metadata.json"

        if not wav_path.exists() or not meta_path.exists():
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            start_sec = meta.get("start_sec", 0.0)

            _, audio = wavfile.read(wav_path)
            if audio.ndim > 1:  # ensure mono
                audio = audio[:, 0]
            if audio.dtype != np.int16:
                audio = audio.astype(np.int16)
            audio_segments.append((start_sec, audio))
        except Exception as e:
            log.warning("Failed to load %s/segment_%04d for full_sound: %s", base_dir.name, seg_id, e)

    # Load speech segments
    if segments_dir.exists():
        for seg_path in sorted(segments_dir.iterdir()):
            if seg_path.is_dir() and seg_path.name.startswith("segment_"):
                try:
                    seg_id = int(seg_path.name.split("_")[1])
                    load_segment(segments_dir, seg_id)
                except ValueError:
                    pass

    # Load audible non-speech segments
    if non_speech_dir.exists():
        for seg_path in sorted(non_speech_dir.iterdir()):
            if seg_path.is_dir() and seg_path.name.startswith("segment_"):
                try:
                    seg_id = int(seg_path.name.split("_")[1])
                    load_segment(non_speech_dir, seg_id)
                except ValueError:
                    pass

    if not audio_segments:
        if full_wav_path.exists():
            full_wav_path.unlink()  # remove stale file if no segments
        return

    # Sort chronologically by start_sec
    audio_segments.sort(key=lambda x: x[0])

    # Concatenate all audio
    full_audio = np.concatenate([audio for _, audio in audio_segments])

    # Write full recording
    try:
        wavfile.write(full_wav_path, sample_rate, full_audio)
        duration = len(full_audio) / sample_rate
        log.info(
            "[full_sound] Updated full_sound.wav | duration=%.2fs | segments=%d",
            duration, len(audio_segments)
        )
    except Exception as e:
        log.error("[full_sound] Failed to write full_sound.wav: %s", e)

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
    max_rtt_history: int = 10
    reconnect_attempts: int = 5
    reconnect_delay: float = 3.0
    vad_model_path: str | None = None  # allow custom model if needed
    vad_threshold: float = 0.3
    min_speech_duration: float = 0.15          # seconds; ignore shorter speech bursts
    min_silence_duration: float = 0.3   # seconds of silence before ending segment
    speech_pad_ms: int = 700                    # NEW: Padding added after silence to avoid cutting off trailing sounds
    max_speech_duration_s: float = float("inf") # NEW: Maximum length of a single utterance (force split if exceeded)

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

# Track when each non-speech segment actually started (wall-clock time)
non_speech_wallclock: dict[int, float] = {}  # segment_num → time.time()

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
    """
    Capture microphone, detect speech/silence, and stream audio chunks to websocket.
    Segments are saved to disk for both speech and non-speech (with energy filtering).
    Tracks VAD confidence, energy, and logs detailed info.
    """
    # Lazy init VAD model path
    model_path = config.vad_model_path
    if model_path is None:
        from pysilero_vad import _DEFAULT_MODEL_PATH
        model_path = _DEFAULT_MODEL_PATH
    vad = SileroVoiceActivityDetector(model_path)
    # --------------------------------------------------------------
    # NEW: Lightweight AGC for VAD robustness
    # Target RMS: ~0.1 (-20 dBFS) in float [-1,1] scale
    # Smoothing: exponential moving average with alpha ≈ 0.1
    # This provides consistent loudness for VAD without full LUFS overhead
    # --------------------------------------------------------------
    TARGET_RMS = 0.1
    SMOOTHING_ALPHA = 0.1
    rolling_rms = 0.01  # initial low value to allow initial boost

    # --- Internal state for speech tracking ---
    speech_start_time: float | None = None
    silence_start_time: float | None = None
    speech_duration_sec: float = 0.0
    current_speech_seconds: float = 0.0

    # VAD confidence stats
    max_vad_confidence = 0.0
    last_vad_confidence = 0.0
    min_vad_confidence = 1.0
    first_vad_confidence = 0.0
    vad_confidence_sum = 0.0
    speech_chunk_count = 0

    # Energy tracking for speech
    speech_energy_sum = 0.0
    speech_energy_sum_squares = 0.0
    max_energy = 0.0
    min_energy = float("inf")

    # NEW variables for speech_pad and max duration handling
    speech_pad_seconds: float = config.speech_pad_ms / 1000.0
    padding_start_time: float | None = None      # When padding period began after min_silence
    forced_end_time: float | None = None         # Absolute time to force-end due to max_speech_duration_s

    # --- File/segment tracking ---
    segments_dir = os.path.join(OUTPUT_DIR, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    non_speech_dir = os.path.join(OUTPUT_DIR, "non_speech_segments")
    os.makedirs(non_speech_dir, exist_ok=True)

    current_segment_num: int | None = None
    current_segment_buffer: bytearray | None = None
    speech_chunks_in_segment: int = 0

    # Non-speech segment tracking
    current_non_speech_num: int | None = None
    current_non_speech_buffer: bytearray | None = None
    non_speech_chunks_in_segment: int = 0
    non_speech_start_time: float | None = None  # monotonic time when current non-speech began
    peak_rms: float = 0.0

    # Energy tracking for non-speech (mirrors speech stats)
    non_speech_energy_sum: float = 0.0
    non_speech_energy_sum_squares: float = 0.0
    non_speech_max_energy: float = 0.0
    non_speech_min_energy: float = float("inf")

    # Per-chunk lists for saving detailed probabilities and RMS
    speech_probs: list[float] = []
    speech_rms_list: list[float] = []
    non_speech_probs: list[float] = []
    non_speech_rms_list: list[float] = []

    # Buffer and stats
    pcm_buffer = bytearray()
    utterances_sent = 0  # now counts full utterances sent, not individual chunks
    speech_chunks_detected = 0
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
                    chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0  # [-1, 1]
                    # --------------------------------------------------------------
                    # NEW: Apply fast AGC using rolling RMS (replaces LUFS)
                    # Compute current chunk RMS
                    current_rms = float(np.sqrt(np.mean(chunk_np ** 2)))
                    # Update rolling RMS (EMA)
                    rolling_rms = (SMOOTHING_ALPHA * current_rms) + ((1 - SMOOTHING_ALPHA) * rolling_rms)
                    # Avoid division by zero
                    gain = TARGET_RMS / (rolling_rms + 1e-8)
                    # Optional: limit max gain to prevent noise amplification (e.g., 10x)
                    gain = min(gain, 10.0)
                    # Apply gain
                    agc_chunk_float = chunk_np * gain
                    # Hard clip to prevent distortion
                    agc_chunk_float = np.clip(agc_chunk_float, -1.0, 1.0)

                    # Use AGC-processed audio for RMS and VAD decisions
                    rms = float(np.sqrt(np.mean(agc_chunk_float ** 2)))
                    has_sound = rms > 50.0

                    # Peak RMS per non-speech segment
                    if speech_start_time is None:
                        if non_speech_start_time is not None:
                            peak_rms = max(peak_rms, rms) if 'peak_rms' in locals() else rms
                        else:
                            peak_rms = rms

                    del pcm_buffer[:VAD_CHUNK_BYTES]
                    total_chunks_processed += 1
                    # VAD receives AGC-processed audio (int16 as expected by Silero)
                    agc_int16 = (agc_chunk_float * 32767.0).astype(np.int16)
                    agc_bytes = agc_int16.tobytes()
                    speech_prob: float = vad(agc_bytes)
                    if speech_prob >= config.vad_threshold:
                        processed += 1
                        speech_chunks_detected += 1
                        last_vad_confidence = speech_prob
                        max_vad_confidence = max(max_vad_confidence, speech_prob)
                        min_vad_confidence = min(min_vad_confidence, speech_prob)

                        silence_start_time = None

                        if speech_start_time is None:
                            speech_start_time = time.monotonic()
                            # Initialize forced end time if max duration is finite
                            forced_end_time = (
                                speech_start_time + config.max_speech_duration_s
                                if config.max_speech_duration_s != float("inf")
                                else None
                            )
                            padding_start_time = None
                            # Start of a speech segment
                            current_segment_num = len([d for d in os.listdir(segments_dir) if d.startswith("segment_")]) + 1
                            segment_start_wallclock[current_segment_num] = time.time()
                            # Set stream_start_time on the very first speech detection,
                            # even if the utterance may later be discarded due to short duration.
                            # This ensures correct relative timing for all subsequent subtitles.
                            global stream_start_time
                            if stream_start_time is None:
                                stream_start_time = segment_start_wallclock[current_segment_num]
                                log.info("[timing] stream_start_time initialized to %.3f (first speech onset)",
                                         stream_start_time)
                            current_segment_buffer = bytearray()
                            speech_chunks_in_segment = 0
                            max_vad_confidence = speech_prob
                            last_vad_confidence = speech_prob
                            first_vad_confidence = speech_prob
                            min_vad_confidence = speech_prob
                            vad_confidence_sum = 0.0
                            speech_chunk_count = 0
                            speech_energy_sum = 0.0
                            speech_energy_sum_squares = 0.0
                            max_energy = rms
                            min_energy = rms
                            log.info("[speech] Speech started | segment_%04d", current_segment_num)
                            speech_duration_sec = 0.0
                            # Reset collection lists for new speech segment
                            speech_probs = []
                            speech_rms_list = []

                        if current_segment_buffer is not None:
                            current_segment_buffer.extend(chunk)
                            speech_chunks_in_segment += 1
                        vad_confidence_sum += speech_prob
                        speech_chunk_count += 1

                        # Collect per-chunk data for later saving
                        speech_probs.append(speech_prob)
                        speech_rms_list.append(rms)

                        max_energy = max(max_energy, rms)
                        min_energy = min(min_energy, rms)
                        speech_energy_sum += rms
                        speech_energy_sum_squares += rms ** 2

                        # Accumulate only — no per-chunk send anymore
                        speech_duration_sec += VAD_CHUNK_SAMPLES / config.sample_rate

                        # No per-chunk logging of sends anymore
                        if rms > 50.0 and speech_prob < config.vad_threshold:
                            log.debug("[no speech] Chunk has audible energy | rms=%.4f | speech_prob=%.3f | samples=%d",
                                      rms, speech_prob, len(chunk_np))

                        # Reset non-speech state
                        non_speech_start_time = None
                        current_non_speech_num = None
                        non_speech_wallclock.pop(current_non_speech_num, None)
                        current_non_speech_buffer = None
                        non_speech_chunks_in_segment = 0
                        peak_rms = 0.0

                    else:
                        # Silence chunk handling
                        if speech_start_time is not None and silence_start_time is None:
                            silence_start_time = time.monotonic()

                        # Non-speech segment management
                        if speech_start_time is None:
                            if non_speech_start_time is None:
                                non_speech_start_time = time.monotonic()
                                current_non_speech_num = (
                                    len([d for d in os.listdir(non_speech_dir) if d.startswith("segment_")]) + 1
                                )
                                non_speech_wallclock[current_non_speech_num] = time.time()
                                current_non_speech_buffer = bytearray()
                                non_speech_chunks_in_segment = 0
                                peak_rms = rms

                                # Reset energy stats
                                non_speech_energy_sum = 0.0
                                non_speech_energy_sum_squares = 0.0
                                non_speech_max_energy = rms
                                non_speech_min_energy = rms

                                if has_sound:
                                    log.info("[non-speech] Non-speech segment started | segment_%04d", current_non_speech_num)

                                # Reset collection lists for new non-speech segment
                                non_speech_probs = []
                                non_speech_rms_list = []

                            if current_non_speech_buffer is not None:
                                current_non_speech_buffer.extend(chunk)
                                non_speech_chunks_in_segment += 1

                                # Collect per-chunk data for non-speech
                                non_speech_probs.append(speech_prob)
                                non_speech_rms_list.append(rms)

                            # Accumulate energy stats
                            non_speech_energy_sum += rms
                            non_speech_energy_sum_squares += rms ** 2
                            non_speech_max_energy = max(non_speech_max_energy, rms)
                            non_speech_min_energy = min(non_speech_min_energy, rms)
                            peak_rms = max(peak_rms, rms)

                            # Save: ≥3s or noise ≥1s
                            elapsed = time.monotonic() - non_speech_start_time
                            if elapsed >= 3.0 or (rms > 100.0 and elapsed >= 1.0):
                                if current_non_speech_buffer and non_speech_chunks_in_segment > 0:
                                    num_samples = len(current_non_speech_buffer) // 2
                                    duration = num_samples / config.sample_rate

                                    # Calculate wallclock-relative start/end (consistent with speech)
                                    base_time = stream_start_time or time.time()
                                    non_speech_wallclock_start = non_speech_wallclock.get(current_non_speech_num, time.time())
                                    start_sec = round(non_speech_wallclock_start - base_time, 3)
                                    end_sec = round(start_sec + duration, 3)

                                    # Energy averages / std
                                    chunk_count = non_speech_chunks_in_segment or 1
                                    avg_rms = non_speech_energy_sum / chunk_count
                                    rms_std = np.sqrt(
                                        (non_speech_energy_sum_squares / chunk_count) - (avg_rms ** 2)
                                    ) if chunk_count > 1 else 0.0

                                    # Save only if audible
                                    if not has_sound and peak_rms > 0.0 and peak_rms <= 50.0:
                                        log.info(
                                            "[non-speech] Skipping save of pure silence segment_%04d | peak_rms=%.4f",
                                            current_non_speech_num, peak_rms
                                        )
                                        non_speech_start_time = None
                                        current_non_speech_num = None
                                        current_non_speech_buffer = None
                                        non_speech_chunks_in_segment = 0
                                        peak_rms = 0.0
                                        non_speech_wallclock.pop(current_non_speech_num, None)
                                        continue

                                    seg_dir = os.path.join(non_speech_dir, f"segment_{current_non_speech_num:04d}")
                                    os.makedirs(seg_dir, exist_ok=True)
                                    wav_path = os.path.join(seg_dir, "sound.wav")
                                    audio_int16 = np.frombuffer(current_non_speech_buffer, dtype=np.int16)
                                    try:
                                        normalized_audio = normalize_loudness(
                                            audio_int16,
                                            sample_rate=config.sample_rate,
                                            return_dtype="int16",
                                            mode=None,  # general mode: -14 LUFS with headroom
                                        )
                                        audio_to_save = normalized_audio.astype(np.int16)
                                        log.info(
                                            "[non-speech] Applied loudness normalization to segment_%04d (target -14 LUFS)",
                                            current_non_speech_num
                                        )
                                    except Exception as e:
                                        log.warning(
                                            "[non-speech] Loudness normalization failed for segment_%04d (%s) – saving original",
                                            current_non_speech_num, e
                                        )
                                        audio_to_save = audio_int16

                                    wavfile.write(wav_path, config.sample_rate, audio_to_save)

                                    metadata = {
                                        "segment_id": current_non_speech_num,
                                        "type": "non_speech",
                                        "start_sec": start_sec,
                                        "end_sec": end_sec,
                                        "duration_sec": round(duration, 3),
                                        "num_chunks": non_speech_chunks_in_segment,
                                        "num_samples": num_samples,
                                        "rms_last_chunk": float(round(rms, 4)),
                                        "peak_rms": float(round(peak_rms, 4)),
                                        "has_audible_sound": has_sound,
                                        "audio_energy": {
                                            "rms_min": float(round(non_speech_min_energy, 4)),
                                            "rms_max": float(round(non_speech_max_energy, 4)),
                                            "rms_ave": float(round(avg_rms, 4)),
                                            "rms_std": float(round(rms_std, 4)),
                                        },
                                    }
                                    meta_path = os.path.join(seg_dir, "metadata.json")
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump(metadata, f, indent=2)

                                    log.info(
                                        "[non-speech] Saved segment_%04d | start=%.3f end=%.3f dur=%.2fs | chunks=%d | peak_rms=%.4f | rms_avg=%.4f",
                                        current_non_speech_num, start_sec, end_sec, duration, non_speech_chunks_in_segment, peak_rms, avg_rms
                                    )

                                    # Progressively update the full concatenated audio
                                    write_full_sound_wav(Path(OUTPUT_DIR), config.sample_rate)

                                    # Save non-speech probabilities and RMS
                                    non_probs_path = os.path.join(seg_dir, "non_speech_probabilities.json")
                                    with open(non_probs_path, "w", encoding="utf-8") as f:
                                        json.dump([round(p, 4) for p in non_speech_probs], f, indent=2)

                                    non_rms_path = os.path.join(seg_dir, "non_speech_rms.json")
                                    with open(non_rms_path, "w", encoding="utf-8") as f:
                                        json.dump([round(r, 4) for r in non_speech_rms_list], f, indent=2)

                                    # Optional simple plots
                                    if len(non_speech_probs) > 1:
                                        time_axis = np.arange(len(non_speech_probs)) * (VAD_CHUNK_SAMPLES / config.sample_rate)
                                        plt.figure(figsize=(8, 3))
                                        plt.plot(time_axis, non_speech_probs, color="gray")
                                        plt.ylim(0, 1)
                                        plt.xlabel("Time (s)")
                                        plt.ylabel("VAD Probability")
                                        plt.title(f"Non-Speech VAD Probability – segment_{current_non_speech_num:04d}")
                                        plt.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(seg_dir, "non_speech_probability_plot.png"))
                                        plt.close()

                                        plt.figure(figsize=(8, 3))
                                        plt.plot(time_axis, non_speech_rms_list, color="red")
                                        plt.xlabel("Time (s)")
                                        plt.ylabel("RMS")
                                        plt.title(f"Non-Speech RMS – segment_{current_non_speech_num:04d}")
                                        plt.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(seg_dir, "non_speech_rms_plot.png"))
                                        plt.close()

                                    # Reset for next period
                                    non_speech_start_time = None
                                    current_non_speech_num = None
                                    current_non_speech_buffer = None
                                    non_speech_chunks_in_segment = 0
                                    peak_rms = 0.0
                                    non_speech_probs = []
                                    non_speech_rms_list = []
                                    non_speech_wallclock.pop(current_non_speech_num, None)
                    # === NEW END-OF-UTTERANCE LOGIC ===
                    current_time = time.monotonic()
                    end_utterance = False
                    end_reason = ""

                    # Trigger 1: Silence-based end with optional speech padding
                    if speech_start_time is not None and silence_start_time is not None:
                        silence_dur = current_time - silence_start_time
                        if silence_dur > config.min_silence_duration:
                            if padding_start_time is None:
                                padding_start_time = current_time  # start padding timer
                            elif current_time - padding_start_time >= speech_pad_seconds:
                                end_utterance = True
                                end_reason = "padding_complete"

                    # Trigger 2: Max speech duration reached (force split)
                    if forced_end_time is not None and current_time >= forced_end_time:
                        end_utterance = True
                        end_reason = "max_duration"

                    if end_utterance:
                        if current_segment_buffer is not None:
                            num_samples = len(current_segment_buffer) // 2
                            duration = num_samples / config.sample_rate
                        else:
                            num_samples = 0
                            duration = 0.0

                        speech_duration_sec = duration  # for min_speech check

                        if duration < config.min_speech_duration:
                            log.info(
                                "[speech] Segment too short (%.3fs < %.3fs) — discarded",
                                speech_duration_sec, config.min_speech_duration
                            )
                            # Short burst → treat as noise and merge into ongoing non-speech if present
                            if (current_non_speech_buffer is not None and non_speech_start_time is not None):
                                current_non_speech_buffer.extend(current_segment_buffer or b"")
                                non_speech_chunks_in_segment += speech_chunks_in_segment

                                # Merge energy stats
                                non_speech_energy_sum += speech_energy_sum
                                non_speech_energy_sum_squares += speech_energy_sum_squares
                                non_speech_max_energy = max(non_speech_max_energy, max_energy)
                                non_speech_min_energy = min(non_speech_min_energy, min_energy if min_energy != float("inf") else non_speech_min_energy)
                                peak_rms = max(peak_rms, max_energy)

                                log.info(
                                    "[non-speech] Merged short speech burst (%.3fs) into ongoing non-speech segment_%04d",
                                    speech_duration_sec, current_non_speech_num
                                )
                            # Discard collected probs/RMS for the short burst
                            speech_probs = []
                            speech_rms_list = []

                            # Full reset after discard
                            speech_start_time = None
                            silence_start_time = None
                            padding_start_time = None
                            forced_end_time = None
                            current_segment_num = None
                            current_segment_buffer = None
                            speech_chunks_in_segment = 0
                            continue

                        else:
                            # Normal (long enough) speech segment → save it
                            log.info(
                                "[speech] Speech segment ended | duration: %.2fs | chunks: %d",
                                duration, speech_chunks_in_segment
                            )
                            log.info(
                                "[speech] End reason: %s | pad_ms=%d | max_s=%s",
                                end_reason,
                                config.speech_pad_ms,
                                config.max_speech_duration_s if config.max_speech_duration_s != float("inf") else "inf"
                            )

                            # === ALL ORIGINAL SEGMENT SAVING AND SENDING CODE (unchanged) ===
                            # Defer local saving until after successful send to server
                            if current_segment_buffer is not None and speech_chunks_in_segment > 0:
                                raw_audio_int16 = np.frombuffer(current_segment_buffer, dtype=np.int16)
                                # First, normalize for sending
                                try:
                                    normalized_for_send = normalize_loudness(
                                        raw_audio_int16,
                                        sample_rate=config.sample_rate,
                                        return_dtype="int16",
                                        mode="speech",
                                    )
                                    audio_to_send = normalized_for_send.astype(np.int16)
                                except Exception as e:
                                    log.warning("Normalization for send failed (%s) – using raw", e)
                                    audio_to_send = raw_audio_int16

                                # Send end_of_utterance marker first
                                try:
                                    await ws.send(json.dumps({"type": "end_of_utterance"}))
                                    log.info("[speech → server] Sent end_of_utterance marker")
                                except websockets.ConnectionClosed:
                                    log.warning("WebSocket closed while sending end marker")
                                    log.info("[speech] Discarding unsent utterance (connection closed)")
                                    # cleanup only, prevent local save
                                    current_segment_buffer = None
                                    current_segment_num = None
                                    speech_chunks_in_segment = 0
                                    non_speech_start_time = None
                                    current_non_speech_num = None
                                    current_non_speech_buffer = None
                                    non_speech_chunks_in_segment = 0
                                    peak_rms = 0.0
                                    non_speech_wallclock.pop(current_non_speech_num or 0, None)
                                    return

                                # Now send audio
                                full_pcm = audio_to_send.tobytes()
                                payload = {
                                    "type": "audio",
                                    "sample_rate": config.sample_rate,
                                    "pcm": base64.b64encode(full_pcm).decode("ascii"),
                                }
                                try:
                                    await ws.send(json.dumps(payload))
                                    log.info(
                                        "[speech → server] Successfully sent utterance segment_%04d | dur=%.2fs",
                                        current_segment_num, len(full_pcm)/2/config.sample_rate
                                    )
                                    utterances_sent += 1
                                except websockets.ConnectionClosed:
                                    log.warning("WebSocket closed while sending audio – discarding local segment")
                                    # cleanup only, prevent local save
                                    current_segment_buffer = None
                                    current_segment_num = None
                                    speech_chunks_in_segment = 0
                                    non_speech_start_time = None
                                    current_non_speech_num = None
                                    current_non_speech_buffer = None
                                    non_speech_chunks_in_segment = 0
                                    peak_rms = 0.0
                                    non_speech_wallclock.pop(current_non_speech_num or 0, None)
                                    return

                                # ONLY NOW save to disk (we know it was sent)
                                if current_segment_num:
                                    segment_dir = os.path.join(segments_dir, f"segment_{current_segment_num:04d}")
                                    os.makedirs(segment_dir, exist_ok=True)
                                    wav_path = os.path.join(segment_dir, "sound.wav")
                                    # Use same normalized audio for saving
                                    wavfile.write(wav_path, config.sample_rate, audio_to_send)
                                    write_full_sound_wav(Path(OUTPUT_DIR), config.sample_rate)

                                    base_time = stream_start_time or segment_start_wallclock[current_segment_num]
                                    relative_start = round(segment_start_wallclock[current_segment_num] - base_time, 3)
                                    relative_end = round(relative_start + duration, 3)
                                    metadata = {
                                        "segment_id": current_segment_num,
                                        "duration_sec": round(duration, 3),
                                        "num_chunks": speech_chunks_in_segment,
                                        "num_samples": num_samples,
                                        "start_sec": relative_start,
                                        "end_sec": relative_end,
                                        "sample_rate": config.sample_rate,
                                        "channels": config.channels,
                                        "vad_confidence": {
                                            "first": round(first_vad_confidence, 4),
                                            "last": round(last_vad_confidence, 4),
                                            "min": round(min_vad_confidence, 4),
                                            "max": round(max_vad_confidence, 4),
                                            "ave": round(vad_confidence_sum / speech_chunk_count, 4)
                                            if speech_chunk_count else 0.0,
                                        },
                                        "audio_energy": {
                                            "rms_min": float(round(min_energy, 4)) if min_energy != float("inf") else 0.0,
                                            "rms_max": float(round(max_energy, 4)),
                                            "rms_ave": float(round(speech_energy_sum / speech_chunk_count, 4))
                                            if speech_chunk_count else 0.0,
                                            "rms_std": float(round(
                                                np.sqrt(
                                                    (speech_energy_sum_squares / speech_chunk_count) -
                                                    (speech_energy_sum / speech_chunk_count) ** 2
                                                ),
                                                4
                                            )) if speech_chunk_count else 0.0,
                                        },
                                    }
                                    speech_meta_path = os.path.join(segment_dir, "speech_meta.json")
                                    with open(speech_meta_path, "w", encoding="utf-8") as f:
                                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                                    all_speech_meta = []
                                    if os.path.exists(ALL_SPEECH_META_PATH):
                                        try:
                                            with open(ALL_SPEECH_META_PATH, "r", encoding="utf-8") as f:
                                                all_speech_meta = json.load(f)
                                        except Exception:
                                            pass
                                    all_speech_meta.append({
                                        "segment_id": current_segment_num,
                                        **metadata,
                                        "duration_sec": round(duration, 3),
                                        "segment_dir": f"segment_{current_segment_num:04d}",
                                        "wav_path": str(wav_path),
                                        "meta_path": str(speech_meta_path),
                                    })
                                    with open(ALL_SPEECH_META_PATH, "w", encoding="utf-8") as f:
                                        json.dump(all_speech_meta, f, indent=2, ensure_ascii=False)

                                    probs_path = os.path.join(segment_dir, "speech_probabilities.json")
                                    with open(probs_path, "w", encoding="utf-8") as f:
                                        json.dump([round(p, 4) for p in speech_probs], f, indent=2)

                                    rms_path = os.path.join(segment_dir, "speech_rms.json")
                                    with open(rms_path, "w", encoding="utf-8") as f:
                                        json.dump([round(r, 4) for r in speech_rms_list], f, indent=2)

                                    if len(speech_probs) > 1:
                                        time_axis = np.arange(len(speech_probs)) * (VAD_CHUNK_SAMPLES / config.sample_rate)
                                        plt.figure(figsize=(8, 3))
                                        plt.plot(time_axis, speech_probs)
                                        plt.ylim(0, 1)
                                        plt.xlabel("Time (s)")
                                        plt.ylabel("VAD Probability")
                                        plt.title(f"Speech VAD Probability – segment_{current_segment_num:04d}")
                                        plt.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(segment_dir, "speech_probability_plot.png"))
                                        plt.close()

                                        plt.figure(figsize=(8, 3))
                                        plt.plot(time_axis, speech_rms_list, color="orange")
                                        plt.xlabel("Time (s)")
                                        plt.ylabel("RMS")
                                        plt.title(f"Speech RMS – segment_{current_segment_num:04d}")
                                        plt.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(segment_dir, "speech_rms_plot.png"))
                                        plt.close()

                                    log.info("[speech] Saved sent segment_%04d to disk", current_segment_num)

                            # Always cleanup state after sending/saving (or if sending fails above)
                            non_speech_start_time = None
                            current_non_speech_num = None
                            current_non_speech_buffer = None
                            non_speech_chunks_in_segment = 0
                            peak_rms = 0.0
                            non_speech_wallclock.pop(current_non_speech_num or 0, None)

                            # === STATE RESET AFTER SUCCESSFUL SEGMENT ===
                            if end_reason != "max_duration":
                                # Normal end → full reset
                                speech_start_time = None
                                silence_start_time = None
                                padding_start_time = None
                                forced_end_time = None
                            else:
                                # Forced split → continue speaking, start new segment immediately
                                speech_start_time = current_time
                                silence_start_time = None
                                padding_start_time = None
                                forced_end_time = (
                                    current_time + config.max_speech_duration_s
                                    if config.max_speech_duration_s != float("inf")
                                    else None
                                )
                                log.info("[speech] Forced split — starting new segment immediately")

                            current_segment_num = None
                            current_segment_buffer = None
                            speech_chunks_in_segment = 0
                            speech_probs = []
                            speech_rms_list = []
                            # non-speech reset already handled above for normal speech case

                    # --- Logging (after both branches) ---
                    # Per instructions, no per-chunk logging of sends anymore
                    # Log audible non-speech chunk only (already done above)

                if processed > 0:
                    log.debug("Processed %d speech chunk(s) this cycle", processed)
                if total_chunks_processed % 100 == 0:
                    status = "SPEAKING" if speech_start_time else "SILENCE"
                    seg_dur = time.monotonic() - speech_start_time if speech_start_time else 0.0
                    log.info(
                        "[status] chunks_processed: %d | sent: %d | detected: %d | state: %s | seg: %.2fs",
                        total_chunks_processed, utterances_sent, speech_chunks_detected, status, seg_dur
                    )
        except asyncio.CancelledError:
            log.info("[task] Streaming task cancelled")
            raise
        finally:
            log.info(
                "[microphone] Stopped | Processed: %d | Detected: %d | Sent: %d",
                total_chunks_processed, speech_chunks_detected, utterances_sent
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

if __name__ == "__main__":
    app = QApplication([])  # Re-uses existing instance if any
    overlay = LiveSubtitlesOverlay.create(app=app, title="Live Japanese Subtitles")
    
    def recording_thread() -> None:
        asyncio.run(main())
    
    Thread(target=recording_thread, daemon=True).start()
    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())