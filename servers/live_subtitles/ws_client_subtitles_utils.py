import json
import os
import time

import numpy as np  # needed for saving wav with frombuffer
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.utils import display_segments

# from rich.logging import RichHandler
from jet.logger import logger as log


def get_timestamp_prefix() -> str:
    """Generate a sortable timestamp prefix (YYYYMMDD-HHMMSS)."""
    return time.strftime("%Y%m%d-%H%M%S")


def find_segments_subdir(
    segments_root: str,
    utterance_id: str | None,
    chunk_index: int,
    create_if_missing: bool = False,
    timestamp_fallback: str | None = None,
) -> str | None:
    """
    Try to locate an existing segment subdirectory matching:
      * utterance_id's last 6 characters
      * chunk_index
    Format expected:  YYYYMMDD-HHMMSS_XXXXXX_N

    If multiple matches are found, returns the most recent (lexicographically largest).
    If none found and create_if_missing=True, creates one using timestamp_fallback or current time.

    Returns full path to the subdir, or None if not found and not creating.
    """
    if not utterance_id or len(utterance_id) < 6:
        utt_short = "noID"
    else:
        utt_short = utterance_id[-6:]

    pattern = f"_{utt_short}_{chunk_index}"
    candidates = [
        d
        for d in os.listdir(segments_root)
        if d.endswith(pattern) and len(d.split("_")) >= 3
    ]

    if candidates:
        # Take the latest (lex largest) if multiple exist due to race / multiple timestamps
        return os.path.join(segments_root, max(candidates))

    if not create_if_missing:
        return None

    # Create new
    ts = timestamp_fallback or get_timestamp_prefix()
    new_name = f"{ts}_{utt_short}_{chunk_index}"
    new_path = os.path.join(segments_root, new_name)
    os.makedirs(new_path, exist_ok=True)
    return new_path


def extract_and_display_buffered_segments(
    _audio_buffer: bytearray,
    min_silence_duration_sec: float,
    min_speech_duration_sec: float,
    max_speech_duration_sec: float,
    is_partial: bool = False,
) -> list[dict]:  # ← better return type hint
    try:
        _audio_np = np.frombuffer(_audio_buffer, dtype=np.int16).copy()

        buffer_segments, all_speech_probs = extract_speech_timestamps(
            _audio_np,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            return_seconds=True,
            time_resolution=3,
            with_scores=True,
            normalize_loudness=False,
            include_non_speech=True,
            double_check=True,
            apply_energy_VAD=True,
        )

        if len(buffer_segments):
            prefix = "Partial" if is_partial else "Complete"
            log.purple(
                f"{prefix} segments ({len(buffer_segments)}):\n"
                f"{json.dumps([{'num': seg['num'], 'duration': seg['duration'], 'prob': seg['prob']} for seg in buffer_segments])}"
            )
            display_segments(buffer_segments, done=not is_partial)
    except Exception as e:
        log.warning(f"Exception in extract_and_display_buffered_segments: {e}")
        return []

    return buffer_segments


def build_segment_metadata(
    *,
    filename: str,
    utterance_id: str | None,
    chunk_index: int,
    is_partial: bool,
    duration_sec: float,
    start_sec: float,
    sample_rate: int,
    channels: int,
    vad_stats: dict | None = None,
    energy_stats: dict | None = None,
    rms_label: str | None = None,
    sent_at: float | None = None,
    num_samples: int | None = None,
) -> dict:
    """Central place to build consistent metadata for both partial & final segments"""
    meta = {
        "filename": filename,
        "utterance_id": utterance_id,
        "chunk_index": chunk_index,
        "is_partial": is_partial,
        "duration_sec": round(duration_sec, 3),
        "start_sec": round(start_sec, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "sent_at": round(sent_at, 3) if sent_at is not None else None,
    }

    if num_samples is not None:
        meta["num_samples"] = num_samples

    if vad_stats:
        meta["vad_confidence"] = {
            k: round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in vad_stats.items()
        }

    if energy_stats:
        meta["audio_energy"] = {
            k: round(v, 5) if isinstance(v, (int, float)) else v
            for k, v in energy_stats.items()
        }

    if rms_label:
        meta["rms_label"] = rms_label

    return meta
