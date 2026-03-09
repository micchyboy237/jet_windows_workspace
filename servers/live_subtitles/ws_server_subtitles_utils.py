"""
Utility helpers for live subtitles WebSocket server.
"""

from __future__ import annotations
from datetime import datetime, timezone
from logger import logger
import wave
from pathlib import Path


# Maximum total stored audio duration
MAX_TOTAL_AUDIO_SECONDS: float = 5 * 60.0


def _get_wav_duration_seconds(path: Path) -> float:
    """
    Read WAV header and return duration in seconds.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        logger.warning("Failed reading WAV duration: %s", path, exc_info=True)
        return 0.0


def enforce_out_dir_duration_limit(out_dir: Path) -> None:
    """
    Ensure total stored audio duration does not exceed MAX_TOTAL_AUDIO_SECONDS.

    Strategy:
    - sort files by modification time (oldest first)
    - accumulate duration from newest → oldest
    - delete files once total duration exceeds limit
    """

    if not out_dir.exists():
        return

    try:
        wav_files = sorted(
            out_dir.glob("*.wav"),
            key=lambda p: p.stat().st_mtime
        )

        if not wav_files:
            return

        total_duration = 0.0
        keep_set: set[Path] = set()

        # walk newest → oldest
        for f in reversed(wav_files):
            dur = _get_wav_duration_seconds(f)
            if total_duration + dur <= MAX_TOTAL_AUDIO_SECONDS:
                keep_set.add(f)
                total_duration += dur
            else:
                break

        # delete older files not kept
        for f in wav_files:
            if f not in keep_set:
                try:
                    f.unlink(missing_ok=True)
                    logger.debug("Deleted old utterance file: %s", f.name)
                except Exception:
                    logger.warning("Failed deleting utterance file: %s", f, exc_info=True)

    except Exception:
        logger.warning("Utterance duration cleanup failed", exc_info=True)


def save_temp_wav(
    audio_bytes: bytes,
    out_dir: Path | None,
    client_id: str = "",
    segment_num: int = 0,
    prefix: str = "utterance",
) -> Path | None:
    """
    Save audio bytes as temporary WAV file when out_dir is provided.
    Returns the saved path (or None if not saved).
    """
    if not out_dir:
        return None

    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{client_id}_{segment_num:04d}_{ts}.wav"
        path = out_dir / filename

        import wave
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)           # 16-bit
            wf.setframerate(16000)       # currently hardcoded — consider passing as arg later
            wf.writeframes(audio_bytes)

        logger.info("Saved temporary utterance WAV: %s  (%.1f kB)", path, len(audio_bytes)/1024)
        return path

    except Exception:
        logger.exception("Failed to save temporary WAV file for %s", client_id)
        return None
