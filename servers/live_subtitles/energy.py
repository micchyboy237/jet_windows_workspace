# energy.py

from typing import Optional, Tuple

import numpy as np
from config import (
    FRAME_LENGTH_MS,
    HOP_LENGTH_MS,
    LOUD_MAX,
    NORMAL_MAX,
    SAMPLE_RATE,
    SILENCE_MAX_THRESHOLD,
    # New loudness thresholds
    VERY_QUIET_MAX,
)


def get_audio_duration(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Return duration of audio in seconds.

    Parameters
    ----------
    samples : np.ndarray
        Input audio samples (1D array).

    Returns
    -------
    float
        Duration in seconds.
    """
    if len(samples) == 0:
        return 0.0

    return float(len(samples) / sample_rate)


def compute_amplitude(samples: np.ndarray) -> float:
    """Compute peak amplitude (max |x|).

    Range: 0.0 (true silence) → 1.0 (maximum possible loudness / 0 dBFS)
    Common values:
      - < 0.01   → very quiet / silence
      - 0.1–0.6  → normal speech
      - > 0.7    → loud speech
    """
    if len(samples) == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def compute_rms_delta(
    rms_values: list[float] | np.ndarray, smoothing: int = 3
) -> float:
    """Compute smoothed first-order derivative (slope) of RMS sequence.
    Positive = rising energy, negative = falling.
    Returns average delta of the last `smoothing` steps (or all if fewer).
    """
    if len(rms_values) < 2:
        return 0.0
    rms_arr = np.asarray(rms_values, dtype=np.float64)
    deltas = np.diff(rms_arr)
    # Average last few deltas for stability
    recent_deltas = deltas[-smoothing:] if len(deltas) >= smoothing else deltas
    return float(np.mean(recent_deltas))


def compute_rms(samples: np.ndarray) -> float:
    """Root Mean Square – best simple measure of perceived loudness/energy.

    Range: 0.0 (true silence) → ~0.707 (full-scale sine wave)
    Typical speech values:
      - < SILENCE_MAX_THRESHOLD     → silence / noise floor
      - SILENCE_MAX_THRESHOLD–VERY_QUIET_MAX → very quiet / breath
      - VERY_QUIET_MAX–NORMAL_MAX → normal conversational speech
      - NORMAL_MAX–LOUD_MAX       → loud speech
      - > LOUD_MAX                → very loud / shouting
    """
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def has_sound(samples: np.ndarray) -> bool:
    """Return True if the audio contains meaningful sound.

    Now aligned with get_loudness_label():
      - rms < SILENCE_MAX_THRESHOLD  → "silent"       → has_sound=False
      - rms >= SILENCE_MAX_THRESHOLD → "very_quiet" and above → has_sound=True
    """
    if len(samples) == 0:
        return False
    rms_value = compute_rms(samples)
    return (
        rms_value >= SILENCE_MAX_THRESHOLD
    )  # Note: >= so exactly SILENCE_MAX_THRESHOLD counts as sound


def rms_to_loudness_label(rms_value: float) -> str:
    """Return a human-readable loudness label based on RMS."""
    if rms_value < SILENCE_MAX_THRESHOLD:
        return "silent"
    elif rms_value < VERY_QUIET_MAX:
        return "very_quiet"
    elif rms_value < NORMAL_MAX:
        return "normal"
    elif rms_value < LOUD_MAX:
        return "loud"
    else:
        return "very_loud"


def trim_silent(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_length_ms: float = FRAME_LENGTH_MS,
    hop_length_ms: float = HOP_LENGTH_MS,
) -> np.ndarray:
    """
    Remove leading and trailing silent portions from audio samples.

    Silence is determined using RMS with the same threshold used by has_sound()
    and rms_to_loudness_label().

    Parameters
    ----------
    samples : np.ndarray
        Input audio samples (1D array).
    sample_rate : int, default=22050
        Sample rate of the audio in Hz. Used to convert ms → samples.
    frame_length_ms : float, default=93.0
        Analysis window size in milliseconds.
    hop_length_ms : float, default=23.0
        Step size between windows in milliseconds.

    Returns
    -------
    np.ndarray
        Trimmed audio with leading and trailing silence removed.
        Returns empty array if the entire signal is silent.
    """
    if len(samples) == 0:
        return np.array([], dtype=samples.dtype)

    # Convert time (ms) to samples
    frame_length = int(round(frame_length_ms * sample_rate / 1000.0))
    hop_length = int(round(hop_length_ms * sample_rate / 1000.0))

    # Safety bounds
    frame_length = max(frame_length, 256)
    hop_length = max(hop_length, 64)
    hop_length = min(hop_length, frame_length // 2)  # ensure at least 50% overlap

    # If audio is shorter than one frame
    if len(samples) < frame_length:
        return (
            samples.copy() if has_sound(samples) else np.array([], dtype=samples.dtype)
        )

    # Override has_sound temporarily with custom threshold if provided
    def is_silent_frame(frame: np.ndarray) -> bool:
        if len(frame) == 0:
            return True
        rms = compute_rms(frame)
        return rms < SILENCE_MAX_THRESHOLD

    # Find start index (first non-silent frame)
    start = 0
    for i in range(0, len(samples) - frame_length + 1, hop_length):
        frame = samples[i : i + frame_length]
        if not is_silent_frame(frame):
            start = i
            break
    else:
        # Entire audio is silent
        return np.array([], dtype=samples.dtype)

    # Find end index (last non-silent frame)
    end = len(samples)
    for i in range(len(samples) - frame_length, -1, -hop_length):
        frame = samples[i : i + frame_length]
        if not is_silent_frame(frame):
            end = i + frame_length
            break

    # Return trimmed copy (preserve original dtype)
    return samples[start:end].copy()


def normalize_energy(
    rms_values: np.ndarray | list[float],
    max_rms: Optional[float] = None,
    # fallback_max now reuses the "normal" threshold so visualization scale
    # stays consistent with loudness classification.
    fallback_max: float = NORMAL_MAX,
    clip: bool = True,
    return_max: bool = False,
) -> np.ndarray | Tuple[np.ndarray, float]:
    """Normalize RMS values to [0, 1] adaptively for visualization.

    - Uses per-segment max_rms when available.
    - fallback_max defaults to NORMAL_MAX for consistency with rms_to_loudness_label().
    """
    rms_arr = np.asarray(rms_values, dtype=np.float64)
    if len(rms_arr) == 0:
        result = np.array([], dtype=np.float64)
        return (result, 0.0) if return_max else result

    effective_max = max_rms if max_rms is not None and max_rms > 0 else np.max(rms_arr)
    effective_max = max(effective_max, fallback_max)

    norm = rms_arr / effective_max
    if clip:
        norm = np.clip(norm, 0.0, 1.0)

    if return_max:
        return norm, float(effective_max)
    return norm


def smooth_signal(
    values: np.ndarray | list[float], window: int = 5, mode: str = "same"
) -> np.ndarray:
    """Simple moving average smoothing for signals (e.g. energy or probability).

    Returns array of the same length as input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window or window < 1:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode=mode)
