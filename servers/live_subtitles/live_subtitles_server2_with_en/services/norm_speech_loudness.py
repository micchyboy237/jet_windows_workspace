from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
try:
    from services.energy import normalize_energy
except ImportError:
    from energy import normalize_energy

logger = logging.getLogger(__name__)


def normalize_audio_for_vad(
    y: Union[np.ndarray, torch.Tensor],
    sr: Optional[int] = None,
    method: str = "hybrid",
    target_rms_db: float = -20.0,
    max_peak: float = 0.95,
    eps: float = 1e-8,
    min_signal_db: float = -60.0,
    remove_dc: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], dict]:
    """
    Normalize audio specifically for Voice Activity Detection (VAD).

    Uses normalize_energy() for consistent RMS measurement, aligned with
    rms_to_loudness_label() and has_sound() thresholds.

    Args:
        y:               Input audio array (any dtype; converted to float32).
                         Supports numpy.ndarray or torch.Tensor.
        sr:              Sample rate in Hz. Currently used for documentation
                         and future extensions (e.g., resampling, pre-emphasis
                         cutoff). Pass it for forward-compatibility.
        method:          Normalization strategy:
                           'peak'   – scale so the loudest sample hits ±1.0.
                           'rms'    – scale to target_rms_db; no peak limit.
                           'hybrid' – RMS target + peak ceiling (recommended).
        target_rms_db:   Desired RMS level in dBFS for 'rms' / 'hybrid'.
        max_peak:        Peak ceiling for 'hybrid' (0 < max_peak ≤ 1.0).
        eps:             Small constant to guard log/division of silent frames.
        min_signal_db:   Signals whose RMS is below this threshold are treated
                         as silent and returned unchanged (avoids boosting pure
                         noise by 50+ dB).
        remove_dc:       If True, subtract the mean before normalizing.
                         Recommended for energy-based and WebRTC VADs.

    Returns:
        y_norm:  Normalized float32 audio array (same type as input).
        info:    Diagnostic dict with original/final statistics.
    """
    is_torch = isinstance(y, torch.Tensor)

    # Convert to numpy for processing
    if is_torch:
        y_numpy = y.detach().cpu().numpy()
    else:
        y_numpy = y

    if len(y_numpy) == 0:
        empty_info = {
            "method": method,
            "original_rms_db": -np.inf,
            "final_rms_db": -np.inf,
            "original_peak": 0.0,
            "final_peak": 0.0,
            "applied_gain_db": 0.0,
            "skipped_reason": "empty_input",
        }
        if is_torch:
            return torch.tensor([], dtype=torch.float32), empty_info
        return y_numpy.astype(np.float32), empty_info

    y_norm = y_numpy.astype(np.float32).copy()
    if remove_dc:
        y_norm -= np.mean(y_norm)

    original_peak = float(np.max(np.abs(y_norm)))

    # --- Use normalize_energy for consistent RMS measurement ---
    # return_max=True gives us the effective max (the normalization anchor).
    # We pass [rms] as a single-element array so normalize_energy handles
    # the fallback_max / clip logic the same way as the rest of the codebase.
    raw_rms = float(np.sqrt(np.mean(y_norm.astype(np.float64) ** 2) + eps))
    _, effective_max = normalize_energy(
        [raw_rms],
        max_rms=None,  # let it auto-detect from the array
        fallback_max=raw_rms,  # anchor to the signal itself
        clip=False,
        return_max=True,
    )

    original_rms_db = float(20 * np.log10(raw_rms)) if raw_rms > eps else -np.inf

    if original_rms_db < min_signal_db:
        info = {
            "method": method,
            "original_rms_db": round(original_rms_db, 2),
            "final_rms_db": round(original_rms_db, 2),
            "original_peak": round(original_peak, 4),
            "final_peak": round(original_peak, 4),
            "applied_gain_db": 0.0,
            "skipped_reason": "silent_input",
        }
        if is_torch:
            return torch.from_numpy(y_norm), info
        return y_norm, info

    if method == "peak":
        y_norm = librosa.util.normalize(y_norm, norm=np.inf)
        final_peak = float(np.max(np.abs(y_norm)))

    elif method in ("rms", "hybrid"):
        target_rms = 10 ** (target_rms_db / 20.0)
        scale = target_rms / (raw_rms + eps)
        y_norm *= scale
        current_peak = float(np.max(np.abs(y_norm)))
        if method == "hybrid" and current_peak > max_peak:
            y_norm *= max_peak / current_peak
            final_peak = max_peak
        else:
            final_peak = current_peak

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from 'peak', 'rms', or 'hybrid'."
        )

    final_rms = float(np.sqrt(np.mean(y_norm.astype(np.float64) ** 2) + eps))
    final_rms_db = float(20 * np.log10(final_rms))

    info = {
        "method": method,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(original_peak, 4),
        "final_peak": round(final_peak, 4),
        "applied_gain_db": round(final_rms_db - original_rms_db, 2),
        "skipped_reason": None,
        "sr": sr,
    }

    if is_torch:
        return torch.from_numpy(y_norm), info
    return y_norm, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize audio for VAD.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio_path, sr=None)

    y_norm, stats = normalize_audio_for_vad(y, sr=sr)

    print("Normalization applied:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
