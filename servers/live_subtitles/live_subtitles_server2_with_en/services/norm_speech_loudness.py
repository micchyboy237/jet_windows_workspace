from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch

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

    Recommended for most pipelines: 'hybrid' method with target_rms_db=-20.
    This provides consistent levels for energy-based, WebRTC, Silero, and
    neural VADs.

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

    # Track input type for return conversion
    is_torch = isinstance(y, torch.Tensor)

    # ------------------------------------------------------------------ #
    # 0. Empty-array guard                                                 #
    # ------------------------------------------------------------------ #
    if len(y) == 0:
        if is_torch:
            return torch.tensor([], dtype=torch.float32), {
                "method": method,
                "original_rms_db": -np.inf,
                "final_rms_db": -np.inf,
                "original_peak": 0.0,
                "final_peak": 0.0,
                "applied_gain_db": 0.0,
                "skipped_reason": "empty_input",
            }
        else:
            return y.astype(np.float32), {
                "method": method,
                "original_rms_db": -np.inf,
                "final_rms_db": -np.inf,
                "original_peak": 0.0,
                "final_peak": 0.0,
                "applied_gain_db": 0.0,
                "skipped_reason": "empty_input",
            }

    # ------------------------------------------------------------------ #
    # 1. Convert to float32                                                #
    # ------------------------------------------------------------------ #
    if is_torch:
        y_norm = y.float().clone()
        # Convert to numpy for librosa operations, then back
        y_numpy = y_norm.numpy()
    else:
        y_norm = y.astype(np.float32).copy()
        y_numpy = y_norm

    # ------------------------------------------------------------------ #
    # 2. DC offset removal (before any level measurement)                  #
    #    Eliminates bias that inflates RMS and confuses energy-based VADs. #
    # ------------------------------------------------------------------ #
    if remove_dc:
        if is_torch:
            y_norm -= torch.mean(y_norm)
            y_numpy = y_norm.numpy()
        else:
            y_norm -= np.mean(y_norm)
            y_numpy = y_norm

    # ------------------------------------------------------------------ #
    # 3. Original statistics                                               #
    # ------------------------------------------------------------------ #
    original_rms = np.sqrt(np.mean(y_numpy**2) + eps)
    original_peak = float(np.max(np.abs(y_numpy)))
    original_rms_db = (
        float(20 * np.log10(original_rms)) if original_rms > eps else -np.inf
    )

    # ------------------------------------------------------------------ #
    # 4. Silence guard                                                     #
    #    Very quiet signals (< min_signal_db) are mostly noise; boosting  #
    #    them by 50+ dB would make the noise floor dominate the VAD.      #
    # ------------------------------------------------------------------ #
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
        return y_norm, info

    # ------------------------------------------------------------------ #
    # 5. Normalization                                                     #
    # ------------------------------------------------------------------ #
    if method == "peak":
        # Scale so the loudest sample reaches ±1.0.
        if is_torch:
            # Manual peak normalization for torch
            max_abs = torch.max(torch.abs(y_norm))
            if max_abs > 0:
                y_norm = y_norm / max_abs
            y_numpy = y_norm.numpy()
        else:
            y_norm = librosa.util.normalize(y_norm, norm=np.inf)
            y_numpy = y_norm

        # Re-measure: all-zeros edge case yields 0.0, not 1.0.
        final_peak = float(np.max(np.abs(y_numpy)))

    elif method in ("rms", "hybrid"):
        target_rms = 10 ** (target_rms_db / 20.0)
        scale = target_rms / (original_rms + eps)

        if is_torch:
            y_norm *= scale
            y_numpy = y_norm.numpy()
        else:
            y_norm *= scale
            y_numpy = y_norm

        # Post-scale peak (this is the true current peak, not pre-scale).
        current_peak = float(np.max(np.abs(y_numpy)))

        if method == "hybrid" and current_peak > max_peak:
            # current_peak > max_peak > 0, so division is safe without eps.
            if is_torch:
                y_norm *= max_peak / current_peak
                y_numpy = y_norm.numpy()
            else:
                y_norm *= max_peak / current_peak
                y_numpy = y_norm
            final_peak = max_peak
        else:
            # 'rms' method, or 'hybrid' where peak is already within limit.
            final_peak = current_peak

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from 'peak', 'rms', or 'hybrid'."
        )

    # ------------------------------------------------------------------ #
    # 6. Final statistics                                                  #
    # ------------------------------------------------------------------ #
    final_rms = np.sqrt(np.mean(y_numpy**2) + eps)
    final_rms_db = float(20 * np.log10(final_rms))

    info = {
        "method": method,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(original_peak, 4),
        "final_peak": round(final_peak, 4),
        "applied_gain_db": round(final_rms_db - original_rms_db, 2),
        "skipped_reason": None,
        # sr preserved for downstream traceability
        "sr": sr,
    }

    return y_norm, info
