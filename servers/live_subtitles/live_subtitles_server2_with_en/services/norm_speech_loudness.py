from __future__ import annotations

import logging
from typing import Optional, Tuple, Union, Literal, get_args

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# LOUDNESS LEVEL PRESETS
# ─────────────────────────────────────────────────────────────────
LOUDNESS_PRESETS = {
    "very_quiet": {
        "target_rms_db": -30.0,
        "max_peak_db": -10.0,
        "description": "Whisper, ASMR, distant speech",
    },
    "quiet": {
        "target_rms_db": -26.0,
        "max_peak_db": -6.0,
        "description": "Soft conversation, close-mic podcast",
    },
    "standard": {
        "target_rms_db": -20.0,
        "max_peak_db": -3.0,
        "description": "Normal conversation, meetings, typical VAD",
    },
    "loud": {
        "target_rms_db": -16.0,
        "max_peak_db": -2.0,
        "description": "Energetic speech, presentations",
    },
    "very_loud": {
        "target_rms_db": -12.0,
        "max_peak_db": -1.0,
        "description": "Broadcast-optimized, processed speech",
    },
    "brickwall": {
        "target_rms_db": -6.0,
        "max_peak_db": -0.3,
        "description": "Maximum digital level (not recommended for VAD)",
    },
}

# Create Literal type from preset keys
LoudnessPreset = Literal["very_quiet", "quiet", "standard", "loud", "very_loud", "brickwall"]


def normalize_audio_for_vad(
    y: Union[np.ndarray, torch.Tensor],
    sr: Optional[int] = None,
    method: str = "hybrid",
    target_rms_db: float = -20.0,
    max_peak: Optional[float] = None,
    max_peak_db: Optional[Union[float, LoudnessPreset]] = None,
    eps: float = 1e-8,
    min_signal_db: float = -60.0,
    remove_dc: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], dict]:
    """
    Normalize audio specifically for Voice Activity Detection (VAD).

    Recommended for most pipelines: 'hybrid' method with target_rms_db=-20
    (the "standard" preset). This provides consistent levels for energy-based,
    WebRTC, Silero, and neural VADs.

    Args:
        y:               Input audio array (any dtype; converted to float32).
                         Supports numpy.ndarray or torch.Tensor.
        sr:              Sample rate in Hz. Currently used for documentation
                         and future extensions (e.g., resampling, pre-emphasis
                         cutoff). Pass it for forward-compatibility.
        method:          Normalization strategy:
                           'peak'   – scale so the loudest sample hits ±1.0 (0 dBFS).
                           'rms'    – scale to target_rms_db; no peak limit.
                           'hybrid' – RMS target + peak ceiling (recommended).
        target_rms_db:   Desired RMS level in dBFS for 'rms' / 'hybrid'.
                         When max_peak_db is a preset string, this overrides
                         the preset's default target_rms_db.
        max_peak:        Peak ceiling as linear amplitude (0 < max_peak ≤ 1.0).
                         Mutually exclusive with max_peak_db.
                         Example: 0.95 = -0.45 dBFS, 0.708 = -3 dBFS.
        max_peak_db:     Peak ceiling specification. Can be:
                         - float: Direct dBFS value (e.g., -3.0, -6.0)
                         - str: Preset name - one of "very_quiet", "quiet",
                           "standard", "loud", "very_loud", "brickwall"
                           When a preset is used, both max_peak_db and 
                           target_rms_db (if not explicitly set) are taken 
                           from the preset.
                         Mutually exclusive with max_peak.
        eps:             Small constant to guard log/division of silent frames.
        min_signal_db:   Signals whose RMS is below this threshold are treated
                         as silent and returned unchanged (avoids boosting pure
                         noise by 50+ dB).
        remove_dc:       If True, subtract the mean before normalizing.
                         Recommended for energy-based and WebRTC VADs.

    Returns:
        y_norm:  Normalized float32 audio array (same type as input).
        info:    Diagnostic dict with original/final statistics, including
                 preset info for traceability.

    Raises:
        ValueError: If both max_peak and max_peak_db are specified, if
                    peak values are out of valid range, or if preset
                    name is invalid.

    Examples:
        # Using preset name
        >>> y_norm, info = normalize_audio_for_vad(y, sr=16000, 
        ...                                        max_peak_db="standard")
        
        # Preset with custom RMS target
        >>> y_norm, info = normalize_audio_for_vad(y, sr=16000, 
        ...                                        max_peak_db="quiet",
        ...                                        target_rms_db=-24.0)
        
        # Direct dBFS value
        >>> y_norm, info = normalize_audio_for_vad(y, sr=16000, 
        ...                                        target_rms_db=-20.0,
        ...                                        max_peak_db=-3.0)
    """

    # ─────────────────────────────────────────────────────────────
    # 0. Resolve max_peak_db from string preset or numeric value
    # ─────────────────────────────────────────────────────────────
    
    # Check mutual exclusivity
    if max_peak is not None and max_peak_db is not None:
        raise ValueError(
            "Specify either 'max_peak' (linear amplitude) or "
            "'max_peak_db' (dBFS or preset name), not both."
        )
    
    preset_name = None
    user_target_rms_db = target_rms_db  # Remember if user explicitly set this
    
    if isinstance(max_peak_db, str):
        # Resolve preset from string
        preset_name = max_peak_db.lower()
        
        if preset_name not in LOUDNESS_PRESETS:
            valid_names = ", ".join(f"'{name}'" for name in LOUDNESS_PRESETS.keys())
            raise ValueError(
                f"Invalid preset name: '{max_peak_db}'. "
                f"Must be one of: {valid_names}"
            )
        
        preset_config = LOUDNESS_PRESETS[preset_name]
        
        # Use preset's max_peak_db
        resolved_max_peak_db = preset_config["max_peak_db"]
        
        # Only override target_rms_db if user didn't explicitly set it
        # (check against default value of -20.0)
        if target_rms_db == -20.0:
            target_rms_db = preset_config["target_rms_db"]
            logger.debug(
                f"Using preset '{preset_name}' target_rms_db: {target_rms_db:.1f}"
            )
        else:
            logger.debug(
                f"Using preset '{preset_name}' max_peak_db: {resolved_max_peak_db:.1f} "
                f"with custom target_rms_db: {target_rms_db:.1f}"
            )
        
        logger.info(
            f"Preset '{preset_name}': {preset_config['description']} "
            f"(RMS: {target_rms_db:.1f} dBFS, Peak: {resolved_max_peak_db:.1f} dBFS)"
        )
        
        max_peak_db = resolved_max_peak_db
    
    # Convert numeric max_peak_db to linear max_peak
    if max_peak_db is not None:
        max_peak = 10 ** (max_peak_db / 20.0)
        logger.debug(
            f"Converted max_peak_db={max_peak_db:.2f} dBFS "
            f"to max_peak={max_peak:.6f} linear"
        )
    elif max_peak is None:
        # Default peak ceiling if nothing specified
        max_peak = 0.95
        logger.debug(f"No peak limit specified, using default max_peak={max_peak}")
    
    # Validate peak range
    if not 0 < max_peak <= 1.0:
        raise ValueError(
            f"max_peak must be in (0, 1.0], got {max_peak}. "
            f"(Equivalent to {20 * np.log10(max_peak):.2f} dBFS)"
        )
    
    # For diagnostics: store the effective dBFS limit
    effective_max_peak_db = 20 * np.log10(max_peak)
    
    # Track input type for return conversion
    is_torch = isinstance(y, torch.Tensor)
    
    # ─────────────────────────────────────────────────────────────
    # 1. Empty-array guard
    # ─────────────────────────────────────────────────────────────
    if len(y) == 0:
        empty_info = {
            "method": method,
            "target_rms_db": target_rms_db,
            "original_rms_db": -np.inf,
            "final_rms_db": -np.inf,
            "original_peak": 0.0,
            "final_peak": 0.0,
            "applied_gain_db": 0.0,
            "max_peak_db": round(effective_max_peak_db, 2),
            "preset": preset_name,
            "skipped_reason": "empty_input",
        }
        if is_torch:
            return torch.tensor([], dtype=torch.float32), empty_info
        else:
            return np.array([], dtype=np.float32), empty_info
    
    # ─────────────────────────────────────────────────────────────
    # 2. Convert to float32
    # ─────────────────────────────────────────────────────────────
    if is_torch:
        y_norm = y.float().clone()
        y_numpy = y_norm.numpy()
    else:
        y_norm = y.astype(np.float32).copy()
        y_numpy = y_norm
    
    # ─────────────────────────────────────────────────────────────
    # 3. DC offset removal
    # ─────────────────────────────────────────────────────────────
    if remove_dc:
        if is_torch:
            y_norm -= torch.mean(y_norm)
            y_numpy = y_norm.numpy()
        else:
            y_norm -= np.mean(y_norm)
            y_numpy = y_norm
    
    # ─────────────────────────────────────────────────────────────
    # 4. Original statistics
    # ─────────────────────────────────────────────────────────────
    original_rms = np.sqrt(np.mean(y_numpy**2) + eps)
    original_peak = float(np.max(np.abs(y_numpy)))
    original_rms_db = (
        float(20 * np.log10(original_rms)) if original_rms > eps else -np.inf
    )
    
    # ─────────────────────────────────────────────────────────────
    # 5. Silence guard
    # ─────────────────────────────────────────────────────────────
    if original_rms_db < min_signal_db:
        info = {
            "method": method,
            "target_rms_db": target_rms_db,
            "original_rms_db": round(original_rms_db, 2),
            "final_rms_db": round(original_rms_db, 2),
            "original_peak": round(original_peak, 4),
            "final_peak": round(original_peak, 4),
            "applied_gain_db": 0.0,
            "max_peak_db": round(effective_max_peak_db, 2),
            "preset": preset_name,
            "skipped_reason": "silent_input",
        }
        return y_norm, info
    
    # ─────────────────────────────────────────────────────────────
    # 6. Normalization
    # ─────────────────────────────────────────────────────────────
    if method == "peak":
        if max_peak < 1.0:
            logger.warning(
                f"Method 'peak' normalizes to 1.0 (0 dBFS), but "
                f"max_peak={max_peak:.4f} ({effective_max_peak_db:.2f} dBFS) "
                f"was specified. Use method='hybrid' for peak limiting."
            )
        
        if is_torch:
            max_abs = torch.max(torch.abs(y_norm))
            if max_abs > 0:
                y_norm = y_norm / max_abs
            y_numpy = y_norm.numpy()
        else:
            y_norm = librosa.util.normalize(y_norm, norm=np.inf)
            y_numpy = y_norm
        
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
        
        current_peak = float(np.max(np.abs(y_numpy)))
        
        if method == "hybrid" and current_peak > max_peak:
            if is_torch:
                y_norm *= max_peak / current_peak
                y_numpy = y_norm.numpy()
            else:
                y_norm *= max_peak / current_peak
                y_numpy = y_norm
            final_peak = max_peak
            logger.debug(
                f"Hybrid peak limiting applied: "
                f"{current_peak:.4f} → {max_peak:.4f} "
                f"({20 * np.log10(current_peak):.2f} dBFS → "
                f"{effective_max_peak_db:.2f} dBFS)"
            )
        else:
            final_peak = current_peak
    
    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from 'peak', 'rms', or 'hybrid'."
        )
    
    # ─────────────────────────────────────────────────────────────
    # 7. Final statistics
    # ─────────────────────────────────────────────────────────────
    final_rms = np.sqrt(np.mean(y_numpy**2) + eps)
    final_rms_db = float(20 * np.log10(final_rms))
    final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -np.inf
    
    info = {
        "method": method,
        "target_rms_db": target_rms_db,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(original_peak, 4),
        "original_peak_db": round(20 * np.log10(original_peak), 2) if original_peak > 0 else -np.inf,
        "final_peak": round(final_peak, 4),
        "final_peak_db": round(final_peak_db, 2),
        "applied_gain_db": round(final_rms_db - original_rms_db, 2),
        "max_peak_db": round(effective_max_peak_db, 2),
        "preset": preset_name,
        "preset_description": LOUDNESS_PRESETS[preset_name]["description"] if preset_name else None,
        "skipped_reason": None,
        "sr": sr,
    }
    
    return y_norm, info


if __name__ == "__main__":
    from main._main_norm_speech_loudness import main
    main()
