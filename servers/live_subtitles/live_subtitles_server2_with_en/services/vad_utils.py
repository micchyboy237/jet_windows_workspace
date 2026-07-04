from typing import List, Union

import matplotlib
import numpy as np

matplotlib.use("Agg")
from rich.console import Console

try:
    from services.vad_config import (
        DEFAULT_PROB_WEIGHT,
        DEFAULT_RMS_WEIGHT,
    )
    from services.audio_config import HOP_SIZE    
except ImportError:
    from vad_config import (
        DEFAULT_PROB_WEIGHT,
        DEFAULT_RMS_WEIGHT,
    )
    from audio_config import HOP_SIZE

console = Console()


# ---------------------------------------------------------------------------
# Reusable hybrid probability computation
# ---------------------------------------------------------------------------


def compute_rms_normalized(
    audio_np: np.ndarray,
    frame_samples: int = HOP_SIZE,
    n_frames: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compute per-frame normalised RMS energy aligned to VAD frames.

    Returns:
        rms_norm:  Normalised RMS array in [0, 1], length = min(n_frames, audio_frames)
        rms_ceil:  The 99th-percentile ceiling used for normalisation.
                   Save this to invert the normalisation later.
    """
    if audio_np.size == 0:
        return np.array([], dtype=np.float32), 1.0

    if n_frames is None:
        n_frames = len(audio_np) // frame_samples

    n_common = min(n_frames, len(audio_np) // frame_samples)
    if n_common == 0:
        return np.array([], dtype=np.float32), 1.0

    frames = audio_np[: n_common * frame_samples].reshape(n_common, frame_samples)
    rms_arr = np.sqrt(np.mean(frames**2, axis=1))
    rms_ceil = float(np.percentile(rms_arr, 99) + 1e-10)
    rms_norm = np.clip(rms_arr / rms_ceil, 0.0, 1.0)

    return rms_norm.astype(np.float32), rms_ceil


def compute_hybrid_probs(
    probs: Union[List[float], np.ndarray],
    audio_np: np.ndarray,
    prob_weight: float = DEFAULT_PROB_WEIGHT,
    rms_weight: float = DEFAULT_RMS_WEIGHT,
    frame_samples: int = HOP_SIZE,
) -> np.ndarray:
    """
    Compute hybrid scores by combining speech probabilities with normalised RMS energy.
    """
    if isinstance(probs, list):
        probs = np.asarray(probs, dtype=np.float32)
    elif not isinstance(probs, np.ndarray):
        raise TypeError("probs must be a list[float] or np.ndarray")

    n_frames = len(probs)
    if n_frames == 0:
        return np.array([], dtype=np.float32)

    rms_norm, _ = compute_rms_normalized(
        audio_np, frame_samples=frame_samples, n_frames=n_frames
    )
    n_common = len(rms_norm)

    if n_common == 0:
        return np.array([], dtype=np.float32)

    hybrid = prob_weight * probs[:n_common] + rms_weight * rms_norm
    if n_frames > n_common:
        pad = prob_weight * probs[n_common:]
        hybrid = np.concatenate([hybrid, pad])

    return hybrid.astype(np.float32)
