from typing import List

import numpy as np


def smooth_vad_probs(probs: List[float], window: int = 20) -> List[float]:
    """Light moving average smoothing to reduce jitter in VAD probabilities."""
    if window <= 1 or len(probs) <= window:
        return probs[:]
    x = np.array(probs, dtype=float)
    smoothed = np.convolve(x, np.ones(window) / window, mode="same")
    smoothed[0] = (x[0] + x[1]) / 2 if len(x) > 1 else x[0]
    if len(x) > 2:
        smoothed[-1] = (x[-1] + x[-2]) / 2
    return smoothed.tolist()


def compute_valley_score(
    min_prob: float,
    mean_prob: float,
    duration_s: float,
    max_duration_ref: float = 1.0,
    w_depth: float = 0.4,
    w_mean: float = 0.4,
    w_duration: float = 0.2,
) -> float:
    """
    Composite score for valley quality. Higher score = stronger silence (safe to cut).

    Args:
        min_prob: Minimum probability in valley.
        mean_prob: Mean probability in valley.
        duration_s: Duration in seconds.
        max_duration_ref: Duration normalization cap.
        w_depth, w_mean, w_duration: Weights.

    Returns:
        float score in [0, 1].
    """
    duration_norm = min(duration_s / max_duration_ref, 1.0)
    score = (
        w_depth * (1.0 - min_prob)
        + w_mean * (1.0 - mean_prob)
        + w_duration * duration_norm
    )
    return float(score)


def compute_trough_score(
    min_prob: float,
    prominence: float,
    width: float,
    max_width_ref: float = 20.0,
    w_depth: float = 0.4,
    w_prominence: float = 0.4,
    w_width: float = 0.2,
) -> float:
    """Score how safe a trough is for cutting. Higher score = safer cut point."""
    depth_score = 1.0 - min_prob
    prominence_norm = min(prominence / 0.5, 1.0) if prominence is not None else 0.0
    width_norm = min(width / max_width_ref, 1.0) if width is not None else 0.0
    score = (
        w_depth * depth_score + w_prominence * prominence_norm + w_width * width_norm
    )
    return float(score)
