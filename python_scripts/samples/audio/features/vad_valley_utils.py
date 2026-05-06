# vad_valley_utils.py

from enum import Enum

import numpy as np


class ThresholdStrategy(str, Enum):
    OTSU = "otsu"  # Optimal bimodal split (recommended default)
    PERCENTILE = "percentile"  # Bottom-N percentile of probs
    DERIVATIVE = "derivative"  # Inflection-based: mean minus k*std of |grad|
    CONTEXT = "context"  # Sliding-window local mean - k*std


# Auto-threshold helpers


def _otsu_threshold(x: np.ndarray) -> float:
    """
    Compute Otsu's optimal threshold on a [0,1] signal.

    Discretises into 256 bins, then maximises inter-class variance between
    the 'silence' class (below threshold) and the 'speech' class (above).
    Falls back to np.median if the signal is unimodal (bimodality coefficient
    below 0.2).
    """
    counts, bin_edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return 0.5

    # Bimodality coefficient: BC = (skew² + 1) / kurtosis
    # BC > 0.555 (= 5/9) suggests bimodality is present.
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        return float(mean)  # flat signal — any threshold is arbitrary
    skew = float(np.mean(((x - mean) / std) ** 3))
    kurt = float(np.mean(((x - mean) / std) ** 4))
    bc = (skew**2 + 1) / (kurt + 1e-9)
    if bc < 0.2:
        # Unimodal — Otsu will be unreliable; fall back to median
        return float(np.median(x))

    # Otsu sweep
    best_thresh = bin_centers[0]
    best_var = -1.0
    weight_bg = 0.0
    sum_bg = 0.0
    total_sum = float(np.dot(counts, bin_centers))

    for i, (w, c) in enumerate(zip(counts, bin_centers)):
        weight_bg += w
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += w * c
        mean_bg = sum_bg / weight_bg
        mean_fg = (total_sum - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = c

    return float(best_thresh)


def _percentile_threshold(x: np.ndarray, percentile: float = 25.0) -> float:
    """
    Set threshold at the given percentile of the probability distribution.
    Default p=25 assumes roughly the bottom quarter of frames are silent.
    """
    return float(np.percentile(x, percentile))


def _derivative_threshold(x: np.ndarray, k: float = 1.0) -> float:
    """
    Inflection-point heuristic: threshold = mean(probs) - k * std(|gradient|).

    The gradient magnitude tells you where rapid transitions occur.  Frames
    well below the mean AND in low-gradient zones are reliably silent.
    Falls back to mean - std when the gradient std is negligible.
    """
    grad = np.abs(np.gradient(x))
    grad_std = float(np.std(grad))
    base = float(np.mean(x))
    if grad_std < 1e-6:
        return max(0.0, base - float(np.std(x)))
    return max(0.0, base - k * grad_std)


def _context_threshold(x: np.ndarray, window: int = 50, k: float = 1.0) -> float:
    """
    Sliding-window context-aware threshold: median of per-window (mean - k*std).

    Each window votes on its local silence boundary; the median vote wins,
    making this robust to outlier windows (e.g. long pure-silence stretches).
    """
    if len(x) < window:
        return _otsu_threshold(x)  # not enough data for windowing
    votes = []
    for start in range(0, len(x) - window + 1, window // 2):  # 50% overlap
        chunk = x[start : start + window]
        votes.append(float(np.mean(chunk) - k * np.std(chunk)))
    return max(0.0, float(np.median(votes)))


def auto_threshold(
    probs: list[float],
    strategy: ThresholdStrategy = ThresholdStrategy.OTSU,
    percentile: float = 25.0,
    derivative_k: float = 1.0,
    context_window: int = 50,
    context_k: float = 1.0,
) -> float:
    """
    Automatically compute a silence/valley threshold from a VAD probability list.

    This value can be used directly as `valley_threshold` in extract_valleys(),
    `threshold` in extract_active_regions(), or negated for `trough_height` in
    extract_troughs().

    Args:
        probs:           Raw VAD probabilities.
        strategy:        Which estimation method to use (default: OTSU).
        percentile:      Used only by PERCENTILE strategy (default 25.0).
        derivative_k:    Sensitivity multiplier for DERIVATIVE strategy.
        context_window:  Frame window size for CONTEXT strategy.
        context_k:       Sensitivity multiplier for CONTEXT strategy.

    Returns:
        A float threshold in [0, 1].
    """
    x = np.array(probs, dtype=float)
    if strategy == ThresholdStrategy.OTSU:
        return _otsu_threshold(x)
    elif strategy == ThresholdStrategy.PERCENTILE:
        return _percentile_threshold(x, percentile)
    elif strategy == ThresholdStrategy.DERIVATIVE:
        return _derivative_threshold(x, derivative_k)
    elif strategy == ThresholdStrategy.CONTEXT:
        return _context_threshold(x, context_window, context_k)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
