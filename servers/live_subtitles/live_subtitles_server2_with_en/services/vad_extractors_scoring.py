from typing import List, Optional, Tuple

import numpy as np
try:
    from services.vad_probs_utils import smooth_vad_probs
    from services.vad_types import (
        TroughToTroughScores,
        TroughToTroughSegment,
    )
except ImportError:
    from vad_probs_utils import smooth_vad_probs
    from vad_types import (
        TroughToTroughScores,
        TroughToTroughSegment,
    )
from rich.console import Console

console = Console()

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

# Color thresholds for Rich formatting
MEAN_PROB_COLORS = {
    "red_max": 0.25,
    "yellow_max": 0.45,
}
CONTENT_COLORS = {
    "red_max": 0.40,
    "yellow_max": 0.55,
}
BOUNDARY_COLORS = {
    "red_max": 0.55,
    "yellow_max": 0.75,
}
FINAL_SCORE_COLORS = {
    "red_max": 0.30,
    "yellow_max": 0.48,
}
DURATION_COLORS = {
    "very_short_max": 0.5,
    "short_max": 3.0,
    "optimal_max": 7.0,
    "long_max": 12.0,
}
QUALITY_LABELS = {
    "red": ("bad", "red"),
    "yellow": ("fair", "yellow"),
    "green": ("good", "bright_green"),
}

# =============================================================================
# SCORING PARAMETERS
# =============================================================================

# Smoothing and trimming
SCORING_SMOOTH_WINDOW: int = 5
CONTENT_EDGE_TRIM_FRAMES: int = 3

# Thresholds - FireRed VAD specific
SPEECH_THRESHOLD: float = 0.3  # Baseline: any prob > 0.3 = speech
HIGH_CONFIDENCE_THRESHOLD: float = 0.5  # Medium confidence: probs > 0.5
SPEECH_CONTINUITY_THRESHOLD: float = 0.4  # For gap detection

# Criteria weights (sum = 1.0)
SPEECH_PRESENCE_WEIGHT: float = 0.25  # % frames > 0.3
CORE_SPEECH_DENSITY_WEIGHT: float = 0.30  # Mean of frames > 0.3
HIGH_CONFIDENCE_RATIO_WEIGHT: float = 0.20  # % frames > 0.5
SPEECH_CONTINUITY_WEIGHT: float = 0.15
BOUNDARY_WEIGHT: float = 0.10
DURATION_WEIGHT: float = 0.10

# Duration scoring thresholds
DURATION_VERY_SHORT_MAX: float = 0.5
DURATION_SHORT_MAX: float = 3.0
DURATION_OPTIMAL_MIN: float = 3.0
DURATION_OPTIMAL_MAX: float = 7.0
DURATION_LONG_MAX: float = 12.0

# Boundary scoring sub-weights
BOUNDARY_VALLEY_WEIGHT: float = 0.35
BOUNDARY_TROUGH_DEPTH_WEIGHT: float = 0.50
BOUNDARY_PROMINENCE_WEIGHT: float = 0.15

# =============================================================================
# FORMATTING UTILITIES
# =============================================================================


def _score_color(value: float, red_max: float, yellow_max: float) -> str:
    """Return a Rich color tag for a score value based on thresholds."""
    if value <= red_max:
        return "red"
    elif value <= yellow_max:
        return "yellow"
    return "green"


def format_score_colored(
    value: float,
    red_max: float,
    yellow_max: float,
    fmt: str = ".3f",
) -> str:
    """Format a score value with Rich color tags based on thresholds."""
    color = _score_color(value, red_max, yellow_max)
    return f"[{color}]{value:{fmt}}[/{color}]"


def format_duration_colored(
    duration_s: float,
    very_short_max: float = 0.5,
    short_max: float = 3.0,
    optimal_max: float = 7.0,
    long_max: float = 12.0,
    fmt: str = ".2f",
) -> str:
    """
    Format a duration value with Rich color tags using 5 distinct levels.
    Color zones:
      - <= very_short_max (0.5s):  red      — too short, likely artifact
      - <= short_max (3.0s):       yellow   — short but usable
      - <= optimal_max (7.0s):     green    — optimal length
      - <= long_max (12.0s):       cyan     — long, may need splitting
      - > long_max:               magenta  — very long, should be split
    """
    if duration_s <= very_short_max:
        color = "red"
    elif duration_s <= short_max:
        color = "yellow"
    elif duration_s <= optimal_max:
        color = "green"
    elif duration_s <= long_max:
        color = "cyan"
    else:
        color = "magenta"
    return f"[{color}]{duration_s:{fmt}}[/{color}]"


def format_duration_score_colored(
    duration_score: float,
    duration_s: float,
    very_short_max: float = 0.5,
    short_max: float = 3.0,
    optimal_max: float = 7.0,
    long_max: float = 12.0,
    fmt: str = ".3f",
) -> str:
    """
    Format a duration score with colors determined by the raw duration.
    Uses the same 5-color system as format_duration_colored.
    """
    if duration_s <= very_short_max:
        color = "red"
    elif duration_s <= short_max:
        color = "yellow"
    elif duration_s <= optimal_max:
        color = "green"
    elif duration_s <= long_max:
        color = "cyan"
    else:
        color = "magenta"
    return f"[{color}]{duration_score:{fmt}}[/{color}]"


def get_quality_label(final_score: float) -> Tuple[str, str]:
    """
    Return (label, rich_color) for a final_score value.
    Uses FINAL_SCORE_COLORS and QUALITY_LABELS to ensure the Quality
    column in the results table is consistent with the Final column colors.
    """
    color = _score_color(
        final_score,
        FINAL_SCORE_COLORS["red_max"],
        FINAL_SCORE_COLORS["yellow_max"],
    )
    return QUALITY_LABELS[color]


# =============================================================================
# SCORING COMPUTATIONS
# =============================================================================


def _compute_speech_presence(
    trimmed_probs: np.ndarray,
    threshold: float = SPEECH_THRESHOLD,
) -> float:
    """
    Speech Presence: Percentage of frames above baseline speech threshold (0.3).

    This ensures that segments with probabilities around 0.3 (weak but present
    speech) still receive non-zero scores. FireRed VAD considers any probability
    above ~0.3 as potential speech activity.

    Args:
        trimmed_probs: 1D numpy array of speech probabilities (already smoothed & edge-trimmed).
        threshold: Baseline speech threshold (default: 0.3).
    Returns:
        float in [0.0, 1.0] - higher = more speech presence.
    """
    if len(trimmed_probs) == 0:
        return 0.0
    return float(np.mean(trimmed_probs >= threshold))


def _compute_core_speech_density(
    trimmed_probs: np.ndarray,
    threshold: float = SPEECH_THRESHOLD,
) -> float:
    """
    Core Speech Density: Mean probability of frames above baseline threshold (0.3).

    This measures the quality of speech in the segment by averaging only the
    frames that exceed the speech threshold, ignoring silent frames and
    boundary noise.

    Args:
        trimmed_probs: 1D numpy array of speech probabilities.
        threshold: Baseline speech threshold (default: 0.3).
    Returns:
        float in [0.0, 1.0] - higher = better quality speech.
    """
    if len(trimmed_probs) == 0:
        return 0.0

    speech_probs = trimmed_probs[trimmed_probs >= threshold]
    if len(speech_probs) == 0:
        return 0.0

    return float(np.mean(speech_probs))


def _compute_high_confidence_ratio(
    trimmed_probs: np.ndarray,
    threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> float:
    """
    High-Confidence Ratio: Percentage of frames above medium-confidence threshold (0.5).

    FireRed VAD typically produces probabilities > 0.75 for clear human speech,
    but medium-confidence speech (0.5-0.75) is also valid. This metric captures
    both strong and medium-confidence speech.

    Args:
        trimmed_probs: 1D numpy array of speech probabilities.
        threshold: High-confidence threshold (default: 0.5).
    Returns:
        float in [0.0, 1.0] - higher = more high-confidence speech.
    """
    if len(trimmed_probs) == 0:
        return 0.0
    return float(np.mean(trimmed_probs >= threshold))


def _compute_speech_continuity(
    trimmed_probs: np.ndarray,
    speech_threshold: float = SPEECH_CONTINUITY_THRESHOLD,
) -> float:
    """
    Speech Continuity: Penalizes long silent gaps within the segment.

    Uses a threshold of 0.4 (adjusted from 0.3 based on analysis that noise
    samples typically have probabilities in the 0.3-0.5 range, so 0.4 provides
    better separation between speech gaps and noise).

    Args:
        trimmed_probs: 1D numpy array of speech probabilities.
        speech_threshold: Frames below this are considered silent (default: 0.4).
    Returns:
        float in [0.0, 1.0] - higher = more continuous speech.
    """
    if len(trimmed_probs) == 0:
        return 0.0

    # Find silent frames (below threshold)
    silent = trimmed_probs < speech_threshold

    # Use diff to find contiguous silent regions
    silent_diff = np.diff(silent.astype(int))

    # Find starts of silent gaps (transition from speech to silence)
    silent_starts = np.where(silent_diff == 1)[0]
    # Find ends of silent gaps (transition from silence to speech)
    silent_ends = np.where(silent_diff == -1)[0]

    # Collect all gap lengths
    gap_lengths = []

    # Check if segment starts with silence
    if len(silent) > 0 and silent[0]:
        if len(silent_ends) > 0:
            gap_lengths.append(silent_ends[0] + 1)
        else:
            gap_lengths.append(len(silent))

    # Check gaps between speech regions
    for start, end in zip(silent_starts, silent_ends):
        gap_lengths.append(end - start + 1)

    # Check if segment ends with silence
    if len(silent) > 0 and silent[-1]:
        if len(silent_starts) > 0:
            last_start = silent_starts[-1]
            if last_start > silent_ends[-1] if len(silent_ends) > 0 else True:
                gap_lengths.append(len(silent) - last_start)

    if not gap_lengths:
        return 1.0  # No silent gaps

    max_gap = max(gap_lengths)
    total_frames = len(trimmed_probs)

    # Normalize gap by total frames
    gap_ratio = max_gap / total_frames

    # Continuity score: 1.0 - gap_ratio
    continuity = float(np.clip(1.0 - gap_ratio, 0.0, 1.0))
    return continuity


def _compute_duration_score(
    dur_s: float,
    very_short_max: float = DURATION_VERY_SHORT_MAX,
    short_max: float = DURATION_SHORT_MAX,
    optimal_min: float = DURATION_OPTIMAL_MIN,
    optimal_max: float = DURATION_OPTIMAL_MAX,
    long_max: float = DURATION_LONG_MAX,
) -> float:
    """
    Compute a duration quality score using a trapezoidal curve.

    Curve shape:
        Score
        1.0 ┤              ┌──────────────────────┐
            │             ╱                        ╲
        0.5 ┤           ╱                            ╲___
            │         ╱                                   ╲___
        0.0 ┤══════════╱                                            ╲═══
            └─────┬─────┬──────────────┬──────────────┬─────────────
                 0.5   3.0            7.0           12.0

    Zones:
      - <= 0.5s:       exponential penalty (artifact region)
      - 0.5-3.0s:      linear ramp 0 → 1
      - 3.0-7.0s:      flat plateau at 1.0 (optimal range)
      - 7.0-12.0s:     linear decay 1.0 → 0.5
      - > 12.0s:       exponential decay from 0.5
    """
    if dur_s <= 0:
        return 0.0
    if dur_s <= very_short_max:
        return float(np.exp(-5.0 * (very_short_max - dur_s) / very_short_max))
    if dur_s <= short_max:
        return float((dur_s - very_short_max) / (short_max - very_short_max))
    if dur_s <= optimal_max:
        return 1.0
    if dur_s <= long_max:
        return float(1.0 - 0.5 * (dur_s - optimal_max) / (long_max - optimal_max))
    excess = dur_s - long_max
    return float(0.5 * np.exp(-0.3 * excess))


def _compute_boundary_score(trough: Optional[dict]) -> float:
    """
    Score a single boundary by combining valley quality, trough depth, and prominence.

    Three factors:
      1. Valley quality (35%): how good is the surrounding silence region?
      2. Trough depth (50%): how silent is the exact cut frame? (1-prob)^2
      3. Trough prominence (15%): how much does the trough stand out?

    Args:
        trough: ValleyTrough dict or None for segment boundaries.
    Returns:
        float in [0.0, 1.0] - higher = better isolated boundary.
    """
    if trough is None:
        return 1.0

    valley_score = float(trough.get("valley", {}).get("final_score", 1.0))
    trough_prob = float(trough.get("prob", 0.5))
    prominence = float(trough.get("prominence", 0.0) or 0.0)

    trough_depth_score = (1.0 - trough_prob) ** 2
    prominence_score = min(prominence / 0.5, 1.0) if prominence else 0.0

    boundary = (
        BOUNDARY_VALLEY_WEIGHT * valley_score
        + BOUNDARY_TROUGH_DEPTH_WEIGHT * trough_depth_score
        + BOUNDARY_PROMINENCE_WEIGHT * prominence_score
    )
    return float(np.clip(boundary, 0.0, 1.0))


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================


def score_trough_to_trough_segments(
    segments: List[TroughToTroughSegment],
    w_speech_presence: float = SPEECH_PRESENCE_WEIGHT,
    w_core_speech: float = CORE_SPEECH_DENSITY_WEIGHT,
    w_high_confidence: float = HIGH_CONFIDENCE_RATIO_WEIGHT,
    w_continuity: float = SPEECH_CONTINUITY_WEIGHT,
    w_duration: float = DURATION_WEIGHT,
    w_boundary: float = BOUNDARY_WEIGHT,
    speech_threshold: float = SPEECH_THRESHOLD,
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    continuity_threshold: float = SPEECH_CONTINUITY_THRESHOLD,
) -> List[TroughToTroughSegment]:
    """
    Compute quality scores for each TroughToTroughSegment in-place.

    Populates two fields on each segment:
      - ``scores`` (TroughToTroughScores): component metrics
      - ``final_score`` (float): overall quality score

    **Tiered Scoring Criteria for FireRed VAD:**
      - speech_presence: % frames > 0.3 (25%) - Any speech activity
      - core_speech_density: Mean of frames > 0.3 (30%) - Speech quality
      - high_confidence_ratio: % frames > 0.5 (20%) - Strong speech
      - speech_continuity: 1 - (longest_gap/duration) (15%) - No gaps
      - boundary_quality_score: MIN of start/end boundaries (10%) - Good cuts
      - duration_score: Trapezoidal curve (10%) - Optimal length

    **Final score:** Weighted sum of all 6 criteria.

    This tiered approach ensures that:
    - Segments with probs around 0.3 still get meaningful scores
    - Segments with probs > 0.5 get additional credit
    - Segments with probs > 0.75 get maximum credit

    Args:
        segments: List of TroughToTroughSegment dicts (mutated in place).
        w_speech_presence: Weight for speech presence. Default 0.25.
        w_core_speech: Weight for core speech density. Default 0.30.
        w_high_confidence: Weight for high-confidence ratio. Default 0.20.
        w_continuity: Weight for speech continuity. Default 0.15.
        w_duration: Weight for duration. Default 0.10.
        w_boundary: Weight for boundary quality. Default 0.10.
        speech_threshold: Threshold for speech presence (default: 0.3).
        high_confidence_threshold: Threshold for high confidence (default: 0.5).
        continuity_threshold: Threshold for gap detection (default: 0.4).

    Returns:
        The same list of segments with ``scores`` and ``final_score`` fields added.
    """
    if not segments:
        console.print(
            "[yellow]score_trough_to_trough_segments: empty list, nothing to score.[/yellow]"
        )
        return segments

    weight_sum = (
        w_speech_presence
        + w_core_speech
        + w_high_confidence
        + w_continuity
        + w_duration
        + w_boundary
    )
    if weight_sum <= 0:
        weight_sum = 1.0

    for idx, seg in enumerate(segments):
        prob_stats = seg.get("prob_stats")
        segment_probs = seg.get("segment_probs")

        # Initialize scores to 0
        speech_presence = 0.0
        core_speech_density = 0.0
        high_confidence_ratio = 0.0
        speech_continuity = 0.0

        if segment_probs and len(segment_probs) > 0:
            # Smooth probabilities for more stable scoring
            smoothed = smooth_vad_probs(segment_probs, window=SCORING_SMOOTH_WINDOW)
            trim = CONTENT_EDGE_TRIM_FRAMES

            # Trim edges to avoid boundary noise
            if len(smoothed) > 2 * trim:
                trimmed = np.array(smoothed[trim:-trim], dtype=float)
            elif len(smoothed) > trim:
                trimmed = np.array(smoothed[trim:], dtype=float)
            else:
                trimmed = np.array(smoothed, dtype=float)

            # Compute tiered speech criteria
            speech_presence = _compute_speech_presence(
                trimmed, threshold=speech_threshold
            )
            core_speech_density = _compute_core_speech_density(
                trimmed, threshold=speech_threshold
            )
            high_confidence_ratio = _compute_high_confidence_ratio(
                trimmed, threshold=high_confidence_threshold
            )
            speech_continuity = _compute_speech_continuity(
                trimmed, speech_threshold=continuity_threshold
            )

            console.print(
                f"[dim]Segment {idx}: {len(trimmed)} content frames "
                f"(trimmed {len(smoothed) - len(trimmed)} edge frames) | "
                f"presence={speech_presence:.3f} | "
                f"core={core_speech_density:.3f} | "
                f"high_conf={high_confidence_ratio:.3f} | "
                f"continuity={speech_continuity:.3f}[/dim]"
            )

        elif prob_stats is not None:
            # Fallback: use prob_stats if segment_probs not available
            probs_list = prob_stats.get("segment_probs", [])
            if probs_list:
                smoothed = smooth_vad_probs(probs_list, window=SCORING_SMOOTH_WINDOW)
                trim = CONTENT_EDGE_TRIM_FRAMES
                if len(smoothed) > 2 * trim:
                    trimmed = np.array(smoothed[trim:-trim], dtype=float)
                elif len(smoothed) > trim:
                    trimmed = np.array(smoothed[trim:], dtype=float)
                else:
                    trimmed = np.array(smoothed, dtype=float)
                speech_presence = _compute_speech_presence(
                    trimmed, threshold=speech_threshold
                )
                core_speech_density = _compute_core_speech_density(
                    trimmed, threshold=speech_threshold
                )
                high_confidence_ratio = _compute_high_confidence_ratio(
                    trimmed, threshold=high_confidence_threshold
                )
                speech_continuity = _compute_speech_continuity(
                    trimmed, speech_threshold=continuity_threshold
                )
            else:
                console.print(
                    f"[dim]Segment {idx}: using prob_stats fallback (no segment_probs)[/dim]"
                )

        # Compute duration score
        duration_score = _compute_duration_score(float(seg["duration_s"]))

        # Compute boundary scores
        start_boundary = _compute_boundary_score(seg.get("trough_start"))
        end_boundary = _compute_boundary_score(seg.get("trough_end"))
        boundary_quality_score = min(start_boundary, end_boundary)

        # Compute final score (weighted sum of all criteria)
        final_score = (
            w_speech_presence * speech_presence
            + w_core_speech * core_speech_density
            + w_high_confidence * high_confidence_ratio
            + w_continuity * speech_continuity
            + w_duration * duration_score
            + w_boundary * boundary_quality_score
        ) / weight_sum
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # Build scores dict
        scores: TroughToTroughScores = {
            "speech_presence": speech_presence,
            "core_speech_density": core_speech_density,
            "high_confidence_ratio": high_confidence_ratio,
            "speech_continuity": speech_continuity,
            "duration_score": duration_score,
            "boundary_quality_score": boundary_quality_score,
            "content_score": final_score,  # content_score = final_score in this system
        }
        seg["scores"] = scores
        seg["final_score"] = final_score

        console.print(
            f"[blue]Segment {idx:03d}: "
            f"final={final_score:.3f} | "
            f"presence={speech_presence:.3f} | "
            f"core={core_speech_density:.3f} | "
            f"high_conf={high_confidence_ratio:.3f} | "
            f"continuity={speech_continuity:.3f} | "
            f"dur={duration_score:.3f} | "
            f"bound={boundary_quality_score:.3f}[/blue]"
        )

    console.print(
        f"[green]score_trough_to_trough_segments: scored {len(segments)} segment(s) "
        f"(smoothing window={SCORING_SMOOTH_WINDOW}, "
        f"edge trim={CONTENT_EDGE_TRIM_FRAMES} frames, "
        f"speech_threshold={speech_threshold}, "
        f"high_confidence_threshold={high_confidence_threshold})[/green]"
    )
    return segments
