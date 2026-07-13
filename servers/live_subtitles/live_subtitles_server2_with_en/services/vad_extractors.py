import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from rich.console import Console

try:
    from services.audio_config import FRAME_SHIFT_MS, SAMPLE_RATE, SILENCE_MAX_THRESHOLD
    from services.audio_utils import AudioInput, load_audio
    from services.energy import trim_silent_frames
    from services.vad_valley_utils import ThresholdStrategy, auto_threshold
    from services.vad_types import (
        TroughToTroughSegment,
        VADSegment,
        ValleyInfo,
        ValleyTrough,
    )
    from services.vad_firered import (
        FireRedVAD,
        extract_speech_timestamps,
    )
    from services.vad_extractors_scoring import score_trough_to_trough_segments
    from services.vad_probs_utils import (
        compute_trough_score,
        compute_valley_score,
        smooth_vad_probs,
    )
except ImportError:
    from audio_config import FRAME_SHIFT_MS, SAMPLE_RATE, SILENCE_MAX_THRESHOLD
    from audio_utils import AudioInput, load_audio
    from energy import trim_silent_frames
    from vad_valley_utils import ThresholdStrategy, auto_threshold
    from vad_types import (
        TroughToTroughSegment,
        VADSegment,
        ValleyInfo,
        ValleyTrough,
    )
    from vad_firered import (
        FireRedVAD,
        extract_speech_timestamps,
    )
    from vad_extractors_scoring import score_trough_to_trough_segments
    from vad_probs_utils import (
        compute_trough_score,
        compute_valley_score,
        smooth_vad_probs,
    )

console = Console()

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


def is_probs_list(obj):
    """Returns True if obj is a list of floats."""
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(x, float) for x in obj)
    )


def _linkify(path: Path):
    return f"[link=file://{path}]{path.name}[/link]"


def load_probs(
    probs_or_audio: list[float] | AudioInput, default_audio: str | Path | None = None
) -> tuple[list[float], Optional[np.ndarray]]:
    """
    Attempts to load probability scores for VAD either from a list of floats,
    from an AudioInput (str path, bytes, os.PathLike, ndarray, or Tensor),
    or as a JSON string. Falls back on default_audio if parsing fails.
    Returns:
        probs (list[float]): VAD probability scores.
        audio_np (Optional[np.ndarray]): Decoded waveform if AudioInput required
            audio loading, otherwise None.
    Raises:
        ValueError: If unable to parse input and no default_audio is provided.
    """
    input_value = probs_or_audio
    audio_np: Optional[np.ndarray] = None

    if is_probs_list(input_value):
        return input_value, None

    if isinstance(input_value, (str, Path)):
        input_path = Path(input_value)
        if input_path.is_file():
            ext = input_path.suffix.lower()
            if ext == ".npy":
                console.print(f"Loading probabilities from: {_linkify(input_path)}")
                np_load = np.load(input_path, allow_pickle=True)
                probs = np_load.tolist() if isinstance(np_load, np.ndarray) else np_load
                if not is_probs_list(probs):
                    raise ValueError(
                        f".npy file does not contain a list of floats: {_linkify(input_path)}"
                    )
                return probs, None
            elif ext in {".json", ".txt"}:
                console.print(f"Loading probabilities from: {_linkify(input_path)}")
                with open(input_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if is_probs_list(loaded):
                    return loaded, None
                else:
                    raise ValueError(
                        f"JSON file is not a list of floats: {_linkify(input_path)}"
                    )
            else:
                console.print(f"Loading audio from: {_linkify(input_path)}")
                audio = load_audio(input_path)
                audio_np = (
                    audio[0] if isinstance(audio, tuple) and len(audio) == 2 else audio
                )
                _, probs = extract_speech_timestamps(
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                    use_hybrid=True,
                )
                if not is_probs_list(probs):
                    raise ValueError(
                        f"Extracted VAD scores are not a list of floats from: {_linkify(input_path)}"
                    )
                return probs, audio_np

    if isinstance(input_value, str):
        try:
            loaded = json.loads(input_value)
            if is_probs_list(loaded):
                return loaded, None
        except Exception:
            pass

    if isinstance(input_value, np.ndarray):
        audio_np = input_value
        _, probs = extract_speech_timestamps(
            audio=audio_np,
            threshold=0.5,
            min_speech_duration_sec=0.250,
            min_silence_duration_sec=0.250,
            with_scores=True,
        )
        if not is_probs_list(probs):
            raise ValueError(
                "Extracted VAD scores are not a list of floats from ndarray input."
            )
        return probs, audio_np

    if isinstance(input_value, bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(input_value)
            tmp.flush()
            audio = load_audio(tmp.name)
            audio_np = (
                audio[0] if isinstance(audio, tuple) and len(audio) == 2 else audio
            )
            _, probs = extract_speech_timestamps(
                audio=audio_np,
                threshold=0.5,
                min_speech_duration_sec=0.250,
                min_silence_duration_sec=0.250,
                with_scores=True,
            )
            if not is_probs_list(probs):
                raise ValueError(
                    "Extracted VAD scores are not a list of floats from bytes input."
                )
            return probs, audio_np

    try:
        import torch

        if isinstance(input_value, torch.Tensor):
            audio_np = input_value.detach().cpu().numpy()
            _, probs = extract_speech_timestamps(
                audio=audio_np,
                threshold=0.5,
                min_speech_duration_sec=0.250,
                min_silence_duration_sec=0.250,
                with_scores=True,
            )
            if not is_probs_list(probs):
                raise ValueError(
                    "Extracted VAD scores are not a list of floats from Tensor input."
                )
            return probs, audio_np
    except ImportError:
        pass

    if default_audio is not None:
        console.print(
            f"[yellow]Input not recognized, falling back to default audio: "
            f"{_linkify(Path(default_audio))}[/yellow]"
        )
        audio = load_audio(default_audio)
        audio_np = audio[0] if isinstance(audio, tuple) and len(audio) == 2 else audio
        _, probs = extract_speech_timestamps(
            audio=audio_np,
            threshold=0.5,
            min_speech_duration_sec=0.250,
            min_silence_duration_sec=0.250,
            with_scores=True,
        )
        if not is_probs_list(probs):
            raise ValueError(
                "Extracted VAD scores are not a list of floats from fallback audio."
            )
        return probs, audio_np

    raise ValueError("Input could not be parsed and no default_audio was provided.")


# ============================================================================
# Standalone extraction functions (no dependency on VADPeakAnalyzer)
# ============================================================================


def _compute_times(frame_idx: int, frame_shift_ms: float) -> Tuple[float, float]:
    """Convert frame index to start/end time in seconds."""
    frame_duration_s = frame_shift_ms / 1000.0
    start_s = frame_idx * frame_duration_s
    end_s = (frame_idx + 1) * frame_duration_s
    return start_s, end_s


def extract_peaks(
    probs_or_audio: List[float] | AudioInput,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[int] = None,
    **kwargs,
) -> List[VADSegment]:
    """
    Extract peaks (local maxima) from VAD probabilities using scipy.signal.find_peaks.
    A peak is a single-frame local maximum in the probability curve.
    This is useful for identifying the strongest speech moments.
    Args:
        probs_or_audio: VAD probability list (values in [0.0, 1.0]) or AudioInput.
        frame_shift_ms: Frame shift in milliseconds.
        height: Minimum probability for a peak (e.g. 0.6).
        distance: Minimum frames between peaks (e.g. 5-20).
        prominence: Required prominence of peaks (e.g. 0.1-0.3).
        width: Minimum width of peaks in frames.
    Returns:
        List[VADSegment]: Detected peak segments (each spans a single frame).
    Logs:
        Logs the number of peaks found and their probabilities.
    """
    probs, _ = load_probs(probs_or_audio)
    if not probs:
        console.print("[cyan]extract_peaks: empty probs list, returning [][/cyan]")
        return []
    x = np.array(probs, dtype=float)
    console.print(
        f"[cyan]extract_peaks: len={len(x)}, height={height}, distance={distance}, "
        f"prominence={prominence}[/cyan]"
    )
    peaks_idx, properties = find_peaks(
        x,
        height=height,
        distance=distance,
        prominence=prominence,
        width=width if width is not None else 0,
        **kwargs,
    )
    console.print(
        f"[cyan]extract_peaks: found {len(peaks_idx)} peaks at indices {peaks_idx.tolist()}[/cyan]"
    )
    segments: List[VADSegment] = []
    for i, peak in enumerate(peaks_idx):
        frame_start = int(peak)
        frame_end = int(peak)
        start_s, end_s = _compute_times(frame_start, frame_shift_ms)
        duration_s = end_s - start_s
        details = {
            "peak_index": int(peak),
            "peak_probability": float(x[peak]),
            "prominence": float(properties.get("prominences", [0])[i])
            if "prominences" in properties
            else None,
            "width": float(properties.get("widths", [0])[i])
            if "widths" in properties
            else None,
            "left_base": int(properties.get("left_bases", [0])[i])
            if "left_bases" in properties
            else None,
            "right_base": int(properties.get("right_bases", [0])[i])
            if "right_bases" in properties
            else None,
        }
        segments.append(
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_length": 1,
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "duration_s": round(duration_s, 4),
                "details": details,
            }
        )
    console.print(
        f"[green]extract_peaks: returning {len(segments)} peak segment(s)[/green]"
    )
    return segments


def extract_troughs(
    probs_or_audio: List[float] | AudioInput,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[int] = None,
    auto_threshold_strategy: ThresholdStrategy = ThresholdStrategy.OTSU,
    **kwargs,
) -> List[VADSegment]:
    """
    Extract troughs (local minima) from VAD probabilities by finding peaks
    on the negated signal.
    A trough is a single-frame local minimum in the probability curve.
    This is useful for identifying the deepest silence points.
    Args:
        probs_or_audio: VAD probability list (values in [0.0, 1.0]) or AudioInput.
        frame_shift_ms: Frame shift in milliseconds.
        height: Maximum probability for a trough (None = auto-computed).
        distance: Minimum frames between troughs (e.g. 5).
        prominence: Required prominence of troughs (e.g. 0.15).
        width: Minimum width of troughs in frames.
        auto_threshold_strategy: Strategy for auto-computing height if None.
            Default: ThresholdStrategy.OTSU.
    Returns:
        List[VADSegment]: Detected trough segments (each spans a single frame).
    Logs:
        Logs the auto-computed height (if applicable), number of troughs found,
        and their probabilities.
    """
    probs, _ = load_probs(probs_or_audio)
    if not probs:
        console.print("[cyan]extract_troughs: empty probs list, returning [][/cyan]")
        return []
    resolved_height = height
    if resolved_height is None:
        resolved_height = auto_threshold(probs, strategy=auto_threshold_strategy)
        console.print(
            f"[cyan]extract_troughs: auto-computed height={resolved_height:.4f} "
            f"via {auto_threshold_strategy.value}[/cyan]"
        )
    x = np.array(probs, dtype=float)
    console.print(
        f"[cyan]extract_troughs: len={len(x)}, height={resolved_height}, "
        f"distance={distance}, prominence={prominence}[/cyan]"
    )
    troughs_idx, properties = find_peaks(
        -x,
        height=-resolved_height if resolved_height is not None else None,
        distance=distance,
        prominence=prominence,
        width=width if width is not None else 0,
        **kwargs,
    )
    console.print(
        f"[cyan]extract_troughs: found {len(troughs_idx)} troughs at indices "
        f"{troughs_idx.tolist()}[/cyan]"
    )
    segments: List[VADSegment] = []
    for i, trough in enumerate(troughs_idx):
        frame_start = int(trough)
        frame_end = int(trough)
        start_s, end_s = _compute_times(frame_start, frame_shift_ms)
        duration_s = end_s - start_s
        details = {
            "trough_index": int(trough),
            "trough_probability": float(x[trough]),
            "prominence": float(properties.get("prominences", [0])[i])
            if "prominences" in properties
            else None,
            "width": float(properties.get("widths", [0])[i])
            if "widths" in properties
            else None,
        }
        segments.append(
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_length": 1,
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "duration_s": round(duration_s, 4),
                "details": details,
            }
        )
    console.print(
        f"[green]extract_troughs: returning {len(segments)} trough segment(s)[/green]"
    )
    return segments


def extract_active_regions(
    probs_or_audio: List[float] | AudioInput,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    threshold: float = 0.3,
    min_duration_s: float = 0.0,
    min_duration_frames: Optional[int] = None,
) -> List[VADSegment]:
    """
    Extract contiguous active (speech) regions where probability >= threshold.
    An "active region" is a run of consecutive frames all at or above the
    threshold — the speech bursts between silences.
    Args:
        probs_or_audio: VAD probability list or AudioInput.
        frame_shift_ms: Frame shift in milliseconds.
        threshold: Minimum probability to count as active (default 0.3).
        min_duration_s: Minimum duration in seconds for an active region to be kept.
        min_duration_frames: Alternative minimum frame count (overrides min_duration_s if provided).
    Returns:
        List[VADSegment]: Detected active region segments.
    Logs:
        Logs the number of active regions found.
    """
    probs, _ = load_probs(probs_or_audio)
    if not probs:
        console.print(
            "[cyan]extract_active_regions: empty probs list, returning [][/cyan]"
        )
        return []
    x = np.array(probs, dtype=float)
    active = x >= threshold
    segments: List[VADSegment] = []
    in_region = False
    region_start = 0
    for i, is_active in enumerate(active):
        if is_active and not in_region:
            in_region = True
            region_start = i
        elif not is_active and in_region:
            _append_active_segment(
                segments, x, region_start, i, threshold, frame_shift_ms
            )
            in_region = False
    if in_region:
        _append_active_segment(
            segments, x, region_start, len(x), threshold, frame_shift_ms
        )
    segments = filter_short_segments(
        segments,
        min_duration_s=min_duration_s,
        min_duration_frames=min_duration_frames,
    )
    console.print(
        f"[green]extract_active_regions: returning {len(segments)} active region(s)[/green]"
    )
    return segments


def _append_active_segment(
    segments: List[VADSegment],
    x: np.ndarray,
    start: int,
    end: int,
    threshold: float,
    frame_shift_ms: float,
) -> None:
    """Helper: build and append one active-region VADSegment."""
    start_s, _ = _compute_times(start, frame_shift_ms)
    _, end_s = _compute_times(end + 1, frame_shift_ms)
    duration_s = end_s - start_s
    region_probs = x[start:end].tolist()

    segments.append(
        {
            "frame_start": start,
            "frame_end": end + 1,
            "frame_length": end - start,
            "start_s": round(start_s, 4),
            "end_s": round(end_s, 4),
            "duration_s": round(duration_s, 4),
            "details": {
                "threshold": threshold,
                "max_probability": float(np.max(x[start:end])),
                "mean_probability": float(np.mean(x[start:end])),
                "frame_count": end - start,
                "region_probs": region_probs,
            },
        }
    )


def extract_valleys(
    probs_or_audio: List[float] | AudioInput,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    threshold: Optional[float] = None,
    min_duration_s: float = 0.0,
    min_duration_frames: Optional[int] = None,
    troughs: Optional[List[VADSegment]] = None,
    auto_threshold_strategy: ThresholdStrategy = ThresholdStrategy.OTSU,
) -> List[VADSegment]:
    """
    Extract contiguous valley (silence) regions where probability < threshold.
    A "valley" is a run of consecutive frames all strictly below the
    threshold — the silence stretches between speech bursts. This is the
    region-based counterpart to extract_troughs(), which finds only the
    single lowest frame inside each dip.
    Relationship to other functions:
        extract_troughs()        → single-frame local minimum inside a dip
        extract_valleys()        → the whole contiguous low-probability region
        extract_active_regions() → the whole contiguous high-probability region
    Args:
        probs_or_audio: VAD probability list or AudioInput.
        frame_shift_ms: Frame shift in milliseconds.
        threshold: Frames strictly below this value are considered silent
                   (default: auto-computed via auto_threshold_strategy).
        min_duration_s: Minimum duration in seconds for a valley to be kept.
        min_duration_frames: Alternative minimum frame count (overrides min_duration_s if set).
        troughs: Optional pre-extracted trough VADSegments. Each trough whose
                 frame index falls within a valley's boundary is attached to
                 that valley's details["troughs"] list.
        auto_threshold_strategy: Strategy for auto-computing threshold if None.
            Default: ThresholdStrategy.OTSU.
    Returns:
        List[VADSegment]: Detected valley segments.
    Logs:
        Logs the auto-computed threshold (if applicable), number of valleys found,
        and which troughs are contained in each valley.
    """
    probs, _ = load_probs(probs_or_audio)
    if not probs:
        console.print("[cyan]extract_valleys: empty probs list, returning [][/cyan]")
        return []
    resolved_threshold = threshold
    if resolved_threshold is None:
        resolved_threshold = auto_threshold(probs, strategy=auto_threshold_strategy)
        console.print(
            f"[cyan]extract_valleys: auto-computed threshold={resolved_threshold:.4f} "
            f"via {auto_threshold_strategy.value}[/cyan]"
        )
    x = np.array(probs, dtype=float)
    silent = x < resolved_threshold
    segments: List[VADSegment] = []
    in_valley = False
    valley_start = 0
    for i, is_silent in enumerate(silent):
        if is_silent and not in_valley:
            in_valley = True
            valley_start = i
        elif not is_silent and in_valley:
            _append_valley_segment(
                segments, x, valley_start, i, resolved_threshold, frame_shift_ms
            )
            in_valley = False
    if in_valley:
        _append_valley_segment(
            segments, x, valley_start, len(x), resolved_threshold, frame_shift_ms
        )
    if troughs and segments:
        for segment in segments:
            v_start = segment["frame_start"]
            v_end = segment["frame_end"]
            contained = [t for t in troughs if v_start <= t["frame_start"] <= v_end]
            segment["details"]["troughs"] = contained
            if contained:
                console.print(
                    f"[cyan]extract_valleys: valley [{v_start}, {v_end}] "
                    f"contains {len(contained)} trough(s) at frames "
                    f"{[t['frame_start'] for t in contained]}[/cyan]"
                )
    segments = filter_short_segments(
        segments,
        min_duration_s=min_duration_s,
        min_duration_frames=min_duration_frames,
    )
    console.print(
        f"[green]extract_valleys: returning {len(segments)} valley segment(s)[/green]"
    )
    return segments


def _append_valley_segment(
    segments: List[VADSegment],
    x: np.ndarray,
    start: int,
    end: int,
    threshold: float,
    frame_shift_ms: float,
) -> None:
    """Helper: build and append one valley VADSegment."""
    start_s, _ = _compute_times(start, frame_shift_ms)
    _, end_s = _compute_times(end + 1, frame_shift_ms)
    duration_s = end_s - start_s
    region_probs = x[start:end].tolist()
    min_prob_frame = int(start + np.argmin(x[start:end]))
    min_prob_s, _ = _compute_times(min_prob_frame, frame_shift_ms)
    frame_length = end - start

    segments.append(
        {
            "frame_start": start,
            "frame_end": end + 1,
            "frame_length": frame_length,
            "start_s": round(start_s, 4),
            "end_s": round(end_s, 4),
            "duration_s": round(duration_s, 4),
            "details": {
                "threshold": threshold,
                "min_probability": float(np.min(x[start:end])),
                "min_prob_frame": min_prob_frame,
                "min_prob_s": round(min_prob_s, 4),
                "mean_probability": float(np.mean(x[start:end])),
                "frame_count": frame_length,
            },
        }
    )


def filter_short_segments(
    segments: List[VADSegment],
    min_duration_s: float = 0.0,
    min_duration_frames: Optional[int] = None,
) -> List[VADSegment]:
    """
    Filter out segments shorter than the specified minimum duration.

    Args:
        segments: List of VADSegment dicts.
        min_duration_s: Minimum duration in seconds.
        min_duration_frames: Alternative minimum frame count (overrides min_duration_s).

    Returns:
        Filtered list of VADSegment dicts.

    Logs:
        Logs how many segments were filtered out.
    """
    if not segments:
        return segments

    before = len(segments)
    if min_duration_frames is not None:
        filtered = [s for s in segments if s["frame_length"] >= min_duration_frames]
    else:
        filtered = [s for s in segments if s["duration_s"] >= min_duration_s]

    removed = before - len(filtered)
    if removed > 0:
        console.print(
            f"[yellow]filter_short_segments: removed {removed} short segment(s), "
            f"kept {len(filtered)}[/yellow]"
        )

    return filtered


# ============================================================================
# Valley trough extraction (updated to use standalone functions)
# ============================================================================


def base_extract_valley_troughs(
    valleys: List[VADSegment], duration_s: float = 0.25
) -> List[ValleyTrough]:
    """
    Extracts the lowest-probability frames (troughs) from a list of VADSegment
    valleys, but only includes valleys that have exactly one trough and
    duration >= duration_s.
    """
    filtered_valleys = [
        valley
        for valley in valleys
        if len(valley["details"].get("troughs", [])) == 1
        and valley["duration_s"] >= duration_s
    ]

    last_frame = max((v["frame_end"] for v in filtered_valleys), default=-1)

    valley_troughs: List[ValleyTrough] = []
    for valley in filtered_valleys:
        details = valley["details"]
        valley_score = details.get("valley_score", 0.0)
        trough_score = details.get("trough_score", 0.0)
        final_score = details.get("final_score", 0.0)

        valley_info: ValleyInfo = {
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "frame_length": valley["frame_length"],
            "start_s": valley["start_s"],
            "end_s": valley["end_s"],
            "duration_s": valley["duration_s"],
            "valley_score": valley_score,
            "trough_score": trough_score,
            "final_score": final_score,
            "global_frame_start": valley["frame_start"],
            "global_frame_end": valley["frame_end"],
            "global_start_s": valley["start_s"],
            "global_end_s": valley["end_s"],
            "global_duration_s": valley["duration_s"],
            "global_valley_score": valley_score,
            "global_trough_score": trough_score,
            "global_final_score": final_score,
            "is_last": valley["frame_end"] >= last_frame,
        }

        valley_troughs.append(
            ValleyTrough(
                frame=details["min_prob_frame"],
                global_frame=details["min_prob_frame"],
                prob=details["min_probability"],
                time_s=details["min_prob_s"],
                global_time_s=details["min_prob_s"],
                valley=valley_info,
            )
        )

    return valley_troughs


def get_best_valley_trough(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 20,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.8,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> Optional[ValleyTrough]:
    """
    Returns the single best ValleyTrough based on the highest final_score.
    Returns None if no suitable trough is found.

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput.
        All other kwargs are forwarded to extract_valley_troughs.
    """
    probs, _ = load_probs(probs_or_audio)
    all_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )
    if not all_troughs:
        return None
    best = max(all_troughs, key=lambda t: t["valley"].get("final_score", 0.0))
    return best


def get_last_valley_trough(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 20,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.8,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> Optional[ValleyTrough]:
    """
    Return the valley trough whose valley covers the last audio frame
    (i.e. ``valley.is_last == True``).

    "Covers the last frame" means the valley's ``frame_end`` reaches
    ``len(probs) - 1`` (accounting for the ``frame_offset`` shift).

    If more than one such trough exists, the one with the highest
    ``final_score`` is returned. Returns ``None`` when no matching
    trough is found.

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput.
        All other kwargs are forwarded to extract_valley_troughs.
    """
    probs, _ = load_probs(probs_or_audio)
    all_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )
    last_troughs = [t for t in all_troughs if t["valley"]["is_last"]]
    if not last_troughs:
        return None
    return max(last_troughs, key=lambda t: t["valley"].get("final_score", 0.0))


SplitResult = Union[
    Tuple[List[float], List[float], ValleyTrough],
    Tuple[List[float], List[float], ValleyTrough, np.ndarray, np.ndarray],
]


def split_best_valley_trough(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 20,
    trough_height: Optional[float] = 0.3,
    trough_prominence: float = 0.0,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.1,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
    trim_silence: bool = True,
    trim_threshold: Optional[float] = SILENCE_MAX_THRESHOLD,
) -> Optional[SplitResult]:
    """
    Split a VAD probability list or audio into two halves at the best valley trough,
    then trim trailing silence from the left half and leading silence from the right half.

    Parameters
    ----------
    trim_silence : bool
        If True, strip silent frames from the inner edges of each half after splitting.
    trim_threshold : float, optional
        VAD probability below which a frame is considered silent for trimming purposes.
        Defaults to ``valley_threshold`` if set, otherwise ``trough_height``, otherwise 0.3.
    """
    probs, audio_np = load_probs(probs_or_audio)
    best_trough = get_best_valley_trough(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )

    if best_trough is None:
        return None

    split_frame: int = best_trough["frame"]
    left_probs: List[float] = probs[:split_frame]
    right_probs: List[float] = probs[split_frame:]

    if audio_np is not None:
        seconds_per_frame = frame_shift_ms / 1000.0
        split_sample = int(split_frame * seconds_per_frame * sample_rate)
        split_sample = max(0, min(split_sample, len(audio_np)))
        left_audio: Optional[np.ndarray] = audio_np[:split_sample]
        right_audio: Optional[np.ndarray] = audio_np[split_sample:]
    else:
        left_audio = None
        right_audio = None

    if trim_silence:
        left_probs, left_audio = trim_silent_frames(
            left_probs,
            left_audio,
            trim_left=False,
            trim_right=True,
            frame_shift_ms=frame_shift_ms,
            sample_rate=sample_rate,
        )
        right_probs, right_audio = trim_silent_frames(
            right_probs,
            right_audio,
            trim_left=True,
            trim_right=False,
            frame_shift_ms=frame_shift_ms,
            sample_rate=sample_rate,
        )

    if audio_np is not None:
        return left_probs, right_probs, best_trough, left_audio, right_audio
    return left_probs, right_probs, best_trough


def extract_valley_troughs(
    probs_or_audio: List[float] | AudioInput,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    smoothing_window: int = 0,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.25,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> List[ValleyTrough]:
    """
    Extract salient valley troughs (composite valley + trough detection) with scoring.

    This is the main entry point for finding the best silence cut points in VAD
    probability curves. It uses the standalone extract_troughs() and extract_valleys()
    functions (no VADPeakAnalyzer dependency).

    Workflow:
        1. Load/parse VAD probabilities from the input.
        2. Optionally smooth the probabilities.
        3. Extract single-frame troughs via extract_troughs().
        4. Extract contiguous valley regions via extract_valleys().
        5. Filter valleys by minimum duration.
        6. Score each valley+trough combination using compute_valley_score()
           and compute_trough_score().
        7. Filter to valleys with exactly one trough.
        8. Build and return ValleyTrough dicts with local/global coordinates,
           including prominence and width of each trough.

    Args:
        probs_or_audio: VAD speech probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs can resolve.
        sample_rate: Audio sample rate in Hz.
        frame_shift_ms: Frame shift in milliseconds.
        smoothing_window: Smoothing window size (0 = disabled).
        trough_height: Min trough height (None = auto-computed via auto_threshold).
        trough_prominence: Trough prominence. Default 0.15.
        trough_distance: Min frames between troughs. Default 5.
        valley_threshold: Valley threshold (None = auto-computed via auto_threshold).
        min_valley_duration_s: Minimum valley duration. Default 0.25.
        min_valley_frames: Min valley frames (overrides duration if set).
        frame_offset: Global frame offset for chunked processing.
        min_trough_offset_s: Min seconds from start for valid trough. Default 0.4.

    Returns:
        List[ValleyTrough]: Detected troughs with enclosing valley info,
        local/global coordinates, composite scores (valley_score × trough_score),
        and per-trough prominence/width values.

    Logs:
        Logs each step: smoothing, trough extraction, valley extraction,
        scoring, filtering, and final result count.
    """
    probs, _ = load_probs(probs_or_audio)

    if smoothing_window:
        smoothed = smooth_vad_probs(probs, window=smoothing_window)
        console.print(
            f"[cyan]extract_valley_troughs: applied smoothing (window={smoothing_window})[/cyan]"
        )
    else:
        smoothed = probs

    # Step 1: Extract single-frame troughs
    troughs = extract_troughs(
        smoothed,
        frame_shift_ms=frame_shift_ms,
        height=trough_height,
        prominence=trough_prominence,
        distance=trough_distance,
    )
    console.print(
        f"[cyan]extract_valley_troughs: extracted {len(troughs)} trough(s)[/cyan]"
    )

    # Step 2: Extract contiguous valley regions, attaching troughs
    valleys = extract_valleys(
        smoothed,
        frame_shift_ms=frame_shift_ms,
        threshold=valley_threshold,
        troughs=troughs,
    )
    console.print(
        f"[cyan]extract_valley_troughs: extracted {len(valleys)} valley(s)[/cyan]"
    )

    # Step 3: Filter valleys by minimum duration
    valleys = filter_short_segments(
        valleys,
        min_duration_s=min_valley_duration_s,
        min_duration_frames=min_valley_frames,
    )
    console.print(
        f"[cyan]extract_valley_troughs: {len(valleys)} valley(s) after duration filter[/cyan]"
    )

    # Step 4: Score each valley and its best trough
    for valley in valleys:
        details = valley["details"]

        # Compute valley score
        valley_score = compute_valley_score(
            min_prob=details.get("min_probability", 1.0),
            mean_prob=details.get("mean_probability", 1.0),
            duration_s=valley["duration_s"],
        )
        details["valley_score"] = valley_score

        # Find best trough (lowest probability) within this valley
        trough_list = details.get("troughs", [])
        if trough_list:
            trough = min(
                trough_list,
                key=lambda t: t.get("details", {}).get("trough_probability", 1.0),
            )
            t_details = trough.get("details", {})

            # Compute trough score using prominence and width
            trough_score = compute_trough_score(
                min_prob=t_details.get("trough_probability", 1.0),
                prominence=t_details.get("prominence", 0.0),
                width=t_details.get("width", 0.0),
            )
            final_score = valley_score * trough_score

            # Store scores and trough metadata in valley details
            details["trough_score"] = trough_score
            details["final_score"] = final_score
            details["trough_prominence"] = t_details.get("prominence")
            details["trough_width"] = t_details.get("width")
            details["trough_probability"] = t_details.get("trough_probability")
        else:
            details["trough_score"] = 0.0
            details["final_score"] = 0.0
            details["trough_prominence"] = None
            details["trough_width"] = None
            details["trough_probability"] = None

    # Step 5: Filter to valleys with exactly one trough and sufficient duration
    filtered_valleys = [
        v
        for v in valleys
        if len(v.get("details", {}).get("troughs", [])) == 1
        and v["duration_s"] >= min_valley_duration_s
    ]
    console.print(
        f"[cyan]extract_valley_troughs: {len(filtered_valleys)} valley(s) with exactly "
        f"one trough and duration >= {min_valley_duration_s}s[/cyan]"
    )

    # Step 6: Build ValleyTrough results with prominence and width
    result: List[ValleyTrough] = []
    seconds_per_frame = frame_shift_ms / 1000.0
    total_frames = len(probs)

    for valley in filtered_valleys:
        details = valley["details"]

        # Skip troughs too close to the start
        local_trough_time_s = details["min_prob_s"]
        if local_trough_time_s < min_trough_offset_s:
            console.print(
                f"[grey62]extract_valley_troughs: skipping trough at "
                f"{local_trough_time_s:.3f}s (below min_trough_offset_s={min_trough_offset_s}s)[/grey62]"
            )
            continue

        global_trough_time_s = local_trough_time_s + (frame_offset * seconds_per_frame)

        # Build valley info with global coordinates
        valley_info: ValleyInfo = {
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "frame_length": valley["frame_length"],
            "start_s": valley["start_s"],
            "end_s": valley["end_s"],
            "duration_s": valley["duration_s"],
            "valley_score": details["valley_score"],
            "trough_score": details["trough_score"],
            "final_score": details["final_score"],
            "global_frame_start": valley["frame_start"] + frame_offset,
            "global_frame_end": valley["frame_end"] + frame_offset,
            "global_start_s": valley["start_s"] + (frame_offset * seconds_per_frame),
            "global_end_s": valley["end_s"] + (frame_offset * seconds_per_frame),
            "global_duration_s": valley["duration_s"],
            "global_valley_score": details["valley_score"],
            "global_trough_score": details["trough_score"],
            "global_final_score": details["final_score"],
            "is_last": valley["frame_end"] >= total_frames - 1,
        }

        # Build the ValleyTrough with prominence and width
        result.append(
            {
                "frame": details["min_prob_frame"],
                "global_frame": details["min_prob_frame"] + frame_offset,
                "prob": details["min_probability"],
                "time_s": local_trough_time_s,
                "global_time_s": global_trough_time_s,
                "valley": valley_info,
                "prominence": details.get("trough_prominence"),
                "width": details.get("trough_width"),
            }
        )

    console.print(
        f"[cyan]extract_valley_troughs: returning {len(result)} valley trough(s)[/cyan]"
    )
    return result


def extract_valley_troughs_from_np_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    vad_threshold: float = 0.3,
    min_speech_duration_sec: float = 0.25,
    min_silence_duration_sec: float = 0.25,
    smoothing_window: int = 20,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
    min_valley_duration_s: float = 0.25,
    temp_dir: str | Path | None = None,
    trough_height: float | None = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    vad: FireRedVAD | None = None,
) -> list[ValleyTrough]:
    """
    Extract valley troughs (strong silence positions) from a raw numpy audio clip.

    This is a high-level utility that computes speech probability curves using
    a VAD, then analyzes the result to return a list of the most prominent
    troughs located in sufficiently silent zones.

    Workflow:
        1. Saves the provided audio (float32, 16kHz recommended) to a temp WAV.
        2. Runs extract_speech_timestamps (FireRed VAD) to obtain framewise
           speech probabilities.
        3. Runs extract_valley_troughs on those probabilities.
        4. Returns the troughs list, each with local and global info.
        5. Always removes the temporary WAV file (even on error/exit).

    Args:
        audio: 1D numpy array of the audio waveform (float32/float64 preferred).
        sample_rate: Sampling rate of audio (Hz).
        vad_threshold: Threshold for considering a frame as speech.
        min_speech_duration_sec: Minimum seconds required for a speech segment.
        min_silence_duration_sec: Minimum required silence (sec) between segments.
        smoothing_window: Smoothing window (frames) for VAD probability smoothing.
        frame_offset: Frame index offset for adjusting global/local outputs.
        min_trough_offset_s: Min time since start before a trough is eligible.
        temp_dir: Optional path for temporary WAV file; defaults to system temp.
        trough_height: Optional minimum height threshold for trough detection.
        trough_prominence: Minimum prominence for detected troughs.
        trough_distance: Minimum distance (frames) between detected troughs.

    Returns:
        List of ValleyTrough dicts, or an empty list on failure.
    """
    if len(audio) == 0:
        return []

    audio = np.asarray(audio, dtype=np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as tmp:
        temp_wav_path = Path(tmp.name)

    try:
        sf.write(str(temp_wav_path), audio, sample_rate, subtype="FLOAT")
        _, probs = extract_speech_timestamps(
            audio=str(temp_wav_path),
            threshold=vad_threshold,
            min_speech_duration_sec=min_speech_duration_sec,
            min_silence_duration_sec=min_silence_duration_sec,
            with_scores=True,
            vad=vad,
        )
        if not probs:
            return []

        troughs = extract_valley_troughs(
            probs_or_audio=probs,
            smoothing_window=smoothing_window,
            frame_offset=frame_offset,
            min_trough_offset_s=min_trough_offset_s,
            min_valley_duration_s=min_valley_duration_s,
            frame_shift_ms=frame_shift_ms,
            trough_height=trough_height,
            trough_prominence=trough_prominence,
            trough_distance=trough_distance,
        )
        return troughs

    finally:
        try:
            if temp_wav_path.exists():
                temp_wav_path.unlink()
        except Exception:
            pass


def extract_trough_to_trough(
    probs_or_audio: List[float] | AudioInput,
    frame_shift_ms: float = FRAME_SHIFT_MS,
    sample_rate: int = SAMPLE_RATE,
    with_audio: bool = False,
    with_scores: bool = False,
    min_duration_s: float = 0.0,
    smoothing_window: int = 0,
    trough_height: Optional[float] = None,
    trough_prominence: float = 0.15,
    trough_distance: int = 5,
    valley_threshold: Optional[float] = None,
    min_valley_duration_s: float = 0.25,
    min_valley_frames: Optional[int] = None,
    frame_offset: int = 0,
    min_trough_offset_s: float = 0.4,
) -> Union[
    List[TroughToTroughSegment],
    List[Tuple[TroughToTroughSegment, np.ndarray]],
    Tuple[List[TroughToTroughSegment], List[float]],
    Tuple[List[Tuple[TroughToTroughSegment, np.ndarray]], List[float]],
]:
    """
    Create segments spanning from one valley trough to the next.

    This function automatically:
    1. Loads/resolves VAD probabilities from the input.
    2. Extracts valley troughs using provided or default parameters.
    3. Creates segments between consecutive troughs (including start-to-first
       and last-to-end).
    4. Scores every segment via score_trough_to_trough_segments().

    For N valley_troughs, this produces N+1 segments:
        segment_0: t=0          → trough[0]
        segment_1: trough[0]    → trough[1]
        ...
        segment_N: trough[N-1]  → end of audio

    Note on smoothing:
        The ``smoothing_window`` parameter only affects trough/valley detection.
        Segment scoring (``score_trough_to_trough_segments``) applies its own
        internal fixed smoothing window to ensure consistent, jitter-free
        quality assessment regardless of this parameter.

    Args:
        probs_or_audio: VAD probabilities or audio input.
        frame_shift_ms: Frame shift in milliseconds.
        sample_rate: Audio sample rate in Hz.
        with_audio: If True, return list of (segment, audio_slice) tuples.
        with_scores: If True, include per-segment VAD probability scores.
        min_duration_s: Minimum segment duration in seconds.
        smoothing_window: Smoothing window size for trough/valley detection.
            Set to 0 or 1 to disable. Does NOT affect segment scoring.
        trough_height: Min trough height (None = auto-computed).
        trough_prominence: Trough prominence. Default 0.15.
        trough_distance: Min frames between troughs. Default 5.
        valley_threshold: Valley threshold (None = auto-computed).
        min_valley_duration_s: Minimum valley duration. Default 0.25.
        min_valley_frames: Min valley frames (overrides duration if set).
        frame_offset: Global frame offset for chunked processing.
        min_trough_offset_s: Min seconds from start for valid trough. Default 0.4.

    Returns:
        Segments with optional audio slices and probability scores.
        Every segment includes ``scores`` (TroughToTroughScores) and
        ``final_score`` fields from score_trough_to_trough_segments().

    Logs:
        Logs number of troughs, segments created, and any filtering.
    """
    probs, audio_np = load_probs(probs_or_audio)

    if not probs:
        console.print(
            "[yellow]extract_trough_to_trough: no probabilities extracted, "
            "returning empty list.[/yellow]"
        )
        if with_scores:
            return ([], probs) if not with_audio else (([], []), probs)
        return [] if not with_audio else []

    # ── Extract valley troughs ──
    valley_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
        smoothing_window=smoothing_window,
        trough_height=trough_height,
        trough_prominence=trough_prominence,
        trough_distance=trough_distance,
        valley_threshold=valley_threshold,
        min_valley_duration_s=min_valley_duration_s,
        min_valley_frames=min_valley_frames,
        frame_offset=frame_offset,
        min_trough_offset_s=min_trough_offset_s,
    )

    if not valley_troughs:
        console.print(
            "[yellow]extract_trough_to_trough: no valley_troughs found, "
            "returning empty list.[/yellow]"
        )
        if with_scores:
            return [], probs
        return [] if not with_audio else []

    if with_audio and audio_np is None:
        raise ValueError(
            "extract_trough_to_trough: with_audio=True requires an audio input "
            "(not just probabilities). Provide an audio file path, numpy array, etc."
        )

    n_frames = len(probs)
    frame_duration_s = frame_shift_ms / 1000.0
    end_time_s = n_frames * frame_duration_s
    end_frame = n_frames - 1

    # ── Build sentinel boundaries for first/last segments ──
    sentinel_start: ValleyTrough = {
        "frame": 0,
        "global_frame": 0,
        "prob": probs[0] if n_frames > 0 else 0.0,
        "time_s": 0.0,
        "global_time_s": 0.0,
        "valley": valley_troughs[0]["valley"].copy(),
        "prominence": None,
        "width": None,
    }
    sentinel_end: ValleyTrough = {
        "frame": end_frame,
        "global_frame": end_frame,
        "prob": probs[-1] if n_frames > 0 else 0.0,
        "time_s": end_time_s,
        "global_time_s": end_time_s,
        "valley": valley_troughs[-1]["valley"].copy(),
        "prominence": None,
        "width": None,
    }

    anchors: List[ValleyTrough] = (
        [sentinel_start] + list(valley_troughs) + [sentinel_end]
    )

    # ── Create segments ──
    segments: List[TroughToTroughSegment] = []
    audio_slices: List[np.ndarray] = []
    total_audio_samples = len(audio_np) if audio_np is not None else 0
    filtered_count = 0
    filtered_durations: List[float] = []

    for idx in range(len(anchors) - 1):
        vt_start = anchors[idx]
        vt_end = anchors[idx + 1]
        is_first = idx == 0
        is_last = idx == len(anchors) - 2

        start_s: float = float(vt_start["global_time_s"])
        end_s: float = float(vt_end["global_time_s"])
        duration_s: float = round(end_s - start_s, 4)

        start_frame: int = int(vt_start["global_frame"])
        end_frame_seg: int = int(vt_end["global_frame"])

        # ── Duration filter ──
        if min_duration_s > 0 and duration_s < min_duration_s:
            filtered_count += 1
            filtered_durations.append(duration_s)
            console.print(
                f"[dim yellow]Segment {idx} [{start_s:.3f}s - {end_s:.3f}s] "
                f"filtered: duration {duration_s:.3f}s < min {min_duration_s:.3f}s"
                f"[/dim yellow]"
            )
            continue

        # ── Probability statistics for this segment ──
        segment_probs: Optional[List[float]] = None
        prob_stats: Optional[Dict[str, float]] = None

        segment_probs_slice = probs[start_frame : end_frame_seg + 1]
        if with_scores:
            segment_probs = segment_probs_slice

        if segment_probs_slice:
            prob_stats = {
                "mean": float(np.mean(segment_probs_slice)),
                "max": float(np.max(segment_probs_slice)),
                "min": float(np.min(segment_probs_slice)),
                "std": float(np.std(segment_probs_slice)),
                "median": float(np.median(segment_probs_slice)),
                "num_frames": len(segment_probs_slice),
            }
            console.print(
                f"[blue]Segment {idx}: probs stats - "
                f"mean={prob_stats['mean']:.4f}, max={prob_stats['max']:.4f}, "
                f"min={prob_stats['min']:.4f}, std={prob_stats['std']:.4f}, "
                f"median={prob_stats['median']:.4f}, "
                f"frames={prob_stats['num_frames']}[/blue]"
            )

        # ── Build segment ──
        segment: TroughToTroughSegment = {
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "start_frame": start_frame,
            "end_frame": end_frame_seg,
            "trough_start": None if is_first else dict(vt_start),
            "trough_end": None if is_last else dict(vt_end),
            "segment_probs": segment_probs if with_scores else None,
            "prob_stats": prob_stats,
        }
        segments.append(segment)

        # ── Extract audio slice if requested ──
        if with_audio and audio_np is not None:
            start_sample = int(start_s * sample_rate)
            end_sample = int(end_s * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(total_audio_samples, end_sample)
            audio_slice = audio_np[start_sample:end_sample]
            audio_slices.append(audio_slice)
            console.print(
                f"[magenta]Segment {idx}: [{start_s:.3f}s - {end_s:.3f}s] "
                f"→ audio samples [{start_sample}:{end_sample}] "
                f"({len(audio_slice)} samples, {len(audio_slice) / sample_rate:.3f}s)"
                f"[/magenta]"
            )

    # ── Duration filter summary ──
    if min_duration_s > 0:
        console.print(
            f"[yellow]extract_trough_to_trough: Filtered {filtered_count} segment(s) "
            f"shorter than {min_duration_s:.3f}s"
            + (
                f" (durations: {[f'{d:.3f}s' for d in filtered_durations]})"
                if filtered_durations
                else ""
            )
            + f". Kept {len(segments)} segment(s).[/yellow]"
        )

    # ── Score all segments (always, uses internal fixed smoothing) ──
    segments = score_trough_to_trough_segments(segments)

    console.print(
        f"[green]extract_trough_to_trough: Created {len(segments)} segments "
        f"from {len(valley_troughs)} trough(s)"
        + (" with audio slices" if with_audio else "")
        + (" with probability scores" if with_scores else "")
        + (f" (min_duration_s={min_duration_s:.3f}s)" if min_duration_s > 0 else "")
        + ".[/green]"
    )

    if with_audio:
        segments_with_audio = list(zip(segments, audio_slices))
        if with_scores:
            return segments_with_audio, probs
        return segments_with_audio
    if with_scores:
        return segments, probs
    return segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from main._main_vad_extractors import main

    main()
