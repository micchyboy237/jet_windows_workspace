# vad_peak_analyzer.py

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from config import FRAME_SHIFT_MS, SAMPLE_RATE
from vad_firered2 import extract_speech_timestamps
from vad_firered_hybrid import FireRedVAD
from vad_peak_analyzer import VADPeakAnalyzer
from vad_types import VADSegment, ValleyInfo, ValleyTrough

import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from _types import AudioInput
from config import FRAME_SHIFT_MS, SAMPLE_RATE
from _main_vad_extractors import main
from vad_peak_analyzer import VADPeakAnalyzer
from vad_types import VADSegment, ValleyInfo, ValleyTrough
from loader import load_audio
from rich.console import Console

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

    # Case 1: Already a list of floats.
    if is_probs_list(input_value):
        return input_value, None

    # Case 2: Path (str | Path): .npy/.json/.txt = scores, otherwise treat as audio.
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
                )
                if not is_probs_list(probs):
                    raise ValueError(
                        f"Extracted VAD scores are not a list of floats from: {_linkify(input_path)}"
                    )
                return probs, audio_np

    # Case 3: JSON string containing scores
    if isinstance(input_value, str):
        try:
            loaded = json.loads(input_value)
            if is_probs_list(loaded):
                return loaded, None
        except Exception:
            pass

    # Case 4: AudioInput objects - bytes, ndarray, Tensor
    # Handle numpy.ndarray: treat as waveform
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

    # Handle bytes: treat as audio file bytes
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

    # Handle Torch Tensor (optional, duck-type)
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

    # If we haven't parsed, fall back to default_audio if provided
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
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs
            can resolve into probabilities.
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
    ``final_score`` is returned.  Returns ``None`` when no matching
    trough is found.

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs
            can resolve into probabilities.
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


# Each half is either (probs, trough) when the input was already a list[float],
# or (probs, trough, audio_np) when the input was an AudioInput (str, bytes,
# os.PathLike, ndarray, or Tensor) that had to be decoded into probabilities —
# so callers can access the raw waveform without re-loading.
_SplitHalf = Union[
    Tuple[List[float], ValleyTrough],
    Tuple[List[float], ValleyTrough, np.ndarray],
]
SplitResult = Tuple[_SplitHalf, _SplitHalf]


def split_best_valley_trough(
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
) -> Optional[SplitResult]:
    """
    Split a VAD probability list into two halves at the best valley trough.

    Calls get_best_valley_trough to find the single most prominent silence
    point, then slices ``probs`` at that frame index.

    Args:
        probs_or_audio: VAD probabilities as a list[float], or an AudioInput
            (str, bytes, os.PathLike, ndarray, or Tensor) that load_probs
            can resolve into probabilities.
        All other kwargs are forwarded directly to get_best_valley_trough.

    Returns:
        When probs_or_audio was already a list[float]::

            (
                (probs[:split_frame], best_trough),
                (probs[split_frame:], best_trough),
            )

        When probs_or_audio was an AudioInput (str, bytes, os.PathLike,
        ndarray, or Tensor) that required decoding, each half also carries
        the waveform::

            (
                (probs[:split_frame], best_trough, audio_np),
                (probs[split_frame:], best_trough, audio_np),
            )

        ``None`` if no suitable valley trough was found.
    """
    probs, audio_np = load_probs(probs_or_audio)
    best_trough = get_best_valley_trough(
        probs_or_audio=probs,  # already resolved — passes list[float] directly
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
        return (left_probs, best_trough, audio_np), (right_probs, best_trough, audio_np)
    return (left_probs, best_trough), (right_probs, best_trough)


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

    Args:
        probs_or_audio: VAD speech probabilities as a list[float] in [0.0, 1.0],
            or an AudioInput (str, bytes, os.PathLike, ndarray, or Tensor)
            that load_probs can resolve into probabilities.
        sample_rate: Audio sample rate in Hz.
        frame_shift_ms: Frame shift in milliseconds.
        smoothing_window: Smoothing window size (0 = disabled).
        trough_height: Min trough height (None = auto).
        trough_prominence: Trough prominence. Default 0.15.
        trough_distance: Min frames between troughs. Default 5.
        valley_threshold: Valley threshold (None = auto).
        min_valley_duration_s: Minimum valley duration. Default 0.25.
        min_valley_frames: Min valley frames (overrides duration if set).
        frame_offset: Global frame offset for chunked processing.
        min_trough_offset_s: Min seconds from start for valid trough. Default 0.4.

    Returns:
        List[ValleyTrough]: Detected troughs with enclosing valley info,
        local/global coordinates, and composite scores (valley_score × trough_score).
    """
    probs, _ = load_probs(probs_or_audio)
    analyzer = VADPeakAnalyzer(
        sample_rate=sample_rate,
        frame_shift_ms=frame_shift_ms,
    )
    smoothed = (
        smooth_vad_probs(probs, window=smoothing_window) if smoothing_window else probs
    )
    troughs = analyzer.extract_troughs(
        smoothed,
        height=trough_height,
        prominence=trough_prominence,
        distance=trough_distance,
    )
    valleys = analyzer.extract_valleys(
        smoothed,
        threshold=valley_threshold,
        troughs=troughs,
    )
    valleys = analyzer.filter_short_segments(
        valleys,
        min_duration_s=min_valley_duration_s,
        min_duration_frames=min_valley_frames,
    )
    for valley in valleys:
        details = valley["details"]
        valley_score = compute_valley_score(
            min_prob=details.get("min_probability", 1.0),
            mean_prob=details.get("mean_probability", 1.0),
            duration_s=valley["duration_s"],
        )
        details["valley_score"] = valley_score
        trough_list = details.get("troughs", [])
        if trough_list:
            trough = trough_list[0]
            t_details = trough.get("details", {})
            trough_score = compute_trough_score(
                min_prob=t_details.get("trough_probability", 1.0),
                prominence=t_details.get("prominence", 0.0),
                width=t_details.get("width", 0.0),
            )
            final_score = valley_score * trough_score
            details["trough_score"] = trough_score
            details["final_score"] = final_score
        else:
            details["trough_score"] = 0.0
            details["final_score"] = 0.0

    filtered_valleys = [
        v
        for v in valleys
        if len(v.get("details", {}).get("troughs", [])) == 1
        and v["duration_s"] >= min_valley_duration_s
    ]

    result: List[ValleyTrough] = []
    seconds_per_frame = frame_shift_ms / 1000.0
    total_frames = len(probs)

    for valley in filtered_valleys:
        details = valley["details"]
        local_trough_time_s = details["min_prob_s"]
        if local_trough_time_s < min_trough_offset_s:
            continue
        global_trough_time_s = local_trough_time_s + (frame_offset * seconds_per_frame)
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
        result.append(
            {
                "frame": details["min_prob_frame"],
                "global_frame": details["min_prob_frame"] + frame_offset,
                "prob": details["min_probability"],
                "time_s": local_trough_time_s,
                "global_time_s": global_trough_time_s,
                "valley": valley_info,
            }
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
    troughs located in sufficiently silent zones. Suitable for downstream
    alignment, trimming, splitting, etc.

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
        min_prob: Minimum probability in valley
        mean_prob: Mean probability in valley
        duration_s: Duration in seconds
        max_duration_ref: Duration normalization cap
        w_depth, w_mean, w_duration: Weights

    Returns:
        float score in [0, 1]
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
