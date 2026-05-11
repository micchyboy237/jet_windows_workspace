# vad_peak_analyzer.py

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from config import FRAME_SHIFT_MS, SAMPLE_RATE
from vad_firered2 import extract_speech_timestamps
from vad_peak_analyzer import VADPeakAnalyzer
from vad_types import VADSegment, ValleyInfo, ValleyTrough

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}

SplitResult = Tuple[Tuple[List[float], ValleyTrough], Tuple[List[float], ValleyTrough]]


def base_extract_valley_troughs(
    valleys: List[VADSegment], duration_s: float = 0.25
) -> List[ValleyTrough]:
    """
    Extracts the lowest-probability frames (troughs) from a list of VADSegment valleys,
    but only includes valleys that have exactly one trough and duration >= duration_s.
    """
    filtered_valleys = [
        valley
        for valley in valleys
        if len(valley["details"].get("troughs", [])) == 1
        and valley["duration_s"] >= duration_s
    ]

    # Determine the last frame across all provided valleys so we can mark
    # whichever valley ends there with is_last=True.
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
            # Global fields
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
    probs: List[float],
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
    """
    all_troughs = extract_valley_troughs(
        probs=probs,
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
    # Get the ValleyTrough dictionary with the highest final_score (stored in valley["final_score"])
    best = max(all_troughs, key=lambda t: t["valley"].get("final_score", 0.0))
    return best


def get_last_valley_trough(
    probs: List[float],
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
    """
    all_troughs = extract_valley_troughs(
        probs=probs,
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


def split_best_valley_trough(
    probs: List[float],
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
        probs: Flat list of VAD frame probabilities.
        All other kwargs are forwarded directly to get_best_valley_trough.

    Returns:
        A tuple of two ``(probs_slice, ValleyTrough)`` pairs::

            (
                (probs[:split_frame], best_trough),   # left half
                (probs[split_frame:], best_trough),   # right half
            )

        ``None`` if no suitable valley trough was found.
    """
    best_trough = get_best_valley_trough(
        probs=probs,
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

    return (left_probs, best_trough), (right_probs, best_trough)


def extract_valley_troughs(
    probs: List[float],
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
    Extract salient valley troughs with composite scoring (valley + trough).
    """
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

    # ── Compute Scores ─────────────────────────────────────────────────────
    for valley in valleys:
        details = valley["details"]

        # Valley score
        valley_score = compute_valley_score(
            min_prob=details.get("min_probability", 1.0),
            mean_prob=details.get("mean_probability", 1.0),
            duration_s=valley["duration_s"],
        )
        details["valley_score"] = valley_score

        # Trough score
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

    # ── Filter ─────────────────────────────────────────────────────────────
    filtered_valleys = [
        v
        for v in valleys
        if len(v.get("details", {}).get("troughs", [])) == 1
        and v["duration_s"] >= min_valley_duration_s
    ]

    # ── Build Result ───────────────────────────────────────────────────────
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
            # Global fields
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
) -> list[ValleyTrough]:
    """
    Extract valley troughs (strong silence positions) from a raw numpy audio clip.

    This is a high-level utility that computes speech probability curves using a VAD, then
    analyzes the result to return a list of the most prominent troughs located in sufficiently
    silent zones. This is suitable for downstream alignment, trimming, splitting, etc.

    Workflow:
        1. Saves the provided audio (float32, 16kHz recommended) to a temporary WAV file.
        2. Runs extract_speech_timestamps (FireRed VAD) to obtain framewise speech probabilities.
        3. Runs extract_valley_troughs on those probabilities to extract valley/trough regions.
        4. Returns the troughs list, each with local and global info.
        5. Always removes the temporary WAV file (even on error/exit).

    Args:
        audio: 1D numpy array of the audio waveform (float32/float64 preferred).
        sample_rate: Sampling rate of audio (Hz).
        vad_threshold: Threshold for considering a frame as speech (used by extract_speech_timestamps).
        min_speech_duration_sec: Minimum number of seconds required to be considered a speech segment.
        min_silence_duration_sec: Minimum required silence (sec) between segments.
        smoothing_window: Smoothing window (in frames) for VAD probability smoothing.
        frame_offset: Frame index offset for adjusting global/local outputs.
        min_trough_offset_s: Minimum time since start of segment before a trough is eligible.
        temp_dir: Optional path (or Path) for placing temporary WAV file; defaults to system temp dir.

    Returns:
        List of ValleyTrough dictionaries like those from extract_valley_troughs, or an empty list on failure.
    """
    if len(audio) == 0:
        return []

    # Ensure audio is float32 and normalized-ish
    audio = np.asarray(audio, dtype=np.float32)
    # Basic safety normalization
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as tmp:
        temp_wav_path = Path(tmp.name)

    try:
        # Save to temp WAV
        sf.write(
            str(temp_wav_path),
            audio,
            sample_rate,
            subtype="FLOAT",  # Preserve full precision
        )

        # Run VAD timestamp extraction (gets probabilities)
        _, probs = extract_speech_timestamps(
            audio=str(temp_wav_path),
            threshold=vad_threshold,
            min_speech_duration_sec=min_speech_duration_sec,
            min_silence_duration_sec=min_silence_duration_sec,
            with_scores=True,
        )

        if not probs:
            return []

        # Extract valley troughs from the probabilities
        troughs = extract_valley_troughs(
            probs=probs,
            smoothing_window=smoothing_window,
            frame_offset=frame_offset,
            min_trough_offset_s=min_trough_offset_s,
            min_valley_duration_s=min_valley_duration_s,
            frame_shift_ms=frame_shift_ms,
        )

        return troughs

    finally:
        # Clean up temporary file
        try:
            if temp_wav_path.exists():
                temp_wav_path.unlink()
        except Exception:
            pass  # Best effort cleanup


def smooth_vad_probs(probs: List[float], window: int = 20) -> List[float]:
    """Light moving average smoothing to reduce jitter in VAD probabilities."""
    if window <= 1 or len(probs) <= window:
        return probs[:]
    x = np.array(probs, dtype=float)
    smoothed = np.convolve(x, np.ones(window) / window, mode="same")
    # Better edge handling
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
    Composite score for valley quality.

    Higher score = stronger silence (safe to cut).

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
    """
    Score how safe a trough is for cutting.

    Higher score = safer cut point.
    """
    depth_score = 1.0 - min_prob

    # Normalize prominence (clip to avoid extreme scaling)
    prominence_norm = min(prominence / 0.5, 1.0) if prominence is not None else 0.0

    # Normalize width (frames)
    width_norm = min(width / max_width_ref, 1.0) if width is not None else 0.0

    score = (
        w_depth * depth_score + w_prominence * prominence_norm + w_width * width_norm
    )

    return float(score)


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Extract valley troughs (strong silence points) from audio or VAD probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio", "-a", type=Path, help="Path to audio file (.wav, .mp3, etc.)"
    )
    input_group.add_argument(
        "--probs",
        "-p",
        type=Path,
        help="Path to .npy file containing VAD probabilities",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=Path(__file__).parent / "generated" / Path(__file__).stem,
        help="Output directory to save JSON results (default: generated/)",
    )

    # Core parameters
    parser.add_argument(
        "--min-duration",
        "-d",
        type=float,
        default=0.25,
        help="Minimum valley duration in seconds",
    )
    parser.add_argument(
        "--min-trough-offset",
        "-t",
        "-to",
        dest="min_trough_offset",
        type=float,
        default=0.4,
        help="Minimum seconds into the valley before a trough is accepted",
    )
    parser.add_argument(
        "--smoothing",
        "-s",
        type=int,
        default=20,
        help="Smoothing window size (0 = disabled)",
    )
    parser.add_argument(
        "--trough-prominence",
        "-T",
        type=float,
        default=0.15,
        help="Minimum prominence for trough detection",
    )
    parser.add_argument(
        "--valley-threshold",
        "-v",
        type=float,
        default=None,
        help="Override valley threshold (None = auto)",
    )

    # VAD parameters (used only with audio input)
    parser.add_argument(
        "--vad-threshold",
        "-V",
        type=float,
        default=0.3,
        help="VAD speech threshold (used only with --audio)",
    )

    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.audio:
            print(f"Processing audio file: {args.audio}")
            troughs = extract_valley_troughs_from_np_audio(
                audio=sf.read(str(args.audio))[0],  # Read audio
                sample_rate=sf.read(str(args.audio))[1],
                smoothing_window=args.smoothing,
                min_trough_offset_s=args.min_trough_offset,
                min_valley_duration_s=args.min_duration,
                vad_threshold=args.vad_threshold,
            )

            # Compose output file path
            output_file = args.output_dir / "valley_troughs.json"
        else:
            # Load probabilities from .npy file
            print(f"Loading probabilities from: {args.probs}")
            probs = np.load(args.probs)
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()

            troughs = extract_valley_troughs(
                probs=probs,
                min_valley_duration_s=args.min_duration,
                smoothing_window=args.smoothing,
                trough_prominence=args.trough_prominence,
                valley_threshold=args.valley_threshold,
                min_trough_offset_s=args.min_trough_offset,
            )

            # Compose output file path
            output_file = args.output_dir / "valley_troughs.json"

        # Output results
        if not troughs:
            print("No valid valley troughs found.")
        else:
            print(f"\nFound {len(troughs)} valley trough(s):\n")

            for i, trough in enumerate(troughs, 1):
                v = trough["valley"]
                print(
                    f"{i:2d}. Time: {trough['time_s']:.3f}s  "
                    f"(Global: {trough.get('global_time_s', trough['time_s']):.3f}s)"
                )
                print(
                    f"    Prob: {trough['prob']:.4f} | "
                    f"Valley Score: {v['valley_score']:.4f} | "
                    f"Trough Score: {v['trough_score']:.4f} | "
                    f"Final Score: {v['final_score']:.4f}"
                )
                print(
                    f"    Duration: {v['duration_s']:.3f}s "
                    f"({v['frame_start']}–{v['frame_end']} frames)\n"
                )

            # Save to JSON if requested
            if args.output_dir:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(troughs, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
