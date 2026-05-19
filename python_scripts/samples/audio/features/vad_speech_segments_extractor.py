from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib
from _main_vad_speech_segments_extractor import main
from _types import SpeechEndReason, SpeechSegment
from vad_firered_hybrid import FireRedVAD
from config import (
    HOP_STEP_S,
    SAMPLE_RATE,
    SILENCE_MAX_THRESHOLD,
)
from vad_config import SAVE_DIR
from vad_types import ValleyTrough
from loader import load_audio

matplotlib.use("Agg")
import numpy as np
import torch
from vad_config import (
    DEFAULT_INCLUDE_NON_SPEECH,
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    DEFAULT_POSTROLL_MAX_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RETURN_SECONDS,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_THRESHOLD,
    DEFAULT_WITH_SCORES,
)
from rich.console import Console

console = Console()

# Single global cached instance to avoid repeated model loading
_global_vad_cache: Optional[FireRedVAD] = None


def get_global_vad(
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    **model_kwargs,
) -> FireRedVAD:
    """Get or create the global cached VAD instance."""
    global _global_vad_cache

    if _global_vad_cache is None:
        with console.status(
            "[bold blue]Loading FireRedVAD model (global cache)...[/bold blue]"
        ):
            _global_vad_cache = FireRedVAD(
                model_dir=SAVE_DIR,
                threshold=threshold,
                min_silence_duration_sec=min_silence_duration_sec,
                min_speech_duration_sec=min_speech_duration_sec,
                max_speech_duration_sec=max_speech_duration_sec,
                smooth_window_size=smooth_window_size,
                max_buffer_sec=max_buffer_sec,
                **model_kwargs,
            )
    else:
        # Optional: update parameters if they differ significantly
        _global_vad_cache.threshold = threshold
        _global_vad_cache.min_silence_duration_sec = min_silence_duration_sec
        # Add other param updates as needed

    return _global_vad_cache


def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
    return_seconds: bool = DEFAULT_RETURN_SECONDS,
    with_scores: bool = DEFAULT_WITH_SCORES,
    include_non_speech: bool = DEFAULT_INCLUDE_NON_SPEECH,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
    preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
    preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
    postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
    postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
    soft_limit_sec: float = DEFAULT_SOFT_LIMIT_SEC,
    soft_limit_min_valley_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    soft_limit_smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    soft_limit_trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    soft_limit_min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    vad: FireRedVAD | None = None,  # if None, uses global cache
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using FireRedVAD with hybrid pre/post-roll.
    """
    from vad_speech_splitter import (
        apply_limit_splits,
        compute_postroll,
        compute_preroll,
    )

    audio_np, sr = load_audio(audio, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"FireRedVAD requires SAMPLE_RATE Hz, got {sr}")

    if vad is None:
        vad = get_global_vad(
            threshold=threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            smooth_window_size=smooth_window_size,
            max_buffer_sec=max_buffer_sec,
            **kwargs,
        )

    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]

    # === Boundary extension + merging (unchanged) ===
    extended_timestamps: list[tuple[float, float]] = []
    total_samples = len(audio_np)

    for start_sec, end_sec in timestamps:
        onset_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        preroll_samples = compute_preroll(
            onset_sample=onset_sample,
            audio_np=audio_np,
            probs=probs,
            sample_rate=sr,
            max_preroll_sec=preroll_max_sec,
            hybrid_threshold=preroll_hybrid_threshold,
            prob_weight=preroll_prob_weight,
            rms_weight=preroll_rms_weight,
        )
        postroll_samples = compute_postroll(
            end_sample=end_sample,
            audio_np=audio_np,
            probs=probs,
            sample_rate=sr,
            max_postroll_sec=postroll_max_sec,
            hybrid_threshold=postroll_hybrid_threshold,
            prob_weight=postroll_prob_weight,
            rms_weight=postroll_rms_weight,
        )

        new_start_sec = max(0.0, (onset_sample - preroll_samples) / sr)
        new_end_sec = min(total_samples / sr, (end_sample + postroll_samples) / sr)
        extended_timestamps.append((new_start_sec, new_end_sec))

    # Merge
    merged: list[tuple[float, float]] = []
    for seg in extended_timestamps:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(tuple(seg))
    timestamps = [tuple(s) for s in merged]

    # === make_segment helper (with new field) ===
    def make_segment(
        num: int,
        start_sec: float,
        end_sec: float,
        seg_type: Literal["speech", "non-speech"],
        end_reason: "SpeechEndReason | None" = None,
        is_ongoing: bool = False,
        last_non_speech_sec: Optional[float] = None,
        best_valley_trough: Optional["ValleyTrough"] = None,
    ) -> SpeechSegment:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        frame_start = int(start_sec / HOP_STEP_S)
        frame_end = int(end_sec / HOP_STEP_S)
        segment_probs_slice = probs[frame_start : frame_end + 1]
        avg_prob = float(np.mean(segment_probs_slice)) if segment_probs_slice else 0.0
        duration_sec = end_sec - start_sec
        start_val = start_sec if return_seconds else start_sample
        end_val = end_sec if return_seconds else end_sample
        return SpeechSegment(
            num=num,
            start=start_val,
            end=end_val,
            duration=duration_sec,
            end_reason=end_reason,
            is_ongoing=is_ongoing,
            last_non_speech_sec=last_non_speech_sec,
            best_valley_trough=best_valley_trough,
            prob=avg_prob,
            frames_length=len(segment_probs_slice),
            frame_start=frame_start,
            frame_end=frame_end,
            type=seg_type,
            segment_probs=segment_probs_slice if with_scores else [],
        )

    # === Build initial segments (same as before) ===
    enhanced: List[SpeechSegment] = []
    current_time = 0.0
    seg_num = 1
    _HARD_LIMIT_TOLERANCE_S = HOP_STEP_S

    if include_non_speech and timestamps and timestamps[0][0] > 0.001:
        enhanced.append(make_segment(seg_num, 0.0, timestamps[0][0], "non-speech"))
        seg_num += 1
        current_time = timestamps[0][0]

    for start_sec, end_sec in timestamps:
        if include_non_speech and start_sec > current_time + 0.01:
            enhanced.append(
                make_segment(seg_num, current_time, start_sec, "non-speech")
            )
            seg_num += 1

        seg_duration = end_sec - start_sec
        initial_reason = (
            "hard_limit"
            if abs(seg_duration - max_speech_duration_sec) <= _HARD_LIMIT_TOLERANCE_S
            else None
        )

        enhanced.append(
            make_segment(
                seg_num,
                start_sec,
                end_sec,
                "speech",
                end_reason=initial_reason,
            )
        )
        seg_num += 1
        current_time = end_sec

    if include_non_speech and current_time < result["dur"] - 0.01:
        enhanced.append(
            make_segment(seg_num, current_time, result["dur"], "non-speech")
        )

    # === Soft limit splitting (preserves "valley") ===
    if soft_limit_sec > 0:
        enhanced = apply_limit_splits(
            segments=enhanced,
            probs=probs,
            sample_rate=sr,
            hop_sec=HOP_STEP_S,
            soft_limit_sec=soft_limit_sec,
            return_seconds=return_seconds,
            with_scores=with_scores,
            make_segment=make_segment,
            smoothing_window=DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
            trough_prominence=DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
            min_valley_duration_s=DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
            min_trough_offset_s=DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
        )

    # === FINAL REFINEMENT - FIXED LOGIC ===
    for i, seg in enumerate(enhanced):
        if seg["type"] != "speech":
            seg.setdefault("is_ongoing", False)
            seg.setdefault("last_non_speech_sec", None)
            continue

        # IMPORTANT: Preserve "valley" and "hard_limit"
        current_reason = seg.get("end_reason")

        start_s = (
            int(seg["start"] * sr)
            if isinstance(seg["start"], (int, float))
            else seg["start"]
        )
        end_s = (
            int(seg["end"] * sr) if isinstance(seg["end"], (int, float)) else seg["end"]
        )
        audio_slice = audio_np[start_s:end_s]

        refined_reason, last_non_speech = _determine_end_reason_with_duration(
            seg.get("segment_probs", []), audio_slice
        )

        # Only override if we have no strong reason yet OR it's a weak "silence"
        if current_reason in (None, "silence"):
            seg["end_reason"] = refined_reason or current_reason

        seg["last_non_speech_sec"] = last_non_speech
        seg["is_ongoing"] = bool(i == len(enhanced) - 1)

    if with_scores:
        return enhanced, probs
    return enhanced


# ---------------------------------------------------------------------------
# extract_speech_audio
# ---------------------------------------------------------------------------


def extract_speech_audio(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float | None = None,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
    preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
    preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
    postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
    postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
    postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using FireRedVAD.

    Both the head (onset) and tail (end) of each segment are extended via a
    hybrid (prob + RMS) pre-roll / post-roll before slicing the audio.

    Returns a flat list of numpy arrays where each array represents one
    complete speech segment in float32 format, normalised to [-1.0, 1.0].
    """
    if sampling_rate != SAMPLE_RATE:
        raise ValueError(f"FireRedVAD requires SAMPLE_RATE Hz, got {sampling_rate}")

    speech_segments = extract_speech_timestamps(
        audio=audio,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        include_non_speech=False,
        smooth_window_size=smooth_window_size,
        max_buffer_sec=max_buffer_sec,
        preroll_max_sec=preroll_max_sec,
        preroll_hybrid_threshold=preroll_hybrid_threshold,
        preroll_prob_weight=preroll_prob_weight,
        preroll_rms_weight=preroll_rms_weight,
        postroll_max_sec=postroll_max_sec,
        postroll_hybrid_threshold=postroll_hybrid_threshold,
        postroll_prob_weight=postroll_prob_weight,
        postroll_rms_weight=postroll_rms_weight,
    )

    audio_np, sr = load_audio(audio=audio, sr=sampling_rate, mono=True)
    if sr != sampling_rate:
        raise ValueError(
            f"Loaded sample rate {sr} does not match requested {sampling_rate}"
        )

    speech_audio_chunks: List[np.ndarray] = []
    for segment in speech_segments:
        start_sec: float = segment["start"]
        end_sec: float = segment["end"]
        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))
        segment_audio = audio_np[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue
        speech_audio_chunks.append(segment_audio.astype(np.float32, copy=False))

    return speech_audio_chunks


def _determine_end_reason_with_duration(
    segment_probs: List[float],
    audio_np_slice: np.ndarray,
    min_silence_frames: int = 8,
) -> tuple[Optional[SpeechEndReason], Optional[float]]:
    """
    Returns (end_reason, last_non_speech_sec)
    - Only suggests 'silence' if there is a clear trailing low-prob + energy tail.
    - Returns None for end_reason if no strong silence evidence.
    """
    if not segment_probs or len(segment_probs) < min_silence_frames:
        return None, None

    tail_probs = np.array(segment_probs[-min_silence_frames:], dtype=np.float32)

    if len(audio_np_slice) < min_silence_frames * 160:
        return ("silence" if float(np.mean(tail_probs)) < 0.35 else None, None)

    # Trailing audio analysis
    tail_audio = audio_np_slice[-min_silence_frames * 160 :]
    frames = tail_audio.reshape(-1, 160)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    is_low_prob = tail_probs < 0.32
    has_energy = (
        rms > SILENCE_MAX_THRESHOLD * 0.4
    )  # has some audio energy (not dead silence)
    is_trailing_non_speech = is_low_prob & has_energy

    # Count consecutive trailing silent-but-energetic frames
    count = 0
    for i in range(len(is_trailing_non_speech) - 1, -1, -1):
        if is_trailing_non_speech[i]:
            count += 1
        else:
            break

    duration_sec = round(count * HOP_STEP_S, 4)

    # Only call it "silence" if we have meaningful trailing non-speech
    if count >= 4:  # at least ~40ms
        return "silence", duration_sec
    else:
        return None, duration_sec if duration_sec > 0 else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
