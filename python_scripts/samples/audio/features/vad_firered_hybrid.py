import io
import json
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from fireredvad.core.constants import SAMPLE_RATE
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from loader import load_audio
from _types import AudioInput, SpeechSegment, WordSegment

console = Console()

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_SEC = 0.250
DEFAULT_MIN_SPEECH_SEC = 0.250
DEFAULT_MAX_SPEECH_SEC = 15.0
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_RETURN_SECONDS = False
DEFAULT_WITH_SCORES = False
DEFAULT_INCLUDE_NON_SPEECH = False

DEFAULT_SMOOTH_WINDOW_SIZE = 5
DEFAULT_MAX_BUFFER_SEC = 1.2

# Pre-roll defaults (head extension — looks backward from onset)
DEFAULT_PREROLL_MAX_SEC = 0.300          # maximum look-back window
DEFAULT_PREROLL_HYBRID_THRESHOLD = 0.15  # hybrid score below which we stop extending
DEFAULT_PREROLL_PROB_WEIGHT = 0.5        # weight for speech probability
DEFAULT_PREROLL_RMS_WEIGHT = 0.5         # weight for normalised RMS energy

# Post-roll defaults (tail extension — looks forward from detected end)
DEFAULT_POSTROLL_MAX_SEC = 0.300         # maximum look-forward window
DEFAULT_POSTROLL_HYBRID_THRESHOLD = 0.15 # hybrid score below which we stop extending
DEFAULT_POSTROLL_PROB_WEIGHT = 0.5       # weight for speech probability
DEFAULT_POSTROLL_RMS_WEIGHT = 0.5        # weight for normalised RMS energy


# ---------------------------------------------------------------------------
# Pre-roll buffer
# ---------------------------------------------------------------------------

class PreRollBuffer:
    """
    Maintains a rolling window of (audio_samples, hybrid_score) pairs aligned
    to 10 ms frames so that, at each speech-segment onset, we can look backward
    and prepend exactly as much audio as the signal warrants.

    Hybrid score  =  prob_weight * smoothed_prob
                   + rms_weight  * rms_norm

    where rms_norm is the per-frame RMS normalised to [0, 1] using the
    long-run 99th-percentile as the ceiling (estimated online via a running
    max with mild decay).
    """

    FRAME_SAMPLES = 160   # 10 ms @ 16 kHz

    def __init__(
        self,
        max_preroll_sec: float = DEFAULT_PREROLL_MAX_SEC,
        hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
        prob_weight: float = DEFAULT_PREROLL_PROB_WEIGHT,
        rms_weight: float = DEFAULT_PREROLL_RMS_WEIGHT,
        sample_rate: int = 16000,
    ) -> None:
        self.max_preroll_sec = max_preroll_sec
        self.hybrid_threshold = hybrid_threshold
        self.prob_weight = prob_weight
        self.rms_weight = rms_weight
        self.sample_rate = sample_rate

        self._max_frames = int(max_preroll_sec * 100)   # 100 frames/s
        self._audio_frames: list[np.ndarray] = []       # deque of FRAME_SAMPLES arrays
        self._hybrid_scores: list[float] = []           # parallel list
        self._rms_running_max: float = 1e-6             # online normaliser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._audio_frames.clear()
        self._hybrid_scores.clear()
        self._rms_running_max = 1e-6

    def push(self, audio_chunk: np.ndarray, prob: float) -> None:
        """
        Feed a chunk of raw audio (any length) together with the corresponding
        smoothed speech probability for that chunk.  The chunk is split into
        FRAME_SAMPLES blocks; a hybrid score is computed for each block and the
        rolling window is updated.
        """
        if len(audio_chunk) == 0:
            return
        audio_chunk = audio_chunk.astype(np.float32)

        # Pad to a multiple of FRAME_SAMPLES
        remainder = len(audio_chunk) % self.FRAME_SAMPLES
        if remainder:
            audio_chunk = np.pad(audio_chunk, (0, self.FRAME_SAMPLES - remainder))

        n_frames = len(audio_chunk) // self.FRAME_SAMPLES
        for i in range(n_frames):
            frame = audio_chunk[i * self.FRAME_SAMPLES: (i + 1) * self.FRAME_SAMPLES]
            rms = float(np.sqrt(np.mean(frame ** 2)))

            # Update running normaliser with mild exponential decay
            self._rms_running_max = max(
                self._rms_running_max * 0.9999, rms + 1e-10
            )
            rms_norm = min(rms / self._rms_running_max, 1.0)

            score = self.prob_weight * prob + self.rms_weight * rms_norm

            self._audio_frames.append(frame)
            self._hybrid_scores.append(score)

        # Keep only the last _max_frames entries
        if len(self._audio_frames) > self._max_frames:
            excess = len(self._audio_frames) - self._max_frames
            self._audio_frames = self._audio_frames[excess:]
            self._hybrid_scores = self._hybrid_scores[excess:]

    def get_preroll(self) -> np.ndarray:
        """
        Scan backward from the most recent frame; return contiguous audio for
        every frame whose hybrid score is above *hybrid_threshold*.  Stops at
        the first frame that falls below the threshold (or at the buffer edge).
        """
        n = len(self._audio_frames)
        if n == 0:
            return np.array([], dtype=np.float32)

        keep = 0
        for i in range(n - 1, -1, -1):
            if self._hybrid_scores[i] >= self.hybrid_threshold:
                keep = n - i          # frames to keep
            else:
                break                 # stop at first sub-threshold frame

        if keep == 0:
            return np.array([], dtype=np.float32)

        frames = self._audio_frames[n - keep:]
        return np.concatenate(frames).astype(np.float32)


# ---------------------------------------------------------------------------
# FireRedVAD wrapper
# ---------------------------------------------------------------------------

class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
        max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
        smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
        max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        console.print(f"[cyan]Loading FireRedVAD (streaming) on {self.device}…[/cyan]")
        frames_per_sec = 100
        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            smooth_window_size=smooth_window_size,
            min_speech_frame=int(min_speech_duration_sec * frames_per_sec),
            max_speech_frame=int(max_speech_duration_sec * frames_per_sec),
            min_silence_frame=int(min_silence_duration_sec * frames_per_sec),
            chunk_max_frame=30000,
        )
        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        self.vad.vad_model.to(self.device)
        console.print("[green]done.[/green]")
        self.sample_rate = SAMPLE_RATE
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0
        self.max_buffer_samples = int(max_buffer_sec * self.sample_rate)

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic range compression / gain normalization."""
        if len(chunk) == 0:
            return chunk.astype(np.float32)
        chunk = chunk.astype(np.float32)
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain
        return chunk

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process incoming audio chunk (any length) and return
        the **latest smoothed speech probability**.
        """
        if len(chunk) == 0:
            return self.last_prob
        chunk = self._normalize_chunk(chunk)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        if len(self.audio_buffer) < 4800:
            return self.last_prob
        to_process = self.audio_buffer[-9600:]
        results = self.vad.detect_chunk(to_process)
        self.audio_buffer = self.audio_buffer[-512:]
        if not results:
            return self.last_prob
        last = results[-1]
        prob = last.smoothed_prob
        self.last_prob = prob
        return prob

    def get_latest_result(self) -> Optional[dict]:
        return None

    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        self.reset()
        frame_results, result = self.vad.detect_full(audio)
        return frame_results, result


# ---------------------------------------------------------------------------
# Pre-roll computation helper
# ---------------------------------------------------------------------------

def _compute_preroll(
    onset_sample: int,
    audio_np: np.ndarray,
    probs: list[float],
    sample_rate: int,
    max_preroll_sec: float,
    hybrid_threshold: float,
    prob_weight: float,
    rms_weight: float,
) -> int:
    """
    Given a speech-segment onset (in samples), look backward through the
    pre-speech audio and find how many additional samples to prepend.

    Strategy
    --------
    1. Build per-frame hybrid scores for up to *max_preroll_sec* before onset.
    2. Walk backward from the onset frame; extend the pre-roll for every
       consecutive frame whose hybrid score >= hybrid_threshold.
    3. Return the number of *samples* to prepend (>= 0).

    The hybrid score per 10 ms frame:
        score = prob_weight * smoothed_prob + rms_weight * rms_norm

    RMS is normalised using the 99th-percentile of the look-back window.
    """
    FRAME_SAMPLES = 160   # 10 ms @ 16 kHz
    hop_sec = 0.010
    max_preroll_samples = int(max_preroll_sec * sample_rate)

    start_sample = max(0, onset_sample - max_preroll_samples)
    lookback_audio = audio_np[start_sample:onset_sample]

    if len(lookback_audio) == 0:
        return 0

    # Align to frame grid
    n_frames = len(lookback_audio) // FRAME_SAMPLES
    if n_frames == 0:
        return 0

    # Trim to frame-aligned length
    lookback_audio = lookback_audio[: n_frames * FRAME_SAMPLES]

    # Per-frame RMS
    frames = lookback_audio.reshape(n_frames, FRAME_SAMPLES)
    rms_arr = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_ceil = np.percentile(rms_arr, 99) + 1e-10
    rms_norm = np.clip(rms_arr / rms_ceil, 0.0, 1.0)

    # Per-frame smoothed prob (align frame index to global prob array)
    onset_frame = int(onset_sample / sample_rate / hop_sec)
    look_start_frame = onset_frame - n_frames

    hybrid = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        global_f = look_start_frame + i
        prob = probs[global_f] if 0 <= global_f < len(probs) else 0.0
        hybrid[i] = prob_weight * prob + rms_weight * float(rms_norm[i])

    # Walk backward from the frame immediately before onset
    keep_frames = 0
    for i in range(n_frames - 1, -1, -1):
        if hybrid[i] >= hybrid_threshold:
            keep_frames = n_frames - i
        else:
            break

    return keep_frames * FRAME_SAMPLES


# ---------------------------------------------------------------------------
# Post-roll computation helper  (symmetric tail extension)
# ---------------------------------------------------------------------------

def _compute_postroll(
    end_sample: int,
    audio_np: np.ndarray,
    probs: list[float],
    sample_rate: int,
    max_postroll_sec: float,
    hybrid_threshold: float,
    prob_weight: float,
    rms_weight: float,
) -> int:
    """
    Given a speech-segment end (in samples), look *forward* through the
    post-speech audio and find how many additional samples to append.

    Strategy
    --------
    1. Build per-frame hybrid scores for up to *max_postroll_sec* after end.
    2. Walk forward from the end frame; extend the post-roll for every
       consecutive frame whose hybrid score >= hybrid_threshold.
    3. Return the number of *samples* to append (>= 0).

    The hybrid score per 10 ms frame:
        score = prob_weight * smoothed_prob + rms_weight * rms_norm

    RMS is normalised using the 99th-percentile of the look-forward window.
    """
    FRAME_SAMPLES = 160   # 10 ms @ 16 kHz
    hop_sec = 0.010
    max_postroll_samples = int(max_postroll_sec * sample_rate)

    stop_sample = min(len(audio_np), end_sample + max_postroll_samples)
    lookahead_audio = audio_np[end_sample:stop_sample]

    if len(lookahead_audio) == 0:
        return 0

    # Align to frame grid
    n_frames = len(lookahead_audio) // FRAME_SAMPLES
    if n_frames == 0:
        return 0

    lookahead_audio = lookahead_audio[: n_frames * FRAME_SAMPLES]

    # Per-frame RMS
    frames = lookahead_audio.reshape(n_frames, FRAME_SAMPLES)
    rms_arr = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_ceil = np.percentile(rms_arr, 99) + 1e-10
    rms_norm = np.clip(rms_arr / rms_ceil, 0.0, 1.0)

    # Per-frame smoothed prob (align frame index to global prob array)
    end_frame = int(end_sample / sample_rate / hop_sec)

    hybrid = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        global_f = end_frame + i
        prob = probs[global_f] if 0 <= global_f < len(probs) else 0.0
        hybrid[i] = prob_weight * prob + rms_weight * float(rms_norm[i])

    # Walk forward from the frame immediately after the detected end
    keep_frames = 0
    for i in range(n_frames):
        if hybrid[i] >= hybrid_threshold:
            keep_frames = i + 1   # extend at least through this frame
        else:
            break                 # stop at first sub-threshold frame

    return keep_frames * FRAME_SAMPLES



def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = DEFAULT_RETURN_SECONDS,
    with_scores: bool = DEFAULT_WITH_SCORES,
    include_non_speech: bool = DEFAULT_INCLUDE_NON_SPEECH,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
    preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
    preroll_prob_weight: float = DEFAULT_PREROLL_PROB_WEIGHT,
    preroll_rms_weight: float = DEFAULT_PREROLL_RMS_WEIGHT,
    postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
    postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    postroll_prob_weight: float = DEFAULT_POSTROLL_PROB_WEIGHT,
    postroll_rms_weight: float = DEFAULT_POSTROLL_RMS_WEIGHT,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using FireRedVAD with symmetric hybrid
    pre-roll (head) and post-roll (tail) boundary extension.

    Both boundaries are extended by a variable amount computed from a
    weighted combination of smoothed speech probability and normalised RMS
    energy (equal 0.5/0.5 weights by default).

    When include_non_speech=True, returns both speech and non-speech segments.
    """
    if max_speech_duration_sec is None:
        max_speech_duration_sec = DEFAULT_MAX_SPEECH_SEC

    audio_np, sr = load_audio(audio, sr=16000, mono=True)
    if sr != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sr}")

    vad = FireRedVAD(
        model_dir=SAVE_DIR,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        smooth_window_size=smooth_window_size,
        max_buffer_sec=max_buffer_sec,
    )

    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]
    hop_sec = 0.010

    # ------------------------------------------------------------------
    # Apply hybrid pre-roll (head) and post-roll (tail) to each segment
    # ------------------------------------------------------------------
    extended_timestamps: list[tuple[float, float]] = []
    total_samples = len(audio_np)
    for start_sec, end_sec in timestamps:
        onset_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        preroll_samples = _compute_preroll(
            onset_sample=onset_sample,
            audio_np=audio_np,
            probs=probs,
            sample_rate=sr,
            max_preroll_sec=preroll_max_sec,
            hybrid_threshold=preroll_hybrid_threshold,
            prob_weight=preroll_prob_weight,
            rms_weight=preroll_rms_weight,
        )
        postroll_samples = _compute_postroll(
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

    # Merge overlapping segments that may arise after pre-roll extension
    merged: list[tuple[float, float]] = []
    for seg in extended_timestamps:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(list(seg))
    timestamps = [tuple(s) for s in merged]

    # ------------------------------------------------------------------
    # Build SpeechSegment objects
    # ------------------------------------------------------------------
    def make_segment(
        num: int,
        start_sec: float,
        end_sec: float,
        seg_type: Literal["speech", "non-speech"],
    ) -> SpeechSegment:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        frame_start = int(start_sec / hop_sec)
        frame_end = int(end_sec / hop_sec)
        segment_probs_slice = probs[frame_start: frame_end + 1]
        avg_prob = float(np.mean(segment_probs_slice)) if segment_probs_slice else 0.0
        duration_sec = end_sec - start_sec
        start_val = start_sec if return_seconds else start_sample
        end_val = end_sec if return_seconds else end_sample
        return SpeechSegment(
            num=num,
            start=start_val,
            end=end_val,
            prob=avg_prob,
            duration=duration_sec,
            frames_length=len(segment_probs_slice),
            frame_start=frame_start,
            frame_end=frame_end,
            type=seg_type,
            segment_probs=segment_probs_slice if with_scores else [],
        )

    enhanced: List[SpeechSegment] = []
    current_time = 0.0
    seg_num = 1

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
        enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
        seg_num += 1
        current_time = end_sec

    total_duration = result["dur"]
    if include_non_speech and current_time < total_duration - 0.01:
        enhanced.append(
            make_segment(seg_num, current_time, total_duration, "non-speech")
        )

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
    preroll_prob_weight: float = DEFAULT_PREROLL_PROB_WEIGHT,
    preroll_rms_weight: float = DEFAULT_PREROLL_RMS_WEIGHT,
    postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
    postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    postroll_prob_weight: float = DEFAULT_POSTROLL_PROB_WEIGHT,
    postroll_rms_weight: float = DEFAULT_POSTROLL_RMS_WEIGHT,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using FireRedVAD.

    Both the head (onset) and tail (end) of each segment are extended via a
    hybrid (prob + RMS) pre-roll / post-roll before slicing the audio.

    Returns a flat list of numpy arrays where each array represents one
    complete speech segment in float32 format, normalised to [-1.0, 1.0].
    """
    if sampling_rate != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sampling_rate}")

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


# ---------------------------------------------------------------------------
# Helpers used by save_segments
# ---------------------------------------------------------------------------

def _frames_from_seconds(sec: float) -> int:
    """Convert seconds to a 10 ms frame index (100 frames per second)."""
    return int(round(sec * 100.0))


def _compute_rms(
    signal: np.ndarray,
    frame_length: int = 160,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Compute per-frame RMS energy aligned to 10 ms frames.
    160 samples @ 16 kHz = exactly 10 ms per frame.
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)
    num_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start: start + frame_length]
        if frame.size:
            rms[i] = float(np.sqrt(np.mean(frame ** 2)))
    return rms


def _generate_plot(
    probs: np.ndarray,
    segment_idx: int,
    duration_sec: float,
    output_path: Path,
    is_dummy: bool = False,
    rms: Optional[np.ndarray] = None,
    hybrid: Optional[np.ndarray] = None,
    hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
) -> None:
    """Save a speech-probability, RMS energy, and hybrid score plot to *output_path*."""
    num_frames = len(probs)
    if num_frames == 0:
        return

    has_rms = rms is not None and len(rms) > 0
    has_hybrid = hybrid is not None and len(hybrid) > 0
    rows = 1 + int(has_rms) + int(has_hybrid)
    fig, axes = plt.subplots(rows, 1, figsize=(9.5, 3.2 * rows), dpi=140)
    if rows == 1:
        axes = [axes]

    label = "Speech probability (dummy)" if is_dummy else "Speech probability"
    color = "#ff7f0e" if is_dummy else "#2ca02c"
    ax = axes[0]
    ax.plot(probs, color=color, linewidth=1.8, label=label)
    ax.fill_between(range(num_frames), probs, color=color, alpha=0.14)
    ax.axhline(
        y=0.4, linestyle="--", color="#d62728", alpha=0.65,
        linewidth=1.2, label="threshold ≈ 0.4",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, num_frames - 1)
    ax.set_ylabel("Speech Probability", fontsize=10.5)
    ax.set_xlabel(
        f"Frame (10 ms)  —  {num_frames} frames ≈ {duration_sec:.1f} s",
        fontsize=10.5,
    )
    ax.set_title(
        f"Segment {segment_idx:03d} — {'Dummy ' if is_dummy else ''}Model Probabilities",
        fontsize=12, pad=12,
    )
    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    ax_idx = 1
    if has_rms:
        ax_rms = axes[ax_idx]
        ax_rms.plot(range(len(rms)), rms, linewidth=1.6, label="RMS energy")
        ax_rms.fill_between(range(len(rms)), rms, alpha=0.15)
        ax_rms.set_ylabel("RMS Energy", fontsize=10.5)
        ax_rms.set_xlabel("Frame (10 ms)", fontsize=10.5)
        ax_rms.set_xlim(0, len(rms) - 1)
        ax_rms.grid(True, alpha=0.28, linestyle="--", zorder=0)
        ax_rms.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
        ax_idx += 1

    if has_hybrid:
        ax_hyb = axes[ax_idx]
        n_hyb = len(hybrid)
        ax_hyb.plot(hybrid, color="#9467bd", linewidth=1.8, label="Hybrid score (0.5·prob + 0.5·RMS)")
        ax_hyb.fill_between(range(n_hyb), hybrid, color="#9467bd", alpha=0.14)
        ax_hyb.axhline(
            y=hybrid_threshold, linestyle="--", color="#d62728", alpha=0.65,
            linewidth=1.2, label=f"threshold = {hybrid_threshold}",
        )
        ax_hyb.set_ylim(-0.03, 1.03)
        ax_hyb.set_xlim(0, n_hyb - 1)
        ax_hyb.set_ylabel("Hybrid Score", fontsize=10.5)
        ax_hyb.set_xlabel("Frame (10 ms)", fontsize=10.5)
        ax_hyb.grid(True, alpha=0.28, linestyle="--", zorder=0)
        ax_hyb.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    fig.tight_layout(pad=0.9)
    plt.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_segments
# ---------------------------------------------------------------------------

def save_segments(
    segments: List[SpeechSegment],
    audio_chunks: List[np.ndarray],
    output_base_dir: Path,
) -> List[SpeechSegment]:
    """
    Persist every speech segment to *output_base_dir/segments/segment_NNN/*.

    For each segment the function writes:
      sound.wav          – 16-kHz PCM-16 audio
      meta.json          – SpeechSegment metadata + probs_info summary
      speech_probs.json  – per-frame probabilities + summary stats
      energies.json      – per-frame RMS energy
      speech_and_rms.png – probability + RMS energy plot
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_base_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    speech_segments = [s for s in segments if s["type"] == "speech"]

    if len(speech_segments) != len(audio_chunks):
        console.print(
            f"[yellow]save_segments: {len(speech_segments)} speech segments but "
            f"{len(audio_chunks)} audio chunks — zipping by position, extras ignored.[/yellow]"
        )

    pairs = list(zip(speech_segments, audio_chunks))
    saved: List[SpeechSegment] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Saving segments + plots…", total=len(pairs))

        for meta, audio_np in pairs:
            idx = meta["num"]
            seg_dir = segments_dir / f"segment_{idx:03d}"
            seg_dir.mkdir(exist_ok=True)

            # ── 1. WAV ────────────────────────────────────────────────────
            wav_path = seg_dir / "sound.wav"
            try:
                torchaudio.save(
                    str(wav_path),
                    torch.from_numpy(audio_np).unsqueeze(0),
                    16000,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            except Exception as exc:
                console.print(f"[red]Failed to save WAV {wav_path}: {exc}[/red]")
                progress.advance(task)
                continue

            # ── 2. Probability array ──────────────────────────────────────
            seg_probs_list: List[float] = meta.get("segment_probs", [])
            seg_probs_arr = np.asarray(seg_probs_list, dtype=np.float32)
            is_dummy = len(seg_probs_arr) == 0

            if is_dummy:
                num_frames = max(1, _frames_from_seconds(meta["duration"]))
                t = np.linspace(0, 1, num_frames)
                base = 0.12 + 0.76 / (1 + np.exp(-14 * (t - 0.48)))
                noise = np.random.default_rng().normal(0, 0.035, num_frames)
                seg_probs_arr = np.clip(base + noise, 0.03, 0.99).astype(np.float32)
                seg_probs_arr *= 0.88 + 0.12 * np.sin(np.pi * t) ** 0.35
                console.print(
                    f"[yellow]Segment {idx:03d}: no probabilities stored — "
                    "using synthetic fallback.[/yellow]"
                )

            # ── 3. probs_info summary stats ───────────────────────────────
            probs_info = {
                "num_frames": int(len(seg_probs_arr)),
                "mean":       float(np.mean(seg_probs_arr)),
                "max":        float(np.max(seg_probs_arr)),
                "min":        float(np.min(seg_probs_arr)),
                "std":        float(np.std(seg_probs_arr)),
                "median":     float(np.median(seg_probs_arr)),
                "frame_rate_hz": 100,
            }

            # ── 4. meta.json ──────────────────────────────────────────────
            meta_to_save = dict(meta)
            meta_to_save["output_path"] = str(
                wav_path.relative_to(output_base_dir)
            )
            meta_to_save["probs_info"] = probs_info
            meta_to_save.pop("segment_probs", None)
            with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta_to_save, fh, indent=2, ensure_ascii=False)

            # ── 5. speech_probs.json ──────────────────────────────────────
            with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "probs": seg_probs_arr.tolist(),
                        "frame_shift_sec": 0.010,
                        "frame_start": meta.get("frame_start", 0),
                        "summary": probs_info,
                        "is_dummy": is_dummy,
                    },
                    fh,
                    indent=2,
                )

            # ── 6. energies.json ──────────────────────────────────────────
            rms = _compute_rms(audio_np)
            with open(seg_dir / "energies.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rms": rms.tolist(),
                        "frame_shift_sec": 0.010,
                        "num_frames": int(len(rms)),
                    },
                    fh,
                    indent=2,
                )

            # ── 7. speech_and_rms.png ─────────────────────────────────────
            # Align rms to the same frame count as probs for the hybrid score.
            # Both are 10 ms frames; length may differ slightly at boundaries.
            n_prob = len(seg_probs_arr)
            n_rms  = len(rms)
            n_min  = min(n_prob, n_rms)
            if n_min > 0:
                rms_ceil = np.percentile(rms[:n_min], 99) + 1e-10
                rms_norm = np.clip(rms[:n_min] / rms_ceil, 0.0, 1.0)
                hybrid_arr = (0.5 * seg_probs_arr[:n_min] + 0.5 * rms_norm).astype(np.float32)
            else:
                hybrid_arr = np.array([], dtype=np.float32)

            _generate_plot(
                probs=seg_probs_arr,
                segment_idx=idx,
                duration_sec=float(meta["duration"]),
                output_path=seg_dir / "speech_and_rms.png",
                is_dummy=is_dummy,
                rms=rms,
                hybrid=hybrid_arr,
                hybrid_threshold=DEFAULT_PREROLL_HYBRID_THRESHOLD,
            )

            meta["output_path"] = meta_to_save["output_path"]
            saved.append(meta)
            progress.advance(task)

    console.print(f"[bold green]✓ Saved {len(saved)} segments[/bold green]")
    console.print(
        f"Output: [link=file://{segments_dir.resolve()}]{segments_dir}[/link]"
    )
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import shutil

    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD + hybrid pre-roll"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"speech threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-ms", "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_SEC,
        help=f"minimum silence duration in seconds (default: {DEFAULT_MIN_SILENCE_SEC})",
    )
    parser.add_argument(
        "-mp", "--min-speech",
        type=float,
        default=DEFAULT_MIN_SPEECH_SEC,
        help=f"minimum speech duration in seconds (default: {DEFAULT_MIN_SPEECH_SEC})",
    )
    parser.add_argument(
        "-mx", "--max-speech",
        type=float,
        default=8.0,
        help="maximum speech duration in seconds",
    )
    parser.add_argument(
        "-sw", "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW_SIZE,
        help=f"smoothing window size (default: {DEFAULT_SMOOTH_WINDOW_SIZE})",
    )
    parser.add_argument(
        "-mb", "--max-buffer-sec",
        type=float,
        default=DEFAULT_MAX_BUFFER_SEC,
        help=f"stream buffer duration in seconds (default: {DEFAULT_MAX_BUFFER_SEC})",
    )
    # Pre-roll args
    parser.add_argument(
        "--preroll-max-sec",
        type=float,
        default=DEFAULT_PREROLL_MAX_SEC,
        help=f"max pre-roll look-back in seconds (default: {DEFAULT_PREROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--preroll-threshold",
        type=float,
        default=DEFAULT_PREROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for pre-roll extension (default: {DEFAULT_PREROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--preroll-prob-weight",
        type=float,
        default=DEFAULT_PREROLL_PROB_WEIGHT,
        help=f"weight for speech prob in hybrid score (default: {DEFAULT_PREROLL_PROB_WEIGHT})",
    )
    parser.add_argument(
        "--preroll-rms-weight",
        type=float,
        default=DEFAULT_PREROLL_RMS_WEIGHT,
        help=f"weight for RMS energy in hybrid score (default: {DEFAULT_PREROLL_RMS_WEIGHT})",
    )
    # Post-roll args
    parser.add_argument(
        "--postroll-max-sec",
        type=float,
        default=DEFAULT_POSTROLL_MAX_SEC,
        help=f"max post-roll look-forward in seconds (default: {DEFAULT_POSTROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--postroll-threshold",
        type=float,
        default=DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for post-roll extension (default: {DEFAULT_POSTROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--postroll-prob-weight",
        type=float,
        default=DEFAULT_POSTROLL_PROB_WEIGHT,
        help=f"weight for speech prob in hybrid score (default: {DEFAULT_POSTROLL_PROB_WEIGHT})",
    )
    parser.add_argument(
        "--postroll-rms-weight",
        type=float,
        default=DEFAULT_POSTROLL_RMS_WEIGHT,
        help=f"weight for RMS energy in hybrid score (default: {DEFAULT_POSTROLL_RMS_WEIGHT})",
    )

    args = parser.parse_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD2 + Hybrid Pre/Post-Roll", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")
    console.print(
        f"[dim]Pre-roll:  max={args.preroll_max_sec}s  "
        f"threshold={args.preroll_threshold}  "
        f"prob_w={args.preroll_prob_weight}  "
        f"rms_w={args.preroll_rms_weight}[/dim]"
    )
    console.print(
        f"[dim]Post-roll: max={args.postroll_max_sec}s  "
        f"threshold={args.postroll_threshold}  "
        f"prob_w={args.postroll_prob_weight}  "
        f"rms_w={args.postroll_rms_weight}[/dim]\n"
    )

    # ── Step 1: detect segments (with per-frame probabilities) ────────────
    segments, speech_probs = extract_speech_timestamps(
        audio_path,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        return_seconds=True,
        with_scores=True,
        include_non_speech=False,
        smooth_window_size=args.smooth_window,
        max_buffer_sec=args.max_buffer_sec,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        seg_type = seg["type"]
        type_color = "bold green" if seg_type == "speech" else "bold red"
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white]"
            f" - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
        )

    if not any(s["type"] == "speech" for s in segments):
        console.print("[red]No speech segments found.[/red]")
        raise SystemExit(0)

    # ── Step 2: extract raw audio for each speech segment ─────────────────
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        smooth_window_size=args.smooth_window,
        max_buffer_sec=args.max_buffer_sec,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
    )

    # ── Step 3: save everything to disk ───────────────────────────────────
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # ── Step 4: write summary JSON files ──────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "all_speech_segments.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        slim = [
            {k: v for k, v in m.items() if k != "segment_probs"}
            for m in saved_metas
        ]
        json.dump(slim, fh, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]✓ Summary saved to:[/bold green] "
        f"[link=file://{summary_path.resolve()}]{summary_path}[/link]"
    )

    all_probs_path = output_dir / "speech_probs.json"
    with open(all_probs_path, "w", encoding="utf-8") as fh:
        json.dump(
            speech_probs if isinstance(speech_probs, list) else [],
            fh,
            indent=2,
        )
    console.print(
        f"[bold green]✓ Full probs saved to:[/bold green] "
        f"[link=file://{all_probs_path.resolve()}]{all_probs_path}[/link]"
    )

    console.rule("Done", style="green")
