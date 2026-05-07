# vad_peak_analyzer.py

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from config import HOP_LENGTH_MS, SAMPLE_RATE
from vad_firered2 import extract_speech_timestamps
from vad_valley_utils import ThresholdStrategy, auto_threshold
from scipy.signal import find_peaks

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


class VADSegment(TypedDict):
    frame_start: int  # Starting frame index (inclusive)
    frame_end: int  # Ending frame index (inclusive)
    frame_length: int  # Number of frames
    start_s: float  # Start time in seconds
    end_s: float  # End time in seconds
    duration_s: float  # Duration in seconds
    details: Dict[str, Any]  # Additional insights (peak/trough properties)


class ValleyInfo(TypedDict):
    frame_start: int
    frame_end: int
    frame_length: int
    start_s: float
    end_s: float
    duration_s: float


class ValleyTrough(TypedDict):
    frame: int
    time_s: float
    prob: float
    valley: ValleyInfo


def base_extract_valley_troughs(
    valleys: List[VADSegment], duration_s: float = 0.25
) -> List[ValleyTrough]:
    """
    Extracts the lowest-probability frames (troughs) from a list of VADSegment valleys,
    but only includes valleys that have exactly one trough and duration >= duration_s.
    Returns a list of ValleyTrough dicts with typed fields.

    Parameters
    ----------
    valleys: list of VADSegment dicts
    duration_s: minimum valley duration (in seconds) to include (default: 0.25)
    """
    filtered_valleys = [
        valley
        for valley in valleys
        if len(valley["details"].get("troughs", [])) == 1
        and valley["duration_s"] >= duration_s
    ]
    valley_troughs: List[ValleyTrough] = [
        ValleyTrough(
            frame=valley["details"]["min_prob_frame"],
            time_s=valley["details"]["min_prob_s"],
            prob=valley["details"]["min_probability"],
            valley=ValleyInfo(
                frame_start=valley["frame_start"],
                frame_end=valley["frame_end"],
                frame_length=valley["frame_length"],
                start_s=valley["start_s"],
                end_s=valley["end_s"],
                duration_s=valley["duration_s"],
            ),
        )
        for valley in filtered_valleys
    ]
    return valley_troughs


def extract_valley_troughs(
    probs: List[float],
    duration_s: float = 0.25,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = HOP_LENGTH_MS,
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
    Extract salient valley troughs (regions of strong silence) from a sequence of speech probability values.

    This function uses a peak/trough analyzer to identify single-trough valleys in speech
    probability arrays (such as those from VAD models) and returns detailed information
    about eligible valleys and their troughs.

    Args:
        probs: List of speech probability values for each frame (0~1), typically from a VAD or speech model.
        duration_s: Minimum duration (in seconds) for a valley to be considered valid.
        sample_rate: Audio sample rate (used for time/frame conversions).
        frame_shift_ms: Size of the step (in milliseconds) between consecutive frames.
        smoothing_window: Window size for smoothing VAD probabilities (set 0 to skip smoothing).
        trough_height: Maximum allowed probability value for a trough (silence valley),
            or None for default/auto.
        trough_prominence: Minimum prominence for a trough to be counted (minimum difference from surroundings).
        trough_distance: Minimum distance (in frames) between troughs.
        valley_threshold: Threshold for classifying a region as a valley.
        min_valley_duration_s: Minimum duration (s) for valleys (applied directly to raw valley candidates).
        min_valley_frames: Minimum number of frames for valleys (None disables; overrides min_valley_duration_s if set).
        frame_offset: Offset for global frame/time output (for combining/joining windowed analyses).
        min_trough_offset_s: Minimum time (in seconds) into a valley segment before the trough can be a candidate
            (e.g., to avoid selecting troughs that are too close to the start of a segment).

    Returns:
        List of ValleyTrough dictionaries, each containing detailed local/global trough and valley info.
        Each entry includes frame, global_frame, prob, time_s, global_time_s, and a nested ValleyInfo.
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

    filtered_valleys = [
        v
        for v in valleys
        if len(v.get("details", {}).get("troughs", [])) == 1
        and v["duration_s"] >= duration_s
    ]

    result: List[ValleyTrough] = []
    seconds_per_frame = frame_shift_ms / 1000.0

    for valley in filtered_valleys:
        local_trough_frame = valley["details"]["min_prob_frame"]

        # === LOCAL TIMES (relative to this segment only) ===
        local_trough_time_s = valley["details"]["min_prob_s"]

        # === NEW FILTER ===
        if local_trough_time_s < min_trough_offset_s:
            continue  # Skip troughs too close to the start of this segment

        # === GLOBAL TIMES ===
        global_trough_time_s = local_trough_time_s + (frame_offset * seconds_per_frame)

        valley_info: ValleyInfo = {
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "frame_length": valley["frame_length"],
            "start_s": valley["start_s"],  # local
            "end_s": valley["end_s"],  # local
            "duration_s": valley["duration_s"],
            "global_frame_start": valley["frame_start"] + frame_offset,
            "global_frame_end": valley["frame_end"] + frame_offset,
            "global_start_s": valley["start_s"] + (frame_offset * seconds_per_frame),
            "global_end_s": valley["end_s"] + (frame_offset * seconds_per_frame),
            "global_duration_s": valley["duration_s"],
        }

        result.append(
            {
                "frame": local_trough_frame,
                "global_frame": local_trough_frame + frame_offset,
                "prob": valley["details"]["min_probability"],
                "time_s": local_trough_time_s,
                "global_time_s": global_trough_time_s,
                "valley": valley_info,
            }
        )

    return result


def extract_valley_troughs_from_np_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_shift_ms: float = HOP_LENGTH_MS,
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


def save_segments_to_subdirs(
    segments: List["VADSegment"],
    category: str,
    probs: List[float],
    output_dir: "Path",
    audio_path: Optional[str],
    sample_rate: int,
    frame_shift_ms: float,
    pad_frames: int = 5,
) -> None:
    """
    For each segment in `segments`, create a numbered subdirectory under
    ``output_dir / category /`` and write three files into it:

        sound.wav  – the audio slice corresponding to the segment's time range
        meta.json  – the VADSegment dict serialised as JSON
        plot.png   – a focused VAD-probability plot for just this segment

    Parameters
    ----------
    segments      : list of VADSegment dicts (from extract_active_regions /
                    extract_valleys or similar).
    category      : subdirectory name, e.g. ``"active_regions"`` or
                    ``"valleys"``.
    probs         : full VAD probability list (used for the zoomed plot).
    output_dir    : root Path under which ``category/segment_NNN/`` dirs are
                    created.
    audio_path    : path to the original audio file, or ``None``.  When
                    ``None`` the function still writes ``meta.json`` and
                    ``plot.png`` but skips ``sound.wav``.
    sample_rate   : audio sample rate in Hz (needed to slice the waveform).
    frame_shift_ms: step size between VAD frames in milliseconds (used to
                    convert frame indices to time and sample positions).
    pad_frames    : number of extra frames to include around the segment in
                    the plot (purely cosmetic, default 5).

    """
    import json

    import soundfile as sf

    frame_duration_s = frame_shift_ms / 1000.0
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Load audio once (only when needed)
    audio_data: Optional[np.ndarray] = None
    file_sr: int = sample_rate
    if audio_path is not None:
        audio_data, file_sr = sf.read(audio_path, always_2d=False)

    x = np.array(probs, dtype=float)
    n_frames = len(x)

    for idx, seg in enumerate(segments):
        seg_dir = cat_dir / f"segment_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # ── meta.json ────────────────────────────────────────────────────────
        meta_path = seg_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(seg, fh, ensure_ascii=False, indent=2)

        # ── sound.wav ────────────────────────────────────────────────────────
        if audio_data is not None:
            start_sample = int(seg["start_s"] * file_sr)
            end_sample = int(seg["end_s"] * file_sr)
            # Clamp to valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            slice_audio = audio_data[start_sample:end_sample]
            wav_path = seg_dir / "sound.wav"
            sf.write(str(wav_path), slice_audio, file_sr)

        # ── plot.png ─────────────────────────────────────────────────────────
        f_start = max(0, seg["frame_start"] - pad_frames)
        f_end = min(n_frames, seg["frame_end"] + pad_frames + 1)
        frames = np.arange(f_start, f_end)
        zoomed = x[f_start:f_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Highlight the actual segment
        ax.axvspan(
            seg["frame_start"],
            seg["frame_end"] + 1,
            alpha=0.20,
            color="green" if category == "active_regions" else "red",
            label=category.replace("_", " ").title(),
        )

        # Overlay attached troughs (if any)
        seg_troughs = seg.get("details", {}).get("troughs", [])
        visible_troughs = [
            t for t in seg_troughs if f_start <= t["frame_start"] < f_end
        ]
        if visible_troughs:
            t_frames = [t["frame_start"] for t in visible_troughs]
            t_probs  = [x[t["frame_start"]] for t in visible_troughs]
            ax.plot(
                t_frames,
                t_probs,
                "rv",          # red downward-pointing triangle
                markersize=9,
                label="Troughs",
                zorder=5,
            )
            for tf, tp in zip(t_frames, t_probs):
                ax.annotate(
                    f"{tp:.2f}",
                    xy=(tf, tp),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="red",
                )

        # Annotate start / end times
        ax.set_title(
            f"{category} · segment {idx:03d}  "
            f"[{seg['start_s']:.3f}s – {seg['end_s']:.3f}s]",
            fontsize=12,
        )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        plot_path = seg_dir / "plot.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"[save_segments_to_subdirs] Wrote {len(segments)} '{category}' segments → {cat_dir}"
    )


class VADPeakAnalyzer:
    """
    Analyzes peaks (local maxima) and troughs (local minima) in VAD speech probabilities.
    Enhanced with optional debug logging for diagnostics.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_shift_ms: float = HOP_LENGTH_MS,
        debug: bool = False,
    ):
        """
        Args:
            sample_rate: Audio sample rate in Hz.
            frame_shift_ms: Frame shift (hop length) in milliseconds between consecutive VAD frames.
            debug: If True, enable debug logging.
        """
        self.sample_rate = sample_rate
        self.frame_shift_ms = frame_shift_ms
        self.frame_duration_s = frame_shift_ms / 1000.0
        self.auto_threshold_strategy: ThresholdStrategy = ThresholdStrategy.OTSU
        """Strategy used when valley_threshold or trough_height is None."""
        self.hop_length = int(sample_rate * self.frame_duration_s)  # samples per frame
        self.debug = debug

        if debug:
            logging.basicConfig(
                level=logging.DEBUG, format="%(levelname)s - %(message)s"
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def _log_debug(self, msg: str, **kwargs):
        if self.debug:
            self.logger.debug(msg, extra=kwargs)

    def _compute_times(self, frame_idx: int) -> Tuple[float, float]:
        """Convert frame index to start/end time in seconds."""
        start_s = frame_idx * self.frame_duration_s
        end_s = (frame_idx + 1) * self.frame_duration_s
        return start_s, end_s

    def extract_peaks(
        self,
        probs: List[float],
        height: Optional[float] = None,
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[VADSegment]:
        """
        Extract peaks (local maxima) from VAD probabilities.

        Recommended params for speech VAD (tune based on your model):
        - height: min probability (e.g. 0.6)
        - distance: min frames between peaks (e.g. 5-20)
        - prominence: how much it stands out (e.g. 0.1-0.3)
        """
        if not probs:
            return []

        x = np.array(probs, dtype=float)
        self._log_debug(
            f"extract_peaks called with height={height}, distance={distance}, prominence={prominence}"
        )
        self._log_debug(f"Input probs: {[round(p, 4) for p in probs]}")

        peaks_idx, properties = find_peaks(
            x,
            height=height,
            distance=distance,
            prominence=prominence,
            # Always compute widths; use `width` only as a minimum filter
            width=width if width is not None else 0,
            **kwargs,
        )

        self._log_debug(f"Raw peaks found at indices: {peaks_idx.tolist()}")
        if len(peaks_idx) > 0:
            self._log_debug(
                f"Peak probabilities: {[round(x[i], 4) for i in peaks_idx]}"
            )
            if "prominences" in properties:
                self._log_debug(
                    f"Prominences: {[round(p, 4) for p in properties['prominences']]}"
                )
            if "left_bases" in properties and "right_bases" in properties:
                for i, idx in enumerate(peaks_idx):
                    left = properties["left_bases"][i]
                    right = properties["right_bases"][i]
                    self._log_debug(
                        f"Peak at {idx}: left_base={left}, right_base={right}, base_range=[{left}:{right + 1}]"
                    )

        segments: List[VADSegment] = []
        for i, peak in enumerate(peaks_idx):
            frame_start = int(peak)
            frame_end = int(peak)

            start_s, end_s = self._compute_times(frame_start)
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

        self._log_debug(f"Returning {len(segments)} peak segments")
        return segments

    def extract_troughs(
        self,
        probs: List[float],
        height: Optional[float] = None,  # None → auto-compute via auto_threshold()
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[VADSegment]:
        """
        Extract troughs (local minima) by finding peaks on the negated signal.
        """
        if not probs:
            return []

        # ── Auto-compute trough_height when not supplied ──────────────────
        resolved_height = height
        if resolved_height is None:
            resolved_height = auto_threshold(
                probs, strategy=self.auto_threshold_strategy
            )
            self._log_debug(
                f"extract_troughs: auto-computed height={resolved_height:.4f} "
                f"via {self.auto_threshold_strategy.value}"
            )
        # ─────────────────────────────────────────────────────────────────

        x = np.array(probs, dtype=float)
        self._log_debug(
            f"extract_troughs called with height={height}, distance={distance}, prominence={prominence}"
        )
        self._log_debug(f"Input probs: {[round(p, 4) for p in probs]}")

        # Negate to turn minima into maxima
        troughs_idx, properties = find_peaks(
            -x,
            height=-resolved_height,  # always set now — never None
            distance=distance,
            prominence=prominence,
            width=width if width is not None else 0,
            **kwargs,
        )

        self._log_debug(f"Raw troughs found at indices: {troughs_idx.tolist()}")
        if len(troughs_idx) > 0:
            self._log_debug(
                f"Trough probabilities: {[round(x[i], 4) for i in troughs_idx]}"
            )
            if "prominences" in properties:
                self._log_debug(
                    f"Prominences: {[round(p, 4) for p in properties['prominences']]}"
                )
            # No left/right base for troughs unless needed

        segments: List[VADSegment] = []
        for i, trough in enumerate(troughs_idx):
            frame_start = int(trough)
            frame_end = int(trough)

            start_s, end_s = self._compute_times(frame_start)
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

        self._log_debug(f"Returning {len(segments)} trough segments")
        return segments

    def extract_active_regions(
        self,
        probs: List[float],
        threshold: float = 0.3,
        min_duration_s: float = 0.0,
        min_duration_frames: Optional[int] = None,
        troughs: Optional[List[VADSegment]] = None,
    ) -> List[VADSegment]:
        """
        Extract contiguous active (speech) regions where probability >= threshold.

        An "active region" is a run of consecutive frames all at or above the
        threshold — think of them as the speech bursts between silences.

        Args:
            probs: VAD probability list.
            threshold: Minimum probability to count as active (default 0.3).
            min_duration_s: Minimum duration in seconds for an active region to be kept.
            min_duration_frames: Alternative minimum frame count (overrides min_duration_s if provided).
            troughs: Optional pre-extracted trough VADSegments (from
                     extract_troughs()).  Each trough whose frame index falls
                     within an active region's [frame_start, frame_end] boundary
                     is attached to that region's details["troughs"] list.

        Returns:
            List of VADSegment dicts, one per contiguous active region.
        """
        if not probs:
            return []

        x = np.array(probs, dtype=float)
        active = x >= threshold  # Boolean mask: True where speech is active

        segments: List[VADSegment] = []
        in_region = False
        region_start = 0

        for i, is_active in enumerate(active):
            if is_active and not in_region:
                # Rising edge — start of a new active region
                in_region = True
                region_start = i
            elif not is_active and in_region:
                # Falling edge — end of the active region (exclusive)
                self._append_active_segment(segments, x, region_start, i, threshold)
                in_region = False

        # Handle region that runs to the very end of the signal
        if in_region:
            self._append_active_segment(segments, x, region_start, len(x), threshold)

        # ── Attach troughs that fall within each active region's frame boundaries ──
        if troughs and segments:
            for segment in segments:
                r_start = segment["frame_start"]
                r_end   = segment["frame_end"]
                contained = [
                    t for t in troughs if r_start <= t["frame_start"] <= r_end
                ]
                segment["details"]["troughs"] = contained
                if contained:
                    self._log_debug(
                        f"extract_active_regions: region [{r_start}, {r_end}] "
                        f"contains {len(contained)} trough(s) at frames "
                        f"{[t['frame_start'] for t in contained]}"
                    )
        # ──────────────────────────────────────────────────────────────────────────

        self._log_debug(f"Returning {len(segments)} active region segments")
        return segments
   

    def _append_active_segment(
        self,
        segments: List[VADSegment],
        x: np.ndarray,
        start: int,
        end: int,  # exclusive index
        threshold: float,
    ) -> None:
        """Helper: build and append one active-region VADSegment."""
        start_s, _ = self._compute_times(start)
        _, end_s = self._compute_times(end - 1)  # last frame's end time
        duration_s = end_s - start_s
        region_probs = x[start:end].tolist()
        segments.append(
            {
                "frame_start": start,
                "frame_end": end - 1,
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

    def merge_active_regions_by_min_silence(
        self,
        active_regions: List[VADSegment],
        min_silence_duration_sec: float = 0.25,
    ) -> List[VADSegment]:
        """
        Merge adjacent active regions if the silence (valley) between them is
        shorter than `min_silence_duration_sec`.

        This is the classic VAD post-processing logic used in many toolkits
        (Silero, NeMo, WhisperX, etc.).
        """
        if len(active_regions) <= 1:
            return active_regions

        merged: List[VADSegment] = []
        current = active_regions[0].copy()

        for next_region in active_regions[1:]:
            # Calculate silence duration between the two active regions
            silence_start_s = current["end_s"]
            silence_end_s = next_region["start_s"]
            silence_duration = silence_end_s - silence_start_s

            if silence_duration < min_silence_duration_sec:
                # Silence is too short → merge the two speech regions
                self._log_debug(
                    f"Merging regions due to short silence: "
                    f"{current['start_s']:.3f}s–{current['end_s']:.3f}s + "
                    f"{next_region['start_s']:.3f}s–{next_region['end_s']:.3f}s "
                    f"(silence = {silence_duration:.3f}s < {min_silence_duration_sec:.3f}s)"
                )
                current = self._merge_two_regions(current, next_region)
            else:
                # Real silence gap → keep current and start new region
                merged.append(current)
                current = next_region.copy()

        merged.append(current)
        return merged

    def extract_valleys(
        self,
        probs: List[float],
        threshold: Optional[float] = None,  # None → auto-compute via auto_threshold()
        min_duration_s: float = 0.0,
        min_duration_frames: Optional[int] = None,
        troughs: Optional[List[VADSegment]] = None,
    ) -> List[VADSegment]:
        """
        Extract contiguous valley (silence) regions where probability < threshold.

        A "valley" is a run of consecutive frames all strictly below the
        threshold — the silence stretches between speech bursts.  This is
        the region-based counterpart to extract_troughs(), which finds only
        the single lowest frame inside each dip.

        Relationship to other methods
        ------------------------------
        extract_troughs()        → single-frame local minimum inside a dip
        extract_valleys()        → the whole contiguous low-probability region
        extract_active_regions() → the whole contiguous high-probability region

        Args:
            probs: VAD probability list.
            threshold: Frames strictly below this value are considered silent
                       (default 0.3).  Frames AT the threshold are NOT included
                       (use > instead of >= to match "below threshold" intent).
            min_duration_s: Minimum duration in seconds for a valley to be kept.
            min_duration_frames: Alternative minimum frame count (overrides min_duration_s if provided).
            troughs: Optional pre-extracted trough VADSegments (from
                     extract_troughs()).  Each trough whose frame index falls
                     within a valley's [frame_start, frame_end] boundary is
                     attached to that valley's details["troughs"] list.  Valley
                     boundaries are never modified.

        Returns:
            List of VADSegment dicts, one per contiguous valley region.
        """
        if not probs:
            return []

        # ── Auto-compute valley threshold when not supplied ───────────────
        resolved_threshold = threshold
        if resolved_threshold is None:
            resolved_threshold = auto_threshold(
                probs, strategy=self.auto_threshold_strategy
            )
            self._log_debug(
                f"extract_valleys: auto-computed threshold={resolved_threshold:.4f} "
                f"via {self.auto_threshold_strategy.value}"
            )
        # ─────────────────────────────────────────────────────────────────

        x = np.array(probs, dtype=float)
        silent = x < resolved_threshold  # Boolean mask: True where frame is silent

        segments: List[VADSegment] = []
        in_valley = False
        valley_start = 0

        for i, is_silent in enumerate(silent):
            if is_silent and not in_valley:
                # Falling edge — entering a silent stretch
                in_valley = True
                valley_start = i
            elif not is_silent and in_valley:
                # Rising edge — leaving the silent stretch
                self._append_valley_segment(
                    segments, x, valley_start, i, resolved_threshold
                )
                in_valley = False

        # Handle valley that runs to the very end of the signal
        if in_valley:
            self._append_valley_segment(
                segments, x, valley_start, len(x), resolved_threshold
            )

        # ── Attach troughs that fall within each valley's frame boundaries ───
        if troughs and segments:
            for segment in segments:
                v_start = segment["frame_start"]
                v_end = segment["frame_end"]
                contained = [t for t in troughs if v_start <= t["frame_start"] <= v_end]
                segment["details"]["troughs"] = contained
                if contained:
                    self._log_debug(
                        f"extract_valleys: valley [{v_start}, {v_end}] "
                        f"contains {len(contained)} trough(s) at frames "
                        f"{[t['frame_start'] for t in contained]}"
                    )
        # ─────────────────────────────────────────────────────────────────────

        self._log_debug(f"Returning {len(segments)} valley segments")
        return segments

    def _append_valley_segment(
        self,
        segments: List[VADSegment],
        x: np.ndarray,
        start: int,
        end: int,  # exclusive index
        threshold: float,
    ) -> None:
        """Helper: build and append one valley VADSegment."""
        start_s, _ = self._compute_times(start)
        _, end_s = self._compute_times(end - 1)  # last frame's end time
        duration_s = end_s - start_s
        region_probs = x[start:end].tolist()
        min_prob_frame = int(start + np.argmin(x[start:end]))
        min_prob_s, _ = self._compute_times(min_prob_frame)

        frame_length = end - start

        segments.append(
            {
                "frame_start": start,
                "frame_end": end - 1,
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
                    # "region_probs": region_probs,
                },
            }
        )

    def merge_active_regions_across_shallow_valleys(
        self,
        active_regions: List[VADSegment],
        probs: List[float],
        min_valley_threshold: Optional[float] = None,
        min_valley_frames: Optional[int] = None,
    ) -> List[VADSegment]:
        """
        Merge adjacent active regions if the valley (gap) between them does not
        pass the minimum valley (silence) threshold — i.e. the dip is not deep enough.

        This implements "active region logic to merge the next region when the
        valley in between doesn't pass min valley threshold".

        Args:
            active_regions: Output from extract_active_regions()
            probs: Original VAD probability list
            min_valley_threshold: If the *minimum* probability in the gap is
                *above* this value, the valley is considered too shallow → merge.
                If None, auto-computed via auto_threshold() using the configured
                strategy (default: Otsu). Defaults to None.
            min_valley_frames: Optional minimum frame length of a valley to be
                considered for merging logic (short gaps are always merged).

        Returns:
            New list of merged VADSegment objects.
        """
        if len(active_regions) <= 1:
            return active_regions

        # ── Auto-compute min_valley_threshold when not supplied ───────────────
        resolved_threshold = min_valley_threshold
        if resolved_threshold is None:
            resolved_threshold = auto_threshold(
                probs, strategy=self.auto_threshold_strategy
            )
            self._log_debug(
                f"merge_active_regions_across_shallow_valleys: auto-computed "
                f"min_valley_threshold={resolved_threshold:.4f} "
                f"via {self.auto_threshold_strategy.value}"
            )
        # ─────────────────────────────────────────────────────────────────────

        x = np.array(probs, dtype=float)
        merged: List[VADSegment] = []
        current = active_regions[0].copy()

        for next_region in active_regions[1:]:
            # Define the valley between current and next_region
            valley_start = current["frame_end"] + 1
            valley_end = next_region["frame_start"] - 1  # inclusive

            if valley_start > valley_end:
                # Overlapping or adjacent regions → merge
                current = self._merge_two_regions(current, next_region)
                continue

            valley_length = valley_end - valley_start + 1

            # If valley is too short, always merge (optional safety)
            if min_valley_frames is not None and valley_length < min_valley_frames:
                current = self._merge_two_regions(current, next_region)
                continue

            # Compute minimum probability in the valley
            valley_probs = x[valley_start : valley_end + 1]
            valley_min = float(np.min(valley_probs))

            if valley_min > resolved_threshold:
                # Valley is too shallow → merge the two active regions
                self._log_debug(
                    f"Merging regions {current['frame_start']}-{current['frame_end']} "
                    f"and {next_region['frame_start']}-{next_region['frame_end']} "
                    f"(valley min={valley_min:.4f} > threshold={resolved_threshold:.4f})"
                )
                current = self._merge_two_regions(current, next_region)
            else:
                # Real silence valley → keep current and start new one
                merged.append(current)
                current = next_region.copy()

        merged.append(current)
        return merged

    def _merge_two_regions(self, reg1: VADSegment, reg2: VADSegment) -> VADSegment:
        """Helper to merge two adjacent VADSegment dicts."""
        merged = reg1.copy()
        merged["frame_end"] = reg2["frame_end"]
        merged["frame_length"] = merged["frame_end"] - merged["frame_start"] + 1
        merged["end_s"] = reg2["end_s"]
        merged["duration_s"] = round(merged["end_s"] - merged["start_s"], 4)

        # Update details
        merged["details"]["max_probability"] = max(
            reg1["details"]["max_probability"], reg2["details"]["max_probability"]
        )
        merged["details"]["mean_probability"] = (
            reg1["details"]["mean_probability"] * reg1["frame_length"]
            + reg2["details"]["mean_probability"] * reg2["frame_length"]
        ) / merged["frame_length"]
        merged["details"]["frame_count"] = merged["frame_length"]
        # Optionally merge region_probs if needed
        return merged

    def filter_short_segments(
        self,
        segments: List[VADSegment],
        min_duration_s: float = 0.0,
        min_duration_frames: Optional[int] = None,
    ) -> List[VADSegment]:
        """Filter out segments shorter than the specified minimum duration."""
        if not segments:
            return segments

        if min_duration_frames is not None:
            return [s for s in segments if s["frame_length"] >= min_duration_frames]
        else:
            return [s for s in segments if s["duration_s"] >= min_duration_s]

    def save_plot(
        self,
        probs: List[float],
        peaks: List[VADSegment],
        troughs: List[VADSegment],
        active_regions: Optional[List[VADSegment]] = None,
        valleys: Optional[List[VADSegment]] = None,
        output_path: str = "vad_peaks_troughs.png",
        title: str = "VAD Probability - Peaks and Troughs",
    ) -> None:
        """
        Save a visualization plot highlighting peaks and troughs.

        Background gradient highlights:
          - Green shading  → active/speech regions (prob >= active threshold)
          - Red shading    → valley/silence regions (prob < valley_threshold)
          - No shading     → transition zones

        Args:
            probs: Original list of VAD probabilities.
            peaks: List of peak segments returned by extract_peaks().
            troughs: List of trough segments returned by extract_troughs().
            active_regions: Optional list from extract_active_regions().
            valleys: Optional list from extract_valleys().
            output_path: Path where the plot image will be saved.
            title: Title of the plot.
        """
        if not probs:
            self._log_debug("Cannot plot: empty probability list")
            return

        x = np.array(probs, dtype=float)
        frames = np.arange(len(x))

        fig, ax = plt.subplots(figsize=(14, 7))

        # ── Gradient background: active (green) regions ──────────────────────
        if active_regions:
            for region in active_regions:
                ax.axvspan(
                    region["frame_start"],
                    region["frame_end"] + 1,
                    alpha=0.15,
                    color="green",
                    label="_nolegend_",
                )

        # ── Gradient background: valley (red) regions ─────────────────────────
        if valleys:
            for v in valleys:
                ax.axvspan(
                    v["frame_start"],
                    v["frame_end"] + 1,
                    alpha=0.12,
                    color="red",
                    label="_nolegend_",
                )

        ax.plot(frames, x, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Threshold reference lines
        if valleys:
            valley_threshold = valleys[0]["details"]["threshold"]
            ax.axhline(
                y=valley_threshold,
                color="red",
                linestyle="--",
                alpha=0.4,
                linewidth=1,
                label=f"Valley threshold ({valley_threshold})",
            )
        if active_regions:
            active_thresh = (
                active_regions[0]["details"]["threshold"] if active_regions else 0.3
            )
            ax.axhline(
                y=active_thresh,
                color="green",
                linestyle="--",
                alpha=0.4,
                linewidth=1,
                label=f"Active threshold ({active_thresh})",
            )

        if peaks:
            peak_indices = [p["frame_start"] for p in peaks]
            peak_probs = [p["details"]["peak_probability"] for p in peaks]
            ax.plot(
                peak_indices, peak_probs, "go", markersize=10, label="Peaks (Speech)"
            )
            for idx, prob in zip(peak_indices, peak_probs):
                ax.annotate(
                    f"{prob:.2f}",
                    xy=(idx, prob),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="green",
                )

        if troughs:
            trough_indices = [t["frame_start"] for t in troughs]
            trough_probs = [t["details"]["trough_probability"] for t in troughs]
            ax.plot(
                trough_indices,
                trough_probs,
                "ro",
                markersize=10,
                label="Troughs (Silence)",
            )
            for idx, prob in zip(trough_indices, trough_probs):
                ax.annotate(
                    f"{prob:.2f}",
                    xy=(idx, prob),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="red",
                )

        # Custom legend patches for shaded regions
        legend_handles = [
            mpatches.Patch(color="green", alpha=0.4, label="Active / Speech region"),
            mpatches.Patch(color="red", alpha=0.4, label="Valley / Silence region"),
        ]

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(
            handles=ax.get_legend_handles_labels()[0] + legend_handles,
            labels=ax.get_legend_handles_labels()[1]
            + ["Active / Speech region", "Valley / Silence region"],
            fontsize=11,
            loc="upper right",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self._log_debug(f"Plot saved successfully to: {output_path}")
        print(f"Plot saved to: {output_path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze VAD speech/voice probabilities and find peaks/troughs"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help=(
            "Path to either:\n"
            "- JSON file with speech probabilities\n"
            "- Audio file (wav/mp3/flac/etc.) to run VAD on\n"
            "If not provided, uses a sample sequence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str((Path(__file__).parent / "generated" / Path(__file__).stem)),
        help="Output directory for generated files (default: ./generated/<script name>)",
    )
    parser.add_argument(
        "--sample-rate",
        "-sr",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--frame-shift-ms",
        "-fsm",
        type=float,
        default=HOP_LENGTH_MS,
        help="Frame shift (hop length) in ms between analysis frames (default: HOP_LENGTH_MS for FireRedVAD)",
    )
    parser.add_argument(
        "--peak-height",
        "-ph",
        type=float,
        default=0.7,
        help="Minimum height for a peak (default: 0.7)",
    )
    parser.add_argument(
        "--peak-prominence",
        "-pp",
        type=float,
        default=0.1,
        help="Minimum prominence for a peak (default: 0.1)",
    )
    parser.add_argument(
        "--peak-distance",
        "-pd",
        type=int,
        default=3,
        help="Minimum distance between peaks in frames (default: 3)",
    )
    parser.add_argument(
        "--trough-height",
        "-th",
        type=float,
        default=None,
        help="Maximum speech probability for a trough (default: None; auto-computed if not set)",
    )
    parser.add_argument(
        "--trough-prominence",
        "-tp",
        type=float,
        default=0.15,
        help="Minimum prominence for a trough (default: 0.15).",
    )
    parser.add_argument(
        "--trough-distance",
        "-td",
        type=int,
        default=5,
        help="Minimum distance between troughs in frames (default: 5).",
    )
    parser.add_argument(
        "--active-threshold",
        "-at",
        type=float,
        default=0.3,
        help="Probability threshold for active/speech regions (default: 0.3)",
    )
    parser.add_argument(
        "--valley-threshold",
        "-vt",
        type=float,
        default=None,
        help="Probability threshold below which regions are valleys (default: None; auto-computed if not set)",
    )
    parser.add_argument(
        "--min-active-duration",
        "-mad",
        type=float,
        default=0.25,
        help="Minimum active speech duration in seconds (default: 0.25s)",
    )
    parser.add_argument(
        "--min-active-frames",
        "-maf",
        type=int,
        default=None,
        help="Minimum active region length in frames (overrides --min-active-duration if set)",
    )
    parser.add_argument(
        "--min-silence-duration",
        "-msd",
        type=float,
        default=0.5,
        help="Minimum silence duration in seconds for merging active regions (default: 0.5s)",
    )
    parser.add_argument(
        "--min-valley-duration",
        "-mvd",
        type=float,
        default=0.25,
        help="Minimum valley/silence duration in seconds (default: 0.25s)",
    )
    parser.add_argument(
        "--min-valley-frames",
        "-mvf",
        type=int,
        default=None,
        help="Minimum valley length in frames (overrides --min-valley-duration if set)",
    )
    parser.add_argument(
        "--smoothing-window",
        "-sw",
        type=int,
        default=0,
        help="Smoothing window size for VAD probabilities (default: 0)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import json
    import shutil

    args = get_args()

    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file is not None:
        input_path = Path(args.input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        suffix = input_path.suffix.lower()

        # ── Audio input ───────────────────────────────────────────────
        if suffix in AUDIO_EXTENSIONS:
            _, probs = extract_speech_timestamps(
                audio=str(input_path),
                threshold=0.3,
                min_speech_duration_sec=0.25,
                min_silence_duration_sec=0.25,
                # threshold=args.active_threshold,
                # min_speech_duration_sec=args.min_active_duration,
                # min_silence_duration_sec=args.min_silence_duration,
                with_scores=True,
            )

        # ── JSON input ────────────────────────────────────────────────
        elif suffix == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                probs = json.load(f)
                if not isinstance(probs, list):
                    raise ValueError("JSON file must contain a list/array of floats.")
                probs = [float(p) for p in probs]

        # ── Unsupported input ─────────────────────────────────────────
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Expected audio file or JSON."
            )
    else:
        # Default sample sequence
        probs = [0.1, 0.15, 0.8, 0.92, 0.85, 0.3, 0.12, 0.05, 0.88, 0.95, 0.7, 0.2]

    # === NEW: Smooth probabilities before analysis ===
    if args.smoothing_window:
        print(f"Original probs length: {len(probs)}")
        probs_smoothed = smooth_vad_probs(
            probs,
            window=args.smoothing_window,
        )
        print("Applied Gaussian smoothing (sigma=1.0)")
    else:
        probs_smoothed = probs

    analyzer = VADPeakAnalyzer(
        sample_rate=args.sample_rate, frame_shift_ms=args.frame_shift_ms
    )

    peaks = analyzer.extract_peaks(
        probs_smoothed,
        height=args.peak_height,
        prominence=args.peak_prominence,
        distance=args.peak_distance,
    )

    troughs = analyzer.extract_troughs(
        probs_smoothed,
        height=args.trough_height,
        prominence=args.trough_prominence,
        distance=args.trough_distance,
    )

    active_regions = analyzer.extract_active_regions(
        probs_smoothed,
        threshold=args.active_threshold,
        # min_duration_s=args.min_active_duration,
        # min_duration_frames=args.min_active_frames,
        troughs=troughs,
    )

    # Depth-based merging
    active_regions = analyzer.merge_active_regions_across_shallow_valleys(
        active_regions,
        probs_smoothed,
        min_valley_threshold=args.valley_threshold,
        # min_valley_threshold=args.valley_threshold
        # * 0.8,  # example: slightly below valley threshold
        min_valley_frames=2,
    )

    # Duration-based merging (most common in real VAD pipelines)
    active_regions = analyzer.merge_active_regions_by_min_silence(
        active_regions,
        min_silence_duration_sec=args.min_silence_duration,
    )

    # Filter by minimum duration
    active_regions = analyzer.filter_short_segments(
        active_regions,
        min_duration_s=args.min_active_duration,
        min_duration_frames=args.min_active_frames,
    )

    valleys = analyzer.extract_valleys(
        probs_smoothed,
        threshold=args.valley_threshold,
        # min_duration_s=args.min_valley_duration,
        # min_duration_frames=args.min_valley_frames,
        troughs=troughs,
    )

    # Filter by minimum duration
    valleys = analyzer.filter_short_segments(
        valleys,
        min_duration_s=args.min_valley_duration,
        min_duration_frames=args.min_valley_frames,
    )

    print("Peaks:", peaks)
    print("Troughs:", troughs)
    print("Active regions:", active_regions)
    print("Valleys:", valleys)

    # ── NEW: per-segment subdirectories (only when audio was provided) ──────
    if (
        args.input_file is not None
        and Path(args.input_file).suffix.lower() in AUDIO_EXTENSIONS
    ):
        save_segments_to_subdirs(
            segments=active_regions,
            category="active_regions",
            probs=probs_smoothed,
            output_dir=output_dir,
            audio_path=args.input_file,
            sample_rate=args.sample_rate,
            frame_shift_ms=args.frame_shift_ms,
        )
        save_segments_to_subdirs(
            segments=valleys,
            category="valleys",
            probs=probs_smoothed,
            output_dir=output_dir,
            audio_path=args.input_file,
            sample_rate=args.sample_rate,
            frame_shift_ms=args.frame_shift_ms,
        )

    analyzer.save_plot(
        probs,
        peaks,
        troughs,
        active_regions=active_regions,
        valleys=valleys,
        output_path=str(output_dir / "vad_analysis_plot.png"),
    )

    if args.smoothing_window:
        analyzer.save_plot(
            probs_smoothed,
            peaks,
            troughs,
            active_regions=active_regions,
            valleys=valleys,
            output_path=str(output_dir / "vad_analysis_plot_smoothed.png"),
        )

    peaks_path = output_dir / "peaks.json"
    with open(peaks_path, "w", encoding="utf-8") as f:
        json.dump(peaks, f, ensure_ascii=False, indent=2)
    print(f"Peaks saved to: {peaks_path.resolve()}")

    troughs_path = output_dir / "troughs.json"
    with open(troughs_path, "w", encoding="utf-8") as f:
        json.dump(troughs, f, ensure_ascii=False, indent=2)
    print(f"Troughs saved to: {troughs_path.resolve()}")

    active_path = output_dir / "active_regions.json"
    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(active_regions, f, ensure_ascii=False, indent=2)
    print(f"Active regions saved to: {active_path.resolve()}")

    valleys_path = output_dir / "valleys.json"
    with open(valleys_path, "w", encoding="utf-8") as f:
        json.dump(valleys, f, ensure_ascii=False, indent=2)
    print(f"Valleys saved to: {valleys_path.resolve()}")

    base_valley_troughs = base_extract_valley_troughs(valleys)
    base_valley_troughs_path = output_dir / "base_valley_troughs.json"
    with open(base_valley_troughs_path, "w", encoding="utf-8") as f:
        json.dump(base_valley_troughs, f, ensure_ascii=False, indent=2)
    print(f"Valley troughs saved to: {base_valley_troughs_path.resolve()}")

    valley_troughs = extract_valley_troughs(
        probs=probs,
        duration_s=args.min_valley_duration,
        sample_rate=args.sample_rate,  # ← add this
        frame_shift_ms=args.frame_shift_ms,  # ← add this (was defaulting to 25ms!)
        frame_offset=args.frame_offset if hasattr(args, "frame_offset") else 0,
        smoothing_window=args.smoothing_window,
    )
    valley_troughs_path = output_dir / "valley_troughs.json"
    with open(valley_troughs_path, "w", encoding="utf-8") as f:
        json.dump(valley_troughs, f, ensure_ascii=False, indent=2)
    print(f"Valley troughs saved to: {valley_troughs_path.resolve()}")
