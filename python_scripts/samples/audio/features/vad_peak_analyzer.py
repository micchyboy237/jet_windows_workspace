# vad_peak_analyzer.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from config import FRAME_SHIFT_MS, SAMPLE_RATE
from vad_firered2 import extract_speech_timestamps
from vad_types import VADSegment, ValleyTrough
from vad_valley_utils import ThresholdStrategy, auto_threshold
from scipy.signal import find_peaks
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


class VADPeakAnalyzer:
    """
    Analyzes peaks (local maxima) and troughs (local minima) in VAD speech probabilities.
    Enhanced with optional debug logging for diagnostics.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_shift_ms: float = FRAME_SHIFT_MS,
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
        _, end_s = self._compute_times(end + 1)  # last frame's end time
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
        _, end_s = self._compute_times(end + 1)  # last frame's end time
        duration_s = end_s - start_s
        region_probs = x[start:end].tolist()
        min_prob_frame = int(start + np.argmin(x[start:end]))
        min_prob_s, _ = self._compute_times(min_prob_frame)

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
        """
        if not probs:
            self._log_debug("Cannot plot: empty probability list")
            return

        x = np.array(probs, dtype=float)
        frames = np.arange(len(x))
        fig, ax = plt.subplots(figsize=(14, 7))

        # Background shading
        if active_regions:
            for region in active_regions:
                ax.axvspan(
                    region["frame_start"],
                    region["frame_end"] + 1,
                    alpha=0.15,
                    color="green",
                )
        if valleys:
            for v in valleys:
                ax.axvspan(
                    v["frame_start"],
                    v["frame_end"] + 1,
                    alpha=0.12,
                    color="red",
                )

        ax.plot(frames, x, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Threshold lines
        if valleys:
            valley_threshold = valleys[0]["details"]["threshold"]
            ax.axhline(
                y=valley_threshold,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Valley threshold ({valley_threshold:.3f})",
            )
        if active_regions:
            active_thresh = active_regions[0]["details"].get("threshold", 0.3)
            ax.axhline(
                y=active_thresh,
                color="green",
                linestyle="--",
                alpha=0.5,
                label=f"Active threshold ({active_thresh:.3f})",
            )

        # Peaks
        if peaks:
            peak_indices = [p["frame_start"] for p in peaks]
            peak_probs = [p["details"]["peak_probability"] for p in peaks]
            ax.plot(peak_indices, peak_probs, "go", markersize=10, label="Peaks")
            for idx, prob in zip(peak_indices, peak_probs):
                ax.annotate(
                    f"{prob:.2f}",
                    xy=(idx, prob),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    color="green",
                    fontsize=9,
                )

        # Troughs
        if troughs:
            trough_indices = [t["frame_start"] for t in troughs]
            trough_probs = [t["details"]["trough_probability"] for t in troughs]
            ax.plot(trough_indices, trough_probs, "ro", markersize=10, label="Troughs")
            for idx, prob in zip(trough_indices, trough_probs):
                ax.annotate(
                    f"{prob:.2f}",
                    xy=(idx, prob),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    color="red",
                    fontsize=9,
                )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="upper right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Rich output with clickable file link
        console.print(
            f"📊 [bold green]Plot saved to:[/bold green] "
            f"[link=file:///{Path(output_path).resolve()}]{Path(output_path).resolve()}[/link]",
            style="green",
        )


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
    """
    import json

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

    # ── Rich summary with clickable link ─────────────────────────────────────
    icon = "🟢" if category == "active_regions" else "🔴"
    console.print(
        f"{icon} [bold]{len(segments)} {category.replace('_', ' ').title()}[/bold] "
        f"segments saved to: "
        f"[link=file:///{cat_dir.resolve()}]{cat_dir.resolve()}[/link]",
        style="green",
    )


def save_valley_trough_segments(
    valley_troughs: List["ValleyTrough"],
    probs: List[float],
    output_dir: "Path",
    audio_path: Optional[str],
    sample_rate: int,
    frame_shift_ms: float,
) -> None:
    """
    For each ValleyTrough, create a segment that spans from the start of the
    audio (t=0) to the trough's position (global_time_s).  Each segment is
    saved as a numbered subdirectory under ``output_dir / "valley_troughs" /``
    containing three files:

        sound.wav  – audio from sample 0 to the trough sample
        meta.json  – the ValleyTrough dict plus derived start_s/end_s/duration_s
        plot.png   – VAD probability from frame 0 to the trough frame
    """
    import json

    cat_dir = output_dir / "valley_troughs"
    cat_dir.mkdir(parents=True, exist_ok=True)

    audio_data: Optional[np.ndarray] = None
    file_sr: int = sample_rate
    if audio_path is not None:
        audio_data, file_sr = sf.read(audio_path, always_2d=False)

    x = np.array(probs, dtype=float)
    n_frames = len(x)

    for idx, vt in enumerate(valley_troughs):
        seg_dir = cat_dir / f"segment_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Derived times: segment always starts at 0, ends at the trough
        start_s: float = 0.0
        end_s: float = float(vt["global_time_s"])
        duration_s: float = round(end_s - start_s, 4)

        # ── meta.json ────────────────────────────────────────────────────────
        meta = dict(vt)  # shallow copy of the ValleyTrough TypedDict
        meta["start_s"] = start_s
        meta["end_s"] = end_s
        meta["duration_s"] = duration_s
        with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        # ── sound.wav ────────────────────────────────────────────────────────
        if audio_data is not None:
            end_sample = int(end_s * file_sr)
            end_sample = min(len(audio_data), end_sample)
            slice_audio = audio_data[0:end_sample]
            sf.write(str(seg_dir / "sound.wav"), slice_audio, file_sr)

        # ── plot.png ─────────────────────────────────────────────────────────
        trough_frame: int = int(vt["global_frame"])
        f_end = min(n_frames, trough_frame + 1)
        frames = np.arange(0, f_end)
        zoomed = x[0:f_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)
        # Shade the entire excerpt in a neutral blue
        ax.axvspan(0, trough_frame, alpha=0.12, color="blue", label="excerpt")
        # Mark the trough itself
        ax.axvline(
            x=trough_frame,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"trough (frame {trough_frame})",
        )
        ax.plot(
            trough_frame,
            x[trough_frame] if trough_frame < n_frames else 0.0,
            "ro",
            markersize=9,
        )
        ax.set_title(
            f"valley_troughs · segment {idx:03d}  [0.000 s – {end_s:.3f} s]",
            fontsize=12,
        )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        plt.savefig(str(seg_dir / "plot.png"), dpi=150, bbox_inches="tight")
        plt.close()

    console.print(
        f"🔻 [bold]{len(valley_troughs)} Valley Troughs[/bold] segments saved to: "
        f"[link=file:///{cat_dir.resolve()}]{cat_dir.resolve()}[/link]",
        style="green",
    )


def save_trough_to_trough_segments(
    valley_troughs: List["ValleyTrough"],
    probs: List[float],
    output_dir: "Path",
    audio_path: Optional[str],
    sample_rate: int,
    frame_shift_ms: float,
) -> None:
    """
    For each ValleyTrough, create a segment spanning from the previous trough
    (or t=0 for the first) up to and including the current trough.
    A final segment is also created from the last trough to the end of the audio.

    Produces N+1 segments for N valley_troughs:
        segment_000: t=0          → trough[0]
        segment_001: trough[0]    → trough[1]
        ...
        segment_N:   trough[N-1]  → end of audio

    Also writes a summary ``trough_to_trough.json`` containing all segment
    metadata in a single list.
    """
    import json

    if not valley_troughs:
        console.print("⚠️  [yellow]trough_to_trough: no troughs provided.[/yellow]")
        return

    cat_dir = output_dir / "trough_to_trough"
    cat_dir.mkdir(parents=True, exist_ok=True)

    audio_data: Optional[np.ndarray] = None
    file_sr: int = sample_rate
    if audio_path is not None:
        audio_data, file_sr = sf.read(audio_path, always_2d=False)

    x = np.array(probs, dtype=float)
    n_frames = len(x)

    # Compute end time/frame from probs length and frame duration
    frame_duration_s = frame_shift_ms / 1000.0
    end_time_s = n_frames * frame_duration_s
    end_frame = n_frames - 1

    # Sentinels: origin at t=0, tail at end of audio
    sentinel_start = {**valley_troughs[0], "global_time_s": 0.0, "global_frame": 0}
    sentinel_end = {
        **valley_troughs[-1],
        "global_time_s": end_time_s,
        "global_frame": end_frame,
    }
    anchors = [sentinel_start] + list(valley_troughs) + [sentinel_end]

    all_segments = []

    # N+1 segments: one per gap between consecutive anchors
    for idx in range(len(anchors) - 1):
        vt_start = anchors[idx]
        vt_end = anchors[idx + 1]

        is_first = idx == 0
        is_last = idx == len(anchors) - 2

        seg_dir = cat_dir / f"segment_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        start_s: float = float(vt_start["global_time_s"])
        end_s: float = float(vt_end["global_time_s"])
        duration_s: float = round(end_s - start_s, 4)

        start_frame: int = int(vt_start["global_frame"])
        end_frame_seg: int = int(vt_end["global_frame"])

        meta = {
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "start_frame": start_frame,
            "end_frame": end_frame_seg,
            "trough_start": None if is_first else dict(vt_start),
            "trough_end": None if is_last else dict(vt_end),
        }

        # ── meta.json ────────────────────────────────────────────────────────
        with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        all_segments.append(meta)

        # ── sound.wav ────────────────────────────────────────────────────────
        if audio_data is not None:
            start_sample = int(start_s * file_sr)
            end_sample = int(end_s * file_sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            slice_audio = audio_data[start_sample:end_sample]
            sf.write(str(seg_dir / "sound.wav"), slice_audio, file_sr)

        # ── plot.png ─────────────────────────────────────────────────────────
        f_start = max(0, start_frame)
        f_end = min(n_frames, end_frame_seg + 1)
        frames = np.arange(f_start, f_end)
        zoomed = x[f_start:f_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)
        ax.axvspan(f_start, f_end, alpha=0.12, color="purple", label="trough span")

        # Start boundary: gray for origin sentinel, red for real trough
        ax.axvline(
            x=start_frame,
            color="gray" if is_first else "red",
            linestyle="--",
            linewidth=1.5,
            label=f"{'origin' if is_first else 'start trough'} (frame {start_frame})",
        )
        if not is_first and start_frame < n_frames:
            ax.plot(start_frame, x[start_frame], "ro", markersize=9)

        # End boundary: gray for tail sentinel, red for real trough
        ax.axvline(
            x=end_frame_seg,
            color="gray" if is_last else "red",
            linestyle="--",
            linewidth=1.5,
            label=f"{'end of audio' if is_last else 'end trough'} (frame {end_frame_seg})",
        )
        if not is_last and end_frame_seg < n_frames:
            ax.plot(end_frame_seg, x[end_frame_seg], "ro", markersize=9)

        ax.set_title(
            f"trough_to_trough · segment {idx:03d}  [{start_s:.3f} s – {end_s:.3f} s]",
            fontsize=12,
        )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        plt.savefig(str(seg_dir / "plot.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── trough_to_trough.json ─────────────────────────────────────────────────
    summary_path = output_dir / "trough_to_trough.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_segments, fh, ensure_ascii=False, indent=2)
    console.print(
        f"   • trough_to_trough.json → "
        f"[link=file:///{summary_path.resolve()}]{summary_path.resolve()}[/link]",
        style="dim",
    )

    console.print(
        f"🟣 [bold]{len(all_segments)} Trough-to-Trough[/bold] segments saved to: "
        f"[link=file:///{cat_dir.resolve()}]{cat_dir.resolve()}[/link]",
        style="green",
    )


def get_args():
    import argparse

    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

    parser = argparse.ArgumentParser(
        description="Analyze VAD speech/voice probabilities and find peaks/troughs"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_AUDIO,
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
        default=SAMPLE_RATE,
        help="Audio sample rate (default: SAMPLE_RATE)",
    )
    parser.add_argument(
        "--frame-shift-ms",
        "-fsm",
        type=float,
        default=FRAME_SHIFT_MS,
        help="Frame shift (hop length) in ms between analysis frames (default: FRAME_SHIFT_MS for FireRedVAD)",
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

    from vad_extractors import (
        base_extract_valley_troughs,
        extract_valley_troughs,
        smooth_vad_probs,
    )

    args = get_args()
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]VAD Peak & Valley Analyzer[/bold cyan]")

    # Load probabilities
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
        console.print("Using default sample probability sequence.", style="dim")
        probs = [0.1, 0.15, 0.8, 0.92, 0.85, 0.3, 0.12, 0.05, 0.88, 0.95, 0.7, 0.2]

    # Smoothing
    if args.smoothing_window:
        probs_smoothed = smooth_vad_probs(probs, window=args.smoothing_window)
        console.print(
            f"Applied smoothing (window={args.smoothing_window})", style="blue"
        )
    else:
        probs_smoothed = probs

    analyzer = VADPeakAnalyzer(
        sample_rate=args.sample_rate, frame_shift_ms=args.frame_shift_ms, debug=False
    )

    # === Analysis ===
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

    valley_troughs = extract_valley_troughs(
        probs_or_audio=probs,
        min_valley_duration_s=args.min_valley_duration,
        sample_rate=args.sample_rate,  # ← add this
        frame_shift_ms=args.frame_shift_ms,  # ← add this (was defaulting to 25ms!)
        frame_offset=args.frame_offset if hasattr(args, "frame_offset") else 0,
        smoothing_window=args.smoothing_window,
    )

    # === Results Summary ===
    console.print("\n[bold cyan]📊 Analysis Summary[/bold cyan]")
    console.print(f"   • Peaks          : {len(peaks)}", style="green")
    console.print(f"   • Troughs        : {len(troughs)}", style="red")
    console.print(f"   • Active regions : {len(active_regions)}", style="green")
    console.print(f"   • Valleys        : {len(valleys)}", style="red")

    # === Save per-segment subdirectories (if audio input) ===
    if args.input_file and Path(args.input_file).suffix.lower() in AUDIO_EXTENSIONS:
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
        save_valley_trough_segments(
            valley_troughs=valley_troughs,
            probs=probs_smoothed,
            output_dir=output_dir,
            audio_path=args.input_file,
            sample_rate=args.sample_rate,
            frame_shift_ms=args.frame_shift_ms,
        )
        save_trough_to_trough_segments(
            valley_troughs=valley_troughs,
            probs=probs_smoothed,
            output_dir=output_dir,
            audio_path=args.input_file,
            sample_rate=args.sample_rate,
            frame_shift_ms=args.frame_shift_ms,
        )

    # === Final Plots & JSONs ===
    analyzer.save_plot(
        probs_smoothed,
        peaks,
        troughs,
        active_regions=active_regions,
        valleys=valleys,
        output_path=str(output_dir / "vad_analysis_plot.png"),
        title="VAD Peak & Trough Analysis",
    )

    if args.smoothing_window:
        analyzer.save_plot(
            probs_smoothed,
            peaks,
            troughs,
            active_regions=active_regions,
            valleys=valleys,
            output_path=str(output_dir / "vad_analysis_plot_smoothed.png"),
            title="VAD Peak & Trough Analysis (Smoothed)",
        )

    # Save JSON files with rich clickable links
    console.print("\n[bold green]💾 Saved Files[/bold green]")

    def save_json(data, filename: str, description: str):
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(
            f"   • {description:18} → "
            f"[link=file:///{path.resolve()}]{path.resolve()}[/link]",
            style="dim",
        )

    save_json(peaks, "peaks.json", "Peaks")
    save_json(troughs, "troughs.json", "Troughs")
    save_json(active_regions, "active_regions.json", "Active Regions")
    save_json(valleys, "valleys.json", "Valleys")

    # Extra valley troughs
    base_valley_troughs = base_extract_valley_troughs(valleys)
    save_json(base_valley_troughs, "base_valley_troughs.json", "Base Valley Troughs")

    save_json(valley_troughs, "valley_troughs.json", "Valley Troughs")

    console.rule("[bold green]Analysis Complete ✓[/bold green]")
