# speech_waves.py

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from _types import AudioInput, SpeechWave
from config import HOP_SIZE, SAMPLE_RATE
from energy import compute_rms_per_frame
from loader import load_audio
from norm_speech_loudness import normalize_audio_for_vad

WaveState = Literal["below", "above"]


@dataclasses.dataclass
class WaveShapeConfig:
    """
    Tunable thresholds that decide whether a probability wave has a real
    mountain shape rather than being a flat plateau or a tiny ripple.

    Attributes:
        min_prominence: How much the peak must rise above the average of the
            two surrounding valley endpoints.
        min_excursion: The minimum difference between the highest and lowest
            probability inside the wave window.
        min_peak_prob: Absolute floor — the peak frame must reach at least
            this probability (guards against waves that never really fire).
        min_frames: Waves shorter than this many frames are discarded.
        min_duration_sec: Minimum wall-clock duration in seconds. Waves
            shorter than this are rejected even if they pass frame and shape
            checks. Derived independently of min_frames so both constraints
            must be satisfied.
        baseline_threshold: Probability threshold used to determine when a
            wave has truly fallen back to baseline/silence level. Used to
            detect wave boundaries and preroll adjustments.
    """

    min_prominence: float = 0.05
    min_excursion: float = 0.04
    min_peak_prob: float = 0.55
    min_frames: int = 3
    min_duration_sec: float = 0.25  # matches default --min-speech-duration of 250 ms
    baseline_threshold: float = 0.1  # threshold for silence/baseline detection


def is_prominent_wave(
    wave_probs: List[float],
    entry_prob: float,
    exit_prob: float,
    cfg: WaveShapeConfig,
) -> tuple[bool, dict]:
    """
    Decide whether a slice of probabilities forms a genuine mountain shape.

    The algorithm:
      1. Baseline = average of entry_prob and exit_prob (the "ground level").
      2. Peak     = maximum probability inside the slice.
      3. Prominence = peak - baseline.
      4. Excursion  = max - min inside the slice (vertical range).

    Returns:
        (passed: bool, diagnostics: dict)
    """
    if not wave_probs:
        return False, {}

    peak_prob = max(wave_probs)
    min_prob = min(wave_probs)
    baseline = (entry_prob + exit_prob) / 2.0
    prominence = peak_prob - baseline
    excursion = peak_prob - min_prob
    n_frames = len(wave_probs)

    passed = (
        prominence >= cfg.min_prominence
        and excursion >= cfg.min_excursion
        and peak_prob >= cfg.min_peak_prob
        and n_frames >= cfg.min_frames
    )

    diagnostics = {
        "baseline": round(baseline, 6),
        "peak_prob": round(peak_prob, 6),
        "prominence": round(prominence, 6),
        "excursion": round(excursion, 6),
        "n_frames": n_frames,
        "shape_passed": passed,
    }
    return passed, diagnostics


def get_speech_waves(
    audio: AudioInput,
    speech_probs: List[float],
    threshold: float = 0.5,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> List[SpeechWave]:
    """
    Identify complete speech waves (rise → sustained high → fall) from FireRedVAD probabilities.

    This function now accepts any AudioInput type and internally uses load_audio()
    for consistent preprocessing (though the audio itself is not processed further here
    unless you need to derive probabilities).
    """
    # Load audio for consistency (ensures correct sample rate and format)
    _, loaded_sr = load_audio(audio, sr=sampling_rate, mono=True)

    # Use the full probability list
    all_waves = check_speech_waves(
        speech_probs=speech_probs,
        threshold=threshold,
        sampling_rate=loaded_sr,  # Use the confirmed sample rate
        shape_cfg=shape_cfg,
    )

    # Filter only valid (complete) waves
    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave.get("is_valid", False):
            valid_waves.append(wave)

    return valid_waves


def check_speech_waves(
    speech_probs: List[float],
    threshold: float = 0.5,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> List[SpeechWave]:
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()

    if not speech_probs:
        return []

    waves: List[SpeechWave] = []
    current_wave: SpeechWave | None = None
    state: WaveState = "below"
    rise_frame_idx: int | None = None

    if speech_probs:
        if speech_probs[0] < shape_cfg.baseline_threshold:
            current_wave = SpeechWave(
                has_risen=False,
                has_multi_passed=False,
                has_fallen=False,
                is_valid=False,
                start_sec=0.0,
                end_sec=0.0,
                details={
                    "frame_start": 0,
                    "frame_end": 0,
                    "frame_len": 0,
                    "duration_sec": 0.0,
                    "min_prob": speech_probs[0],
                    "max_prob": speech_probs[0],
                    "avg_prob": speech_probs[0],
                    "std_prob": 0.0,
                    "composite_score": 0.0,
                },
            )
            state = "below"

        elif speech_probs[0] >= threshold:
            state = "above"

    for i, prob in enumerate(speech_probs):
        frame_time_sec = i * HOP_SIZE / sampling_rate

        if state == "below":
            if prob >= threshold:
                rise_frame_idx = i

                # ── Preroll: walk back from rise_frame_idx until we find a
                #    frame strictly below baseline_threshold (or hit index 0).
                preroll_start = rise_frame_idx
                while preroll_start > 0 and speech_probs[preroll_start - 1] >= shape_cfg.baseline_threshold:
                    preroll_start -= 1
                preroll_start_sec = preroll_start * HOP_SIZE / sampling_rate

                current_wave = SpeechWave(
                    has_risen=current_wave["has_risen"] if current_wave else True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=preroll_start_sec,
                    end_sec=preroll_start_sec,
                    details={
                        "frame_start": preroll_start,
                        "frame_end": preroll_start,
                        "frame_len": 0,
                        "duration_sec": 0.0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "avg_prob": prob,
                        "std_prob": 0.0,
                        "composite_score": 0.0,
                    },
                )

                state = "above"
        else:
            if prob >= threshold:
                if current_wave is not None:
                    current_wave["has_multi_passed"] = True
            else:
                if current_wave is not None:
                    if prob <= shape_cfg.baseline_threshold:
                        current_wave["has_fallen"] = True

                    # frame_start uses the preroll-adjusted value stored in details
                    frame_start = current_wave["details"]["frame_start"]
                    frame_end = i
                    wave_probs = speech_probs[frame_start:frame_end]
                    frame_len = frame_end - frame_start

                    # entry_prob: the frame immediately before the preroll start
                    entry_prob = (
                        speech_probs[frame_start - 1] if frame_start > 0 else 0.0
                    )
                    exit_prob = prob

                    shape_ok, shape_diag = is_prominent_wave(
                        wave_probs, entry_prob, exit_prob, shape_cfg
                    )

                    duration_sec = frame_time_sec - current_wave["start_sec"]
                    duration_ok = duration_sec >= shape_cfg.min_duration_sec

                    current_wave["is_valid"] = (
                        current_wave["has_risen"]
                        and current_wave["has_multi_passed"]
                        and current_wave["has_fallen"]
                        and shape_ok
                        and duration_ok
                    )
                    current_wave["end_sec"] = frame_time_sec
                    current_wave["details"] = {
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "frame_len": frame_len,
                        "duration_sec": duration_sec,
                        "min_prob": min(wave_probs) if wave_probs else 0.0,
                        "max_prob": max(wave_probs) if wave_probs else 0.0,
                        "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                        "std_prob": statistics.stdev(wave_probs)
                        if frame_len > 1
                        else 0.0,
                        "duration_ok": duration_ok,
                        **shape_diag,
                        "composite_score": 0.0,
                    }
                    current_wave["details"]["composite_score"] = (
                        _compute_composite_score(current_wave)
                    )

                if prob < shape_cfg.baseline_threshold:
                    waves.append(current_wave)
                    current_wave = None
                    rise_frame_idx = None
                    state = "below"

    # Handle a wave that never fell back below the threshold
    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False
        current_wave["end_sec"] = len(speech_probs) * HOP_SIZE / sampling_rate

        if rise_frame_idx is not None:
            # frame_start is already preroll-adjusted in details
            frame_start = current_wave["details"]["frame_start"]
            frame_end = len(speech_probs)
            wave_probs = speech_probs[frame_start:frame_end]
            frame_len = frame_end - frame_start
            duration_sec = current_wave["end_sec"] - current_wave["start_sec"]
            entry_prob = speech_probs[frame_start - 1] if frame_start > 0 else 0.0
            exit_prob = threshold
            shape_ok, shape_diag = is_prominent_wave(
                wave_probs, entry_prob, exit_prob, shape_cfg
            )
            current_wave["details"] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_len": frame_len,
                "duration_sec": duration_sec,
                "min_prob": min(wave_probs) if wave_probs else 0.0,
                "max_prob": max(wave_probs) if wave_probs else 0.0,
                "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
                "duration_ok": False,
                **shape_diag,
                "composite_score": 0.0,
            }
            current_wave["details"]["composite_score"] = _compute_composite_score(
                current_wave
            )

        waves.append(current_wave)

    return waves


def save_wave_audio(
    audio_np: np.ndarray,
    sampling_rate: int,
    frame_start: int,
    frame_end: int,
    output_path: Path,
    hop_size: int = HOP_SIZE,
) -> None:
    """Extract and save audio chunk for a wave based on frame indices."""
    start_sample = frame_start * hop_size
    end_sample = (frame_end + 1) * hop_size
    wave_audio = audio_np[start_sample:end_sample]
    wavfile.write(output_path, sampling_rate, wave_audio)


def _compute_composite_score(wave: SpeechWave) -> float:
    """
    Composite quality score for ranking speech waves.

    Formula:
        score = avg_prob * prominence * log1p(duration_sec) * (1 + 0.3 * excursion)

    Rationale for each term:
    - avg_prob: rewards sustained confidence across the whole wave, not just
      a single spike; a wave hovering at 0.95 outranks one that spikes once
      and sits at 0.55.
    - prominence: the mountain height above the noise floor (peak minus
      baseline); guards against flat plateaus that happen to be above threshold.
    - log1p(duration_sec): duration reward with diminishing returns so long
      but featureless segments don't dominate short, sharp utterances.
      log1p(1 s) ≈ 0.69, log1p(3 s) ≈ 1.39, log1p(10 s) ≈ 2.40.
    - (1 + 0.3 * excursion): small multiplicative bonus for shape sharpness;
      high excursion means the wave truly rises and falls rather than
      lingering as a flat plateau. Coefficient 0.3 caps the bonus at ×1.3
      (when excursion = 1.0) so it modulates rather than dominates.
    """
    d = wave["details"]
    avg_prob = d.get("avg_prob", 0.0)
    prominence = d.get("prominence", d["max_prob"])
    duration_sec = d.get("duration_sec", 0.0)
    excursion = d.get("excursion", 0.0)
    return avg_prob * prominence * math.log1p(duration_sec) * (1.0 + 0.3 * excursion)


def save_wave_plot(
    probs: List[float],
    rms_values: List[float],
    output_path: Path,
    wave_num: int,
    seg_num: int,
    wave: Optional[SpeechWave] = None,
    threshold: float = 0.5,
    hop_size: int = HOP_SIZE,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> None:
    """
    Create a two-panel visualization for a single speech wave.

    Top panel — VAD probability:
    - X-axis in milliseconds (real time, not frame index)
    - Above-threshold region shaded in light blue
    - Vertical dashed markers at wave start and end
    - Baseline shown as a horizontal dashed line with label
    - Peak annotated with a dot and probability label
    - Metric text-box: peak, avg, prominence, excursion, baseline, composite,
      duration (drawn in the upper-right corner so it never overlaps the curve)

    Bottom panel — RMS energy:
    - Normalised to [0, 1] within the plot window for readability at any
      absolute amplitude; annotated with "(normalised)" on the y-axis
    - Same x-axis and time markers as the top panel
    """
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()
    
    baseline_threshold = shape_cfg.baseline_threshold

    # --- align arrays --------------------------------------------------------
    min_length = min(len(probs), len(rms_values))
    probs_aligned = probs[:min_length]
    rms_aligned = rms_values[:min_length]

    # Convert frame indices to milliseconds
    ms_per_frame = hop_size / sampling_rate * 1000.0
    frames = np.arange(min_length)
    times_ms = frames * ms_per_frame

    # --- pull wave metadata --------------------------------------------------
    d = wave["details"] if wave is not None else {}
    peak_prob = d.get("max_prob", max(probs_aligned) if probs_aligned else 0.0)
    avg_prob = d.get("avg_prob", 0.0)
    prominence = d.get("prominence", 0.0)
    excursion = d.get("excursion", 0.0)
    baseline = d.get("baseline", 0.0)
    duration_s = d.get("duration_sec", min_length * hop_size / sampling_rate)
    composite = _compute_composite_score(wave) if wave is not None else 0.0

    # Wave start/end in milliseconds relative to the wave window origin
    # (frame_start is absolute; the slice already starts there, so t=0 in
    # the plot is the wave's own first frame)
    wave_start_ms = 0.0
    wave_end_ms = duration_s * 1000.0

    # --- normalise RMS -------------------------------------------------------
    rms_arr = np.array(rms_aligned, dtype=float)
    rms_max = rms_arr.max() if rms_arr.size and rms_arr.max() > 0 else 1.0
    rms_norm = rms_arr / rms_max

    # --- figure setup --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6]},
    )
    fig.subplots_adjust(hspace=0.08, left=0.09, right=0.97, top=0.92, bottom=0.11)

    # ── TOP PANEL: VAD probability ──────────────────────────────────────────
    # Above-threshold shading
    ax1.fill_between(
        times_ms,
        probs_aligned,
        threshold,
        where=[p >= threshold for p in probs_aligned],
        alpha=0.18,
        color="#2196F3",
        interpolate=True,
        label=None,
    )

    # Probability curve
    ax1.plot(times_ms, probs_aligned, color="#1565C0", linewidth=1.4, zorder=3)

    # Threshold line
    ax1.axhline(
        y=threshold,
        color="#E53935",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"Threshold ({threshold:.2f})",
    )

    # Baseline threshold line
    ax1.axhline(
        y=baseline_threshold,
        color="#6D4C41",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label=f"Baseline threshold ({baseline_threshold:.3f})",
    )

    # Wave start / end vertical markers
    ax1.axvline(
        wave_start_ms,
        color="#4CAF50",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Wave start",
    )
    ax1.axvline(
        wave_end_ms,
        color="#FF7043",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Wave end",
    )

    # Peak annotation
    if probs_aligned:
        peak_frame = int(np.argmax(probs_aligned))
        peak_ms = times_ms[peak_frame]
        ax1.plot(
            peak_ms,
            probs_aligned[peak_frame],
            "o",
            color="#E53935",
            markersize=5,
            zorder=5,
        )
        ax1.annotate(
            f"{probs_aligned[peak_frame]:.3f}",
            xy=(peak_ms, probs_aligned[peak_frame]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#E53935",
            zorder=6,
        )

    # Metric text-box (upper-right corner)
    metrics_text = (
        f"peak:        {peak_prob:.3f}\n"
        f"avg:         {avg_prob:.3f}\n"
        f"prominence:  {prominence:.3f}\n"
        f"excursion:   {excursion:.3f}\n"
        f"baseline:    {baseline:.3f}\n"
        f"duration:    {duration_s:.2f} s\n"
        f"composite:   {composite:.4f}"
    )
    ax1.text(
        0.985,
        0.97,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=7.5,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#BDBDBD",
            alpha=0.88,
            linewidth=0.6,
        ),
        zorder=7,
    )

    ax1.set_ylabel("VAD probability", fontsize=9)
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.85, edgecolor="#BDBDBD")
    ax1.set_title(
        f"Segment {seg_num:03d}  ·  Wave {wave_num:03d}  ·  {duration_s * 1000:.0f} ms",
        fontsize=10,
        pad=6,
    )

    # ── BOTTOM PANEL: normalised RMS energy ─────────────────────────────────
    ax2.fill_between(times_ms[: len(rms_norm)], rms_norm, alpha=0.25, color="#388E3C")
    ax2.plot(times_ms[: len(rms_norm)], rms_norm, color="#2E7D32", linewidth=1.2)

    ax2.axvline(
        wave_start_ms, color="#4CAF50", linestyle="--", linewidth=1.0, alpha=0.7
    )
    ax2.axvline(wave_end_ms, color="#FF7043", linestyle="--", linewidth=1.0, alpha=0.7)

    ax2.set_xlabel("Time (ms)", fontsize=9)
    ax2.set_ylabel("RMS energy\n(normalised)", fontsize=8)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_wave_data(
    wave: SpeechWave,
    audio_np: np.ndarray,
    speech_probs: List[float],
    sampling_rate: int,
    output_dir: Path,
    seg_num: int,
    wave_num: int,
    hop_size: int = HOP_SIZE,
    threshold: float = 0.5,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> None:
    """Save all wave-related data to the specified directory."""
    wave_dir = output_dir / f"segment_{seg_num:03d}_wave_{wave_num:03d}"
    wave_dir.mkdir(parents=True, exist_ok=True)

    # Extract frame info
    frame_start = wave["details"]["frame_start"]
    frame_end = wave["details"]["frame_end"]

    # Save wave audio
    wav_path = wave_dir / "sound.wav"
    save_wave_audio(audio_np, sampling_rate, frame_start, frame_end, wav_path, hop_size)

    # Save wave probabilities slice
    wave_probs = speech_probs[frame_start:frame_end]
    probs_path = wave_dir / "speech_probs.json"
    with open(probs_path, "w") as f:
        json.dump(wave_probs, f, indent=2)

    # Calculate and save RMS energies
    rms_values = compute_rms_per_frame(audio_np, hop_size, frame_start, frame_end)
    energies_path = wave_dir / "energies.json"
    with open(energies_path, "w") as f:
        json.dump(rms_values, f, indent=2)

    # Save wave metadata
    wave_json_path = wave_dir / "wave.json"
    wave_copy = wave.copy()
    wave_copy["segment_num"] = seg_num
    wave_copy["wave_num"] = wave_num
    with open(wave_json_path, "w") as f:
        json.dump(wave_copy, f, indent=2)

    # Create and save visualization (pass full wave context)
    plot_path = wave_dir / "wave_plot.png"
    save_wave_plot(
        probs=wave_probs,
        rms_values=rms_values,
        output_path=plot_path,
        wave_num=wave_num,
        seg_num=seg_num,
        wave=wave,
        threshold=threshold,
        hop_size=hop_size,
        sampling_rate=sampling_rate,
        shape_cfg=shape_cfg,
    )


# ── Reporting helpers ──


def _find_parent_segment(wave: SpeechWave, segments: list) -> int:
    """
    Find which segment a wave belongs to based on time overlap.
    Returns 1-based segment number.
    """
    wave_start = wave["start_sec"]
    wave_end = wave["end_sec"]
    
    for seg in segments:
        seg_start = seg.get("start_sec", 0.0)
        seg_end = seg.get("end_sec", 0.0)
        
        # Check for any time overlap between wave and segment
        if wave_start <= seg_end and wave_end >= seg_start:
            return seg.get("num", seg.get("segment_num", 1))
    
    # Fallback to first segment if no match found
    return 1


def _build_wave_report(
    wave: SpeechWave,
    wave_idx: int,
    waves_dir: Path,
    segments: list,
) -> dict:
    """
    Flatten one SpeechWave into a clean, self-contained report dict.
    Used for both summary.json rows and top_5_waves.json entries.
    """
    parent_seg_num = _find_parent_segment(wave, segments)

    dir_name = f"segment_{parent_seg_num:03d}_wave_{wave_idx:03d}"
    wav_abs = (waves_dir / dir_name / "sound.wav").resolve()
    plot_abs = (waves_dir / dir_name / "wave_plot.png").resolve()

    d = wave["details"]
    return {
        # ── identity ──────────────────────────────────────────────────
        "wave": wave_idx,
        "dir": dir_name,
        # ── timing ────────────────────────────────────────────────────
        "start_sec": round(wave["start_sec"], 4),
        "end_sec": round(wave["end_sec"], 4),
        "dur_sec": round(d["duration_sec"], 4),
        # ── Plot file ────────────────────────────────────────────────
        "plot_path": str(plot_abs),
        # ── audio file ────────────────────────────────────────────────
        "sound_path": str(wav_abs),
        # ── probability scores ────────────────────────────────────────
        "scores": {
            "min_prob": round(d["min_prob"], 6),
            "max_prob": round(d["max_prob"], 6),
            "avg_prob": round(d["avg_prob"], 6),
            "std_prob": round(d["std_prob"], 6),
            "baseline": round(d.get("baseline", 0.0), 6),
            "prominence": round(d.get("prominence", 0.0), 6),
            "excursion": round(d.get("excursion", 0.0), 6),
            "composite": round(_compute_composite_score(wave), 6),
        },
    }


def _top5_reports(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
) -> list[dict]:
    """
    Return the 5 waves with the highest composite score, already serialised
    as report dicts (not raw SpeechWave objects).

    Composite score (see _compute_composite_score for full rationale):
        avg_prob * prominence * log1p(duration_sec) * (1 + 0.3 * excursion)

    - avg_prob rewards sustained confidence across the whole wave (not just
      a single spike).
    - prominence measures mountain height above the noise floor.
    - log1p(duration_sec) applies a duration bonus with diminishing returns.
    - (1 + 0.3 * excursion) gives a small multiplicative bonus for waves
      that genuinely rise and fall rather than sitting as flat plateaus.
    """
    indexed = list(enumerate(speech_waves, 1))  # [(1, wave), (2, wave), …]
    ranked = sorted(
        indexed, key=lambda iv: _compute_composite_score(iv[1]), reverse=True
    )
    return [
        _build_wave_report(wave, idx, waves_dir, segments) for idx, wave in ranked[:5]
    ]


def build_summary_rows(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
) -> list[dict]:
    """
    Build a flat list of report dicts — one per valid wave — used for both
    the rich summary table and summary.json.
    """
    return [
        _build_wave_report(wave, idx, waves_dir, segments)
        for idx, wave in enumerate(speech_waves, 1)
    ]


if __name__ == "__main__":
    import argparse

    from file_utils import save_file
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from vad_firered2 import extract_speech_timestamps
    # from vad_tenvad import extract_speech_timestamps

    console = Console()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    DEFAULT_AUDIO = str(
        Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
    )

    parser = argparse.ArgumentParser(
        description="Extract and analyse speech waves from audio using FireRedVAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="Input audio file path.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help="Output results directory.",
    )

    # ── VAD core ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="VAD probability threshold (above = speech).",
    )
    parser.add_argument(
        "-s",
        "--hop-size",
        type=int,
        default=160,
        help="Frame hop size in samples (160 = 10 ms at 16 kHz).",
    )

    # ── VAD segment filtering ─────────────────────────────────────────────────
    parser.add_argument(
        "-d",
        "--min-speech-duration",
        type=int,
        default=250,
        metavar="MS",
        help=(
            "Minimum speech segment duration in ms passed to the VAD and "
            "also used as the wave-level min_duration_sec floor."
        ),
    )
    parser.add_argument(
        "-g",
        "--min-silence-duration",
        type=int,
        default=100,
        metavar="MS",
        help="Minimum silence gap between segments in ms.",
    )
    parser.add_argument(
        "-ns",
        "--include-non-speech",
        action="store_true",
        help="Include non-speech segments in the VAD output.",
    )

    # ── WaveShapeConfig ───────────────────────────────────────────────────────
    parser.add_argument(
        "-p",
        "--min-prominence",
        type=float,
        default=WaveShapeConfig.min_prominence,
        metavar="FLOAT",
        help=(
            "Minimum prominence: how much the peak must rise above the "
            "average of the entry/exit probabilities."
        ),
    )
    parser.add_argument(
        "-e",
        "--min-excursion",
        type=float,
        default=WaveShapeConfig.min_excursion,
        metavar="FLOAT",
        help=(
            "Minimum excursion: minimum difference between the highest and "
            "lowest probability inside the wave window."
        ),
    )
    parser.add_argument(
        "-P",
        "--min-peak-prob",
        type=float,
        default=WaveShapeConfig.min_peak_prob,
        metavar="FLOAT",
        help=(
            "Minimum peak probability: absolute floor the peak frame must "
            "reach for the wave to be considered valid."
        ),
    )
    parser.add_argument(
        "-f",
        "--min-frames",
        type=int,
        default=WaveShapeConfig.min_frames,
        metavar="N",
        help="Minimum number of frames a wave must span.",
    )
    parser.add_argument(
        "-b",
        "--baseline-threshold",
        type=float,
        default=WaveShapeConfig.baseline_threshold,
        metavar="FLOAT",
        help=(
            "Probability threshold used to determine when a wave has truly "
            "fallen back to baseline/silence level. Used for wave boundary "
            "detection and preroll adjustments."
        ),
    )

    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help=(
            "Normalize audio before VAD processing. Applies RMS-based normalization "
            "to improve VAD performance on low-volume or variable-level recordings."
        ),
    )

    args = parser.parse_args()

    # ── Build shape config from args ──────────────────────────────────────────
    # min_duration_sec is always driven by --min-speech-duration so the VAD
    # segment floor and the wave-level floor are always in sync.
    shape_cfg = WaveShapeConfig(
        min_prominence=args.min_prominence,
        min_excursion=args.min_excursion,
        min_peak_prob=args.min_peak_prob,
        min_frames=args.min_frames,
        min_duration_sec=args.min_speech_duration / 1000,
        baseline_threshold=args.baseline_threshold,
    )

    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load audio for wave extraction
    audio_np, sr = load_audio(args.input, sr=SAMPLE_RATE, mono=True)

    if args.normalize:
        audio_np_norm, vad_stats = normalize_audio_for_vad(audio_np, sr)
        audio_np = audio_np_norm

    segments, scores = extract_speech_timestamps(
        audio=audio_np,
        include_non_speech=args.include_non_speech,
        threshold=args.threshold,
        min_speech_duration_sec=args.min_speech_duration / 1000,
        min_silence_duration_sec=args.min_silence_duration / 1000,
        with_scores=True,
    )

    speech_waves = get_speech_waves(
        args.input,
        scores,
        threshold=args.threshold,
        shape_cfg=shape_cfg,
    )

    # Save main JSON files
    save_file(segments, args.output_dir / "segments.json")
    save_file(scores, args.output_dir / "speech_probs.json")
    save_file(speech_waves, args.output_dir / "speech_waves.json")
    # save_file(vad_stats, args.output_dir / "vad_stats.json")

    # Create waves directory and save individual wave files
    waves_dir = args.output_dir / "waves"
    waves_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold]Generating files for {len(speech_waves)} valid speech waves...[/bold]"
    )

    for wave_idx, wave in enumerate(speech_waves, 1):
        parent_seg_num = _find_parent_segment(wave, segments)

        save_wave_data(
            wave=wave,
            audio_np=audio_np,
            speech_probs=scores,
            sampling_rate=sr,
            output_dir=waves_dir,
            seg_num=parent_seg_num,
            wave_num=wave_idx,
            hop_size=args.hop_size,
            threshold=args.threshold,
            shape_cfg=shape_cfg,
        )

    # ── Summary table & JSON ──────────────────────────────────────────────────
    rows = build_summary_rows(speech_waves, waves_dir, segments)
    save_file(rows, args.output_dir / "summary.json")

    # ── Top-5 waves ───────────────────────────────────────────────────────────
    top5 = _top5_reports(speech_waves, waves_dir, segments)
    save_file(top5, args.output_dir / "top_5_waves.json")

    table = Table(
        title=f"Speech Waves Summary  ({len(rows)} valid waves)",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Dir", style="cyan", justify="left", no_wrap=True)
    table.add_column("Start (s)", style="white", justify="right", no_wrap=True)
    table.add_column("End (s)", style="white", justify="right", no_wrap=True)
    table.add_column("Dur (s)", style="yellow", justify="right", no_wrap=True)
    table.add_column("Prominence", style="magenta", justify="right", no_wrap=True)
    table.add_column("Composite", style="bright_cyan", justify="right", no_wrap=True)
    table.add_column("Baseline", style="blue", justify="right", no_wrap=True)
    table.add_column("Peak prob", style="green", justify="right", no_wrap=True)
    table.add_column("Sound", style="bright_black", justify="left")

    top5_dirs = {w["dir"] for w in top5}

    for r in rows:
        is_top5 = r["dir"] in top5_dirs
        row_style = "bold" if is_top5 else ""
        star = "★ " if is_top5 else "  "

        dir_cell = f"[link=file://{r['plot_path']}]{r['dir']}[/link]"
        sound_cell = f"[link=file://{r['sound_path']}]▶️[/link]"

        table.add_row(
            f"{star}{r['wave']}",
            dir_cell,
            f"{r['start_sec']:.2f}",
            f"{r['end_sec']:.2f}",
            f"{r['dur_sec']:.2f}",
            f"{r['scores']['prominence']:.3f}",
            f"{r['scores']['composite']:.4f}",
            f"{r['scores']['baseline']:.3f}",
            f"{r['scores']['max_prob']:.3f}",
            sound_cell,
            style=row_style,
        )

    console.print()
    console.print(table)
    console.print()

    summary_path = (args.output_dir / "summary.json").resolve()
    top5_path = (args.output_dir / "top_5_waves.json").resolve()

    console.print(
        f"[bold green]✓[/bold green] All wave files saved under : [cyan]{waves_dir}[/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] summary.json              : "
        f"[cyan][link=file://{summary_path}]{summary_path}[/link][/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] top_5_waves.json          : "
        f"[cyan][link=file://{top5_path}]{top5_path}[/link][/cyan]"
    )
