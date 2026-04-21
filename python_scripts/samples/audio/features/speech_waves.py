from __future__ import annotations

import dataclasses
import json
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
    """

    min_prominence: float = 0.05
    min_excursion: float = 0.04
    min_peak_prob: float = 0.55
    min_frames: int = 3


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
    """
    Analyze speech probabilities from FireRedVAD and return complete wave
    metadata. Updated for 10ms hop length (HOP_SIZE samples per frame).

    Now uses prominence-based shape validation so flat plateaus (low excursion,
    low prominence) are rejected even when all frames exceed the threshold.
    """
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()

    if not speech_probs:
        return []

    waves: List[SpeechWave] = []
    current_wave: SpeechWave | None = None
    state: WaveState = "below"
    rise_frame_idx: int | None = None

    # Handle case where probabilities start already above threshold
    if speech_probs and speech_probs[0] >= threshold:
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
            },
        )
        state = "above"

    for i, prob in enumerate(speech_probs):
        # Frame time in seconds
        frame_time_sec = i * HOP_SIZE / sampling_rate

        if state == "below":
            if prob >= threshold:
                rise_frame_idx = i
                current_wave = SpeechWave(
                    has_risen=True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=frame_time_sec,
                    end_sec=frame_time_sec,
                    details={
                        "frame_start": i,
                        "frame_end": i,
                        "frame_len": 0,
                        "duration_sec": 0.0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "avg_prob": prob,
                        "std_prob": 0.0,
                    },
                )
                state = "above"

        else:  # state == "above"
            if prob >= threshold:
                if current_wave is not None:
                    current_wave["has_multi_passed"] = True
            else:
                if current_wave is not None:
                    current_wave["has_fallen"] = True
                    # ------ shape-based validation ------------
                    frame_start = rise_frame_idx if rise_frame_idx is not None else 0
                    frame_end = i
                    wave_probs = speech_probs[frame_start:frame_end]
                    frame_len = frame_end - frame_start

                    # Entry/Exit for baseline
                    entry_prob = (
                        speech_probs[frame_start - 1] if frame_start > 0 else 0.0
                    )
                    exit_prob = prob  # dropped below threshold
                    shape_ok, shape_diag = is_prominent_wave(
                        wave_probs, entry_prob, exit_prob, shape_cfg
                    )

                    current_wave["is_valid"] = (
                        current_wave["has_risen"]
                        and current_wave["has_multi_passed"]
                        and shape_ok
                    )
                    current_wave["end_sec"] = frame_time_sec

                    # Finalize details for complete wave
                    current_wave["details"] = {
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "frame_len": frame_len,
                        "duration_sec": current_wave["end_sec"]
                        - current_wave["start_sec"],
                        "min_prob": min(wave_probs) if wave_probs else 0.0,
                        "max_prob": max(wave_probs) if wave_probs else 0.0,
                        "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                        "std_prob": statistics.stdev(wave_probs)
                        if frame_len > 1
                        else 0.0,
                        **shape_diag,
                    }
                    waves.append(current_wave)

                current_wave = None
                rise_frame_idx = None
                state = "below"

    # Handle unfinished wave at the end of the sequence
    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False  # incomplete waves are never valid
        current_wave["end_sec"] = len(speech_probs) * HOP_SIZE / sampling_rate

        if rise_frame_idx is not None:
            frame_start = rise_frame_idx
            frame_end = len(speech_probs)
            wave_probs = speech_probs[frame_start:frame_end]
            frame_len = frame_end - frame_start

            entry_prob = speech_probs[frame_start - 1] if frame_start > 0 else 0.0
            # If it never fell, threshold as proxy for exit prob
            exit_prob = threshold
            shape_ok, shape_diag = is_prominent_wave(
                wave_probs, entry_prob, exit_prob, shape_cfg
            )

            current_wave["details"] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_len": frame_len,
                "duration_sec": current_wave["end_sec"] - current_wave["start_sec"],
                "min_prob": min(wave_probs) if wave_probs else 0.0,
                "max_prob": max(wave_probs) if wave_probs else 0.0,
                "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
                **shape_diag,
            }
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


def save_wave_plot(
    probs: List[float],
    rms_values: List[float],
    output_path: Path,
    wave_num: int,
    seg_num: int,
) -> None:
    """Create visualization plot for wave probabilities and energy.
    Handles potential length mismatches between probs and rms_values."""

    # Ensure arrays have the same length by taking the minimum length
    min_length = min(len(probs), len(rms_values))
    probs_aligned = probs[:min_length]
    rms_aligned = rms_values[:min_length]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    frames = np.arange(min_length)

    # Plot probabilities
    ax1.plot(frames, probs_aligned, color="blue", linewidth=1)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax1.set_ylabel("VAD Probability")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Segment {seg_num:03d} - Wave {wave_num:03d} (Valid: {wave_num})")
    ax1.legend()

    # Plot RMS energy
    ax2.plot(frames, rms_aligned, color="green", linewidth=1)
    ax2.set_xlabel("Frame Index (relative to wave)")
    ax2.set_ylabel("RMS Energy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
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

    # Create and save visualization
    plot_path = wave_dir / "wave_plot.png"
    save_wave_plot(wave_probs, rms_values, plot_path, wave_num, seg_num)


# ── Reporting helpers ──


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
    frame_start = wave["details"]["frame_start"]
    parent_seg_num = 0
    for seg in segments:
        if seg["frame_start"] <= frame_start <= seg["frame_end"]:
            parent_seg_num = seg["num"]
            break

    dir_name = f"segment_{parent_seg_num:03d}_wave_{wave_idx:03d}"
    wav_abs = (waves_dir / dir_name / "sound.wav").resolve()
    plot_abs = (waves_dir / dir_name / "wave_plot.png").resolve()
    short = _shorten_path(str(wav_abs))

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
        "sound_short": short,
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
        },
    }


def _top5_reports(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
    duration_weight: float = 0.5,
) -> list[dict]:
    """
    Return the 5 waves with the highest composite score, already serialised
    as report dicts (not raw SpeechWave objects).
    Composite score = prominence * log(1 + duration_sec * duration_weight)
    This rewards waves that are both prominent and long, while the log scale
    prevents very long but flat waves from dominating short, sharp ones.
    Set duration_weight=0 to rank by prominence only (legacy behaviour).
    """
    import math
    indexed = list(enumerate(speech_waves, 1))  # [(1, wave), (2, wave), …]
    def _composite(wave):
        d = wave["details"]
        prominence = d.get("prominence", d["max_prob"])
        duration_sec = d.get("duration_sec", 0.0)
        return prominence * math.log1p(duration_sec * duration_weight)
    ranked = sorted(indexed, key=lambda iv: _composite(iv[1]), reverse=True)
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


def _shorten_path(path_str: str) -> str:
    """
    Show only the last 2 components of a path to keep the table columns narrow.
    E.g. segment_001_wave_003/sound.wav
    """
    parts = Path(path_str).parts
    if len(parts) <= 2:
        return path_str
    return "/".join(parts[-2:])


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
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    DEFAULT_AUDIO = (
        r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
    )
    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Input audio file path (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output results dir (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="VAD probability threshold"
    )
    parser.add_argument(
        "-s", "--hop-size", type=int, default=160, help="Frame hop size in samples"
    )
    parser.add_argument(
        "--min-speech-duration",
        "-d",
        type=int,
        default=250,
        help="Minimum speech segment duration in ms",
    )
    parser.add_argument(
        "--min-silence-duration",
        "-g",
        type=int,
        default=100,
        help="Minimum silence duration in ms",
    )
    parser.add_argument(
        "--include-non-speech",
        "-n",
        action="store_true",
        help="Include non-speech segments",
    )
    args = parser.parse_args()

    # segments, scores = extract_speech_timestamps(
    #     audio=args.input,
    #     include_non_speech=args.include_non_speech,
    #     hop_size=args.hop_size,
    #     threshold=args.threshold,
    #     min_speech_duration_ms=args.min_speech_duration,
    #     min_silence_duration_ms=args.min_silence_duration,
    #     with_scores=True,
    # )
    segments, scores = extract_speech_timestamps(
        audio=args.input,
        include_non_speech=args.include_non_speech,
        threshold=args.threshold,
        min_speech_duration_sec=args.min_speech_duration / 1000,
        min_silence_duration_sec=args.min_silence_duration / 1000,
        # max_speech_duration_sec
        with_scores=True,
    )

    # Load audio for wave extraction
    audio_np, sr = load_audio(args.input, sr=SAMPLE_RATE, mono=True)

    speech_waves = get_speech_waves(args.input, scores, threshold=args.threshold)

    # Save main JSON files
    save_file(segments, OUTPUT_DIR / "segments.json")
    save_file(scores, OUTPUT_DIR / "speech_probs.json")
    save_file(speech_waves, OUTPUT_DIR / "speech_waves.json")

    # Create waves directory and save individual wave files
    waves_dir = OUTPUT_DIR / "waves"
    waves_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold]Generating files for {len(speech_waves)} valid speech waves...[/bold]"
    )

    for wave_idx, wave in enumerate(speech_waves, 1):
        wave_frame_start = wave["details"]["frame_start"]

        parent_seg_num = 1
        for seg in segments:
            if seg["frame_start"] <= wave_frame_start <= seg["frame_end"]:
                parent_seg_num = seg["num"]
                break

        save_wave_data(
            wave=wave,
            audio_np=audio_np,
            speech_probs=scores,
            sampling_rate=sr,
            output_dir=waves_dir,
            seg_num=parent_seg_num,
            wave_num=wave_idx,
            hop_size=args.hop_size,
        )

    # ── summary table & JSON ──────────────────────────────────────────────────
    rows = build_summary_rows(speech_waves, waves_dir, segments)
    save_file(rows, OUTPUT_DIR / "summary.json")

    # ── top-5 waves (built after waves_dir exists and dirs are known) ─────────
    top5 = _top5_reports(speech_waves, waves_dir, segments)
    save_file(top5, OUTPUT_DIR / "top_5_waves.json")

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
    table.add_column("Peak prob", style="green", justify="right", no_wrap=True)
    table.add_column("Sound", style="bright_black", justify="left")

    top5_dirs = {w["dir"] for w in top5}

    for r in rows:
        is_top5 = r["dir"] in top5_dirs
        row_style = "bold" if is_top5 else ""
        star = "★ " if is_top5 else "  "

        dir_cell = f"[link=file://{r['plot_path']}]{r['dir']}[/link]"
        sound_cell = f"[link=file://{r['sound_path']}]{r['sound_short']}[/link]"

        table.add_row(
            f"{star}{r['wave']}",
            dir_cell,
            f"{r['start_sec']:.2f}",
            f"{r['end_sec']:.2f}",
            f"{r['dur_sec']:.2f}",
            f"{r['scores']['prominence']:.3f}",
            f"{r['scores']['max_prob']:.3f}",
            sound_cell,
            style=row_style,
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[bold green]✓[/bold green] All wave files saved under : [cyan]{waves_dir}[/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] summary.json              : [cyan][link=file://{(OUTPUT_DIR / 'summary.json').resolve()}]{(OUTPUT_DIR / 'summary.json').resolve()}[/link][/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] top_5_waves.json          : [cyan][link=file://{(OUTPUT_DIR / 'top_5_waves.json').resolve()}]{(OUTPUT_DIR / 'top_5_waves.json').resolve()}[/link][/cyan]"
    )
