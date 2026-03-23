from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
from audio_utils import load_audio
from joblib import Parallel, delayed
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy.signal import correlate, fftconvolve

console = Console()


class AudioMatchSample(TypedDict):
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float


class AudioMatchResult(TypedDict):
    a_sample: AudioMatchSample
    b_sample: AudioMatchSample
    duration: float
    confidence: float


def _validate_inputs(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
) -> None:
    if long_signal.ndim != 1 or short_signal.ndim != 1:
        raise ValueError("Signals must be 1D mono arrays.")
    if short_signal.size == 0 or long_signal.size == 0:
        raise ValueError("Signals must not be empty.")
    # Removed check: short_signal.size > long_signal.size


def _ensure_search_order(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Ensure the first signal is the longer one for correlation.
    This allows the search logic to work regardless of which file
    is longer without raising errors.
    Returns the (longer, shorter, swapped: bool).
    """
    if signal_b.size > signal_a.size:
        return signal_b, signal_a, True
    return signal_a, signal_b, False


def _compute_normalized_cross_correlation(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Compute normalized cross-correlation using FFT for speed."""
    # Use float64 to avoid precision issues in FFT-based NCC
    long_signal = long_signal.astype(np.float64)
    short_signal = short_signal.astype(np.float64)

    m = short_signal.size
    short_mean = short_signal.mean()
    short_zero = short_signal - short_mean
    short_energy = np.sum(short_zero**2)
    if short_energy == 0:
        return np.zeros(long_signal.size - m + 1, dtype=np.float64)

    if verbose:
        console.rule("Computing normalized cross-correlation (FFT-based)")
        start_t = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]FFT-based NCC...", total=3)

        # Use correlate with fft method → handles flip internally
        numerator = correlate(long_signal, short_zero, mode="valid", method="fft")
        progress.advance(task)

        long_sum = fftconvolve(long_signal, np.ones(m), mode="valid")
        progress.advance(task)

        long_sq_sum = fftconvolve(long_signal**2, np.ones(m), mode="valid")
        progress.advance(task)

    long_mean = long_sum / m
    # Prevent negative energy due to floating-point errors
    long_energy = np.maximum(long_sq_sum - m * (long_mean**2), 0.0)

    denominator = np.sqrt(long_energy * short_energy)
    denominator[denominator == 0] = np.inf

    ncc = numerator / denominator
    ncc = np.clip(ncc, -1.0, 1.0)

    if verbose:
        elapsed = time.perf_counter() - start_t
        console.print(f"[dim]NCC computation took {elapsed:.1f} seconds[/dim]")

    return ncc


def find_audio_offset(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    verbose: bool = True,
    confidence_threshold: float = 0.8,
    tie_break_epsilon: float = 1e-9,
) -> Optional[AudioMatchResult]:
    long_signal, short_signal, swapped = _ensure_search_order(long_signal, short_signal)
    _validate_inputs(long_signal, short_signal)

    ncc = _compute_normalized_cross_correlation(
        long_signal=long_signal,
        short_signal=short_signal,
        verbose=verbose,
    )

    max_score = float(np.max(ncc))

    if max_score < confidence_threshold:
        return None

    # Find all indices close to max score
    candidate_indices = np.where(np.abs(ncc - max_score) <= tie_break_epsilon)[0]

    # Choose earliest match deterministically
    best_index = int(candidate_indices[0])

    start_sample = best_index
    end_sample = best_index + short_signal.size

    long_sample: AudioMatchSample = {
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_time": start_sample / sample_rate,
        "end_time": end_sample / sample_rate,
    }

    short_sample: AudioMatchSample = {
        "start_sample": 0,
        "end_sample": short_signal.size,
        "start_time": 0.0,
        "end_time": short_signal.size / sample_rate,
    }

    if swapped:
        a_sample, b_sample = short_sample, long_sample
    else:
        a_sample, b_sample = long_sample, short_sample

    duration = a_sample["end_time"] - a_sample["start_time"]

    return AudioMatchResult(
        a_sample=a_sample,
        b_sample=b_sample,
        duration=duration,
        confidence=max_score,
    )


def find_audio_offsets(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    verbose: bool = True,
    confidence_threshold: float = 0.8,
    min_distance_samples: Optional[int] = None,
    tie_break_epsilon: float = 1e-9,
) -> list[AudioMatchResult]:
    """
    Find all high-confidence occurrences of short_signal inside long_signal.

    Returns a list of matches sorted by starting position.
    Uses greedy non-maximum suppression to avoid reporting near-duplicate/strongly overlapping detections.
    """
    long_signal, short_signal, swapped = _ensure_search_order(long_signal, short_signal)
    _validate_inputs(long_signal, short_signal)

    if min_distance_samples is None:
        # Allow very small overlaps only (e.g. 50 samples), forbid most near-duplicates
        min_distance_samples = short_signal.size - 50

    ncc = _compute_normalized_cross_correlation(
        long_signal=long_signal,
        short_signal=short_signal,
        verbose=verbose,
    )

    # Find all positions above threshold
    candidate_mask = (
        ncc >= confidence_threshold - tie_break_epsilon
    )  # slight relaxation for ties
    if not np.any(candidate_mask):
        return []

    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_scores = ncc[candidate_indices]

    # Sort candidates by descending score (best first)
    sorted_order = np.argsort(-candidate_scores)
    sorted_candidates = candidate_indices[sorted_order]
    sorted_scores = candidate_scores[sorted_order]

    selected: list[int] = []
    used = np.zeros(len(ncc), dtype=bool)

    # Greedy NMS: take best remaining, suppress neighborhood
    for idx, score in zip(sorted_candidates, sorted_scores):
        if used[idx]:
            continue

        selected.append(idx)

        # Suppress nearby peaks (symmetric window)
        start_suppress = max(0, idx - min_distance_samples)
        end_suppress = min(len(ncc), idx + min_distance_samples + 1)
        used[start_suppress:end_suppress] = True

    # Build final results sorted by start position
    results: list[AudioMatchResult] = []

    for start in sorted(selected):
        score = float(ncc[start])

        long_sample: AudioMatchSample = {
            "start_sample": start,
            "end_sample": start + short_signal.size,
            "start_time": start / sample_rate,
            "end_time": (start + short_signal.size) / sample_rate,
        }

        short_sample: AudioMatchSample = {
            "start_sample": 0,
            "end_sample": short_signal.size,
            "start_time": 0.0,
            "end_time": short_signal.size / sample_rate,
        }

        if swapped:
            a_sample, b_sample = short_sample, long_sample
        else:
            a_sample, b_sample = long_sample, short_sample

        duration = a_sample["end_time"] - a_sample["start_time"]

        results.append(
            AudioMatchResult(
                a_sample=a_sample,
                b_sample=b_sample,
                duration=duration,
                confidence=score,
            )
        )

    return results


def _process_partial_subclip(
    sub_clip: np.ndarray,
    sub_start: int,
    long_signal: np.ndarray,
    sample_rate: int,
    confidence_threshold: float,
    min_distance_samples: int | None,
    tie_break_epsilon: float,
):
    """
    Worker function for joblib. Must be top-level so it can be pickled.
    """
    matches = find_audio_offsets(
        long_signal=long_signal,
        short_signal=sub_clip,
        sample_rate=sample_rate,
        verbose=False,
        confidence_threshold=confidence_threshold,
        min_distance_samples=min_distance_samples,
        tie_break_epsilon=tie_break_epsilon,
    )

    return [(m, sub_start) for m in matches]


def extract_sliding_subsignals(
    signal: np.ndarray,
    min_length: int,
    max_length: int,
    step: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """
    Generate overlapping sub-signals from signal[start:start+length].
    Returns list of (sub_signal, original_start_idx)
    """
    if step is None:
        step = max(1, (max_length - min_length) // 8)
    subs = []
    for length in range(min_length, max_length + 1, step):
        for start in range(0, signal.size - length + 1, step):
            subs.append((signal[start : start + length], start))
    return subs


def find_partial_audio_matches(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    verbose: bool = True,
    confidence_threshold: float = 0.75,  # usually lower for partial
    min_match_fraction: float = 0.5,
    max_match_fraction: float = 1.0,
    length_step_fraction: float = 0.1,
    min_distance_samples: int | None = None,
    tie_break_epsilon: float = 1e-9,
    max_subclips: int | None = 60,
) -> list["AudioMatchResult"]:
    """
    Find partial matches by trying multiple sub-clips of short_signal.
    Returns sorted list of best non-overlapping partial matches.
    """
    long_signal, short_signal, _ = _ensure_search_order(long_signal, short_signal)
    _validate_inputs(long_signal, short_signal)

    min_len = max(64, int(len(short_signal) * min_match_fraction))
    max_len = int(len(short_signal) * max_match_fraction)
    step = max(1, int(len(short_signal) * length_step_fraction))

    sub_clips = extract_sliding_subsignals(short_signal, min_len, max_len, step=step)

    # Optional: early exit if full-length match is very confident
    full_matches = find_audio_offsets(
        long_signal,
        short_signal,
        sample_rate,
        verbose=False,
        confidence_threshold=0.90,
        min_distance_samples=min_distance_samples,
    )
    if full_matches and max(m["confidence"] for m in full_matches) >= 0.92:
        # Return full_matches directly (already in AudioMatchResult/dict-like form)
        return full_matches

    # Use new picklable worker
    if max_subclips is not None and len(sub_clips) > max_subclips:
        sub_clips = sub_clips[:max_subclips]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_process_partial_subclip)(
            clip,
            start,
            long_signal,
            sample_rate,
            confidence_threshold,
            min_distance_samples,
            tie_break_epsilon,
        )
        for clip, start in sub_clips
    )
    all_matches: list[tuple["AudioMatchResult", int]] = [
        item for sublist in results for item in sublist
    ]

    # Sort by start position, then prefer longer + higher conf
    all_matches.sort(
        key=lambda x: (
            x[0]["a_sample"]["start_sample"],
            -x[0]["a_sample"]["end_sample"] + x[0]["a_sample"]["start_sample"],
            -x[0]["confidence"],
        )
    )

    # Simple greedy non-overlap suppression (can be refined)
    selected: list["AudioMatchResult"] = []
    used_ranges = []

    for match, sub_start in all_matches:
        s = match["a_sample"]["start_sample"]
        e = match["a_sample"]["end_sample"]
        if any(s < ue and e > us for us, ue in used_ranges):
            continue
        selected.append(match)
        used_ranges.append((s, e))

    return sorted(selected, key=lambda x: x["a_sample"]["start_sample"])


from pathlib import Path
import argparse
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def search_audio(audio1, audio2, **kwargs):
    try:
        long_signal, sr_long = load_audio(audio1)
        short_signal, sr_short = load_audio(audio2)

        if sr_long != sr_short:
            console.print(
                f"[red]Warning:[/red] Sample rates differ"
                # f"\n({sr_long} Hz vs {sr_short} Hz). Using {sr_long} Hz for timing."
            )

        sample_rate = sr_long

        # ── Total durations ───────────────────────────────────────
        total_long_sec  = len(long_signal)  / sample_rate
        total_short_sec = len(short_signal) / sample_rate

        console.print("\n[bold cyan]Audio Files:[/bold cyan]")
        console.print(f"  Long audio (A)   :  [bold]{total_long_sec:8.2f}s[/bold]")
        console.print(f"  Short clip  (B)  :  [bold]{total_short_sec:8.2f}s[/bold]")
        console.print()

        console.rule("Searching for partial matches")

        matches = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=sample_rate,
            verbose=True,
            confidence_threshold=kwargs.get("threshold", 0.75),
            min_match_fraction=kwargs.get("min_fraction", 0.50),
            length_step_fraction=0.18 if kwargs.get("quick", False) else 0.10,
            max_subclips=35 if kwargs.get("quick", False) else 80,
        )

        if not matches:
            console.print(
                f"[bold red]No partial matches found[/bold red] above confidence {kwargs.get('threshold', 0.75):.2f} "
                f"and min length fraction {kwargs.get('min_fraction', 0.50):.2f}"
            )
            console.print("[dim]Try lowering --threshold or --min-fraction[/dim]")
            return

        # ───────────────────────────────────────────────────────────
        #               Result Display
        # ───────────────────────────────────────────────────────────
        for i, m in enumerate(matches, 1):
            a = m["a_sample"]
            b = m["b_sample"]
            match_duration = m["duration"]
            confidence = m["confidence"]

            percent_of_A = (match_duration / total_long_sec * 100) if total_long_sec > 0 else 0
            percent_of_B = (match_duration / total_short_sec * 100) if total_short_sec > 0 else 0

            # Header for this match
            console.print(f"\n[bold green]Match {i}   Confidence: {confidence:.4f}[/bold green]")

            console.print(
                f"  Matched segment duration: [bold yellow]{match_duration:6.2f} seconds[/bold yellow]\n"
                f"  • Covers [bold]{percent_of_A:5.1f}%[/bold] of long audio (A)\n"
                f"  • Covers [bold]{percent_of_B:5.1f}%[/bold] of short clip (B)"
            )

            # Table with total durations included
            table = Table(show_header=True, header_style="bold magenta", show_lines=True, border_style="dim")
            table.add_column("Signal",     justify="center", width=12)
            table.add_column("Start (s)",  justify="right")
            table.add_column("End (s)",    justify="right")
            table.add_column("Total (s)",  justify="right", style="cyan")
            table.add_column("% of Total", justify="right", style="bright_cyan")

            # Row for long audio (A)
            table.add_row(
                "[bold]A (long)[/bold]",
                f"{a['start_time']:.2f}",
                f"{a['end_time']:.2f}",
                f"{total_long_sec:6.2f}",
                f"{percent_of_A:5.1f}%"
            )

            # Row for short clip (B)
            table.add_row(
                "B (clip)",
                f"{b['start_time']:.2f}",
                f"{b['end_time']:.2f}",
                f"{total_short_sec:6.2f}",
                f"{percent_of_B:5.1f}%"
            )

            console.print(table)

        # Final summary
        match_word = "match" if len(matches) == 1 else "matches"
        console.print(
            f"\n[italic dim]{len(matches)} partial {match_word} found "
            f"(conf ≥ {kwargs.get('threshold', 0.75):.2f}, matched ≥ {kwargs.get('min_fraction', 0.50):.0%} of short clip)[/italic dim]"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}", style="red")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Find partial occurrences of a short audio clip inside a longer audio file.\n"
                    "Only some segments of the short clip may match.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("long_audio", type=str, help="Path to long audio file (WAV)")
    parser.add_argument("short_clip", type=str, help="Path to short clip to search for (WAV)")
    parser.add_argument(
        "-t", "--threshold",
        type=float, default=0.75,
        help="Minimum confidence threshold for partial matches (default: 0.75)"
    )
    parser.add_argument(
        "-m", "--min-fraction",
        type=float, default=0.50, metavar="FRAC",
        help="Minimum fraction of the short clip that must match (default: 0.50)"
    )
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Faster mode: coarser steps, fewer sub-clips"
    )

    args = parser.parse_args()

    # Convert paths (optional but cleaner)
    long_path = Path(args.long_audio)
    short_path = Path(args.short_clip)

    # Pass all relevant arguments as kwargs
    search_audio(
        long_path,
        short_path,
        threshold=args.threshold,
        min_fraction=args.min_fraction,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()