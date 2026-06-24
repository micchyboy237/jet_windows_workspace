from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Union

from audio_utils import load_audio
from audio_config import SAMPLE_RATE
from norm_speech_loudness import normalize_audio_for_vad
from rich.console import Console
from speech_waves import (
    DEFAULT_THRESHOLD,
    WaveShapeConfig,
    get_speech_waves,
    build_summary_rows,
    find_parent_segment,
    top5_reports,
    save_wave_data,
)
from rich.table import Table, box

# Create a shared console instance for consistent styling
console = Console()


def save_file(
    data: Any,
    file_path: Union[str, Path],
    *,
    indent: int = 2,
    log_success: bool = True,
) -> Path:
    """
    Save JSON-serializable data to a file with rich-formatted success logging.

    💡 Simple analogy: Like hitting "Save" in a text editor, but with a
    pretty confirmation message that shows exactly where your file went.

    Args:
        data: Any data that can be converted to JSON (dict, list, str, etc.)
        file_path: Where to save the file (string or Path object)
        indent: How many spaces to use for JSON formatting (default: 2)
        log_success: Whether to print a success message (default: True)

    Returns:
        Path: The absolute path of the saved file (useful for chaining or logging)

    Raises:
        OSError: If the file cannot be written (permissions, disk full, etc.)
        TypeError: If data cannot be serialized to JSON

    Example:
        >>> save_file({"name": "test"}, "output/data.json")
        ✓ Saved: /full/path/to/output/data.json
    """
    # 🗂️ Step 1: Normalize the path
    path = Path(file_path)

    # 📁 Step 2: Make sure the folder exists (like mkdir -p)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 💾 Step 3: Write the data as nicely formatted JSON
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    # 🎯 Step 4: Get the full absolute path for the log message
    absolute_path = path.resolve()

    # 🎨 Step 5: Show a pretty success message with rich
    if log_success:
        # Rich lets us make file paths clickable!
        # Format: [link=file:///absolute/path]display text[/link]
        console.print(
            f"[bold green]✓[/bold green] Saved: "
            f"[link=file://{absolute_path}]{absolute_path}[/link]"
        )

    # 🔙 Step 6: Return the path so callers can use it if needed
    return absolute_path


def main():
    import argparse

    from rich import box
    from rich.console import Console
    from rich.table import Table
    from vad_firered import extract_speech_timestamps
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
        default=DEFAULT_THRESHOLD,
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
        parent_seg_num = find_parent_segment(wave, segments)

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
    top5 = top5_reports(speech_waves, waves_dir, segments)
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
    table.add_column("Excursion", style="magenta", justify="right", no_wrap=True)
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
            f"{r['scores']['excursion']:.3f}",
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
        f"[bold green]✓[/bold green] All wave files saved under : "
        f"[cyan][link=file://{waves_dir}]{waves_dir}[/link][/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] summary.json              : "
        f"[cyan][link=file://{summary_path}]{summary_path}[/link][/cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] top_5_waves.json          : "
        f"[cyan][link=file://{top5_path}]{top5_path}[/link][/cyan]"
    )


if __name__ == "__main__":
    main()
