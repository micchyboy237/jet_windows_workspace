from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Union
from audio_utils import load_audio
from audio_config import SAMPLE_RATE
from norm_speech_loudness import normalize_audio_for_vad
from rich.console import Console
from rich.table import Table, box
from speech_waves import (
    WaveShapeConfig,
    get_speech_waves,
    build_summary_rows,
    find_parent_segment,
    top5_reports,
    save_wave_data,
    DEFAULT_THRESHOLD,
    DEFAULT_MIN_PROMINENCE,
    DEFAULT_MIN_EXCURSION,
    DEFAULT_MIN_PEAK_PROB,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_BASELINE_THRESHOLD,
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_MIN_SILENCE_DURATION_MS,
)
from vad_firered import extract_speech_timestamps

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
)


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
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)
    absolute_path = path.resolve()
    if log_success:
        console.print(
            f"[bold green]✓[/bold green] Saved: "
            f"[link=file://{absolute_path}]{absolute_path}[/link]"
        )
    return absolute_path


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract and analyse speech waves from audio using FireRedVAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument(
        "-d",
        "--min-speech-duration",
        type=int,
        default=DEFAULT_MIN_SPEECH_DURATION_MS,
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
        default=DEFAULT_MIN_SILENCE_DURATION_MS,
        metavar="MS",
        help="Minimum silence gap between segments in ms.",
    )
    parser.add_argument(
        "-ns",
        "--include-non-speech",
        action="store_true",
        help="Include non-speech segments in the VAD output.",
    )
    parser.add_argument(
        "-p",
        "--min-prominence",
        type=float,
        default=DEFAULT_MIN_PROMINENCE,
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
        default=DEFAULT_MIN_EXCURSION,
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
        default=DEFAULT_MIN_PEAK_PROB,
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
        default=DEFAULT_MIN_FRAMES,
        metavar="N",
        help="Minimum number of frames a wave must span.",
    )
    parser.add_argument(
        "-b",
        "--baseline-threshold",
        type=float,
        default=DEFAULT_BASELINE_THRESHOLD,
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
    console.print(f"[dim]Parsed arguments: {vars(args)}[/dim]")
    return args


def main():
    args = get_args()

    # Use all default variables for WaveShapeConfig — no overrides from CLI needed
    # since CLI args already default to the same constants.
    shape_cfg = WaveShapeConfig(
        min_prominence=args.min_prominence,
        min_excursion=args.min_excursion,
        min_peak_prob=args.min_peak_prob,
        min_frames=args.min_frames,
        # min_duration_sec=DEFAULT_MIN_DURATION_SEC,  # Use the dedicated default, not derived from VAD ms
        min_duration_sec=args.min_speech_duration / 1000,
        baseline_threshold=args.baseline_threshold,
    )
    console.print(f"[dim]WaveShapeConfig: {shape_cfg}[/dim]")

    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    audio_np, sr = load_audio(args.input, sr=SAMPLE_RATE, mono=True)
    console.print(f"[dim]Loaded audio: shape={audio_np.shape}, sr={sr}[/dim]")

    if args.normalize:
        audio_np_norm, vad_stats = normalize_audio_for_vad(audio_np, sr)
        audio_np = audio_np_norm
        console.print(f"[dim]Audio normalized. VAD stats: {vad_stats}[/dim]")

    segments, scores = extract_speech_timestamps(
        audio=audio_np,
        include_non_speech=args.include_non_speech,
        threshold=args.threshold,
        min_speech_duration_sec=args.min_speech_duration / 1000,
        min_silence_duration_sec=args.min_silence_duration / 1000,
        with_scores=True,
    )
    console.print(f"[dim]VAD produced {len(segments)} segments, {len(scores)} scores[/dim]")

    speech_waves = get_speech_waves(
        args.input,
        scores,
        threshold=args.threshold,
        sampling_rate=sr,
        shape_cfg=shape_cfg,
    )
    console.print(f"[dim]Detected {len(speech_waves)} valid speech waves[/dim]")

    save_file(segments, args.output_dir / "segments.json")
    save_file(scores, args.output_dir / "speech_probs.json")
    save_file(speech_waves, args.output_dir / "speech_waves.json")

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

    rows = build_summary_rows(speech_waves, waves_dir, segments)
    save_file(rows, args.output_dir / "summary.json")

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
