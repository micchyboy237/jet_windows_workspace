import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import box
from nemo_titanet_with_outliers import (
    detect_multi_speakers,
    MultiSpeakerResult,
    TimelineSegment,
)

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_AUDIO = str(Path(r"~\.cache\files\audio\recording_3_speakers.wav").expanduser().resolve())
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
SPEAKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
RICH_SPEAKER_COLORS = [
    "blue", "dark_orange", "green", "red",
    "magenta", "gold3", "plum3", "grey70",
]
BAR_WIDTH = 25

def get_args():
    parser = argparse.ArgumentParser(
        description="Automatic speaker labeling with NeMo TitaNet-Large embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_path", type=str, nargs="?", default=DEFAULT_AUDIO,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--model-name", type=str, default="titanet_large",
        help="NeMo pretrained speaker embedding model name"
    )
    parser.add_argument(
        "-o", "--output-dir", default=str(OUTPUT_DIR), type=Path,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=2.0,
        help="Window duration in seconds for embedding extraction"
    )
    parser.add_argument(
        "-s", "--step", type=float, default=0.75,
        help="Window step in seconds for sliding window"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=16,
        help="Number of windows embedded per forward pass"
    )
    parser.add_argument(
        "-e", "--min-energy-percentile", type=float, default=15.0,
        help="Skip the quietest N%% of windows before embedding. 0 disables."
    )
    parser.add_argument(
        "-m", "--min-segment-duration", type=float, default=1.0,
        help="Minimum duration in seconds for a speaker segment"
    )
    parser.add_argument(
        "-c", "--clustering-method", type=str, choices=["agglomerative", "spectral"],
        default="agglomerative", help="Clustering method for speaker grouping"
    )
    parser.add_argument(
        "-t", "--merge-threshold", type=float, default=0.55,
        help="Similarity threshold for merging speaker clusters"
    )
    parser.add_argument(
        "-a", "--assign-threshold", type=float, default=0.55,
        help="Minimum similarity threshold for assigning a frame to a speaker"
    )
    parser.add_argument(
        "--outlier-method",
        type=str,
        choices=["mahalanobis", "zscore", "isolation_forest", "dbscan"],
        default=None,
        help="Method for outlier detection (default: None)",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.99,
        help="Threshold for outlier detection (default: 0.99)",
    )
    args = parser.parse_args()
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args

def print_results(results: MultiSpeakerResult, output_dir: Path, assign_threshold: float = 0.55) -> None:
    """Pretty-print a MultiSpeakerResult using Rich console with clickable file:// links."""
    n_speakers = results["n_speakers"]
    speaker_stats = results["speaker_stats"]
    centroids = results["centroids"]
    confidences = results["confidences"]
    timeline = results["timeline"]
    outlier_mask = results.get("outlier_mask", None)

    console.print()
    console.rule("[bold green]✅ FINAL RESULT")
    console.print(f"[bold]{n_speakers} SPEAKERS DETECTED[/bold]")
    console.rule()

    if outlier_mask is not None:
        console.print(f"\n[bold]🚨 OUTLIER DETECTION[/bold]")
        console.print(f"Outliers detected: {np.sum(outlier_mask)} / {len(outlier_mask)} windows")

    console.print("\n[bold]📊 SPEAKER STATISTICS[/bold]")
    stats_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    stats_table.add_column("Speaker", style="bold", width=12)
    stats_table.add_column("Duration", justify="right", width=12)
    stats_table.add_column("Frames", justify="right", width=8)
    stats_table.add_column("Consistency", justify="right", width=16)
    stats_table.add_column("Quality", width=10)

    sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['duration'], reverse=True)
    for i, (speaker_id, stats) in enumerate(sorted_speakers):
        quality_emoji = "✅" if stats['quality'] == 'good' else "⚠️"
        speaker_label = chr(65 + i)
        color = RICH_SPEAKER_COLORS[speaker_id % len(RICH_SPEAKER_COLORS)]
        stats_table.add_row(
            f"[{color}]{speaker_label}[/{color}]",
            f"{stats['duration']:.1f}s ({stats['frames_percent']:.1f}%)",
            str(stats['n_frames']),
            f"{stats['avg_similarity']:.3f} ± {stats['std_similarity']:.3f}",
            f"{quality_emoji} {stats['quality'].upper()}",
        )
    console.print(stats_table)

    console.print("\n[bold]📅 SPEAKER TIMELINE[/bold]")
    console.print("[dim]Ctrl+Click ▶️ to play audio  |  Ctrl+Click 📁 to open segment folder[/dim]\n")
    tl_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    tl_table.add_column("#", justify="right", style="dim", width=4)
    tl_table.add_column("Start → End", width=16)
    tl_table.add_column("Speaker", width=11)
    tl_table.add_column("Dur", justify="right", width=6)
    tl_table.add_column("Bar", width=BAR_WIDTH + 2)
    tl_table.add_column("", width=8)

    for idx, seg in enumerate(timeline):
        bar_fill = int(seg['duration'] / 32 * BAR_WIDTH)
        bar_full = min(bar_fill, BAR_WIDTH)
        bar_empty = max(0, BAR_WIDTH - bar_full)
        color = RICH_SPEAKER_COLORS[seg["speaker_id"] % len(RICH_SPEAKER_COLORS)]
        hex_color = SPEAKER_COLORS[seg["speaker_id"] % len(SPEAKER_COLORS)]
        bar_text = Text()
        bar_text.append("█" * bar_full, style=color)
        bar_text.append("░" * bar_empty, style="dim")
        seg_dir = f"segments/seg_{idx:03d}"
        wav_abs = (output_dir / seg_dir / "sound.wav").resolve().as_uri()
        folder_abs = (output_dir / seg_dir).resolve().as_uri()
        icons = Text()
        icons.append("▶️", style=f"link {wav_abs}")
        icons.append("  ", style="")
        icons.append("📁", style=f"link {folder_abs}")
        tl_table.add_row(
            f"{idx:03d}",
            f"{seg['start']:5.1f}s → {seg['end']:5.1f}s",
            f"[{color}]{seg['speaker_label']}[/{color}]",
            f"{seg['duration']:.1f}s",
            bar_text,
            icons,
        )
    console.print(tl_table)

    if len(centroids) >= 2:
        console.print("\n[bold]🔍 SPEAKER SEPARATION QUALITY[/bold]")
        speaker_list = list(centroids.keys())
        between_sims = []
        for i, sp1 in enumerate(speaker_list):
            for sp2 in speaker_list[i + 1:]:
                sim = np.dot(centroids[sp1], centroids[sp2])
                between_sims.append(sim)
        avg_between = np.mean(between_sims) if between_sims else 0
        avg_intra = np.mean([stats['avg_similarity'] for stats in speaker_stats.values()])
        margin = avg_intra - avg_between
        sep_table = Table(box=box.SIMPLE, show_header=False)
        sep_table.add_column("Metric", style="bold")
        sep_table.add_column("Value")
        sep_table.add_row("Intra-speaker similarity", f"{avg_intra:.3f}")
        sep_table.add_row("Between-speaker similarity", f"{avg_between:.3f}")
        sep_table.add_row("Separation margin", f"{margin:.3f}")
        console.print(sep_table)
        if margin > 0.3:
            console.print("   ✅ [green]EXCELLENT separation[/green] - speakers are very distinct")
        elif margin > 0.2:
            console.print("   ✅ [green]GOOD separation[/green] - speakers are distinguishable")
        elif margin > 0.1:
            console.print("   ⚠️  [yellow]MODERATE separation[/yellow] - some confusion possible")
        else:
            console.print("   ❌ [red]POOR separation[/red] - speakers sound similar")

    console.print("\n[bold]💡 FINAL ASSESSMENT[/bold]")
    at_threshold_pct = np.sum(confidences >= assign_threshold) / len(confidences) * 100
    strict_pct = np.sum(confidences > 0.7) / len(confidences) * 100
    assess_table = Table(box=box.SIMPLE, show_header=False)
    assess_table.add_column("Metric", style="bold")
    assess_table.add_column("Value")
    assess_table.add_row("Speakers detected", str(n_speakers))
    assess_table.add_row(
        f"Frames meeting threshold (≥{assign_threshold})",
        f"{at_threshold_pct:.1f}%"
    )
    assess_table.add_row(
        "Frames high-confidence (>0.7)",
        f"{strict_pct:.1f}%"
    )
    console.print(assess_table)

def save_outputs(results: MultiSpeakerResult, audio_path: str, output_dir: Path) -> None:
    """Save speaker labeling results to structured output."""
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    timeline = results["timeline"]
    outlier_mask = results.get("outlier_mask", None)

    logger.info(f"Saving {len(timeline)} segments to {segments_dir}")

    if outlier_mask is not None:
        outlier_dir = output_dir / "outliers"
        outlier_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving outlier information to {outlier_dir}")
        outlier_info = {
            "total_windows": len(outlier_mask),
            "outliers": int(np.sum(outlier_mask)),
            "outlier_indices": [i for i, is_outlier in enumerate(outlier_mask) if is_outlier],
        }
        outlier_path = outlier_dir / "outliers.json"
        outlier_path.write_text(json.dumps(outlier_info, indent=2))
        logger.info(f"Saved outlier info: {outlier_path}")

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if not ffmpeg_available:
        logger.warning(
            "ffmpeg not found in PATH – segment .wav files will NOT be extracted. "
            "Install ffmpeg or add it to your PATH."
        )
    for idx, seg in enumerate(timeline):
        seg_dir = segments_dir / f"seg_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        wav_path = seg_dir / "sound.wav"
        if ffmpeg_available:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(seg["start"]),
                "-t", str(seg["duration"]),
                "-i", audio_path,
                "-c", "copy",
                str(wav_path),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=30)
                logger.debug(f"  Extracted {wav_path} ({seg['duration']:.1f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"  ffmpeg failed for seg_{idx:03d}: {e}")
            except subprocess.TimeoutExpired:
                logger.error(f"  ffmpeg timed out for seg_{idx:03d}")
        else:
            wav_path.touch()
            logger.debug(f"  Placeholder created: {wav_path}")
        segment_info = {
            "segment_index": idx,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "speaker_id": seg["speaker_id"],
            "speaker_label": seg["speaker_label"],
            "audio_file": str(wav_path.relative_to(output_dir)),
        }
        info_path = seg_dir / "segment_info.json"
        info_path.write_text(json.dumps(segment_info, indent=2))
        logger.debug(f"  Saved {info_path}")

    global_summary = {
        "audio_source": str(audio_path),
        "n_speakers": results["n_speakers"],
        "total_segments": len(timeline),
        "total_duration": sum(seg["duration"] for seg in timeline),
        "segments": timeline,
        "speaker_stats": {
            str(k): v for k, v in results["speaker_stats"].items()
        },
        "outliers": {
            "total_windows": len(outlier_mask),
            "outliers": int(np.sum(outlier_mask)),
        } if outlier_mask is not None else None,
    }
    summary_path = output_dir / "segments.json"
    summary_path.write_text(json.dumps(global_summary, indent=2, default=str))
    logger.info(f"Saved global summary: {summary_path}")

    plot_path = output_dir / "segments_plot.png"
    _plot_timeline(timeline, results["n_speakers"], plot_path)
    logger.info(f"Saved timeline plot: {plot_path}")

    html_path = output_dir / "segments_timeline.html"
    _save_html_timeline(timeline, results["n_speakers"], output_dir, html_path)
    logger.info(f"Saved HTML timeline: {html_path}")

def _plot_timeline(
    timeline: list[TimelineSegment],
    n_speakers: int,
    save_path: Path,
) -> None:
    """Generate a horizontal Gantt-style timeline plot of speaker segments."""
    fig, ax = plt.subplots(figsize=(14, max(3, n_speakers * 0.8)))
    seen: dict[int, str] = {}
    for seg in timeline:
        sid = seg["speaker_id"]
        if sid not in seen:
            seen[sid] = seg["speaker_label"]
    speaker_labels = [seen[sid] for sid in sorted(seen.keys())]
    label_to_y = {label: i for i, label in enumerate(speaker_labels)}
    for seg in timeline:
        y = label_to_y[seg["speaker_label"]]
        color = SPEAKER_COLORS[seg["speaker_id"] % len(SPEAKER_COLORS)]
        ax.barh(
            y,
            width=seg["duration"],
            left=seg["start"],
            height=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if seg["duration"] > 1.5:
            ax.text(
                seg["start"] + seg["duration"] / 2,
                y,
                f"{seg['duration']:.1f}s",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=SPEAKER_COLORS[sid % len(SPEAKER_COLORS)], label=label)
        for sid, label in zip(sorted(seen.keys()), speaker_labels)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    ax.set_yticks(range(len(speaker_labels)))
    ax.set_yticklabels(speaker_labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Speaker Timeline – {n_speakers} speaker(s), {len(timeline)} segments")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _save_html_timeline(
    timeline: list[TimelineSegment],
    n_speakers: int,
    output_dir: Path,
    save_path: Path,
) -> None:
    """Generate an interactive HTML timeline report with clickable segments."""
    seen: dict[int, str] = {}
    for seg in timeline:
        sid = seg["speaker_id"]
        if sid not in seen:
            seen[sid] = seg["speaker_label"]
    speaker_labels = [seen[sid] for sid in sorted(seen.keys())]
    rows_html: list[str] = []
    for idx, seg in enumerate(timeline):
        color = SPEAKER_COLORS[seg["speaker_id"] % len(SPEAKER_COLORS)]
        seg_dir = f"segments/seg_{idx:03d}"
        wav_rel = f"{seg_dir}/sound.wav"
        wav_abs = (output_dir / wav_rel).resolve().as_posix()
        folder_abs = (output_dir / seg_dir).resolve().as_posix()
        bar_pct = max(2.0, min(100.0, seg["duration"] / 32.0 * 100.0))
        rows_html.append(f"""
        <tr style="border-left: 4px solid {color};">
            <td class="idx">{idx:03d}</td>
            <td class="icon-cell">
                <a href="file://{wav_abs}" title="Play segment audio (ctrl+click to open)"
                   class="icon-link">▶️</a>
            </td>
            <td class="icon-cell">
                <a href="file://{folder_abs}" title="Open segment folder (ctrl+click to open)"
                   class="icon-link">📁</a>
            </td>
            <td>
                <span class="speaker-tag" style="background: {color}15; color: {color};">
                    {seg['speaker_label']}
                </span>
            </td>
            <td class="time">{seg['start']:.1f}s</td>
            <td class="time">{seg['end']:.1f}s</td>
            <td class="dur">{seg['duration']:.1f}s</td>
            <td class="bar-cell">
                <div class="bar-bg">
                    <div class="bar-fill" style="width: {bar_pct:.1f}%; background: {color};"></div>
                </div>
            </td>
        </tr>""")
    total_duration = sum(seg["duration"] for seg in timeline)
    legend_items: list[str] = []
    for sid in sorted(seen.keys()):
        color = SPEAKER_COLORS[sid % len(SPEAKER_COLORS)]
        label = seen[sid]
        legend_items.append(
            f'<li><span class="color-swatch" style="background: {color};"></span>{label}</li>'
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Speaker Timeline – {n_speakers} speaker(s)</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 960px;
            margin: 40px auto;
            padding: 24px;
            background: #fafafa;
            color: #333;
        }}
        h1 {{ margin-bottom: 4px; font-size: 1.6em; }}
        .summary {{
            color: #666;
            margin-bottom: 24px;
            line-height: 1.5;
        }}
        .summary small {{ color: #999; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        th {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #e0e0e0;
            font-size: 0.78em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #999;
            background: #f5f5f5;
        }}
        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: middle;
        }}
        tr:hover {{ background: #fafcff; }}
        .idx {{
            text-align: center;
            font-weight: 600;
            color: #aaa;
            font-size: 0.85em;
            width: 44px;
        }}
        .icon-cell {{
            text-align: center;
            width: 40px;
        }}
        .icon-link {{
            text-decoration: none;
            font-size: 1.15em;
            opacity: 0.7;
            transition: opacity 0.15s;
        }}
        .icon-link:hover {{ opacity: 1; }}
        .speaker-tag {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .time {{ text-align: right; font-variant-numeric: tabular-nums; width: 64px; }}
        .dur {{ text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; width: 64px; }}
        .bar-cell {{ width: 210px; }}
        .bar-bg {{
            background: #eee;
            border-radius: 4px;
            width: 100%;
            height: 12px;
            overflow: hidden;
        }}
        .bar-fill {{
            border-radius: 4px;
            height: 100%;
            min-width: 2px;
            transition: width 0.3s ease;
        }}
        .speakers-section {{
            margin-top: 32px;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .speakers-section h2 {{
            margin-bottom: 12px;
            font-size: 1.1em;
        }}
        .speakers-section ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .color-swatch {{
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 4px;
            margin-right: 8px;
            vertical-align: middle;
        }}
        .footer {{
            margin-top: 24px;
            text-align: center;
            color: #bbb;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <h1>🎙️ Speaker Timeline</h1>
    <p class="summary">
        <strong>{n_speakers}</strong> speaker(s) &nbsp;·&nbsp;
        <strong>{len(timeline)}</strong> segments &nbsp;·&nbsp;
        <strong>{total_duration:.1f}s</strong> total duration
        <br>
        <small>Ctrl+Click ▶️ to play audio &nbsp;|&nbsp; Ctrl+Click 📁 to open segment folder</small>
    </p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>▶️</th>
                <th>📁</th>
                <th>Speaker</th>
                <th>Start</th>
                <th>End</th>
                <th>Duration</th>
                <th>Bar</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    <div class="speakers-section">
        <h2>Speakers</h2>
        <ul>
            {''.join(legend_items)}
        </ul>
    </div>
    <p class="footer">
        Generated by NeMo TitaNet-Large multi-speaker labeling
    </p>
</body>
</html>"""
    save_path.write_text(html, encoding="utf-8")
    logger.info(f"Saved HTML timeline: {save_path}")

def main():
    args = get_args()
    logger.info(f"Processing: {args.audio_path}")
    results = detect_multi_speakers(
        audio_path=args.audio_path,
        model_name=args.model_name,
        duration=args.duration,
        step=args.step,
        batch_size=args.batch_size,
        min_energy_percentile=args.min_energy_percentile,
        min_segment_duration=args.min_segment_duration,
        method=args.clustering_method,
        merge_threshold=args.merge_threshold,
        assign_threshold=args.assign_threshold,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
    )
    print_results(results, output_dir=args.output_dir, assign_threshold=args.assign_threshold)
    save_outputs(results, audio_path=args.audio_path, output_dir=args.output_dir)
    logger.info("Done.")

if __name__ == "__main__":
    main()