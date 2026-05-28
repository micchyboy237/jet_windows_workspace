"""Main entry point for speaker embedding analysis with short buffer support."""
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path(
        "~/Desktop/Jet_Files/Jet_Windows_Workspace/python_scripts/samples/audio"
        "/features/generated/speech_waves/waves/"
    )
    .expanduser()
    .resolve()
)


def main():
    from audio_utils import resolve_audio_paths, resolve_audio_paths_as_tensor_list
    from pyannote.audio import Inference, Model
    from segment_speaker_labeler_short_buffer import (
        SegmentSpeakerLabeler,
        DEFAULT_THRESHOLD_SAME,
        DEFAULT_THRESHOLD_POSSIBLE,
        DEFAULT_THRESHOLD_NEW_SPEAKER,
        DEFAULT_MIN_EMBEDDING_DURATION,
        DEFAULT_BUFFER_SIMILARITY,
        DEFAULT_MAX_BUFFER_SLOTS,
        DEFAULT_BUFFER_TTL,
    )

    console = Console()

    parser = argparse.ArgumentParser(
        description="Speaker embedding analysis with short buffer support."
    )
    parser.add_argument(
        "speakers",
        nargs="*",
        default=[DEFAULT_AUDIO],
        help="Paths to speaker WAV files or directories.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "-t", "--threshold-same",
        type=float,
        default=DEFAULT_THRESHOLD_SAME,
        help=f"Similarity threshold for strong match (default: {DEFAULT_THRESHOLD_SAME})",
    )
    parser.add_argument(
        "-tp", "--threshold-possible",
        type=float,
        default=DEFAULT_THRESHOLD_POSSIBLE,
        help=f"Similarity threshold for possible match (default: {DEFAULT_THRESHOLD_POSSIBLE})",
    )
    parser.add_argument(
        "-tn", "--threshold-new-speaker",
        type=float,
        default=DEFAULT_THRESHOLD_NEW_SPEAKER,
        help=f"Similarity threshold for new speaker creation (default: {DEFAULT_THRESHOLD_NEW_SPEAKER})",
    )
    # Short buffer arguments
    parser.add_argument(
        "--enable-short-buffer",
        action="store_true",
        default=True,
        help="Enable short embedding buffering (default: True)",
    )
    parser.add_argument(
        "--no-short-buffer",
        action="store_false",
        dest="enable_short_buffer",
        help="Disable short embedding buffering",
    )
    parser.add_argument(
        "--min-embedding-duration",
        type=float,
        default=DEFAULT_MIN_EMBEDDING_DURATION,
        help=f"Minimum audio duration before bypassing buffer (default: {DEFAULT_MIN_EMBEDDING_DURATION}s)",
    )
    parser.add_argument(
        "--buffer-similarity",
        type=float,
        default=DEFAULT_BUFFER_SIMILARITY,
        help=f"Similarity threshold for merging into same buffer slot (default: {DEFAULT_BUFFER_SIMILARITY})",
    )
    parser.add_argument(
        "--max-buffer-slots",
        type=int,
        default=DEFAULT_MAX_BUFFER_SLOTS,
        help=f"Maximum concurrent buffer slots (default: {DEFAULT_MAX_BUFFER_SLOTS})",
    )
    parser.add_argument(
        "--buffer-ttl",
        type=float,
        default=DEFAULT_BUFFER_TTL,
        help=f"Buffer slot TTL in seconds (default: {DEFAULT_BUFFER_TTL}s)",
    )
    args = parser.parse_args()

    sample_rate = 16000

    console.print(Panel.fit(
        "[bold cyan]Speaker Embedding Analysis Tool[/bold cyan]\n"
        "pyannote/embedding + Dynamic Speaker Labeling + Short Buffer",
        title="🚀 Speaker Analysis",
        border_style="blue"
    ))

    # Buffer configuration summary
    if args.enable_short_buffer:
        console.print(
            f"[dim]📦 Short buffer enabled: min_duration={args.min_embedding_duration}s, "
            f"merge_threshold={args.buffer_similarity:.2f}, "
            f"max_slots={args.max_buffer_slots}, "
            f"ttl={args.buffer_ttl}s[/dim]"
        )
    else:
        console.print("[dim]📦 Short buffer disabled[/dim]")

    console.print("\n[yellow]Scanning audio files...[/yellow]")
    audio_files = resolve_audio_paths(
        args.speakers,
        recursive=True,
        includes=["**/sound.wav"],
    )
    waveforms = resolve_audio_paths_as_tensor_list(
        audio_files,
        sr=sample_rate,
    )
    audio_data = list(zip(waveforms, audio_files))

    # Show duration distribution before processing
    durations = []
    for waveform, _ in audio_data:
        if waveform.dim() == 1:
            dur = waveform.shape[0] / sample_rate
        else:
            dur = waveform.shape[-1] / sample_rate
        durations.append(dur)
    
    short_count = sum(1 for d in durations if d < args.min_embedding_duration)
    long_count = sum(1 for d in durations if d >= args.min_embedding_duration)
    console.print(
        f"[dim]Found {len(audio_data)} segments: "
        f"[yellow]{short_count} short[/yellow] (<{args.min_embedding_duration}s), "
        f"[green]{long_count} long[/green] (≥{args.min_embedding_duration}s)[/dim]"
    )

    with console.status("[bold green]Loading pyannote embedding model...[/bold green]", spinner="dots"):
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")

    labeler = SegmentSpeakerLabeler(
        embedding_model=inference,
        threshold_same=args.threshold_same,
        threshold_possible=args.threshold_possible,
        threshold_new_speaker=args.threshold_new_speaker,
        enable_short_buffer=args.enable_short_buffer,
        min_embedding_duration=args.min_embedding_duration,
        buffer_similarity_threshold=args.buffer_similarity,
        max_buffer_slots=args.max_buffer_slots,
        buffer_ttl=args.buffer_ttl,
        debug=True,
    )

    console.print(f"\n[bold]Processing {len(audio_data)} audio segments...[/bold]\n")

    # Track buffered segments for final summary
    buffered_segments = []
    promoted_segments = []

    # Build structured results grouped by segment
    segment_groups = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as task_progress:
        task = task_progress.add_task("Analyzing speakers...", total=len(audio_data))
        for i, (waveform, filepath_str) in enumerate(audio_data):
            filepath = Path(filepath_str)
            filename = filepath.name
            dir_path = str(filepath.parent)
            timestamp = float(i)  # Use index as timestamp for sequential processing

            if waveform.dim() == 1:
                duration = waveform.shape[0] / sample_rate
            else:
                duration = waveform.shape[-1] / sample_rate

            # Pass segment_duration to enable buffer routing
            segment_results = labeler.label_segments(
                waveform, sample_rate, timestamp,
                segment_duration=duration,  # KEY: Pass duration for buffer decision
            )

            # Check if this segment was buffered
            is_buffered = segment_results[0].get("is_buffered", False)
            if is_buffered:
                buffered_segments.append({
                    "index": i + 1,
                    "file": str(filepath),
                    "duration": duration,
                })

            # Build primary + alternatives for this segment
            matches = []
            for j, match in enumerate(segment_results):
                match_entry = {
                    "label": match["label"],
                    "confidence": match["confidence"],
                    "match_type": match.get("match_type", "unknown"),
                    "is_primary": match.get("is_primary", False),
                    "is_new_speaker": match.get("is_new_speaker", False),
                    "is_buffered": match.get("is_buffered", False),
                    "segment_duration": match.get("segment_duration", duration),
                    "rank": j + 1 if not match.get("is_primary") else 0,
                }
                matches.append(match_entry)

            segment_groups.append({
                "index": i + 1,
                "file": str(filepath),
                "filename": filename,
                "dir": dir_path,
                "duration": duration,
                "is_buffered": is_buffered,
                "matches": matches,
            })
            task_progress.advance(task)

    # Force-check for any remaining ready buffer slots
    if args.enable_short_buffer:
        console.print("\n[dim]Checking for remaining buffered segments...[/dim]")
        promoted_results = labeler.check_and_promote_buffered()
        if promoted_results:
            console.print(f"[green]Promoted {len(promoted_results)} additional buffered slots[/green]")
            for result in promoted_results:
                promoted_segments.append(result)

    # Build flat results list for JSON export
    results = []
    for group in segment_groups:
        for match in group["matches"]:
            results.append({
                "index": group["index"],
                "file": group["file"],
                "filename": group["filename"],
                "dir": group["dir"],
                "duration": group["duration"],
                "is_buffered": group["is_buffered"],
                "label": match["label"],
                "confidence": match["confidence"],
                "match_type": match["match_type"],
                "is_primary": match["is_primary"],
                "is_new_speaker": match.get("is_new_speaker", False),
                "is_buffered_match": match.get("is_buffered", False),
                "segment_duration": match.get("segment_duration", group["duration"]),
                "rank": match["rank"],
            })

    # Render results table
    table = Table(
        title="🎤 Speaker Analysis Results",
        show_lines=True,
        expand=False,
        title_justify="left"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Dir", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Status", justify="center")  # New column for buffer status
    table.add_column("Rank", justify="center")
    table.add_column("Speaker", style="green", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Match Type", justify="center")
    table.add_column("Primary", justify="center")
    table.add_column("▶️ Play", justify="center")

    for group in segment_groups:
        for idx, match in enumerate(group["matches"]):
            is_first = (idx == 0)

            # Only show segment index on first row
            index_str = str(group["index"]) if is_first else ""

            # Show directory name with terminal link on first row
            dir_name = Path(group["dir"]).name
            dir_link = f"[link=file://{group['dir']}]{dir_name}[/link]" if is_first else ""

            # Only show duration on first row
            duration_str = f"{group['duration']:.2f}s" if is_first else ""

            # Show buffer status on first row
            if is_first:
                if group["is_buffered"]:
                    status_str = "[yellow]📦 BUFFERED[/yellow]"
                elif group["duration"] < args.min_embedding_duration:
                    status_str = "[green]🚀 PROMOTED[/green]"
                else:
                    status_str = "[dim]→ direct[/dim]"
            else:
                status_str = ""

            # Only show play link on first row
            play_link = f"[link=file://{group['file']}]▶️ Play[/link]" if is_first else ""

            # Confidence color coding
            if match["confidence"] == 0.0 and match.get("is_buffered_match"):
                conf_color = "dim"
                conf_display = "—"
            else:
                conf_color = "green" if match["confidence"] > 0.7 else "yellow" if match["confidence"] > 0.4 else "red"
                conf_display = f"{match['confidence']:.3f}"

            primary_marker = "⭐" if match["is_primary"] else ""
            rank_str = f"#{match['rank']}" if match["rank"] > 0 else "—"

            table.add_row(
                index_str,
                dir_link,
                duration_str,
                status_str,
                rank_str,
                f"[bold]{match['label']}[/bold]",
                f"[{conf_color}]{conf_display}[/{conf_color}]",
                match["match_type"],
                primary_marker,
                play_link,
            )

    console.print(table)

    # Calculate statistics
    primary_results = [r for r in results if r["is_primary"]]
    unique_speakers = len({r["label"] for r in results if r["label"] != "BUFFERED"})
    total_duration = sum(r["duration"] for r in primary_results)
    buffered_count = sum(1 for r in results if r.get("is_buffered"))
    promoted_count = sum(
        1 for r in primary_results 
        if r.get("match_type") != "buffered_short" 
        and r["duration"] < args.min_embedding_duration
    )

    # Buffer statistics
    buffer_status = labeler.get_buffer_status()
    buffer_stats = buffer_status["statistics"] if buffer_status else {}

    # Summary panel
    summary_lines = [
        f"Total segments: [bold]{len(segment_groups)}[/bold]",
        f"Total results (incl. alternatives): [bold]{len(results)}[/bold]",
        f"Total duration: [bold]{total_duration:.1f}s[/bold]",
        f"Unique speakers: [bold cyan]{unique_speakers}[/bold cyan]",
        f"Average matches per segment: [bold]{len(results) / max(len(segment_groups), 1):.1f}[/bold]",
    ]

    if args.enable_short_buffer:
        summary_lines.extend([
            "",
            "[bold]Short Buffer Stats:[/bold]",
            f"  Segments buffered: [yellow]{buffered_count}[/yellow]",
            f"  Slots promoted: [green]{buffer_stats.get('total_promoted', 0)}[/green]",
            f"  Active slots remaining: [dim]{buffer_stats.get('active_slots', 0)}[/dim]",
            f"  Merged into existing: {buffer_stats.get('total_merged_into_slot', 0)}",
            f"  New slots created: {buffer_stats.get('total_new_slots_created', 0)}",
            f"  Discarded (evicted/expired): {buffer_stats.get('total_discarded', 0)}",
        ])

    console.print(Panel(
        "\n".join(summary_lines),
        title="Summary",
        border_style="green",
        padding=(1, 2),
    ))

    # Save outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Health status (now includes buffer info)
    health_status = labeler.get_health_status()
    if buffer_status:
        health_status["short_buffer"] = buffer_status
    output_file = output_dir / "health_status.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(health_status, f, indent=2, ensure_ascii=False)

    # Full results
    output_file = output_dir / "speaker_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary by segment
    summary_file = output_dir / "speaker_analysis_summary.json"
    summary_data = [
        {
            "file": r["file"],
            "filename": r["filename"],
            "dir": r["dir"],
            "duration": r["duration"],
            "is_buffered": r.get("is_buffered", False),
            "primary_speaker": r["label"],
            "confidence": r["confidence"],
            "match_type": r["match_type"],
            "alternatives_count": len([x for x in results if x["file"] == r["file"]]) - 1,
        }
        for r in primary_results
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Buffer-specific report
    if args.enable_short_buffer and buffer_stats:
        buffer_report_file = output_dir / "buffer_report.json"
        buffer_report = {
            "configuration": {
                "min_embedding_duration": args.min_embedding_duration,
                "buffer_similarity_threshold": args.buffer_similarity,
                "max_slots": args.max_buffer_slots,
                "ttl": args.buffer_ttl,
            },
            "statistics": buffer_stats,
            "active_slots": buffer_status["active_slots"],
            "buffered_segments": buffered_segments,
        }
        with open(buffer_report_file, "w", encoding="utf-8") as f:
            json.dump(buffer_report, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]Buffer report saved to:[/dim] [blue]{buffer_report_file}[/blue]")

    console.print(f"\n[dim]Detailed results saved to:[/dim] [blue]{output_file}[/blue]")
    console.print(f"[dim]Summary saved to:[/dim] [blue]{summary_file}[/blue]")

    # Warn if segments are still buffered
    if buffered_count > 0 and args.enable_short_buffer:
        console.print(
            f"\n[yellow]⚠️  {buffered_count} segments remain buffered "
            f"(insufficient total duration for promotion)[/yellow]"
        )


if __name__ == "__main__":
    main()
