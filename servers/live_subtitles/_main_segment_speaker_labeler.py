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
    from segment_speaker_labeler import (
        SegmentSpeakerLabeler,
        DEFAULT_THRESHOLD_SAME,
        DEFAULT_THRESHOLD_POSSIBLE,
        DEFAULT_THRESHOLD_NEW_SPEAKER,
    )

    console = Console()

    parser = argparse.ArgumentParser(
        description="Speaker embedding analysis with rich terminal output."
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
    args = parser.parse_args()

    sample_rate = 16000

    console.print(Panel.fit(
        "[bold cyan]Speaker Embedding Analysis Tool[/bold cyan]\n"
        "pyannote/embedding + Dynamic Speaker Labeling",
        title="🚀 Speaker Analysis",
        border_style="blue"
    ))

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

    with console.status("[bold green]Loading pyannote embedding model...[/bold green]", spinner="dots"):
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")

    labeler = SegmentSpeakerLabeler(
        embedding_model=inference,
        threshold_same=args.threshold_same,
        threshold_possible=args.threshold_possible,
        threshold_new_speaker=args.threshold_new_speaker,
        debug=True
    )

    console.print(f"\n[bold]Processing {len(audio_data)} audio segments...[/bold]\n")

    # Build structured results grouped by segment
    segment_groups = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Analyzing speakers...", total=len(audio_data))
        for i, (waveform, filepath_str) in enumerate(audio_data):
            filepath = Path(filepath_str)
            filename = filepath.name
            dir_path = str(filepath.parent)  # Get directory path
            timestamp = 0.0

            if waveform.dim() == 1:
                duration = waveform.shape[0] / sample_rate
            else:
                duration = waveform.shape[-1] / sample_rate

            segment_results = labeler.label_segments(
                waveform, sample_rate, timestamp
            )

            # Build primary + alternatives for this segment
            matches = []
            for j, match in enumerate(segment_results):
                matches.append({
                    "label": match["label"],
                    "confidence": match["confidence"],
                    "match_type": match.get("match_type", "unknown"),
                    "is_primary": match.get("is_primary", False),
                    "is_new_speaker": match.get("is_new_speaker", False),
                    "rank": j + 1 if not match.get("is_primary") else 0,
                })

            segment_groups.append({
                "index": i + 1,
                "file": str(filepath),
                "filename": filename,
                "dir": dir_path,  # Add directory path
                "duration": duration,
                "matches": matches,
            })
            progress.advance(task)

    # Build flat results list for JSON export
    results = []
    for group in segment_groups:
        for match in group["matches"]:
            results.append({
                "index": group["index"],
                "file": group["file"],
                "filename": group["filename"],
                "dir": group["dir"],  # Add directory to results
                "duration": group["duration"],
                "label": match["label"],
                "confidence": match["confidence"],
                "match_type": match["match_type"],
                "is_primary": match["is_primary"],
                "is_new_speaker": match["is_new_speaker"],
                "rank": match["rank"],
            })

    table = Table(
        title="🎤 Speaker Analysis Results",
        show_lines=True,
        expand=False,
        title_justify="left"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Dir", style="cyan")  # Changed from "Filename" to "Dir"
    table.add_column("Duration", justify="right")
    table.add_column("Rank", justify="center")
    table.add_column("Speaker", style="green", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Match Type", justify="center")
    table.add_column("Primary", justify="center")
    table.add_column("▶️ Play", justify="center")

    # Render table with proper grouping - only show segment info on first row
    for group in segment_groups:
        for idx, match in enumerate(group["matches"]):
            is_first = (idx == 0)

            # Only show segment index on first row
            index_str = str(group["index"]) if is_first else ""

            # Show directory name with terminal link on first row
            dir_name = Path(group["dir"]).name  # Get just the directory name
            dir_link = f"[link=file://{group['dir']}]{dir_name}[/link]" if is_first else ""
            
            # Only show duration on first row
            duration_str = f"{group['duration']:.2f}s" if is_first else ""

            # Only show play link on first row
            play_link = f"[link=file://{group['file']}]▶️ Play[/link]" if is_first else ""

            conf_color = "green" if match["confidence"] > 0.7 else "yellow" if match["confidence"] > 0.4 else "red"
            primary_marker = "⭐" if match["is_primary"] else ""
            rank_str = f"#{match['rank']}" if match["rank"] > 0 else "—"

            table.add_row(
                index_str,
                dir_link,  # Changed from filename_str to dir_link
                duration_str,
                rank_str,
                f"[bold]{match['label']}[/bold]",
                f"[{conf_color}]{match['confidence']:.3f}[/{conf_color}]",
                match["match_type"],
                primary_marker,
                play_link
            )

    console.print(table)

    primary_results = [r for r in results if r["is_primary"]]
    unique_speakers = len({r["label"] for r in results})
    total_duration = sum(r["duration"] for r in primary_results)

    console.print(Panel(
        f"Total segments: [bold]{len(segment_groups)}[/bold]\n"
        f"Total results (incl. alternatives): [bold]{len(results)}[/bold]\n"
        f"Total duration: [bold]{total_duration:.1f}s[/bold]\n"
        f"Unique speakers: [bold cyan]{unique_speakers}[/bold cyan]\n"
        f"Average matches per segment: [bold]{len(results) / max(len(segment_groups), 1):.1f}[/bold]",
        title="Summary",
        border_style="green",
        padding=(1, 2)
    ))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    health_status = labeler.get_health_status()
    output_file = output_dir / "health_status.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(health_status, f, indent=2, ensure_ascii=False)

    output_file = output_dir / "speaker_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_file = output_dir / "speaker_analysis_summary.json"
    summary_data = [
        {
            "file": r["file"],
            "filename": r["filename"],
            "dir": r["dir"],  # Add directory to summary
            "duration": r["duration"],
            "primary_speaker": r["label"],
            "confidence": r["confidence"],
            "match_type": r["match_type"],
            "alternatives_count": len([x for x in results if x["file"] == r["file"]]) - 1,
        }
        for r in primary_results
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[dim]Detailed results saved to:[/dim] [blue]{output_file}[/blue]")
    console.print(f"[dim]Summary saved to:[/dim] [blue]{summary_file}[/blue]")


if __name__ == "__main__":
    main()
