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
    from audio_utils import resolve_audio_paths_as_tensor_list
    from pyannote.audio import Inference, Model
    from segment_speaker_labeler import (
        SegmentSpeakerLabeler, 
        DEFAULT_THRESHOLD_SAME, 
        DEFAULT_THRESHOLD_POSSIBLE
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
    # Threshold arguments with shorthands
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
    args = parser.parse_args()

    sample_rate = 16000

    console.print(Panel.fit(
        "[bold cyan]Speaker Embedding Analysis Tool[/bold cyan]\n"
        "pyannote/embedding + Dynamic Speaker Labeling",
        title="🚀 Speaker Analysis",
        border_style="blue"
    ))

    console.print("\n[yellow]Scanning audio files...[/yellow]")
    
    audio_files = []
    for path_str in args.speakers:
        p = Path(path_str).expanduser().resolve()
        if p.is_dir():
            for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac"]:
                audio_files.extend(p.glob(f"**/{ext}"))
        elif p.is_file():
            audio_files.append(p)
    
    audio_files = sorted(set(str(f) for f in audio_files))

    waveforms = resolve_audio_paths_as_tensor_list(
        args.speakers, 
        sr=sample_rate, 
        recursive=True
    )

    audio_data = list(zip(waveforms, audio_files))

    # Load model
    with console.status("[bold green]Loading pyannote embedding model...[/bold green]", spinner="dots"):
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")

    labeler = SegmentSpeakerLabeler(
        embedding_model=inference,
        threshold_same=args.threshold_same,
        threshold_possible=args.threshold_possible,
        debug=True
    )

    console.print(f"\n[bold]Processing {len(audio_data)} audio segments...[/bold]\n")

    results = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Analyzing speakers...", total=len(audio_data))

        for i, (waveform, filepath_str) in enumerate(audio_data):
            filepath = Path(filepath_str)
            filename = filepath.name
            timestamp = 0.0

            # Calculate duration
            if waveform.dim() == 1:
                duration = waveform.shape[0] / sample_rate
            else:
                duration = waveform.shape[-1] / sample_rate

            label, confidence, metadata = labeler.label_segment(
                waveform, sample_rate, timestamp
            )

            results.append({
                "index": i + 1,
                "file": str(filepath),
                "filename": filename,
                "duration": duration,
                "label": label,
                "confidence": confidence,
                "match_type": metadata.get("match_type", "unknown"),
            })

            progress.advance(task)

    # === Auto-sizing Table (no fixed widths) ===
    table = Table(
        title="🎤 Speaker Analysis Results",
        show_lines=True,
        expand=False,          # Content-based sizing
        title_justify="left"
    )

    table.add_column("#", justify="right")
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Duration", justify="right")
    table.add_column("Speaker", style="green", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Match Type", justify="center")
    table.add_column("▶️ Play", justify="center")

    for r in results:
        conf_color = "green" if r["confidence"] > 0.7 else "yellow" if r["confidence"] > 0.4 else "red"
        
        duration_str = f"{r['duration']:.2f}s"
        play_link = f"[link=file://{r['file']}]▶️ Play[/link]"

        table.add_row(
            str(r["index"]),
            r["filename"],
            duration_str,
            f"[bold]{r['label']}[/bold]",
            f"[{conf_color}]{r['confidence']:.3f}[/{conf_color}]",
            r["match_type"],
            play_link
        )

    console.print(table)

    # Summary
    unique_speakers = len({r["label"] for r in results})
    total_duration = sum(r["duration"] for r in results)
    
    console.print(Panel(
        f"Total segments: [bold]{len(results)}[/bold]\n"
        f"Total duration: [bold]{total_duration:.1f}s[/bold]\n"
        f"Unique speakers: [bold cyan]{unique_speakers}[/bold cyan]",
        title="Summary",
        border_style="green",
        padding=(1, 2)
    ))

    # Save results
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "speaker_analysis.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[dim]Results saved to:[/dim] [blue]{output_file}[/blue]")


if __name__ == "__main__":
    main()
