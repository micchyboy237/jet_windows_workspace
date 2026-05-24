"""
Demonstration script for SpeakerLabeler using a local audio file.
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install as install_rich_traceback
import os

def main():
    from speaker_labeler import SpeakerLabeler

    # Install rich traceback handler
    install_rich_traceback(show_locals=True)

    console = Console()

    parser = argparse.ArgumentParser(
        description="Demonstrate SpeakerLabeler with pyannote/segmentation-3.0"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace access token (or set HF_TOKEN env var)",
        default=None
    )
    parser.add_argument(
        "--min-duration-on",
        type=float,
        default=0.1,
        help="Minimum speech segment duration (default: 0.1s)"
    )
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=0.3,
        help="Minimum non-speech gap duration (default: 0.3s)"
    )
    
    args = parser.parse_args()
    
    # Validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {args.audio_file}[/red]")
        return
    
    console.print(Panel.fit(
        "[bold yellow]SpeakerLabeler Demo[/bold yellow]\n"
        f"[cyan]Model: pyannote/segmentation-3.0[/cyan]\n"
        f"[cyan]Audio: {audio_path}[/cyan]",
        border_style="yellow"
    ))
    
    try:
        # Initialize the SpeakerLabeler
        console.print("\n[bold]Step 1: Initializing SpeakerLabeler[/bold]")
        labeler = SpeakerLabeler(
            hf_token=args.token or os.environ.get("HF_TOKEN"),
            min_duration_on=args.min_duration_on,
            min_duration_off=args.min_duration_off,
        )
        
        # Process the audio file
        console.print(f"\n[bold]Step 2: Processing audio file[/bold]")
        console.print(f"[dim]File: {audio_path}[/dim]")
        console.print(f"[dim]Size: {audio_path.stat().st_size / 1024:.1f} KB[/dim]")
        
        results = labeler.label_speakers(str(audio_path))
        
        # Display results
        console.print(f"\n[bold]Step 3: Displaying Results[/bold]")
        labeler.display_results(results)
        
        # Additional insights
        console.print("\n[bold green]✓ Processing completed successfully![/bold green]")
        
        # Model capabilities used
        console.print(Panel.fit(
            "[bold]Capabilities Demonstrated:[/bold]\n"
            "• Voice Activity Detection (VAD)\n"
            "• Overlapped Speech Detection (OSD)\n"
            "• Speaker Segmentation (Powerset encoding)\n"
            "• Progressive Reference Building\n"
            "• Multi-format Audio Support",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"\n[red bold]Error: {e}[/red bold]")
        console.print_exception()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 