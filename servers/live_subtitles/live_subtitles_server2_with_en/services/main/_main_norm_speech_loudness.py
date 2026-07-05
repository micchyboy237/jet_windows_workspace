import argparse
import shutil
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track

from norm_speech_loudness import normalize_audio_for_vad

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/sub_audio/start_5s_recording_1_speaker.wav").expanduser().resolve()
)

# ─────────────────────────────────────────────────────────────────
# LOUDNESS LEVELS TO GENERATE
# ─────────────────────────────────────────────────────────────────
LOUDNESS_PRESETS = [
    "very_quiet",
    "quiet", 
    "standard",
    "loud",
    "very_loud",
    "brickwall",
]

PRESET_DISPLAY = {
    "very_quiet": {"emoji": "🤫", "description": "Whisper, ASMR, distant speech"},
    "quiet": {"emoji": "🤐", "description": "Soft conversation, close-mic podcast"},
    "standard": {"emoji": "🗣️", "description": "Normal conversation, meetings, typical VAD"},
    "loud": {"emoji": "📢", "description": "Energetic speech, presentations"},
    "very_loud": {"emoji": "🔊", "description": "Broadcast-optimized, processed speech"},
    "brickwall": {"emoji": "💥", "description": "Maximum digital level (not recommended for VAD)"},
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate all loudness variants of an audio file for VAD testing"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sample rate (default: keep original)",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        default=True,
        help="Also save the original unmodified audio (default: True)",
    )
    parser.add_argument(
        "--no-original",
        action="store_false",
        dest="keep_original",
        help="Don't save the original audio",
    )
    return parser.parse_args()


def load_audio(audio_path: str, target_sr: int = None):
    """Load audio file and return waveform + sample rate."""
    console.print(f"[bold cyan]Loading audio:[/bold cyan] {audio_path}")
    
    # Load with librosa (handles many formats)
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Get original duration
    duration = len(y) / sr
    
    console.print(f"  Sample rate: [green]{sr}[/green] Hz")
    console.print(f"  Duration: [green]{duration:.2f}[/green] seconds")
    console.print(f"  Samples: [green]{len(y):,}[/green]")
    console.print(f"  Original RMS: [green]{20 * np.log10(np.sqrt(np.mean(y**2))):.2f}[/green] dBFS")
    console.print(f"  Original Peak: [green]{20 * np.log10(np.max(np.abs(y))):.2f}[/green] dBFS")
    console.print()
    
    return y, sr


def save_audio(y: np.ndarray, sr: int, output_path: Path, normalize_to_16bit: bool = True):
    """Save audio to file, optionally converting to 16-bit PCM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize_to_16bit:
        # Convert float32 [-1, 1] to int16 for standard WAV
        y_save = np.clip(y, -1.0, 1.0)
        sf.write(str(output_path), y_save, sr, subtype='PCM_16')
    else:
        sf.write(str(output_path), y, sr, subtype='FLOAT')
    
    file_size = output_path.stat().st_size
    return file_size


def display_results_table(results: list):
    """Display a rich table with all generated variants."""
    table = Table(
        title="🎤 Audio Loudness Variants Generated",
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
    )
    
    table.add_column("Level", style="bold", width=15)
    table.add_column("RMS (dBFS)", justify="right", width=12)
    table.add_column("Peak (dBFS)", justify="right", width=12)
    table.add_column("Gain (dB)", justify="right", width=10)
    table.add_column("Peak Limit", justify="right", width=12)
    table.add_column("File", style="dim", width=40)
    table.add_column("Size", justify="right", width=10)
    
    for result in results:
        emoji = result.get("emoji", "🎵")
        table.add_row(
            f"{emoji} {result['level']}",
            f"{result['final_rms_db']:.2f}",
            f"{result['final_peak_db']:.2f}",
            f"{result['applied_gain_db']:+.2f}",
            f"{result['max_peak_db']:.2f}",
            result['filename'],
            f"{result['file_size'] / 1024:.1f} KB",
        )
    
    console.print()
    console.print(table)
    console.print()


def main():
    args = get_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    
    # Clean and recreate output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   Audio Loudness Variant Generator        ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")
    console.print()
    
    # Load original audio
    y_original, sr = load_audio(audio_path, args.sr)
    
    # Store results for summary table
    results = []
    
    # ─────────────────────────────────────────────────────────────
    # Save original (unmodified)
    # ─────────────────────────────────────────────────────────────
    if args.keep_original:
        console.print("[bold yellow]💾 Saving original audio...[/bold yellow]")
        original_path = output_dir / "original.wav"
        file_size = save_audio(y_original, sr, original_path)
        
        original_rms_db = 20 * np.log10(np.sqrt(np.mean(y_original**2)) + 1e-8)
        original_peak_db = 20 * np.log10(np.max(np.abs(y_original)) + 1e-8)
        
        results.append({
            "level": "Original",
            "emoji": "📁",
            "final_rms_db": original_rms_db,
            "final_peak_db": original_peak_db,
            "applied_gain_db": 0.0,
            "max_peak_db": 0.0,
            "filename": "original.wav",
            "file_size": file_size,
        })
        console.print(f"  ✅ Saved: [dim]{original_path}[/dim]")
        console.print()
    
    # ─────────────────────────────────────────────────────────────
    # Generate all loudness variants
    # ─────────────────────────────────────────────────────────────
    console.print("[bold yellow]🎛️  Generating loudness variants...[/bold yellow]")
    console.print()
    
    for preset_name in track(LOUDNESS_PRESETS, description="Processing...", console=console):
        display = PRESET_DISPLAY[preset_name]
        emoji = display["emoji"]
        description = display["description"]
        
        # Apply normalization using preset string directly
        y_norm, info = normalize_audio_for_vad(
            y_original,
            sr=sr,
            method="hybrid",
            max_peak_db=preset_name,  # ← Just pass the preset name!
            remove_dc=True,
        )
        
        # Convert to numpy if torch tensor
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.numpy()
        
        console.print(
            f"  {emoji} [bold]{preset_name.replace('_', ' ').title()}[/bold] "
            f"[dim]({description})[/dim]"
        )
        console.print(
            f"    Target: RMS={info['target_rms_db']:+.1f} dBFS, "
            f"Peak limit={info['max_peak_db']:+.1f} dBFS"
        )
        
        # Save normalized audio
        filename = f"{preset_name}.wav"
        output_path = output_dir / filename
        file_size = save_audio(y_norm, sr, output_path)
        
        # Display transformation details
        console.print(
            f"    RMS: [cyan]{info['original_rms_db']:.2f}[/cyan] → "
            f"[cyan]{info['final_rms_db']:.2f}[/cyan] dBFS "
            f"(Gain: [yellow]{info['applied_gain_db']:+.2f}[/yellow] dB)"
        )
        console.print(
            f"    Peak: [cyan]{info['original_peak_db']:.2f}[/cyan] → "
            f"[cyan]{info['final_peak_db']:.2f}[/cyan] dBFS"
        )
        
        # Check if normalization was skipped
        if info.get("skipped_reason"):
            console.print(f"    ⚠️  [bold red]Skipped: {info['skipped_reason']}[/bold red]")
        
        console.print(f"    ✅ Saved: [dim]{output_path}[/dim]")
        console.print()
        
        results.append({
            "level": preset_name.replace("_", " ").title(),
            "emoji": emoji,
            "final_rms_db": info["final_rms_db"],
            "final_peak_db": info["final_peak_db"],
            "applied_gain_db": info["applied_gain_db"],
            "max_peak_db": info["max_peak_db"],
            "filename": filename,
            "file_size": file_size,
        })
    
    # ─────────────────────────────────────────────────────────────
    # Display summary table
    # ─────────────────────────────────────────────────────────────
    display_results_table(results)
    
    # Summary statistics
    total_files = len(results)
    total_size = sum(r["file_size"] for r in results)
    
    console.print(f"[bold green]✅ Generated {total_files} audio variants[/bold green]")
    console.print(f"[bold green]📁 Output directory: {output_dir}[/bold green]")
    console.print(f"[bold green]💾 Total size: {total_size / 1024 / 1024:.2f} MB[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
