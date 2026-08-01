"""Test script for combine_audio_paths with segment info output."""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import soundfile as sf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from audio_utils import resolve_audio_paths, combine_audio_paths, SAMPLE_RATE, SegmentInfo

# ── Output directory ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Default audio path ────────────────────────────────────────────
DEFAULT_AUDIO = str(
    Path(
        r"~\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles"
        r"\live_subtitles_server2_with_en\services\generated"
        r"\test_extract_trough_to_trough_double_check"
    )
    .expanduser()
    .resolve()
)

console = Console()


def _log_saved(file_path: Path, description: str = "") -> None:
    """Log a saved file with clickable resource link showing base name."""
    label = f"{description}: " if description else ""
    console.print(
        f"[green]✓ Saved {label}[/green]"
        f"[link=file://{file_path.as_posix()}]{file_path.name}[/link]"
    )


# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Combine audio files with segment info output."
)
parser.add_argument(
    "speakers",
    nargs="*",
    default=[DEFAULT_AUDIO],
    help="Paths to speaker WAV files or directories.",
)
parser.add_argument(
    "--gap",
    type=float,
    default=2.0,
    help="Silence gap between segments in seconds (default: 2.0)",
)
args = parser.parse_args()
gap = args.gap

# ── Collect audio files ───────────────────────────────────────────
console.print("\n[yellow]Scanning audio files...[/yellow]")
audio_files = resolve_audio_paths(
    args.speakers,
    recursive=True,
    includes=["**/sound.wav"],
)

if not audio_files:
    console.print("[red]No audio files found![/red]")
    raise SystemExit(1)

console.print(f"[green]Found {len(audio_files)} audio files[/green]")

# ── Show found files ──────────────────────────────────────────────
if len(audio_files) <= 20:
    table = Table(title="Audio Files Found")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("File Path", style="green")
    for i, file_path in enumerate(audio_files, 1):
        table.add_row(str(i), str(Path(file_path).name))
    console.print(table)

# ── Combine with segment info ─────────────────────────────────────
console.print(f"[yellow]Combining with {gap}s gap...[/yellow]")
combined_audio, segments = combine_audio_paths(
    audio_files,
    gap=gap,
    return_segments=True,
)

# ═══════════════════════════════════════════════════════════════════
# Save ALL results under OUTPUT_DIR
# ═══════════════════════════════════════════════════════════════════

# 1. Combined WAV file
wav_path = OUTPUT_DIR / "combined_audio.wav"
sf.write(wav_path, combined_audio, SAMPLE_RATE)
_log_saved(wav_path, "Combined WAV")

# 2. Segment info as JSON (programmatic use)
segments_json = []
for seg in segments:
    segments_json.append({
        "index": seg.index,
        "source": seg.source,
        "start_sample": seg.start_sample,
        "end_sample": seg.end_sample,
        "start_time": round(seg.start_time, 4),
        "end_time": round(seg.end_time, 4),
        "duration": round(seg.duration, 4),
    })

json_path = OUTPUT_DIR / "segments.json"
with open(json_path, "w") as f:
    json.dump(segments_json, f, indent=2)
_log_saved(json_path, "Segment JSON")

# 3. Human-readable text summary
total_gap_duration = gap * (len(segments) - 1) if len(segments) > 1 else 0
total_audio_duration = len(combined_audio) / SAMPLE_RATE

txt_path = OUTPUT_DIR / "summary.txt"
with open(txt_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  COMBINED AUDIO SEGMENT SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"  Generated: {datetime.now().isoformat()}\n")
    f.write(f"  Total files: {len(audio_files)}\n")
    f.write(f"  Gap between segments: {gap}s\n")
    f.write(f"  Total gap duration: {total_gap_duration:.3f}s\n")
    f.write(f"  Total audio duration: {total_audio_duration:.3f}s\n")
    f.write(f"  Sample rate: {SAMPLE_RATE} Hz\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {'Idx':<5} {'Source':<45} {'Start':>8} {'End':>8} {'Dur':>8}\n")
    f.write("-" * 70 + "\n")
    for seg in segments:
        f.write(
            f"  {seg.index:<5} "
            f"{seg.source:<45} "
            f"{seg.start_time:>8.3f} "
            f"{seg.end_time:>8.3f} "
            f"{seg.duration:>8.3f}\n"
        )
    f.write("=" * 70 + "\n")
_log_saved(txt_path, "Text summary")

# ═══════════════════════════════════════════════════════════════════
# Rich console display
# ═══════════════════════════════════════════════════════════════════

console.print(f"\n[blue]Total Duration: {total_audio_duration:.2f} seconds[/blue]")

# Segment timeline table
console.print("\n[bold yellow]Segment Timeline:[/bold yellow]")
table = Table(title="Audio Segments", show_header=True, header_style="bold cyan")
table.add_column("Idx", justify="right", style="dim")
table.add_column("Source", style="green", max_width=50)
table.add_column("Start (s)", justify="right")
table.add_column("End (s)", justify="right")
table.add_column("Duration (s)", justify="right")

for seg in segments:
    table.add_row(
        str(seg.index),
        seg.source,
        f"{seg.start_time:.3f}",
        f"{seg.end_time:.3f}",
        f"{seg.duration:.3f}",
    )

console.print(table)

# Summary panel
console.print(Panel.fit(
    f"[bold]Output directory:[/bold] {OUTPUT_DIR}\n"
    f"  combined_audio.wav  — merged audio file\n"
    f"  segments.json      — machine-readable segment metadata\n"
    f"  summary.txt        — human-readable timeline\n\n"
    f"[bold]Stats:[/bold]\n"
    f"  Total segments: {len(segments)}\n"
    f"  Gap between segments: {gap}s\n"
    f"  Total gap duration: {total_gap_duration:.2f}s\n"
    f"  Total audio duration: {total_audio_duration:.2f}s\n"
    f"  Sample rate: {SAMPLE_RATE} Hz",
    title="Summary"
))
