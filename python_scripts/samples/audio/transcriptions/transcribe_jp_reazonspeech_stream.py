"""
Streaming transcription with clean Rich output + fixed final transcript panel
"""
import time
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box   # ← Added for explicit box styles

from reazonspeech.espnet.asr.transcribe import (
    load_model,
    transcribe_stream,
)
from reazonspeech.espnet.asr.audio import audio_from_path
from reazonspeech.espnet.asr.interface import TranscribeConfig

# ------------------------------------------------------------------
console = Console(highlight=False, soft_wrap=True)

DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

parser = argparse.ArgumentParser(description="Streaming transcription with Rich output")
parser.add_argument(
    "audio_path",
    nargs="?",
    default=DEFAULT_AUDIO_PATH,
    help=f"Path to audio file (default: {DEFAULT_AUDIO_PATH})"
)
args = parser.parse_args()
audio_path = args.audio_path

# ------------------------------------------------------------------
# Starting header - using fit is good
console.print(
    Panel.fit(
        f"🎙️ Starting Streaming Transcription\n[dim]Audio: {audio_path}[/dim]",
        style="bold cyan",
        box=box.ROUNDED,      # Explicit rounded box for consistency
    )
)

model = load_model()
audio = audio_from_path(audio_path)

console.print("Model loaded. Starting transcription...\n", style="dim")

start_time = time.perf_counter()
full_transcript: list[str] = []

# Suppress tqdm progress from reazonspeech
import os
os.environ["TQDM_DISABLE"] = "1"

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    console=console,
    transient=True,
) as progress:
    task = progress.add_task("[cyan]Transcribing...", total=None)

    for segment in transcribe_stream(
        model,
        audio,
        config=TranscribeConfig(verbose=False)
    ):
        now = time.perf_counter()
        duration = segment.end_seconds - segment.start_seconds
        delta = now - start_time if not full_transcript else now - (start_time + sum(s.end_seconds - s.start_seconds for s in []))  # simplified
        elapsed = now - start_time

        clean_text = segment.text.strip()
        if clean_text:
            full_transcript.append(clean_text)

        speed_color = "green" if delta < 0.4 else "yellow" if delta < 1.0 else "red"

        time_range = Text.assemble(
            "[", (f"{segment.start_seconds:6.2f}", "cyan"),
            " → ",
            (f"{segment.end_seconds:6.2f}", "cyan"),
            "] "
        )
        text_part = Text(f"{clean_text} ", style="bold white")
        timing_part = Text.assemble(
            "(", ("+", speed_color), (f"{(now - (start_time + (elapsed - delta))):.3f}s" if 'last_time' in locals() else f"{delta:.3f}s", speed_color),
            " | dur=", (f"{duration:.2f}s", "dim"),
            " | total=", (f"{elapsed:.2f}s", "dim"), ")"
        )

        console.print(time_range, text_part, timing_part)
        sys.stdout.flush()
        progress.update(task, advance=1)

# ====================== FINAL TRANSCRIPT (FIXED) ======================
console.print("\n" + "═" * 70, style="bold magenta")

final_text_str = "".join(full_transcript)

# Wrap in Text for better control over wrapping
final_content = Text(final_text_str, style="white", no_wrap=False)

console.print("[bold green]📝 Final Full Transcript[/bold green]\n")
console.print(final_content)

console.print(
    f"\n✅ [bold green]Transcription completed successfully![/bold green] "
    f"Total time: [cyan]{time.perf_counter() - start_time:.2f}s[/cyan]"
)
