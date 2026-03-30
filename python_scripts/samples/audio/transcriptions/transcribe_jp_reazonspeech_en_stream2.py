"""
Streaming Japanese transcription + English translation (Clean output)
"""

import time
import sys
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from reazonspeech.espnet.asr.transcribe import load_model, transcribe_stream
from reazonspeech.espnet.asr.audio import audio_from_path
from reazonspeech.espnet.asr.interface import TranscribeConfig

from translators.translate_jp_en_shisa_lfm import translate_japanese_to_english

console = Console(highlight=False, soft_wrap=True)

DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

parser = argparse.ArgumentParser(description="Streaming JP transcription + EN translation")
parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH)
args = parser.parse_args()
audio_path = args.audio_path

console.print(
    Panel.fit(f"🎙️ Starting Streaming Transcription + Translation\n[dim]Audio: {audio_path}[/dim]",
              style="bold cyan", box=box.ROUNDED)
)

# Load ASR
model = load_model()
audio = audio_from_path(audio_path)
console.print("✅ ASR Model loaded.\n", style="dim")

start_time = time.perf_counter()
full_segments: list[dict] = []
previous_time = start_time


# === Streaming Loop ===
import os
os.environ["TQDM_DISABLE"] = "1"

with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(),
              console=console, transient=True) as progress:
    
    task = progress.add_task("[cyan]Transcribing & Translating...", total=None)

    for segment in transcribe_stream(model, audio, config=TranscribeConfig(verbose=False)):
        now = time.perf_counter()
        process_delta = now - previous_time
        previous_time = now
        elapsed = now - start_time

        clean_ja = segment.text.strip()
        en_result = translate_japanese_to_english(clean_ja)
        en_text = en_result["text"]

        if clean_ja:
            full_segments.append({
                "start": segment.start_seconds,
                "end": segment.end_seconds,
                "ja": clean_ja,
                "en": en_text
            })

        speed_color = "green" if process_delta < 0.6 else "yellow" if process_delta < 1.5 else "red"

        # Clean output as requested
        console.print(
            Text.assemble(
                "[", (f"{segment.start_seconds:6.2f}", "cyan"),
                " → ", (f"{segment.end_seconds:6.2f}", "cyan"),
                f"]  (+{process_delta:.2f}s)", style=speed_color
            )
        )
        console.print(f"   JA: {clean_ja}", style="bold white")
        console.print(f"   EN: {en_text or '(no translation)'}", style="bold green")

        sys.stdout.flush()
        progress.update(task, advance=1)

# === Final Output ===
console.print("\n" + "═" * 80, style="bold magenta")
console.print("[bold green]📝 Final Bilingual Transcript[/bold green]\n")

for seg in full_segments:
    console.print(Text.assemble("[", (f"{seg['start']:6.2f}", "cyan"), " → ", (f"{seg['end']:6.2f}", "cyan"), "]"))
    console.print(f"   JA: {seg['ja']}", style="white")
    console.print(f"   EN: {seg['en']}", style="green")
    console.print()

console.print(f"\n✅ Done! Total time: [cyan]{time.perf_counter() - start_time:.2f}s[/cyan] | Segments: [cyan]{len(full_segments)}[/cyan]")
