# python_scripts/client/whisper_batch_examples.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
import logging


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    duration_sec: float
    detected_language: Optional[str] = None
    detected_language_prob: Optional[float] = None
    transcription: str
    translation: Optional[str] = None
    segments: Optional[List[TranscriptionSegment]] = None


async def fetch_batch_transcribe(
    client: httpx.AsyncClient,
    audio_paths: List[Path | str],
    translate: bool = False,
) -> List[TranscriptionResponse]:
    """Reusable function to perform batch transcription (± translation) using an existing AsyncClient."""
    files = [("files", (Path(p).name, open(p, "rb"), "audio/wav")) for p in audio_paths]
    endpoint = "/batch/transcribe_translate" if translate else "/batch/transcribe"

    response = await client.post(endpoint, files=files)
    response.raise_for_status()
    return [TranscriptionResponse(**item) for item in response.json()]


# Rich logging setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("whisper_examples")


async def example_simple_batch() -> None:
    """Example 1: Minimal reusable batch transcription (no context manager)."""
    log.info("[bold magenta]Starting Example 1: Simple Batch Transcription[/]")
    log.info("Fetching up to 4 WAV files from audio directory for basic transcription")

    base_url = "http://127.0.0.1:8001"
    audio_dir = Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio")
    audio_files = list(audio_dir.glob("**/*.wav"))[:4]
    log.info(f"Selected audio files: {[p.name for p in audio_files]}")

    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        results = await fetch_batch_transcribe(client, audio_files, translate=False)

    # Pretty print results
    table = Table(title="Simple Batch Transcription")
    table.add_column("File")
    table.add_column("Duration (s)")
    table.add_column("Language")
    table.add_column("Text (truncated)")

    for path, result in zip(audio_files, results):
        lang = (
            f"{result.detected_language} ({result.detected_language_prob:.2f})"
            if result.detected_language_prob
            else "?"
        )
        text_preview = result.transcription[:80] + ("..." if len(result.transcription) > 80 else "")
        table.add_row(path.name, f"{result.duration_sec:.1f}", lang, text_preview)

    rprint(table)
    log.info("[bold green]Example 1 completed[/]\n")


async def example_with_segments() -> None:
    """Example 2: Display per-segment transcription for each file."""
    log.info("[bold magenta]Starting Example 2: Batch Transcription with Segments[/]")
    log.info("Processing 2 files to display detailed timestamped segments")

    base_url = "http://127.0.0.1:8001"
    audio_dir = Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio")
    audio_files = list(audio_dir.glob("**/*.wav"))[:2]  # Fewer files for clearer segment output
    log.info(f"Selected audio files: {[p.name for p in audio_files]}")

    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        results = await fetch_batch_transcribe(client, audio_files, translate=False)

    for path, result in zip(audio_files, results):
        rprint(f"\n[bold cyan]File:[/] {path.name} | Duration: {result.duration_sec:.1f}s")
        if result.segments:
            for seg in result.segments:
                rprint(f"  [{seg.start:6.2f} - {seg.end:6.2f}] {seg.text}")
        else:
            rprint("  [dim]No segments available[/]")

    log.info("[bold green]Example 2 completed[/]\n")


async def example_translate_batch() -> None:
    """Example 3: Batch transcribe + translate (e.g., Japanese → English)."""
    log.info("[bold magenta]Starting Example 3: Batch Transcribe + Translate[/]")
    log.info("Performing transcription and translation on up to 3 audio files")

    base_url = "http://127.0.0.1:8001"
    audio_dir = Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio")
    audio_files = list(audio_dir.glob("**/*.wav"))[:3]
    log.info(f"Selected audio files: {[p.name for p in audio_files]}")

    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        results = await fetch_batch_transcribe(client, audio_files, translate=True)

    table = Table(title="Batch Transcribe + Translate")
    table.add_column("File")
    table.add_column("Original")
    table.add_column("Translation")

    for path, result in zip(audio_files, results):
        orig = result.transcription[:60] + ("..." if len(result.transcription) > 60 else "")
        trans = (result.translation or "[none]")[:60] + ("..." if result.translation and len(result.translation) > 60 else "")
        table.add_row(path.name, orig, trans)

    rprint(table)
    log.info("[bold green]Example 3 completed[/]\n")


async def main() -> None:
    log.info("[bold green]Running Whisper batch client examples...[/]\n")
    await example_simple_batch()
    # await example_with_segments()
    # await example_translate_batch()
    log.info("[bold green]All examples finished.[/]")


if __name__ == "__main__":
    asyncio.run(main())