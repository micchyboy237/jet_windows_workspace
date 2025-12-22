# client/whisper_batch_client.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


class TranscriptionResponse(BaseModel):
    duration_sec: float
    detected_language: Optional[str] = None
    detected_language_prob: Optional[float] = None
    transcription: str
    translation: Optional[str] = None


class BatchTranscriptionResult(BaseModel):
    responses: List[TranscriptionResponse]


class WhisperBatchClient:
    """Reusable async HTTPX client for the Whisper batch transcription server.

    Designed for high concurrency, connection pooling, and clean lifecycle management.
    Supports both transcribe-only and transcribe+translate endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 300.0,
        limits: Optional[httpx.Limits] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout)
        self.limits = limits or httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "WhisperBatchClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        await self.close()

    async def start(self) -> None:
        """Initialize the async client (connection pooling)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=self.limits,
                follow_redirects=True,
            )

    async def close(self) -> None:
        """Close the client and all connections."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not started. Use async with WhisperBatchClient() or await client.start()")
        return self._client

    async def batch_transcribe(
        self,
        audio_paths: List[Path | str],
        translate: bool = False,
    ) -> List[TranscriptionResponse]:
        """Send multiple audio files for batch transcription (± translation).

        Args:
            audio_paths: List of local audio file paths.
            translate: If True, use the transcribe+translate endpoint.

        Returns:
            Ordered list of TranscriptionResponse matching input order.
        """
        files = [("files", (Path(p).name, open(p, "rb"))) for p in audio_paths]

        endpoint = "/batch/transcribe_translate" if translate else "/batch/transcribe"

        response = await self.client.post(endpoint, files=files)  # type: ignore[arg-type]
        response.raise_for_status()

        result = BatchTranscriptionResult(responses=response.json())
        return result.responses


# Example usage script
async def main() -> None:
    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio"
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("**/*.wav"))[:8]  # Example: up to 8 files

    if not audio_files:
        rprint("[red]No audio files found in samples/[/red]")
        return

    rprint(f"[bold green]Found {len(audio_files)} audio files[/bold green]")

    async with WhisperBatchClient(base_url="http://127.0.0.1:8001") as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Transcribing batch...", total=1)

            results = await client.batch_transcribe(
                audio_paths=audio_files,
                translate=False,  # Set True for transcription + translation to English
            )

            progress.update(task, advance=1)

    # Pretty output with rich
    table = Table(title="Batch Transcription Results")
    table.add_column("File", style="cyan")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Language", style="magenta")
    table.add_column("Text", style="green")

    for file_path, result in zip(audio_files, results):
        lang = f"{result.detected_language} ({result.detected_language_prob:.2f})" if result.detected_language_prob else result.detected_language or "?"
        table.add_row(
            file_path.name,
            f"{result.duration_sec:.1f}",
            lang,
            result.transcription[:100] + ("..." if len(result.transcription) > 100 else ""),
        )

    rprint(table)


if __name__ == "__main__":
    asyncio.run(main())