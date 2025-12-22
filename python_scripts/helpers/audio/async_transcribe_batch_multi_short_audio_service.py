from pathlib import Path
from typing import List, Optional, AsyncGenerator, TypedDict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from faster_whisper import WhisperModel
from utils.audio_utils import resolve_audio_paths

# Configure rich logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("transcribe")


class TranscriptionResult(TypedDict):
    """TypedDict representing the result for a single transcribed file."""
    index: int
    audio_path: str
    transcription: str
    success: bool  # True if transcription succeeded and produced non-empty text


def transcribe_file(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str] = None,
) -> str:
    """Transcribe a single file and return text."""
    log.info(f"Starting transcription: [bold cyan]{audio_path}[/bold cyan]")
    segments, _ = model.transcribe(audio_path, language=language)
    text = " ".join(segment.text.strip() for segment in segments)
    log.info(f"Completed: [bold green]{audio_path}[/bold green]")
    return text


async def batch_transcribe_files_async(
    audio_paths: List[str],
    max_workers: int = 4,
    output_dir: Optional[str] = None,
    language: Optional[str] = None,
) -> AsyncGenerator[TranscriptionResult, None]:
    """
    Async generator that yields a typed TranscriptionResult
    as soon as each file is processed.

    Provides immediate user feedback while maintaining parallel execution.
    """
    if not audio_paths:
        log.warning("No audio files provided for transcription.")
        return

    log.info("Loading Whisper model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] on CUDA (float32)")
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Transcriptions will be saved to: [bold yellow]{output_path.resolve()}[/bold yellow]")

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue]"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[bold blue]Transcribing files...", total=len(audio_paths))

            tasks = {
                loop.run_in_executor(
                    executor,
                    transcribe_file,
                    model,
                    path,
                    language,
                ): (idx, path)
                for idx, path in enumerate(audio_paths)
            }

            while tasks:
                done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                for future in done:
                    idx, path = tasks.pop(future)
                    try:
                        transcription = future.result()
                        success = bool(transcription.strip())

                        if output_path:
                            txt_path = output_path / f"{Path(path).stem}.txt"
                            txt_path.write_text(transcription, encoding="utf-8")
                            log.debug(f"Saved: [dim]{txt_path}[/dim]")

                        yield TranscriptionResult(
                            index=idx,
                            audio_path=path,
                            transcription=transcription,
                            success=success,
                        )
                    except Exception as exc:
                        log.error(f"[bold red]Failed[/bold red] {path}: {exc}")

                        yield TranscriptionResult(
                            index=idx,
                            audio_path=path,
                            transcription="",
                            success=False,
                        )
                    finally:
                        progress.update(task, advance=1)

    log.info("[bold green]Batch transcription completed[/bold green]")


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio"
    files = resolve_audio_paths(audio_dir)
    files = files[:5]  # Temporarily limit for testing

    async def main():
        async for result in batch_transcribe_files_async(
            files, max_workers=4, output_dir=str(OUTPUT_DIR), language="ja"
        ):
            status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
            print(f"\n{status} #[bold cyan]{result['index'] + 1}[/bold cyan]: {Path(result['audio_path']).name}")
            preview = result["transcription"][:300] + ("..." if len(result["transcription"]) > 300 else "")
            print(f"Preview: {preview}\n")

    asyncio.run(main())