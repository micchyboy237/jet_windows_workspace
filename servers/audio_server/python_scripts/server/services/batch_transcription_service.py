# servers/audio_server/python_scripts/server/services/batch_transcription_service.py
from __future__ import annotations

import numpy as np
from typing import List

from faster_whisper import WhisperModel
from ..utils.audio_utils import load_audio
from ..utils.logger import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from ..services.transcribe_service import transcribe_audio
from ..services.translate_service import translate_text
from .transcriber_types import TranscriptionResult, BatchResult

log = get_logger("batch_transcription")

_MODEL: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load and cache the faster-whisper model used for batch processing."""
    global _MODEL
    if _MODEL is None:
        log.info(
            "[bold yellow]Loading faster-whisper model[/] [cyan]large-v3[/] "
            "[blue]device=cuda[/] [green]compute_type=int8_float16[/]"
        )
        _MODEL = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="int8_float16",
        )
        log.info("[bold green]Batch model loaded and cached[/]")
    else:
        log.info("[dim]Batch model cache hit[/]")
    return _MODEL


def _transcribe_single(audio_bytes: bytes) -> TranscriptionResult:
    """Internal helper to transcribe a single audio byte stream."""
    result = transcribe_audio(audio_bytes)
    return result


def batch_transcribe_bytes(audio_bytes_list: List[bytes]) -> List[BatchResult]:
    """
    Batch transcribe a list of in-memory audio byte streams (transcribe-only).

    Uses threading for parallel transcription with rich progress bar.
    """
    if not audio_bytes_list:
        log.warning("No audio bytes provided for batch transcription.")
        return []

    results: list[BatchResult] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue]"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[bold blue]Batch transcribing audio bytes...", total=len(audio_bytes_list)
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_transcribe_single, audio_bytes): idx
                for idx, audio_bytes in enumerate(audio_bytes_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(
                        BatchResult(
                            audio_bytes=audio_bytes_list[idx],
                            language=result.get("language"),
                            language_prob=result.get("language_prob"),
                            text=result["text"],
                            translation=None,
                        )
                    )
                except Exception as exc:
                    log.error(f"[bold red]Failed batch item {idx}[/bold red]: {exc}")
                    results.append(
                        BatchResult(
                            audio_bytes=audio_bytes_list[idx],
                            language=None,
                            language_prob=None,
                            text="",
                            translation=None,
                        )
                    )
                finally:
                    progress.update(task, advance=1)

    log.info("[bold green]Batch transcription completed[/bold green]")
    return results


def batch_transcribe_and_translate_bytes(audio_bytes_list: List[bytes]) -> List[BatchResult]:
    """
    Batch transcribe + translate a list of in-memory audio byte streams.

    First performs batch transcription, then translates non-empty results.
    """
    transcribe_results = batch_transcribe_bytes(audio_bytes_list)

    for item in transcribe_results:
        if item["text"].strip():
            translated_list = translate_text(text=item["text"], device="cuda")
            item["translation"] = " ".join(translated_list).strip() or None
        else:
            item["translation"] = None

    log.info("[bold green]Batch transcription + translation completed[/bold green]")
    return transcribe_results