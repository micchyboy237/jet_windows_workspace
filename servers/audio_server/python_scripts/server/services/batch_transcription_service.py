# servers/audio_server/python_scripts/server/services/batch_transcription_service.py
from __future__ import annotations

import numpy as np
from typing import List

from faster_whisper import WhisperModel, BatchedInferencePipeline
from ..utils.audio_utils import load_audio, resample_audio
from ..utils.logger import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TransferSpeedColumn

from ..services.transcribe_service import transcribe_audio  # kept for compatibility, but batch uses direct pipeline
from ..services.translate_service import translate_text
from .transcriber_types import TranscriptionResult, BatchResult

log = get_logger("batch_transcription")

DEFAULT_TRANSCRIBE_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float32"

_MODEL: WhisperModel | None = None


def _get_model(model: str = DEFAULT_TRANSCRIBE_MODEL, device: str = DEFAULT_DEVICE, compute_type: str = DEFAULT_COMPUTE_TYPE) -> WhisperModel:
    """Lazy-load and cache the faster-whisper model used for batch processing."""
    global _MODEL
    if _MODEL is None:
        log.info(
            f"[bold yellow]Loading faster-whisper model[/] [cyan]{model}[/] "
            f"[blue]device={device}[/] [green]compute_type={compute_type}[/]"
        )
        _MODEL = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )
        log.info("[bold green]Batch model loaded and cached[/]")
    else:
        log.info("[dim]Batch model cache hit[/]")
    return _MODEL

def _get_batched_pipeline() -> BatchedInferencePipeline:
    """Lazy-load and return a batched inference pipeline for higher throughput."""
    model = _get_model()
    return BatchedInferencePipeline(
        model=model,
        batch_size=4,  # Tune as needed for your GPU
    )


def _transcribe_single(audio_bytes: bytes) -> TranscriptionResult:
    """Internal helper to transcribe a single audio byte stream."""
    result = transcribe_audio(audio_bytes)
    return result


def batch_transcribe_bytes(audio_bytes_list: List[bytes]) -> List[BatchResult]:
    """
    Batch transcribe a list of in-memory audio byte streams (transcribe-only).

    Uses batched pipeline if possible, falls back to threaded single if needed.
    """
    if not audio_bytes_list:
        log.warning("No audio bytes provided for batch transcription.")
        return []  # type: ignore

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
            "[bold blue]Batch transcribing audio files...", total=len(audio_bytes_list)
        )
        try:
            pipeline = _get_batched_pipeline()
            audio_arrays = [
                resample_audio(load_audio(ab), target_sr=16000)
                for ab in audio_bytes_list
            ]
            batch_results = pipeline.transcribe(
                audio_arrays,
                batch_size=4,
                vad_filter=True,
                beam_size=5,
                word_timestamps=False,
                suppress_blank=True,
                temperature=0.0,
            )
            for idx, (segments_gen, info) in enumerate(batch_results):
                # The transcribe method yields (generator of Segment, TranscriptionInfo) per input
                segments = list(segments_gen)
                full_text = " ".join(seg.text.strip() for seg in segments).strip()
                results.append(
                    BatchResult(
                        audio_bytes=audio_bytes_list[idx],
                        language=info.language,
                        language_prob=info.language_probability,
                        text=full_text,
                        translation=None,
                    )
                )
                progress.update(task, advance=1)
        except Exception as exc:
            log.error(f"[bold red]Batch processing failed[/bold red]: {exc}")
            # Fallback to single-by-single if batch fails (rare)
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
                    except Exception as fallback_exc:
                        log.error(f"[bold red]Failed batch item {idx}[/bold red]: {fallback_exc}")
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
        # Accept both dicts and BatchResult pydantic objects
        text_val = item["text"] if isinstance(item, dict) else getattr(item, "text", "")
        if text_val and text_val.strip():
            translated_list = translate_text(text=text_val, device="cuda")
            translation_val = " ".join(translated_list).strip() or None
            if isinstance(item, dict):
                item["translation"] = translation_val
            else:
                item.translation = translation_val
        else:
            if isinstance(item, dict):
                item["translation"] = None
            else:
                item.translation = None

    log.info("[bold green]Batch transcription + translation completed[/bold green]")
    return transcribe_results