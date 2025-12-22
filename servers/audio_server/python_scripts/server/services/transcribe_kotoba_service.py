# servers\audio_server\python_scripts\server\services\transcribe_kotoba_service.py
from __future__ import annotations

import numpy as np
from typing import Any, TypedDict

from faster_whisper import WhisperModel

from ..utils.audio_utils import load_audio
from ..utils.logger import get_logger

from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn



from transformers import AutoTokenizer

from .translator_types import (
    Translator,
    TRANSLATOR_MODEL_PATH,
    TRANSLATOR_TOKENIZER,
)

log = get_logger("kotoba_transcriber")

class KotobaTranscriptionResult(TypedDict):
    language: str
    language_prob: float
    text: str

_MODEL: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        log.info("[bold yellow]Loading Kotoba Whisper model[/] [cyan]kotoba-tech/kotoba-whisper-v2.0-faster[/] [blue]device=cuda[/] [green]compute_type=float32[/]")
        _MODEL = WhisperModel(
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            device="cuda",
            compute_type="float32",
        )
        log.info("[bold green]Kotoba Whisper model loaded and cached[/]")
    else:
        log.info("[dim]Kotoba model cache hit[/]")
    return _MODEL

def transcribe_kotoba_audio(
    audio: Any,
) -> KotobaTranscriptionResult:
    """
    Transcribe Japanese audio using the Kotoba Whisper faster-whisper model.
    
    Args:
        audio: Audio input compatible with load_audio (bytes, path, np.ndarray, etc.)
    
    Returns:
        Dict containing detected language, probability, and transcribed text.
    """
    model = _get_model()
    
    waveform: np.ndarray = load_audio(audio)  # Already float32, mono, 16kHz
    
    log.info("[bold cyan]Kotoba transcribe[/] processing → {:.2f}s audio".format(len(waveform) / 16000))
    
    segments, info = model.transcribe(
        waveform,
        language="ja",
        chunk_length=30,  # Better context than 15s
        condition_on_previous_text=False,  # Reduces repetitions/hallucinations
        temperature=0.0,  # More deterministic
    )
    
    text = " ".join(segment.text for segment in segments).strip()
    
    log.info(f"[bold green]Kotoba transcribed[/] → {text}")
    
    return {
        "language": info.language,
        "language_prob": info.language_probability,
        "text": text,
    }

def transcribe_file(model: WhisperModel, audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe a single file and return text."""
    log.info(f"Starting transcription: [bold cyan]{audio_path}[/bold cyan]")
    segments, _ = model.transcribe(audio_path, language=language)  # or keep 5
    text = " ".join(segment.text for segment in segments)
    log.info(f"Completed: [bold green]{audio_path}[/bold green]")
    return text

def batch_transcribe_files(
    audio_paths: List[str],
    max_workers: int = 4,
    output_dir: str | None = None,
    language: Optional[str] = None,
) -> List[str]:
    """Process multiple files in parallel with rich progress tracking and logging.
    
    Args:
        audio_paths: List of audio file paths to transcribe.
        max_workers: Number of parallel workers.
        output_dir: If provided, saves each transcription to a .txt file in this directory
                    using the same base name as the audio file.
    
    Returns:
        List of transcriptions in the same order as input paths.
    """
    if not audio_paths:
        log.warning("No audio files provided for transcription.")
        return []

    # log.info("Loading model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] on CPU (int8, 12 threads)")
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
        # cpu_threads=12,
    )

    results: List[str] = [None] * len(audio_paths)  # Preserve order
    path_to_index = {path: idx for idx, path in enumerate(audio_paths)}

    # Prepare output directory if saving is requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Transcriptions will be saved to: [bold yellow]{output_path.resolve()}[/bold yellow]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue]"),  # Added processed/total count
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[bold blue]Transcribing files...", total=len(audio_paths))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(transcribe_file, model, path, language): path
                for path in audio_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results[path_to_index[path]] = result

                    # Save to file if output_dir is provided
                    if output_dir:
                        stem = Path(path).stem
                        txt_path = output_path / f"{stem}.txt"
                        txt_path.write_text(result, encoding="utf-8")
                        log.debug(f"Saved transcription: [dim]{txt_path}[/dim]")

                except Exception as exc:
                    log.error(f"[bold red]Failed[/bold red] {path}: {exc}")
                    results[path_to_index[path]] = ""
                finally:
                    progress.update(task, advance=1)

    log.info("[bold green]Batch transcription completed[/bold green]")
    return results

def transcribe_and_translate_file(
    model: WhisperModel,
    translator: Translator,
    tokenizer: "AutoTokenizer",
    audio_path: str,
    language: Optional[str] = None,
) -> str:
    """Transcribe a single file to Japanese text, then translate to English."""
    log.info(f"Starting transcription + translation: [bold cyan]{audio_path}[/bold cyan]")
    segments, _ = model.transcribe(audio_path, language=language or "ja", beam_size=5, vad_filter=True)
    ja_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

    if not ja_text:
        log.warning(f"No Japanese text detected in {audio_path}")
        return ""

    # Tokenize Japanese text
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(ja_text))

    # Translate batch (single sentence)
    results = translator.translate_batch([source_tokens])
    en_tokens = results[0].hypotheses[0]  # Best hypothesis
    en_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(en_tokens), skip_special_tokens=True)

    log.info(f"Completed: [bold green]{audio_path}[/bold green]")
    return en_text

def batch_transcribe_and_translate_files(
    audio_paths: List[str],
    max_workers: int = 4,
    output_dir: str | None = None,
    language: Optional[str] = "ja",
) -> List[str]:
    """Process multiple files in parallel: transcribe (ja) → translate (en)."""
    if not audio_paths:
        log.warning("No audio files provided.")
        return []

    log.info("Loading Whisper model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] on CUDA (float32)")
    whisper_model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    # Load shared translator (thread-safe for read-only inference)
    translator = Translator(TRANSLATOR_MODEL_PATH, device="cpu", compute_type="int8", inter_threads=max_workers)
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    log.info("Loaded shared OPUS-MT ja→en translator")

    results: List[str] = [None] * len(audio_paths)
    path_to_index = {path: idx for idx, path in enumerate(audio_paths)}

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Translations will be saved to: [bold yellow]{output_path.resolve()}[/bold yellow]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue]"),  # Added processed/total count
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[bold blue]Transcribing + translating files...", total=len(audio_paths))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    transcribe_and_translate_file,
                    whisper_model,
                    translator,
                    tokenizer,
                    path,
                    language,
                ): path
                for path in audio_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results[path_to_index[path]] = result
                    if output_dir:
                        stem = Path(path).stem
                        txt_path = output_path / f"{stem}_en.txt"
                        txt_path.write_text(result, encoding="utf-8")
                        log.debug(f"Saved English translation: [dim]{txt_path}[/dim]")

                except Exception as exc:
                    log.error(f"[bold red]Failed[/bold red] {path}: {exc}")
                    results[path_to_index[path]] = ""
                finally:
                    progress.update(task, advance=1)

    log.info("[bold green]Batch transcription + translation completed[/bold green]")
    return results
