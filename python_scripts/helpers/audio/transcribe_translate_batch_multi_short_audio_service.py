from pathlib import Path
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from faster_whisper import WhisperModel
from utils.audio_utils import resolve_audio_paths

import ctranslate2
from transformers import AutoTokenizer

from translator_types import (
    Device,
    BatchType,
    TranslationOptions,
    Translator,
)

TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"

# Configure rich logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("transcribe")

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

if __name__ == "__main__":
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio"
    output_dir = OUTPUT_DIR

    language = "ja"

    files = resolve_audio_paths(audio_dir)
    files = files[:20]  # Temporarily limit for testing

    # Change to translate as well
    transcriptions = batch_transcribe_and_translate_files(files, max_workers=4, output_dir=output_dir, language=language)

    # Optional: print first English translation as preview
    if transcriptions:
        print("\nPreview of first English translation:")
        print(transcriptions[0][:500] + "..." if len(transcriptions[0]) > 500 else transcriptions[0])
