import logging
import asyncio
import dataclasses
from pathlib import Path
from typing import List, Optional, AsyncGenerator, Generator, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from faster_whisper import WhisperModel
from utils.audio_utils import resolve_audio_paths
from transformers import AutoTokenizer
from translator_types import Translator

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

class WordDict(TypedDict):
    start: float
    end: float
    word: str
    probability: float

class SegmentDict(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[WordDict]]
    temperature: Optional[float]

class TranslationResult(TypedDict):
    """TypedDict representing the result for a single processed file."""
    index: int
    audio_path: str
    translation: str
    success: bool  # True if translation succeeded and produced non-empty text


def transcribe_and_translate_file(
    model: WhisperModel,
    translator: Translator,
    tokenizer: "AutoTokenizer",
    audio_path: str,
    language: Optional[str] = None,
) -> str:
    """Transcribe a single file to Japanese text, then translate to English."""
    log.info(f"Starting transcription + translation: [bold cyan]{audio_path}[/bold cyan]")

    segments_iter, _ = model.transcribe(audio_path, language=language or "ja", beam_size=5, vad_filter=False)

    segments: List[SegmentDict] = []
    for s in segments_iter:
        segments.append(dataclasses.asdict(s))

    ja_text = " ".join(segment["text"].strip() for segment in segments if segment["text"].strip())
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
    output_dir: Optional[str] = None,
    language: Optional[str] = "ja",
) -> Generator[TranslationResult, None, None]:
    """
    Synchronous generator that yields TranslationResult as soon as each file completes.
    Provides immediate feedback without requiring async/await.
    """
    if not audio_paths:
        log.warning("No audio files provided.")
        return

    log.info("Loading Whisper model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] on CUDA (float32)")
    whisper_model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    translator = Translator(
        TRANSLATOR_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=max_workers,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    log.info("Loaded shared OPUS-MT ja→en translator")

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Translations will be saved to: [bold yellow]{output_path.resolve()}[/bold yellow]")

    path_to_index = {path: idx for idx, path in enumerate(audio_paths)}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue]"),
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
                idx = path_to_index[path]
                try:
                    result = future.result()
                    success = bool(result.strip())

                    if output_path:
                        txt_path = output_path / f"{Path(path).stem}_en.txt"
                        txt_path.write_text(result, encoding="utf-8")
                        log.debug(f"Saved: [dim]{txt_path}[/dim]")

                    yield TranslationResult(
                        index=idx,
                        audio_path=path,
                        translation=result,
                        success=success,
                    )
                except Exception as exc:
                    log.error(f"[bold red]Failed[/bold red] {path}: {exc}")

                    yield TranslationResult(
                        index=idx,
                        audio_path=path,
                        translation="",
                        success=False,
                    )
                finally:
                    progress.update(task, advance=1)

    log.info("[bold green]Batch transcription + translation completed[/bold green]")


async def batch_transcribe_and_translate_files_async(
    audio_paths: List[str],
    max_workers: int = 4,
    output_dir: Optional[str] = None,
    language: Optional[str] = "ja",
) -> AsyncGenerator[TranslationResult, None]:
    """
    Async generator that yields a typed TranslationResult dictionary
    as soon as each file is processed.
    Provides immediate user feedback while maintaining parallel execution.
    """
    if not audio_paths:
        log.warning("No audio files provided.")
        return

    log.info("Loading Whisper model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] on CUDA (float32)")
    whisper_model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    translator = Translator(
        TRANSLATOR_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=max_workers,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    log.info("Loaded shared OPUS-MT ja→en translator")

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Translations will be saved to: [bold yellow]{output_path.resolve()}[/bold yellow]")

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
            task = progress.add_task("[bold blue]Transcribing + translating files...", total=len(audio_paths))

            tasks = {
                loop.run_in_executor(
                    executor,
                    transcribe_and_translate_file,
                    whisper_model,
                    translator,
                    tokenizer,
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
                        result = future.result()
                        success = bool(result.strip())

                        if output_path:
                            txt_path = output_path / f"{Path(path).stem}_en.txt"
                            txt_path.write_text(result, encoding="utf-8")
                            log.debug(f"Saved: [dim]{txt_path}[/dim]")

                        log.info(f"Completed: [bold green]{path}[/bold green]")

                        yield TranslationResult(
                            index=idx,
                            audio_path=path,
                            translation=result,
                            success=success,
                        )
                    except Exception as exc:
                        log.error(f"[bold red]Failed[/bold red] {path}: {exc}")

                        yield TranslationResult(
                            index=idx,
                            audio_path=path,
                            translation="",
                            success=False,
                        )
                    finally:
                        progress.update(task, advance=1)

    log.info("[bold green]Batch transcription + translation completed[/bold green]")


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio"
    files = resolve_audio_paths(audio_dir)
    files = files[:5]  # Temporarily limit for testing

    print("=== Synchronous Generator Version ===")
    for result in batch_transcribe_and_translate_files(
        files, max_workers=4, output_dir=str(OUTPUT_DIR), language="ja"
    ):
        status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
        print(f"\n{status} #[bold cyan]{result['index'] + 1}[/bold cyan]: {Path(result['audio_path']).name}")
        preview = result["translation"][:300] + ("..." if len(result["translation"]) > 300 else "")
        print(f"Preview: {preview}\n")

    # Uncomment to test async version
    # print("=== Async Version ===")
    # async def main():
    #     async for result in batch_transcribe_and_translate_files_async(
    #         files, max_workers=4, output_dir=str(OUTPUT_DIR), language="ja"
    #     ):
    #         status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
    #         print(f"\n{status} #[bold cyan]{result['index'] + 1}[/bold cyan]: {Path(result['audio_path']).name}")
    #         preview = result["translation"][:300] + ("..." if len(result["translation"]) > 300 else "")
    #         print(f"Preview: {preview}\n")
    
    # asyncio.run(main())