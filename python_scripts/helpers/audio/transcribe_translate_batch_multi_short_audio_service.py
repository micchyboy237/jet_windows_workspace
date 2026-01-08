# transcribe_translate_batch_multi_short_audio_service

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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"

log = logging.getLogger(__name__)

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

class DetailedTranslationResult(TypedDict):
    """Enhanced result with full transcription details, timing, and quality metrics."""
    index: int
    audio_path: str
    japanese_text: str                  # Raw transcribed Japanese
    translation: str                    # English translation
    success: bool

    # Timing (in milliseconds)
    start_ms: int
    end_ms: int
    duration_ms: int

    # Quality scores (normalized where possible)
    avg_logprob: float                  # Higher is better
    no_speech_prob: float               # Lower is better
    compression_ratio: float            # Closer to 1.0 is better

    # Full segment list for advanced use / debugging
    segments: List[SegmentDict]

# For backward compatibility as in the diff:
TranslationResult = DetailedTranslationResult

# ──────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_to_japanese(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str] = None,
) -> tuple[str, List[SegmentDict]]:
    """
    Perform transcription and return both the concatenated Japanese text
    and the full list of segments with timing and quality metrics.
    """
    log.info(f"Starting transcription: [bold cyan]{audio_path}[/bold cyan]")
    segments_iter, _ = model.transcribe(
        audio_path,
        language=language or "ja",
        beam_size=5,
        vad_filter=False,
    )
    segments: List[SegmentDict] = [dataclasses.asdict(s) for s in segments_iter]

    ja_text = " ".join(
        segment["text"].strip() for segment in segments if segment["text"].strip()
    ).strip()

    if not ja_text:
        log.warning(f"No Japanese text detected in {audio_path}")

    log.debug(f"Transcribed Japanese (len={len(ja_text)}): {ja_text[:80]}...")
    return ja_text, segments


def translate_to_english(
    translator: Translator,
    tokenizer: AutoTokenizer,
    japanese_text: str,
) -> str:
    """Translate Japanese text to English using OPUS-MT (CPU operation)."""
    if not japanese_text:
        return ""

    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(japanese_text))
    results = translator.translate_batch([source_tokens])
    en_tokens = results[0].hypotheses[0]
    en_text = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(en_tokens),
        skip_special_tokens=True
    )
    return en_text


def process_single_result(
    path: str,
    transcription_result: tuple[str, List[SegmentDict]],
    translator: Translator,
    tokenizer: AutoTokenizer,
    output_path: Optional[Path],
    idx: int,
) -> DetailedTranslationResult:
    """Create enriched result with timing, scores, and full segment data."""
    ja_text, segments = transcription_result

    en_text = translate_to_english(translator, tokenizer, ja_text)
    success = bool(en_text.strip())

    if output_path and success:
        txt_path = output_path / f"{Path(path).stem}_en.txt"
        txt_path.write_text(en_text, encoding="utf-8")
        log.debug(f"Saved translation: [dim]{txt_path}[/dim]")

    # Derive timing and aggregate scores
    if segments:
        start_ms = int(min(seg["start"] for seg in segments) * 1000)
        end_ms = int(max(seg["end"] for seg in segments) * 1000)
        duration_ms = end_ms - start_ms

        avg_logprob = float(sum(seg["avg_logprob"] for seg in segments) / len(segments))
        no_speech_prob = float(sum(seg["no_speech_prob"] for seg in segments) / len(segments))
        compression_ratio = float(sum(seg["compression_ratio"] for seg in segments) / len(segments))
    else:
        start_ms = end_ms = duration_ms = 0
        avg_logprob = no_speech_prob = compression_ratio = 0.0

    return DetailedTranslationResult(
        index=idx,
        audio_path=path,
        japanese_text=ja_text,
        translation=en_text,
        success=success,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
        segments=segments,
    )

# ──────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSORS
# ──────────────────────────────────────────────────────────────────────────────

def batch_transcribe_and_translate_files(
    audio_paths: List[str],
    max_workers: int = 4,
    output_dir: Optional[str] = None,
    language: Optional[str] = "ja",
) -> Generator[TranslationResult, None, None]:
    """
    Synchronous generator that yields TranslationResult as soon as each file completes.
    Transcription runs in threads; translation runs in main thread.
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
                    transcribe_to_japanese,
                    whisper_model,
                    path,
                    language,
                ): path
                for path in audio_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                idx = path_to_index[path]
                try:
                    transcription_result = future.result()
                    result = process_single_result(
                        path=path,
                        transcription_result=transcription_result,
                        translator=translator,
                        tokenizer=tokenizer,
                        output_path=output_path,
                        idx=idx,
                    )
                    status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
                    log.info(f"{status} #{idx+1}: {Path(path).name}")
                    log.info(f"Japanese : {result['japanese_text']}")
                    log.info(f"English  : {result['translation']}")
                    log.info(
                        f"Timing   : {result['start_ms']}ms → {result['end_ms']}ms "
                        f"({result['duration_ms']}ms)"
                    )
                    log.info(
                        f"Scores   : logprob={result['avg_logprob']:.3f} | "
                        f"no_speech={result['no_speech_prob']:.3f} | "
                        f"compression={result['compression_ratio']:.2f}"
                    )
                    yield result
                except Exception as exc:
                    log.error(f"[bold red]Failed[/bold red] {path}: {exc}")
                    yield TranslationResult(
                        index=idx,
                        audio_path=path,
                        japanese_text="",
                        translation="",
                        success=False,
                        start_ms=0,
                        end_ms=0,
                        duration_ms=0,
                        avg_logprob=0.0,
                        no_speech_prob=0.0,
                        compression_ratio=0.0,
                        segments=[],
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
    Async generator version with the same separation of transcription/translation.
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
                    transcribe_to_japanese,
                    whisper_model,
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
                        transcription_result = future.result()
                        result = process_single_result(
                            path=path,
                            transcription_result=transcription_result,
                            translator=translator,
                            tokenizer=tokenizer,
                            output_path=output_path,
                            idx=idx,
                        )
                        yield result
                    except Exception as exc:
                        log.error(f"[bold red]Failed[/bold red] {path}: {exc}")
                        yield TranslationResult(
                            index=idx,
                            audio_path=path,
                            japanese_text="",
                            translation="",
                            success=False,
                            start_ms=0,
                            end_ms=0,
                            duration_ms=0,
                            avg_logprob=0.0,
                            no_speech_prob=0.0,
                            compression_ratio=0.0,
                            segments=[],
                        )
                    finally:
                        progress.update(task, advance=1)

    log.info("[bold green]Batch transcription + translation completed[/bold green]")


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\jet_scripts\modules\generated\speech_segmentation\segments"
    files = resolve_audio_paths(audio_dir, recursive=True)

    print("=== Synchronous Generator Version ===")
    for result in batch_transcribe_and_translate_files(
        files, max_workers=4, output_dir=str(OUTPUT_DIR), language="ja"
    ):
        status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
        print(f"{status} #{result['index']+1}: {Path(result['audio_path']).name}")
        print(f"Japanese : {result['japanese_text']}")
        print(f"English  : {result['translation']}")
        print(
            f"Timing   : {result['start_ms']}ms → {result['end_ms']}ms "
            f"({result['duration_ms']}ms)"
        )
        print(
            f"Scores   : logprob={result['avg_logprob']:.3f} | "
            f"no_speech={result['no_speech_prob']:.3f} | "
            f"compression={result['compression_ratio']:.2f}"
        )
        print()