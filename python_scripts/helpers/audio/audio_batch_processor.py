# audio_batch_processor.py

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, Coroutine

from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import print as rprint  # optional, for colored prints if needed

from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from translator_types import Translator

# Update import to use new interface (DetailedTranslationResult as TranslationResult)
from transcribe_translate_batch_multi_short_audio_service import (
    DetailedTranslationResult as TranslationResult,  # Keep name for backward compatibility
    transcribe_to_japanese,
    process_single_result,
    TRANSLATOR_MODEL_PATH,
    TRANSLATOR_TOKENIZER,
)

log = logging.getLogger(__name__)

Callback = Callable[[TranslationResult], Coroutine[None, None, None]]


class AudioBatchProcessor:
    """
    Asynchronous batch processor that queues audio inputs (paths or bytes) and processes them
    in fixed-size batches using a background worker task.

    - Accepts str, Path, or bytes (bytes will be written to a temporary file)
    - Allows per-item async callback for immediate result handling
    - Single shared Whisper model (GPU) + shared translator (CPU)
    - Configurable batch_size and max_concurrent_transcriptions
    """

    def __init__(
        self,
        batch_size: int = 4,
        max_concurrent_transcriptions: int = 4,
        output_dir: Optional[str] = None,
        language: Optional[str] = "ja",
    ):
        self.batch_size = batch_size
        self.max_workers = max_concurrent_transcriptions
        self.output_dir = output_dir
        self.language = language

        self._queue: Queue[tuple[str, Callback | None, bool]] = Queue()  # path, callback, is_temp_file
        self._worker_task: asyncio.Task[None] | None = None
        self._shutdown = False

        # Lazily loaded in worker
        self._whisper_model: WhisperModel | None = None
        self._translator: Translator | None = None
        self._tokenizer: AutoTokenizer | None = None

        self._output_path: Path | None = Path(output_dir) if output_dir else None
        if self._output_path:
            self._output_path.mkdir(parents=True, exist_ok=True)

    async def add_audio(
        self,
        audio: str | Path | bytes,
        callback: Callback | None = None,
    ) -> None:
        """
        Add an audio item to the processing queue.

        - str | Path → treated as file path
        - bytes → written to temporary .wav file (will be deleted after processing)
        - callback → optional async callable that receives TranslationResult as soon as ready
        """
        if self._shutdown:
            raise RuntimeError("Processor has been shut down")

        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with open(f.name, "wb") as wav_file:
                    wav_file.write(audio)
                temp_path = f.name
            is_temp = True
        else:
            temp_path = str(Path(audio).resolve())
            is_temp = False

        if not self._worker_task or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

        self._queue.put((temp_path, callback, is_temp))

    async def _load_models(self) -> None:
        if self._whisper_model is None:
            log.info("Loading Whisper model on CUDA...")
            self._whisper_model = WhisperModel(
                "kotoba-tech/kotoba-whisper-v2.0-faster",
                device="cuda",
                compute_type="float32",
            )
            self._translator = Translator(
                TRANSLATOR_MODEL_PATH,
                device="cpu",
                compute_type="int8",
                inter_threads=self.max_workers,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
            log.info("Models loaded")

    async def _worker(self) -> None:
        await self._load_models()

        loop = asyncio.get_event_loop()
        items: list[tuple[int, str, Callback | None, bool]] = []  # idx, path, callback, is_temp
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[bold blue]{task.completed}/{task.total}[/bold blue]"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task_id = progress.add_task("[bold blue]Processing queued audio...", total=None)

                while not self._shutdown or not self._queue.empty():
                    # Collect up to batch_size items from queue
                    while len(items) < self.batch_size and not self._queue.empty():
                        path, cb, is_temp = self._queue.get()
                        items.append((len(items), path, cb, is_temp))

                    if not items:
                        await asyncio.sleep(0.1)
                        continue

                    # Submit transcription jobs to thread pool
                    tasks = [
                        loop.run_in_executor(
                            executor,
                            transcribe_to_japanese,
                            self._whisper_model,
                            path,
                            self.language,
                        )
                        for idx, path, cb, is_temp in items
                    ]

                    # Map asyncio tasks → metadata
                    task_to_meta = {
                        task: (idx, path, cb, is_temp)
                        for task, (idx, path, cb, is_temp) in zip(tasks, items)
                    }

                    pending = set(tasks)

                    # Process completions in order they finish
                    while pending:
                        done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )

                        for task in done:
                            idx, path, cb, is_temp = task_to_meta[task]

                            try:
                                # CHANGED: Now expects tuple (ja_text, segments)
                                ja_result = task.result()
                                result = process_single_result(
                                    path=path,
                                    transcription_result=ja_result,
                                    translator=self._translator,  # type: ignore[arg-type]
                                    tokenizer=self._tokenizer,    # type: ignore[arg-type]
                                    output_path=self._output_path,
                                    idx=idx,
                                )

                                status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
                                log.info(f"{status} #{idx + 1}: {Path(path).name}")
                                # Optionally: print more detailed preview than just 'translation'
                                log.info(
                                    f"Preview: {result['translation'][:120]}{'...' if len(result['translation']) > 120 else ''}"
                                )

                                if cb:
                                    await cb(result)

                            except Exception as exc:
                                log.error(f"[bold red]Failed[/bold red] {path}: {exc}")

                                error_result: TranslationResult = TranslationResult(
                                    index=idx,
                                    audio_path=path,
                                    translation="",
                                    success=False,
                                )

                                if cb:
                                    await cb(error_result)

                            finally:
                                # Advance progress using the correct Rich TaskID (int)
                                progress.advance(task_id)
                                completed_count += 1

                                # Clean up temporary file if it was created from bytes
                                if is_temp:
                                    try:
                                        os.unlink(path)
                                    except Exception:
                                        pass

                    # Clear current batch
                    items.clear()

                # Final progress update
                progress.update(
                    task_id,
                    description=f"[bold green]Completed {completed_count} audio file{'s' if completed_count != 1 else ''}[/bold green]",
                )

        log.info("[bold green]Audio batch processing completed[/bold green]")

    async def shutdown(self, wait: bool = True) -> None:
        """Signal shutdown and optionally wait for remaining items to process."""
        self._shutdown = True
        if wait and self._worker_task:
            await self._worker_task
