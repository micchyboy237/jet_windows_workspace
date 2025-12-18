from pathlib import Path
from typing import List, Optional, Literal
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from faster_whisper import WhisperModel
from utils.audio_utils import resolve_audio_paths
import torch
import psutil

# Configure rich logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("transcribe")

# Configuration constants for dynamic fallback
CPU_LOAD_THRESHOLD = 85.0      # % average CPU load to trigger GPU attempt
CPU_CHECK_INTERVAL = 2.0       # seconds between load checks
CPU_SAMPLES_FOR_DECISION = 3   # how many high-load samples needed before switching


def get_current_cpu_load() -> float:
    """Return current CPU percent load (blocking for ~1 interval)."""
    return psutil.cpu_percent(interval=1)


def should_use_gpu() -> Literal["cuda", "cpu"]:
    """
    Dynamically decide device based on current system CPU load.
    If CPU is consistently high, try CUDA (fallback to CPU if unavailable).
    """
    if not torch.cuda.is_available():
        log.info("CUDA not available → using CPU")
        return "cpu"

    high_load_count = 0
    log.info("Checking CPU load to decide device (threshold ≥85% over 3 samples)...")

    for i in range(CPU_SAMPLES_FOR_DECISION):
        load = get_current_cpu_load()
        log.info(f"CPU load sample {i+1}/{CPU_SAMPLES_FOR_DECISION}: [bold yellow]{load:.1f}%[/bold yellow]")
        if load >= CPU_LOAD_THRESHOLD:
            high_load_count += 1
        time.sleep(CPU_CHECK_INTERVAL)

    if high_load_count >= CPU_SAMPLES_FOR_DECISION:
        log.info("[bold green]High sustained CPU load detected → offloading to CUDA[/bold green]")
        return "cuda"
    else:
        log.info("CPU load acceptable → using CPU (faster startup for small batches)")
        return "cpu"


def transcribe_file(model: WhisperModel, audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe a single file and return text."""
    log.info(f"Starting transcription: [bold cyan]{audio_path}[/bold cyan]")
    segments, _ = model.transcribe(audio_path, language=language, beam_size=1)
    text = " ".join(segment.text for segment in segments)
    log.info(f"Completed: [bold green]{audio_path}[/bold green]")
    return text


def batch_transcribe_files(
    audio_paths: List[str],
    max_workers: int = 4,
    output_dir: str | None = None,
    language: Optional[str] = None,
) -> List[str]:
    """Process multiple files with dynamic device selection, rich progress and logging."""
    if not audio_paths:
        log.warning("No audio files provided for transcription.")
        return []

    # Dynamic device decision
    device = should_use_gpu()
    compute_type: Literal["float16", "int8_float16", "int8"] = (
        "float16" if device == "cuda" else "int8"
    )

    if device == "cuda":
        log.info(
            f"Loading model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] on "
            f"[bold green]CUDA[/bold green] ({torch.cuda.get_device_name(0)}) with {compute_type}"
        )
    else:
        log.info(
            "Loading model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] "
            "on CPU (int8, 12 threads)"
        )

    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device=device,
        compute_type=compute_type,
        cpu_threads=12 if device == "cpu" else None,
    )

    # Adjust workers: GPU inference is usually fastest with 1-2 workers due to serialization
    effective_workers = 1 if device == "cuda" else max_workers
    if effective_workers != max_workers:
        log.info(f"Adjusted max_workers to {effective_workers} for {device.upper()} efficiency")

    results: List[str] = [None] * len(audio_paths)
    path_to_index = {path: idx for idx, path in enumerate(audio_paths)}

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Transcriptions will be saved to: [bold yellow]{output_path.resolve()}[/bold yellow]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[bold blue]Transcribing files...", total=len(audio_paths))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(transcribe_file, model, path, language): path
                for path in audio_paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results[path_to_index[path]] = result
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


if __name__ == "__main__":
    import shutil
    from pathlib import Path
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\generated\extract_parquet_data\audio"
    output_dir = OUTPUT_DIR
    language = "ja"
    files = resolve_audio_paths(audio_dir)
    files = files[:5]  # Temporarily limit for testing
    transcriptions = batch_transcribe_files(files, max_workers=4, output_dir=output_dir, language=language)
    if transcriptions:
        print("\nPreview of first transcription:")
        print(transcriptions[0][:500] + "..." if len(transcriptions[0]) > 500 else transcriptions[0])