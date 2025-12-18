from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import shutil
from pathlib import Path
import logging
import asyncio  # <-- added
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from faster_whisper import WhisperModel

from python_scripts.server.utils.logger import get_logger

log = get_logger("batch_transcription")

router = APIRouter(
    prefix="/batch_transcribe",  # optional – makes URLs cleaner if desired
    tags=["batch_transcription"],
)

# Global model shared across requests
model: Optional[WhisperModel] = None

class BatchTranscribeResponse(BaseModel):
    transcriptions: List[str]
    filenames: List[str]

@router.post("", response_model=BatchTranscribeResponse)  # → /batch_transcribe
async def batch_transcribe(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple audio files to transcribe"),
    language: Optional[str] = None,
    max_workers: int = 4,
):
    """Batch transcribe uploaded audio files (multipart/form-data)."""
    if not files:
        raise HTTPException(status_code=400, detail="No audio files provided")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet – please wait for server startup")

    temp_dir = Path(tempfile.mkdtemp(prefix="batch_transcribe_"))
    file_stems: List[str] = []

    try:
        for upload_file in files:
            if upload_file.content_type not in [
                "audio/mpeg", "audio/wav", "audio/ogg", "audio/webm", "audio/flac"
            ]:
                log.warning(f"Potentially unsupported content-type: {upload_file.content_type}")

            suffix = Path(upload_file.filename).suffix or ".audio"
            temp_file = temp_dir / f"{upload_file.filename}{suffix}"
            with open(temp_file, "wb") as f:
                shutil.copyfileobj(upload_file.file, f)
            file_stems.append(temp_file.name)

        transcriptions = await asyncio.get_running_loop().run_in_executor(
            None,
            process_batch,
            temp_dir,
            file_stems,
            language,
            max_workers,
        )

        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

        return BatchTranscribeResponse(
            transcriptions=transcriptions,
            filenames=[f.filename for f in files],
        )

    except Exception as e:
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        log.exception("Error during batch_transcribe")
        raise HTTPException(status_code=500, detail=str(e))

class AudioItem(BaseModel):
    filename: str = Field(..., description="Original filename (used for ordering and logging)")
    audio_bytes: bytes = Field(..., description="Raw audio file content")

class BatchTranscribeBytesRequest(BaseModel):
    audios: List[AudioItem]
    language: Optional[str] = None
    max_workers: int = 4

@router.post("/bytes", response_model=BatchTranscribeResponse)  # → /batch_transcribe/bytes
async def batch_transcribe_bytes(
    background_tasks: BackgroundTasks,
    payload: BatchTranscribeBytesRequest,
):
    """Batch transcribe audio sent as raw bytes in JSON payload."""
    if not payload.audios:
        raise HTTPException(status_code=400, detail="No audio items provided")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    temp_dir = Path(tempfile.mkdtemp(prefix="batch_bytes_"))
    file_paths: List[str] = []

    try:
        for item in payload.audios:
            suffix = Path(item.filename).suffix or ".wav"
            temp_file = temp_dir / f"{item.filename}{suffix}"
            temp_file.write_bytes(item.audio_bytes)
            file_paths.append(str(temp_file))

        transcriptions = await asyncio.get_running_loop().run_in_executor(
            None,
            process_batch,
            temp_dir,
            [Path(p).name for p in file_paths],
            payload.language,
            payload.max_workers,
        )

        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

        return BatchTranscribeResponse(
            transcriptions=transcriptions,
            filenames=[item.filename for item in payload.audios],
        )

    except Exception as e:
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        log.exception("Error in batch_transcribe_bytes")
        raise HTTPException(status_code=500, detail=str(e))

def transcribe_file(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe a single file using the global model."""
    assert model is not None
    log.info(f"Transcribing: [bold cyan]{Path(audio_path).name}[/bold cyan]")
    segments, _ = model.transcribe(audio_path, language=language, beam_size=1)
    text = " ".join(segment.text.strip() for segment in segments)
    log.info(f"Completed: [bold green]{Path(audio_path).name}[/bold green]")
    return text

def process_batch(
    temp_dir: Path,
    file_stems: List[str],
    language: Optional[str],
    max_workers: int,
) -> List[str]:
    """Core batch processing with rich progress bar."""
    audio_paths = [str(temp_dir / stem) for stem in file_stems]
    results: List[str] = [""] * len(audio_paths)
    path_to_index = {path: idx for idx, path in enumerate(audio_paths)}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[bold blue]Batch transcribing...", total=len(audio_paths))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(transcribe_file, path, language): path
                for path in audio_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results[path_to_index[path]] = result
                except Exception as exc:
                    log.error(f"[bold red]Transcription failed[/bold red] {Path(path).name}: {exc}")
                    results[path_to_index[path]] = ""
                finally:
                    progress.update(task, advance=1)

    log.info("[bold green]Batch transcription completed[/bold green]")
    return results

# Exposed startup function called from main.py
async def load_model() -> None:
    global model
    log.info("Loading model [bold magenta]kotoba-tech/kotoba-whisper-v2.0-faster[/bold magenta] (CUDA int8)")
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="int8",
    )
    log.info("[bold green]Model loaded successfully[/bold green]")