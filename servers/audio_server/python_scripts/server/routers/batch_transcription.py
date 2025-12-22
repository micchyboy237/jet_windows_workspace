# servers/audio_server/python_scripts/server/routers/batch_transcription.py
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Request, HTTPException, UploadFile

from python_scripts.server.models.responses import TranscriptionResponse, TranscriptionSegment
from python_scripts.server.services.batch_transcription_service import (
    batch_transcribe_bytes,
    batch_transcribe_and_translate_bytes,
)
from python_scripts.server.utils.audio_utils import load_audio
from python_scripts.server.utils.logger import get_logger

log = get_logger("batch_transcription_router")

router = APIRouter(prefix="/batch", tags=["batch"])


@router.post("/transcribe", response_model=List[TranscriptionResponse])
async def batch_transcribe(request: Request):
    """
    Batch transcription endpoint (transcribe-only, no translation).

    Accepts multipart/form-data with multiple audio files under any field name.
    Returns a list of TranscriptionResponse objects in the order the files were received.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=400,
            detail="This endpoint requires multipart/form-data with multiple audio files",
        )

    form = await request.form()
    audio_files: List[UploadFile] = [
        f for f in form.values() if hasattr(f, "filename") and f.filename
    ]
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files found in multipart form")

    audio_bytes_list: List[bytes] = []
    for upload_file in audio_files:
        content = await upload_file.read()
        audio_bytes_list.append(content)

    log.info(f"[bold cyan]Batch transcribe[/] request → {len(audio_bytes_list)} files")

    results = batch_transcribe_bytes(audio_bytes_list)

    responses: List[TranscriptionResponse] = []
    for result in results:
        duration_sec = (
            len(load_audio(result["audio_bytes"])) / 16000
            if result["audio_bytes"]
            else 0.0
        )
        response = TranscriptionResponse(
            duration_sec=round(duration_sec, 3),
            detected_language=result.get("language"),
            detected_language_prob=(
                round(result.get("language_prob"), 4)
                if result.get("language_prob")
                else None
            ),
            transcription=result["text"].strip(),
            translation=None,
        )
        # Attach segments if available (for streaming-capable clients)
        if "segments" in result and result["segments"]:
            response.segments = [
                TranscriptionSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                )
                for seg in result["segments"]
            ]
        responses.append(response)

    log.info(f"[bold green]Batch transcription completed[/] → {len(responses)} files")
    return responses


@router.post("/transcribe_translate", response_model=List[TranscriptionResponse])
async def batch_transcribe_translate(request: Request):
    """
    Batch transcription + translation endpoint.

    Accepts multipart/form-data with multiple audio files under any field name.
    Returns a list of TranscriptionResponse objects (with translation) in the order the files were received.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=400,
            detail="This endpoint requires multipart/form-data with multiple audio files",
        )

    form = await request.form()
    audio_files: List[UploadFile] = [
        f for f in form.values() if hasattr(f, "filename") and f.filename
    ]
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files found in multipart form")

    audio_bytes_list: List[bytes] = []
    for upload_file in audio_files:
        content = await upload_file.read()
        audio_bytes_list.append(content)

    log.info(f"[bold cyan]Batch transcribe+translate[/] request → {len(audio_bytes_list)} files")

    results = batch_transcribe_and_translate_bytes(audio_bytes_list)

    responses: List[TranscriptionResponse] = []
    for result in results:
        duration_sec = (
            len(load_audio(result["audio_bytes"])) / 16000
            if result["audio_bytes"]
            else 0.0
        )
        response = TranscriptionResponse(
            duration_sec=round(duration_sec, 3),
            detected_language=result.get("language"),
            detected_language_prob=(
                round(result.get("language_prob"), 4)
                if result.get("language_prob")
                else None
            ),
            transcription=result["text"].strip(),
            translation=result["translation"].strip() if result.get("translation") else None,
        )
        # Attach segments if available
        if "segments" in result and result["segments"]:
            response.segments = [
                TranscriptionSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                )
                for seg in result["segments"]
            ]
        responses.append(response)

    log.info(f"[bold green]Batch transcribe+translate completed[/] → {len(responses)} files")
    return responses