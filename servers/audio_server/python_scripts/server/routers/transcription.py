# servers/audio_server/python_scripts/server/routers/transcription.py
from __future__ import annotations
import numpy as np
import dataclasses
import time
from typing import Optional, Literal, Annotated, Union
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, Depends
from faster_whisper import WhisperModel
from starlette.requests import ClientDisconnect
from pydantic import BaseModel
from python_scripts.server.models.responses import TranscriptionResponse
from python_scripts.server.services.transcribe_service import transcribe_audio
from python_scripts.server.services.transcribe_service_old import get_transcriber
from python_scripts.server.services.translate_service import translate_text
from python_scripts.server.services.whisper_ct2_transcriber import QuantizedModelSizes
from python_scripts.server.utils.audio_utils import load_audio
from python_scripts.server.utils.logger import get_logger
from python_scripts.server.services.transcribe_kotoba_service import transcribe_kotoba_audio

# Configure logger for this module
log = get_logger("transcription")

router = APIRouter()

# Helper function to DRY the audio extraction
async def _extract_audio_content(request: Request) -> tuple[bytes, int]:
    """Shared logic to extract audio bytes from multipart or raw body (identical to existing endpoints)."""
    content_type = request.headers.get("content-type", "")
    is_multipart = content_type.startswith("multipart/form-data")

    if is_multipart:
        try:
            form = await request.form()
        except ClientDisconnect:
            log.info("Client disconnected during multipart form parsing")
            raise HTTPException(
                status_code=499,
                detail="Client closed connection during upload"
            )
        except Exception as exc:
            log.warning(f"Multipart parsing failed: {exc.__class__.__name__}")
            raise HTTPException(400, "Invalid multipart form data") from exc

        upload_file: Optional[UploadFile] = None
        for field in ["file", "audio", "data", "upload"]:
            if field in form and hasattr(form[field], "filename"):
                upload_file = form[field]
                break
        if not upload_file:
            raise HTTPException(400, "No audio file found in multipart form")

        try:
            content = await upload_file.read()
        except ClientDisconnect:
            log.info("Client disconnected while reading uploaded file content")
            raise HTTPException(499, "Client closed connection during file upload")
    else:
        try:
            content = await request.body()
        except ClientDisconnect:
            log.info("Client disconnected during raw body read")
            raise HTTPException(499, "Client closed connection during upload")

        if not content:
            raise HTTPException(400, "Empty body")

    # ──────────────────────────────────────────────────────────────
    # The rest remains unchanged (content-type detection, PCM→WAV conversion, etc.)
    # ...
    content_type_detected = request.headers.get("content-type", "").split(";", 1)[0].strip()
    if content_type_detected in ("application/octet-stream", "") or content_type_detected.endswith("/octet-stream"):
        try:
            audio_np = np.frombuffer(content, dtype=np.float32)
            if audio_np.size > 1000 and not np.all(audio_np == 0):
                import io, wave
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(4)
                    wf.setframerate(16000)
                    wf.writeframes(content)
                wav_buffer.seek(0)
                content = wav_buffer.read()
                log.info(f"[bold blue]Converted raw PCM to WAV[/] → {len(audio_np):,} samples")
        except Exception as e:
            log.warning(f"Raw PCM conversion failed: {e}")

    return content, len(content)

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: Request):
    """
    Transcribe-only endpoint – same universal file reception as /transcribe_translate
    but returns only transcription (no translation to English).
    """
    content, size = await _extract_audio_content(request)
    log.info(f"[bold cyan]Transcribe-only[/] request → {size/1024/1024:.2f} MB")

    result = transcribe_audio(
        content,
        # device="cuda",
    )

    duration_sec = len(load_audio(content)) / 16000

    response_data = {
        "duration_sec": round(duration_sec, 3),
        "detected_language": result.get("language"),
        "detected_language_prob": round(result.get("language_prob"), 4) if result.get("language_prob") else None,
        "transcription": result["text"].strip(),
        "translation": None,
    }

    log.info(f"[bold green]Transcribed[/] → {response_data['transcription'][:60]}...")

    return TranscriptionResponse(**response_data)

@router.post("/transcribe_translate", response_model=TranscriptionResponse)
async def transcribe_translate(request: Request):
    """
    Universal file receiver – works with:
      • multipart/form-data  → any field name (data, file, audio, upload, …)
      • raw binary body      → curl --data-binary, mobile apps, etc.
    Zero 422 errors. Zero assumptions.
    """
    content, size = await _extract_audio_content(request)

    device = "cuda"
    log.info(f"[bold cyan]Transcribe+Translate[/] request → {size/1024/1024:.2f} MB")

    # Run transcription
    transcribe_result = transcribe_audio(
        content,
        # device="cuda",
    )

    transcription_text = transcribe_result["text"].strip()
    detected_language = transcribe_result.get("language")
    detected_language_prob = transcribe_result.get("language_prob")

    duration_sec = len(load_audio(content)) / 16000  # safe approximate duration

    translation_text = None
    if transcription_text:
        translated_list = translate_text(text=transcription_text, device=device)
        translation_text = " ".join(translated_list).strip() or None

    response_data = {
        "duration_sec": round(duration_sec, 3),
        "detected_language": detected_language,
        "detected_language_prob": round(detected_language_prob, 4) if detected_language_prob else None,
        "transcription": transcription_text,
        "translation": translation_text,
    }

    log.info(f"[bold green]Transcribed[/] → {response_data['transcription'][:60]}...")
    if response_data["translation"]:
        log.info(f"[bold magenta]Translated[/] → {response_data['translation'][:60]}...")

    return TranscriptionResponse(**response_data)

@router.post("/transcribe_translate_old", response_model=TranscriptionResponse)
async def transcribe_translate_old(request: Request):
    """
    Universal file receiver – works with:
      • multipart/form-data  → any field name (data, file, audio, upload, …)
      • raw binary body      → curl --data-binary, mobile apps, etc.
    Zero 422 errors. Zero assumptions.
    """
    content, size = await _extract_audio_content(request)

    model_size = "small"
    compute_type = "int8_float16"
    device = "cuda"

    log.info(f"[bold cyan]Transcribe+Translate (old)[/] request → {size/1024/1024:.2f} MB")

    t = get_transcriber(model_size, compute_type, device)
    # Run transcription only; we handle translation separately for consistency
    result = t.transcribe(content, detect_language=True, translate_to_english=False)
    log.info(f"[bold green]CT2[/] [dim]transcribed[/] → [white]{result['transcription'][:60]}...[/white]")

    # Translate only if we have transcription text
    translation_text: Optional[str] = None
    if result["transcription"].strip():
        try:
            translated_list = translate_text(text=result["transcription"], device=device)
            translation_text = " ".join(translated_list).strip()
            # Handle case where model returns empty list
            if not translation_text:
                translation_text = None
        except Exception as e:
            log.warning(f"Translation failed: {e}")
            translation_text = None
    else:
        translation_text = None

    result["translation"] = translation_text
    log.info(f"[bold magenta]Translated[/] → [white]{(translation_text or '')[:60]}...[/white]")

    return TranscriptionResponse(**result)

# --- Kotoba endpoints ---
@router.post("/transcribe_kotoba", response_model=TranscriptionResponse)
async def transcribe_kotoba(request: Request):
    """
    Japanese-specialized transcription using Kotoba Whisper (no translation).
    Accepts the same universal audio input as other endpoints.
    """
    content, size = await _extract_audio_content(request)
    
    log.info(f"[bold cyan]Kotoba transcribe-only[/] request → {size/1024/1024:.2f} MB")
    
    result = transcribe_kotoba_audio(content)
    
    duration_sec = len(load_audio(content)) / 16000
    
    response_data = {
        "duration_sec": round(duration_sec, 3),
        "detected_language": result["language"],
        "detected_language_prob": round(result["language_prob"], 4),
        "transcription": result["text"].strip(),
        "translation": None,
    }
    
    log.info(f"[bold green]Kotoba transcribed[/] → {response_data['transcription'][:60]}...")
    return TranscriptionResponse(**response_data)

@router.post("/transcribe_translate_kotoba", response_model=TranscriptionResponse)
async def transcribe_translate_kotoba(request: Request):
    """
    Japanese-specialized transcription + translation to English using Kotoba Whisper.
    Accepts the same universal audio input as other endpoints.
    """
    content, size = await _extract_audio_content(request)
    
    log.info(f"[bold cyan]Kotoba transcribe+translate[/] request → {size/1024/1024:.2f} MB")
    
    result = transcribe_kotoba_audio(content)
    
    transcription_text = result["text"].strip()
    duration_sec = len(load_audio(content)) / 16000
    
    translation_text = None
    if transcription_text:
        translated_list = translate_text(text=transcription_text, device="cuda")
        translation_text = " ".join(translated_list).strip() or None
    
    response_data = {
        "duration_sec": round(duration_sec, 3),
        "detected_language": result["language"],
        "detected_language_prob": round(result["language_prob"], 4),
        "transcription": transcription_text,
        "translation": translation_text,
    }
    
    log.info(f"[bold green]Kotoba transcribed[/] → {response_data['transcription'][:60]}...")
    if response_data["translation"]:
        log.info(f"[bold magenta]Translated[/] → {response_data['translation'][:60]}...")
    
    return TranscriptionResponse(**response_data)
