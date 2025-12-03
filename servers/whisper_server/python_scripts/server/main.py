from __future__ import annotations
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from .transcribe_service import get_transcriber
from helpers.audio.whisper_ct2_transcriber import QuantizedModelSizes
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%^(message^)s", datefmt="[^%X]", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("whisper-api")

app = FastAPI(title="Whisper CTranslate2 FastAPI Server", version="1.0.0")

class TranscriptionResponse(BaseModel):
    audio_path: str
    duration_sec: float
    detected_language: Optional[str] = None
    detected_language_prob: Optional[float] = None
    transcription: str
    translation: Optional[str] = None

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model_size: QuantizedModelSizes = Query("large-v2"),
    compute_type: str = Query("int8_float16"),
    device: str = Query("cpu"),
):
    if not file.filename.lower().endswith((".wav",".mp3",".m4a",".flac",".ogg")):
        raise HTTPException(400, "Unsupported file format")
    content = await file.read()
    tmp = Path("temp_uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)
    try:
        t = get_transcriber(model_size, compute_type, device)
        result = t(tmp, detect_language=True, translate_to_english=False)
        log.info(f"[Transcribe] {result['detected_language']} {result['duration_sec']:.1f}s")
        return TranscriptionResponse(**result)
    finally:
        if tmp.exists(): tmp.unlink()

@app.post("/translate", response_model=TranscriptionResponse)
async def translate_audio(
    file: UploadFile = File(...),
    model_size: QuantizedModelSizes = Query("large-v2"),
    compute_type: str = Query("int8_float16"),
    device: str = Query("cpu"),
):
    if not file.filename.lower().endswith((".wav",".mp3",".m4a",".flac",".ogg")):
        raise HTTPException(400, "Unsupported file format")
    content = await file.read()
    tmp = Path("temp_uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)
    try:
        t = get_transcriber(model_size, compute_type, device)
        result = t(tmp, detect_language=True, translate_to_english=True)
        log.info(f"[Translate] {result['detected_language']} -^> en")
        return TranscriptionResponse(**result)
    finally:
        if tmp.exists(): tmp.unlink()

@app.get("/")
async def root(): return {"message": "Whisper CTranslate2 API ready"}
