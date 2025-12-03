import numpy as np
import dataclasses
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, Depends
from typing import Optional, Literal
from pathlib import Path
from faster_whisper import WhisperModel

from python_scripts.server.models.responses import TranscriptionResponse
from python_scripts.server.transcribe_service import get_transcriber
from python_scripts.server.whisper_ct2_transcriber import QuantizedModelSizes
from python_scripts.server.utils.streaming_model import get_streaming_model

router = APIRouter()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model_size: QuantizedModelSizes = Query("large-v2"),
    compute_type: str = Query("int8"),
    device: str = Query("cpu"),
):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format")

    content = await file.read()
    tmp = Path("temp_uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)

    try:
        t = get_transcriber(model_size, compute_type, device)
        result = t(tmp, detect_language=True, translate_to_english=False)
        return TranscriptionResponse(**result)
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/translate", response_model=TranscriptionResponse)
async def translate_audio(
    file: UploadFile = File(...),
    model_size: QuantizedModelSizes = Query("large-v2"),
    compute_type: str = Query("int8"),
    device: str = Query("cpu"),
):
    # Same logic as above, just translate_to_english=True
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format")

    content = await file.read()
    tmp = Path("temp_uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)

    try:
        t = get_transcriber(model_size, compute_type, device)
        result = t(tmp, detect_language=True, translate_to_english=True)
        return TranscriptionResponse(**result)
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/transcribe_stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    task: Literal["transcribe", "translate"] = "transcribe",
    model: WhisperModel = Depends(get_streaming_model),
):
    content = await file.read()
    audio = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio,
        language=language,
        task=task,
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
    )
    text = " ".join(seg.text for seg in segments).strip()

    return {
        "text": text,
        "language": info.language,
        "language_prob": round(info.language_probability, 4),
        "duration": info.duration,
        "processing_time_sec": info.duration / info.duration_after_vad,
    }


@router.post("/transcribe_chunk")
async def transcribe_chunk(
    request: Request,
    duration_sec: Optional[float] = Query(None),  # kept for backward compatibility, ignored when task=translate
    task: Literal["transcribe", "translate"] = Query("transcribe", description="Task to perform: transcribe or translate to English"),
    model: WhisperModel = Depends(get_streaming_model),
):
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(400, "Empty audio data")

    audio_np = np.frombuffer(body_bytes, dtype=np.float32)
    if audio_np.size == 0:
        raise HTTPException(400, "Invalid audio data")

    segments_iter, info = model.transcribe(
        audio_np,
        language=None,
        task=task,                                     # now dynamic
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
        temperature=0.0,
    )

    segments = []
    words = []
    for segment_num, seg in enumerate(segments_iter, start=1):
        seg_dict = dataclasses.asdict(seg)
        seg_words = seg_dict.pop("words")

        # Inject segment_num into each word of this segment
        for word in seg_words:
            word["segment_num"] = segment_num

        segments.append(seg_dict)
        words.extend(seg_words)
    full_text = " ".join(seg["text"] for seg in segments).strip()

    return {
        "text": full_text or "",
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_sec": len(audio_np) / 16000,
        "segments": segments,
        "words": words,
        "task": task,                                  # optional: expose what was performed
    }
