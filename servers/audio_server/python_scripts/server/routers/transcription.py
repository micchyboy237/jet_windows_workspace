# servers/audio_server/python_scripts/server/routers/transcription.py
from __future__ import annotations
import numpy as np
import dataclasses
from typing import Optional, Literal, Annotated
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, Depends
from faster_whisper import WhisperModel
from pydantic import BaseModel
from python_scripts.server.models.responses import TranscriptionResponse
from python_scripts.server.services.transcribe_service import get_transcriber
from python_scripts.server.services.whisper_ct2_transcriber import QuantizedModelSizes
from python_scripts.server.utils.logger import get_logger
from python_scripts.server.utils.streaming_model import get_streaming_model

# Configure logger for this module
log = get_logger("transcription")

router = APIRouter()

# Response models for streaming endpoints
class StreamingResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    processing_time_sec: float

class ChunkResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration_sec: float
    segments: list[dict]
    words: list[dict]
    task: Literal["transcribe", "translate"]


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model_size: QuantizedModelSizes = Query("large-v3", description="CTranslate2 model size"),
    compute_type: str = Query("int8", description="Compute type for CTranslate2"),
    device: str = Query("cpu", description="Device for CTranslate2 (cpu/cuda)"),
):
    """High-quality transcription using CTranslate2 (recommended for final results)."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format. Use WAV, MP3, M4A, FLAC, or OGG.")

    content = await file.read()
    tmp = Path("temp_uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)

    try:
        log.info(
            f"[bold cyan]Transcribe (CT2)[/] [white]processing[/] "
            f"[dim]→[/] [yellow]'{file.filename}'[/] "
            f"[green]{len(content)/1024/1024:.2f} MB[/] | "
            f"[cyan]{model_size}[/] | [blue]{device}[/]"
        )
        
        t = get_transcriber(model_size, compute_type, device)
        result = t(tmp, detect_language=True, translate_to_english=False)
        log.info(f"[bold green]CT2[/] [dim]transcribed[/] → [white]{result['transcription'][:60]}...[/white]")
        
        return TranscriptionResponse(**result)
    except Exception as e:
        log.error(f"[bold red]CT2[/] [dim]transcribe failed[/] → {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/translate", response_model=TranscriptionResponse)
async def translate_audio(
    file: UploadFile = File(...),
    model_size: QuantizedModelSizes = Query("large-v3", description="CTranslate2 model size"),
    compute_type: str = Query("int8", description="Compute type for CTranslate2"),
    device: str = Query("cpu", description="Device for CTranslate2 (cpu/cuda)"),
):
    """Translate audio to English using CTranslate2 (best quality)."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format. Use WAV, MP3, M4A, FLAC, or OGG.")

    content = await file.read()
    tmp = Path("temp_uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)

    try:
        log.info(
            f"[bold magenta]Translate (CT2)[/] [white]processing[/] "
            f"[dim]→[/] [yellow]'{file.filename}'[/] "
            f"[green]{len(content)/1024/1024:.2f} MB[/] | "
            f"[cyan]{model_size}[/] | [blue]{device}[/]"
        )
        
        t = get_transcriber(model_size, compute_type, device)
        result = t(tmp, detect_language=True, translate_to_english=True)
        log.info(f"[bold green]CT2[/] [dim]translated[/] → [white]{result['translation'][:60]}...[/white]")
        
        return TranscriptionResponse(**result)
    except Exception as e:
        log.error(f"[bold red]CT2[/] [dim]translate failed[/] → {e}")
        raise HTTPException(500, f"Translation failed: {str(e)}")
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/transcribe_stream", response_model=StreamingResponse)
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language code (auto-detect if omitted)"),
    task: Literal["transcribe", "translate"] = Query("transcribe", description="Task to perform"),
    model_size: Annotated[str, Query(description="faster-whisper model size")] = "large-v3",
    compute_type: Annotated[str, Query(description="int8, int8_float16, float16")] = "int8",
    device: Annotated[str, Query(description="cpu or cuda")] = "cpu",
    model: WhisperModel = Depends(get_streaming_model),
):
    """Low-latency streaming transcription using faster-whisper (single file upload)."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format. Use WAV, MP3, M4A, FLAC, or OGG.")

    content = await file.read()
    
    # Handle different audio formats
    if content[:4] == b'RIFF':  # WAV
        # Skip WAV header (44 bytes for standard WAV)
        audio_start = 44
        audio_data = content[audio_start:]
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        # Assume raw or other format - try to load via librosa in background
        # For now, treat as raw int16 (common for streaming)
        audio = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Basic validation
    if len(audio) == 0:
        raise HTTPException(400, "Empty or invalid audio data")

    log.info(
        f"[bold blue]Stream (faster-whisper)[/] [white]processing[/] "
        f"[dim]→[/] [yellow]'{file.filename}'[/] "
        f"[green]{len(content)/1024/1024:.2f} MB[/] | "
        f"[cyan]{model_size}[/] | [blue]{device}[/] | "
        f"[dim]{task}[/] lang={language or 'auto'}"
    )

    try:
        start_time = time.time()
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
        
        # Collect all segments
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())
        
        text = " ".join(text_parts).strip()
        duration = info.duration
        processing_time = time.time() - start_time
        
        log.info(
            f"[bold green]Stream[/] [dim]{task}ed[/] "
            f"→ [white]{text[:60]}{'...' if len(text) > 60 else ''}[/] "
            f"[dim]({duration:.1f}s audio → {processing_time:.2f}s)[/]"
        )

        return StreamingResponse(
            text=text,
            language=info.language or "unknown",
            language_probability=round(info.language_probability, 4),
            duration=duration,
            processing_time_sec=round(processing_time, 4),
        )
    except Exception as e:
        log.error(f"[bold red]Stream[/] [dim]{task} failed[/] → {e}")
        raise HTTPException(500, f"Streaming transcription failed: {str(e)}")


@router.post("/transcribe_chunk", response_model=ChunkResponse)
async def transcribe_chunk(
    request: Request,
    duration_sec: Optional[float] = Query(None, description="Expected duration in seconds (optional)"),
    task: Literal["transcribe", "translate"] = Query("transcribe", description="Task to perform: transcribe or translate to English"),
    model_size: Annotated[str, Query(description="faster-whisper model size")] = "large-v3",
    compute_type: Annotated[str, Query(description="int8, int8_float16, float16")] = "int8",
    device: Annotated[str, Query(description="cpu or cuda")] = "cpu",
    model: WhisperModel = Depends(get_streaming_model),
):
    """Real-time chunk transcription for raw PCM 16kHz float32 audio (microphone/streaming)."""
    
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(400, "Empty audio data received")
    
    if len(body_bytes) == 0:
        raise HTTPException(400, "No audio data in request body")
    
    # Expect raw float32, 16kHz little-endian
    try:
        audio_np = np.frombuffer(body_bytes, dtype=np.float32)
    except ValueError as e:
        log.error(f"[bold red]Chunk[/] [dim]invalid audio format[/] → {e}")
        raise HTTPException(400, "Invalid audio format. Expected raw PCM float32, 16kHz.")
    
    if audio_np.size == 0:
        raise HTTPException(400, "Audio data is empty")
    
    # Calculate duration
    sample_rate = 16000
    actual_duration = len(audio_np) / sample_rate
    
    log.info(
        f"[bold purple]Chunk (faster-whisper)[/] [white]processing[/] "
        f"[dim]→[/] [green]{actual_duration:.2f}s[/] audio | "
        f"[cyan]{model_size}[/] | [blue]{device}[/] | "
        f"[dim]{task}[/]"
    )

    try:
        start_time = time.time()
        segments_iter, info = model.transcribe(
            audio_np,
            language=None,
            task=task,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,
            temperature=0.0,
        )
        
        # Process segments and words
        segments = []
        words = []
        for segment_num, seg in enumerate(segments_iter, start=1):
            seg_dict = dataclasses.asdict(seg)
            seg_words = seg_dict.pop("words", [])
            
            # Add segment number to words
            for word in seg_words:
                word["segment_num"] = segment_num
            
            segments.append(seg_dict)
            words.extend(seg_words)
        
        full_text = " ".join(seg["text"] for seg in segments).strip()
        processing_time = time.time() - start_time
        
        log.info(
            f"[bold green]Chunk[/] [dim]{task}ed[/] "
            f"→ [white]{full_text[:60]}{'...' if len(full_text) > 60 else ''}[/] "
            f"[dim]({actual_duration:.2f}s → {processing_time:.2f}s)[/]"
        )

        return ChunkResponse(
            text=full_text or "",
            language=info.language or "unknown",
            language_probability=round(info.language_probability, 4),
            duration_sec=round(actual_duration, 4),
            segments=segments,
            words=words,
            task=task,
        )
    except Exception as e:
        log.error(f"[bold red]Chunk[/] [dim]{task} failed[/] → {e}")
        raise HTTPException(500, f"Chunk transcription failed: {str(e)}")