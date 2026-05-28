"""
Transcription route for audio-to-text conversion.
"""
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.transcribe_funasr import transcribe_audio, TranscriptionResult
from rich.console import Console

console = Console()
router = APIRouter(tags=["transcription"])


@router.post("/transcribe")
async def transcribe_endpoint(
    audio_file: UploadFile = File(..., description="Audio file (WAV, PCM int16 recommended)"),
    sample_rate: int = Form(16000, description="Sample rate of the audio"),
    hotwords: Optional[str] = Form(None, description="Optional hotwords for better recognition"),
    language: str = Form(
        "auto",
        description="Language code (e.g., 'ja' for Japanese, 'en' for English, 'auto' for auto-detect)",
    ),
):
    """Transcribe audio → text (REST API)"""
    try:
        console.print(f"[info]Received file upload: {audio_file.filename} ({audio_file.content_type})[/info]")
        audio_bytes = await audio_file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        
        console.print(f"[info]Audio size: {len(audio_bytes)/1024:.1f} KB | Sample rate: {sample_rate} Hz[/info]")
        
        result: TranscriptionResult = transcribe_audio(
            audio_bytes=audio_bytes,
            language=language,
            sample_rate=sample_rate,
            hotwords=hotwords,
        )
        
        return {
            "success": True,
            "transcription": result.get("text", ""),
            "metadata": result.get("metadata", {}),
            "word_segments": result.get("word_segments", []),
            "phrase_segments": result.get("phrase_segments", []),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Transcription endpoint failed: {e}[/error]")
        import traceback
        console.print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
