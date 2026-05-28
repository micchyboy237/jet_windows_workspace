"""
Request and response Pydantic models.
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 encoded PCM int16 audio (optional if file uploaded)")
    sample_rate: int = Field(16000, description="Sample rate of the audio")
    hotwords: Optional[str] = Field(None, description="Hotwords for ASR")


class TranscribeResponse(BaseModel):
    success: bool
    transcription: str
    speaker_label: str = "SPEAKER_UNKNOWN"
    speaker_confidence: float = 0.0
    metadata: Dict[str, Any]
    word_segments: list = []
    phrase_segments: list = []


class TranslateRequest(BaseModel):
    japanese_text: str = Field(..., description="Japanese text to translate")
    history: Optional[list] = Field(default=None, description="Conversation history for context")
    temperature: Optional[float] = Field(0.35, ge=0.0, le=1.0)


class TranslateResponse(BaseModel):
    success: bool
    en_text: str
    quality: str = "N/A"
    log_prob: Optional[float] = None
    confidence: Optional[float] = None
