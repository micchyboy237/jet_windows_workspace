from pydantic import BaseModel
from typing import Optional

class TranscriptionResponse(BaseModel):
    audio_path: str
    duration_sec: float
    detected_language: Optional[str] = None
    detected_language_prob: Optional[float] = None
    transcription: str
    translation: Optional[str] = None