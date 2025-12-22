from typing import List, Optional
from pydantic import BaseModel

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    duration_sec: float
    detected_language: Optional[str] = None
    detected_language_prob: Optional[float] = None
    transcription: str
    translation: Optional[str] = None
    segments: Optional[List[TranscriptionSegment]] = None  # Added: per-segment breakdown