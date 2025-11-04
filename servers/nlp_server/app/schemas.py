# app/schemas.py
from pydantic import BaseModel, Field
from typing import List

class AnnotateRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to annotate")

class WordAnnotation(BaseModel):
    text: str
    lemma: str
    pos: str
    ner: str | None = None

class SentenceAnnotation(BaseModel):
    text: str
    tokens: List[str]
    words: List[WordAnnotation]

class AnnotateResponseItem(BaseModel):
    sentences: List[SentenceAnnotation]

class AnnotateResponse(BaseModel):
    results: List[AnnotateResponseItem]