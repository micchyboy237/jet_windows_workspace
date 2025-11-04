# app/routes.py
from fastapi import APIRouter, HTTPException
from typing import List
from .schemas import AnnotateRequest, AnnotateResponse
from .pipeline import nlp_processor

router = APIRouter()

@router.get("/health", summary="Check server health")
def health_check():
    return {"status": "ok"}

@router.post("/annotate", response_model=AnnotateResponse, summary="Annotate list of texts")
def annotate(request: AnnotateRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    try:
        raw_results = nlp_processor.annotate(request.texts)
        return AnnotateResponse(results=raw_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Annotation failed: {e}")