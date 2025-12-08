from __future__ import annotations

import logging
from fastapi import APIRouter, Query
from pydantic import BaseModel
from python_scripts.server.services.translate_service import translate_text
from python_scripts.server.utils.logger import get_logger

log = get_logger("translation")

router = APIRouter(prefix="/translate", tags=["translation"])


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    original: str
    translation: str


@router.post("/", response_model=TranslateResponse)
async def translate_text_endpoint(
    request: TranslateRequest,
    device: str = Query("cuda", description="Device to run the translation model on (cpu/cuda)"),
):
    """
    Pure text-to-text translation using the cached Opus-MT CTranslate2 model.
    Useful when you already have a transcription and only need translation.
    """
    if not request.text.strip():
        return TranslateResponse(original="", translation="")

    try:
        translated = translate_text(
            text=request.text,
            device=device,
        )
        log.info(
            f"[bold magenta]Text translated[/] → [white]{translated[:80]}{'...' if len(translated) > 80 else ''}[/]"
        )
        return TranslateResponse(original=request.text, translation=translated)
    except Exception as e:
        log.error(f"[bold red]Text translation failed[/] → {e}")
        raise