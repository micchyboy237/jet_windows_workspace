from __future__ import annotations

import logging
from typing import List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from python_scripts.server.services.translate_service import translate_text
from python_scripts.server.utils.logger import get_logger

log = get_logger("translation")

router = APIRouter()

class TranslateRequest(BaseModel):
    text: str | List[str] = Field(..., description="Single text string or list of strings to translate")
    device: str = Field("cuda", description="Device to run the translation model on (cpu/cuda)")

class TranslateResponse(BaseModel):
    original: str
    translation: str

class BatchTranslateResponse(BaseModel):
    results: List[TranslateResponse]

@router.post("/translate", response_model=TranslateResponse | BatchTranslateResponse)
async def translate(
    request: TranslateRequest,
):
    """
    Pure text-to-text translation using the cached Opus-MT CTranslate2 model.
    Now supports both single string and list of strings (batch translation).
    """
    texts = request.text if isinstance(request.text, list) else [request.text]
    device = request.device

    if not any(t.strip() for t in texts):
        empty_result = TranslateResponse(original="", translation="")
        return empty_result if isinstance(request.text, str) else BatchTranslateResponse(results=[empty_result])

    try:
        translated_list = translate_text(
            text=texts,  # now passes list
            device=device,
        )
        results = [
            TranslateResponse(original=orig, translation=trans)
            for orig, trans in zip(texts, translated_list)
        ]

        log.info(
            f"[bold magenta]Translated {len(results)} text(s)[/] → "
            f"[white]{translated_list[0][:60]}{'...' if len(translated_list[0]) > 60 else ''}[/]"
        )

        # Return single object if input was single string
        if isinstance(request.text, str):
            return results[0]
        return BatchTranslateResponse(results=results)

    except Exception as e:
        log.error(f"[bold red]Text translation failed[/] → {e}")
        raise