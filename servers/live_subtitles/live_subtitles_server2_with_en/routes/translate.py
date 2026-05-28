"""
Translation route for Japanese-to-English text translation.
"""
from fastapi import APIRouter, HTTPException
from models.schemas import TranslateRequest, TranslateResponse
from services.translate_jp_en_llm_prefixed import translate_japanese_to_english
from rich.console import Console

console = Console()
router = APIRouter(tags=["translation"])


@router.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest):
    """Translate Japanese text to English only (REST API)."""
    try:
        if not request.japanese_text or not request.japanese_text.strip():
            raise HTTPException(status_code=400, detail="japanese_text is required and cannot be empty")
        
        result = translate_japanese_to_english(
            text=request.japanese_text.strip(),
            history=request.history,
            temperature=request.temperature or 0.35,
        )
        
        return {
            "success": True,
            "en_text": result["text"],
            "quality": result.get("quality", "N/A"),
            "log_prob": result.get("log_prob"),
            "confidence": result.get("confidence"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Translation endpoint error: {e}[/error]")
        raise HTTPException(status_code=500, detail=str(e))
