import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).parent.parent

PAGES = {
    "/": BASE_DIR / "client.html",
    "/partial": BASE_DIR / "demo_partial_range.html",
    "/full": BASE_DIR / "demo_full_range.html",
}

logger = logging.getLogger("audio_streaming.pages_router")

router = APIRouter()


def _serve_html(file_path: Path, label: str) -> HTMLResponse:
    """Shared file-read + response logic for every HTML page route."""
    if not os.path.exists(file_path):
        logger.error(f"{label} not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"{label} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"Served {label}")
    return HTMLResponse(content=content)


@router.get("/", response_class=HTMLResponse)
async def serve_client_html():
    """Serves the original client.html UI (native <audio> tag, no howler.js)."""
    return _serve_html(PAGES["/"], "client.html")


@router.get("/partial", response_class=HTMLResponse)
async def serve_partial_demo():
    """Serves the howler.js partial-range (progressive streaming) demo."""
    return _serve_html(PAGES["/partial"], "demo_partial_range.html")


@router.get("/full", response_class=HTMLResponse)
async def serve_full_demo():
    """Serves the howler.js full-range (complete download) demo."""
    return _serve_html(PAGES["/full"], "demo_full_range.html")
