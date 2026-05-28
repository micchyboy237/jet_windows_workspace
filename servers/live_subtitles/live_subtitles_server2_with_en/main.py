"""
Live Japanese Subtitles Server 2 - Main Application Entry Point.

Refactored with separate route files for better maintainability.
"""
import shutil
import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Core state imports
from core.state import (
    get_segment_index_path,
)
from services.live_subtitles_server_utils import load_segment_counter

# Route imports
from routes.websocket import websocket_endpoint
from routes.speakers import router as speakers_router
from routes.transcribe import router as transcribe_router
from routes.translate import router as translate_router

# ---- Console Setup ----
console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "success": "green bold",
            "warning": "yellow",
            "error": "red bold",
            "value": "white bold",
            "time": "magenta bold",
            "number": "bright_white",
            "uuid": "bright_blue",
            "speaker": "bright_green",
        }
    )
)

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("live_subtitles")

# Suppress uvicorn logs to avoid duplicate logging
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = True

# ---- Output Directory ----
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- FastAPI App ----
app = FastAPI(title="Live Japanese Subtitles Server 2")

# ---- Register WebSocket Route (direct registration) ----
# IMPORTANT: Use add_api_websocket_route for direct handler registration
# The decorator wrapper in main.py was causing 403 errors because
# FastAPI couldn't properly resolve the WebSocket handler chain.
app.add_api_websocket_route("/ws/live-subtitles", websocket_endpoint)

# ---- Register REST Routes ----
app.include_router(speakers_router)
app.include_router(transcribe_router)
app.include_router(translate_router)


def initialize_detector():
    """Initialize the audio language detector."""
    from core.state import set_audio_language_detector
    from services.audio_language_detector import AudioLanguageDetector
    
    console.print("Initializing AudioLanguageDetector...")
    detector = AudioLanguageDetector()
    set_audio_language_detector(detector)
    console.print("Detector initialized successfully!\n")


# ---- Main Entry Point ----
if __name__ == "__main__":
    initialize_detector()
    
    segment_index_path = get_segment_index_path()
    segment_counter = load_segment_counter(segment_index_path)
    console.print(
        f"[info]Segment counter initialized: {segment_counter} "
        f"(next will be segment_{segment_counter + 1:03d})[/info]"
    )
    
    logger.info("🚀 Starting [bold cyan]Live Japanese Subtitles Server 2[/]")
    logger.info("WebSocket endpoint → [bold]ws://0.0.0.0:8000/ws/live-subtitles[/]")
    logger.info("REST endpoints:")
    logger.info("   POST /transcribe")
    logger.info("   POST /translate")
    logger.info("   GET  /speakers")
    logger.info("   GET  /speakers/status")
    logger.info("   GET  /speakers/similarities")
    logger.info("   POST /speakers/consolidate")
    logger.info("   POST /speakers/reset")
    logger.info("   POST /speakers/merge")
    logger.info("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )