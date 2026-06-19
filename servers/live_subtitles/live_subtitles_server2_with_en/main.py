# servers\live_subtitles\live_subtitles_server2_with_en\main.py
"""
Live Japanese Subtitles Server 2 - Main Application Entry Point.
Refactored with separate route files for better maintainability.
"""
import shutil
import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from core.state import (
    get_segment_index_path,
)
from services.live_subtitles_server_utils import load_segment_counter
from routes.websocket import websocket_endpoint
from routes.speakers import router as speakers_router
from routes.transcribe import router as transcribe_router
from routes.translate import router as translate_router
from routes.tagger import router as tagger_router
from routes.global_reset import router as global_reset_router

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("live_subtitles")

# Create FastAPI app
app = FastAPI(title="Live Japanese Subtitles Server 2")

# ===== Configure Static File Serving =====
# Get the directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Mount static files directory
static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ===== WebSocket and API Routes =====
app.add_api_websocket_route("/ws/live-subtitles", websocket_endpoint)
app.include_router(speakers_router)
app.include_router(transcribe_router)
app.include_router(translate_router)
app.include_router(tagger_router)
app.include_router(global_reset_router)

def initialize_detector():
    """Initialize the audio language detector."""
    from core.state import set_audio_language_detector
    from services.audio_language_detector import AudioLanguageDetector
    
    console.print("Initializing AudioLanguageDetector...")
    detector = AudioLanguageDetector()
    set_audio_language_detector(detector)
    console.print("Detector initialized successfully!\n")

def initialize_labeler():
    import json
    from core.state import (
        get_speaker_labeler,
        get_audio_tagger,
        get_speaker_state_path,
        set_embedding_inference,
        set_speaker_labeler,
    )
    from pyannote.audio import Inference, Model
    from services.segment_speaker_labeler import SegmentSpeakerLabeler
    from services.embedding_model_factory import (
        EmbeddingModelType,
        create_embedding_model,
        list_available_models,
    )

    MODEL_TYPE = EmbeddingModelType.PYANNOTE

    console.print(f"[bold]Available embedding models:[/bold]")
    for name, info in list_available_models().items():
        console.print(f"  • {name} (dim={info['embedding_dim']})")

    with console.status(
        f"[bold green]Loading embedding model '{MODEL_TYPE.value}'...[/bold green]",
        spinner="dots",
    ):
        embedding_inference = create_embedding_model(MODEL_TYPE)

    set_embedding_inference(embedding_inference)

    speaker_state_path = get_speaker_state_path()
    tagger = get_audio_tagger()
    if speaker_state_path.exists():
        try:
            with open(speaker_state_path, "r") as f:
                state = json.load(f)
            labeler = SegmentSpeakerLabeler.from_dict(
                state,
                embedding_model=embedding_inference,
                audio_tagger=tagger,
            )
            set_speaker_labeler(labeler)
            console.print(
                f"[success]Restored speaker state: "
                f"{labeler.speaker_count} speaker(s), "
                f"{labeler.total_segments_processed} segments processed[/success]"
            )
            return labeler
        except Exception as e:
            console.print(
                f"[warning]Could not restore speaker state: {e}[/warning]"
            )

    labeler = SegmentSpeakerLabeler(
        embedding_model=embedding_inference,
        debug=True,
    )
    set_speaker_labeler(labeler)
    console.print("[success]Speaker labeler initialized[/success]")

    return labeler


def initialize_tagger():
    """
    Initialize the audio tagger at startup.
    This pre-loads the ONNX model so the first tagging request
    doesn't have to wait for model loading.
    """
    from core.state import set_audio_tagger
    from services.audio_tagger import AudioTagger
    
    console.print("Initializing AudioTagger...")
    try:
        tagger = AudioTagger(
            debug=False,
        )
        set_audio_tagger(tagger)
        console.print("AudioTagger initialized successfully!\n")
    except FileNotFoundError as e:
        console.print(f"[warning]AudioTagger model files not found: {e}[/warning]")
        console.print("[warning]Audio tagging endpoints will be available but may fail on first request.[/warning]")
        console.print("[warning]Download models from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models[/warning]\n")
    except Exception as e:
        console.print(f"[warning]Could not initialize AudioTagger: {e}[/warning]")
        console.print("[warning]Audio tagging endpoints will initialize on first request.[/warning]\n")

def cleanup_on_shutdown():
    """
    Cleanup resources on server shutdown.
    Saves speaker state and performs any necessary cleanup.
    """
    from core.state import (
        save_speaker_state,
        get_speaker_labeler,
        get_audio_tagger,
    )
    
    console.print("[info]Performing cleanup before shutdown...[/info]")
    
    if get_speaker_labeler() is not None:
        save_speaker_state()
        console.print("[success]Speaker state saved.[/success]")
    
    tagger = get_audio_tagger()
    if tagger is not None:
        try:
            tagger.reset()
            console.print("[success]AudioTagger resources freed.[/success]")
        except Exception as e:
            console.print(f"[warning]Error resetting AudioTagger: {e}[/warning]")
    
    console.print("[success]Cleanup complete.[/success]")

import atexit
atexit.register(cleanup_on_shutdown)

if __name__ == "__main__":
    initialize_detector()
    initialize_tagger()
    initialize_labeler()
    
    segment_index_path = get_segment_index_path()
    segment_counter = load_segment_counter(segment_index_path)
    console.print(
        f"[info]Segment counter initialized: {segment_counter} "
        f"(next will be segment_{segment_counter + 1:03d})[/info]"
    )
    
    logger.info("🚀 Starting [bold cyan]Live Japanese Subtitles Server 2[/]")
    logger.info("")
    logger.info("📡 WebSocket endpoint:")
    logger.info("   [bold]ws://0.0.0.0:8000/ws/live-subtitles[/]")
    logger.info("")
    logger.info("📋 REST endpoints:")
    logger.info("   POST /transcribe")
    logger.info("   POST /translate")
    logger.info("")
    logger.info("📋 Speaker Labeling endpoints:")
    logger.info("   GET  /speakers")
    logger.info("   GET  /speakers/status")
    logger.info("   GET  /speakers/similarities")
    logger.info("   POST /speakers/consolidate")
    logger.info("   POST /speakers/reset")
    logger.info("   POST /speakers/merge")
    logger.info("   GET  /speakers/dashboard")
    logger.info("   GET  /speakers/plots")
    logger.info("   GET  /speakers/plot/{plot_name}")
    logger.info("   GET  /speakers/data/export")
    logger.info("")
    logger.info("🎵 Audio Tagging endpoints:")
    logger.info("   GET  /tags [HTML]")
    logger.info("   GET  /tags/config")
    logger.info("   GET  /tags/chunks")
    logger.info("   GET  /tags/dashboard [HTML]")
    logger.info("   POST /tags/audio")
    logger.info("   POST /tags/chunks")
    logger.info("   POST /tags/speech-check")
    logger.info("   POST /tags/config/update")
    logger.info("")
    logger.info("🔄 Global Reset endpoints:")
    logger.info("   POST /global/reset")
    logger.info("   GET  /global/status")
    logger.info("")
    logger.info("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
