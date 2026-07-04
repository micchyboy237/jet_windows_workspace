import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.audio_router import router as audio_router, AUDIO_DIR
from routers.pages_router import router as pages_router, TEMPLATES_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Audio Streaming Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Range", "Content-Type"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)
app.include_router(audio_router)
app.include_router(pages_router)

if __name__ == "__main__":
    port = 8001
    logger.info(f"Server listening on http://localhost:{port}")
    logger.info(f"Audio directory: {AUDIO_DIR}")
    logger.info(f"Templates directory: {TEMPLATES_DIR}")
    logger.info("Routes: / (native), /partial (partial range), /full (full range)")
    logger.info("API: /audio, /audio-list, /health")
    uvicorn.run(app, host="0.0.0.0", port=port)
