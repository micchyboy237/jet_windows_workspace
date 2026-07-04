import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.audio_router import router as audio_router
from routers.pages_router import router as pages_router

BASE_DIR = Path(__file__).parent
audio_file_path = str(BASE_DIR / "sample_audio.wav")
client_html_path = str(BASE_DIR / "client.html")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("audio_streaming.server")

app = FastAPI(title="Audio Streaming Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Range", "Content-Type"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

# Mount routers. audio_router owns /audio + /health,
# pages_router owns all HTML page routes (/, /partial, /full).
app.include_router(audio_router)
app.include_router(pages_router)


if __name__ == "__main__":
    port = 8001
    logger.info(f"Server listening on http://localhost:{port}")
    logger.info(f"Audio file: {audio_file_path}")
    logger.info(f"HTML client: {client_html_path}")
    logger.info("Routes: / (client), /partial (demo), /full (demo), /audio, /health")
    uvicorn.run(app, host="0.0.0.0", port=port)
