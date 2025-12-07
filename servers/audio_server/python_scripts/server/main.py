from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rich.logging import RichHandler

from .routers import transcription, health

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=False)],
)
log = logging.getLogger("whisper-api")

app = FastAPI(
    title="Whisper CTranslate2 FastAPI Server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(transcription.router)

# Optional: prefix if you want /v1/transcribe etc.
# app.include_router(transcription.router, prefix="/v1")