from __future__ import annotations
import time
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from python_scripts.server.routers import health, transcription, translation, batch_transcription
# Import the startup function from the batch router
from python_scripts.server.routers.batch_transcription import load_model as load_batch_model

from python_scripts.server.utils.logger import get_logger

log = get_logger("main")
log.info("[bold green]Whisper CTranslate2 + faster-whisper Server Starting...[/bold green]")

app = FastAPI(
    title="Whisper CTranslate2 & faster-whisper FastAPI Server",
    description="High-performance transcription with CTranslate2 (best quality) and faster-whisper (low-latency streaming)",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""
    icons = {
        "/": "Health",
        "/transcribe": "Transcribe (CT2)",
        "/transcribe_stream": "Stream (faster-whisper)",
        "/transcribe_chunk": "Chunk (faster-whisper)",
        "/translate": "Text → EN (Opus-MT)",
        "/sample": "Sample (multipart/raw)",
        "/batch_transcribe": "Batch Transcribe (multipart)",
        "/batch_transcribe_bytes": "Batch Transcribe (bytes)",
    }
    icon = icons.get(path, "Request")
    content_type = request.headers.get("content-type", "")
    is_multipart = content_type.startswith("multipart/form-data")
    if is_multipart:
        log.info(
            f"[bold cyan]{icon}[/] [white]{method} {path}[/] "
            f"[dim]→ multipart/form-data[/] "
            f"[dim]from[/] [blue]{client_ip}[/] {query}"
        )
    else:
        log.info(
            f"[bold magenta]{icon}[/] [white]{method} {path}[/] "
            f"[dim]from[/] [blue]{client_ip}[/] {query or '[no params]'}"
        )
    response = await call_next(request)
    process_time = time.time() - start_time
    status_color = "[bold green]" if response.status_code < 400 else "[bold red]"
    log.info(
        f"{status_color}→ {response.status_code}[/] [dim]| {process_time:.3f}s[/]"
    )
    return response

app.include_router(health.router)
app.include_router(transcription.router)
app.include_router(batch_transcription.router)
app.include_router(translation.router)

# Load the faster-whisper model at application startup
@app.on_event("startup")
async def startup_event():
    await load_batch_model()

log.info("[bold green]Server ready — http://0.0.0.0:8001[/bold green]")
log.info("[dim]Interactive docs:[/] [underline blue]http://127.0.0.1:8001/docs[/underline blue]")