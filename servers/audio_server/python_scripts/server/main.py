# servers/audio_server/python_scripts/server/main.py
from __future__ import annotations

import time
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console

# Import routers
from python_scripts.server.routers import transcription, health
from python_scripts.server.utils.logger import get_logger

# Configure shared rich logger
log = get_logger("main")

# Startup banner
log.info("[bold green]Whisper CTranslate2 + faster-whisper Server Starting...[/bold green]")

app = FastAPI(
    title="Whisper CTranslate2 & faster-whisper FastAPI Server",
    description="High-performance transcription with CTranslate2 (best quality) and faster-whisper (low-latency streaming)",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Allow all origins (adjust in production if needed)
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

    # Icon mapping for visual clarity
    icons = {
        "/transcribe": "Transcribe (CT2)",
        "/translate": "Translate (CT2)",
        "/transcribe_stream": "Stream (faster-whisper)",
        "/transcribe_chunk": "Chunk (faster-whisper)",
        "/": "Health",
    }
    icon = icons.get(path, "Request")

    filename = "—"
    file_size_mb = 0.0
    is_multipart = False

    content_type = request.headers.get("content-type", "")
    if content_type and content_type.startswith("multipart/form-data"):
        try:
            body = await request.body()
            if body:
                form = await request.form()
                file = form.get("file")
                if file and hasattr(file, "filename") and file.filename:
                    filename = file.filename
                    file_size_mb = len(body) / (1024 * 1024)
                    is_multipart = True

                # Restore body for downstream consumption
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                request._receive = receive  # type: ignore
        except Exception as e:
            log.error(f"Could not parse form in middleware: {e}")

    # Request incoming log
    if is_multipart:
        log.info(
            f"[bold cyan]{icon}[/] [white]{method} {path}[/] "
            f"[dim]→[/] [yellow]'{filename}'[/] "
            f"[green]{file_size_mb:.2f} MB[/] "
            f"[dim]from[/] [blue]{client_ip}[/] {query}"
        )
    else:
        log.info(
            f"[bold magenta]{icon}[/] [white]{method} {path}[/] "
            f"[dim]from[/] [blue]{client_ip}[/] {query or '[no params]'}"
        )

    try:
        response = await call_next(request)
    except Exception as exc:
        process_time = time.time() - start_time
        log.exception(
            f"[bold red]Exception[/] {method} {path} | {process_time:.3f}s → {exc}"
        )
        raise

    process_time = time.time() - start_time
    status_color = "[bold green]" if response.status_code < 400 else "[bold red]"

    log.info(
        f"{status_color}→ {response.status_code}[/] [dim]| {process_time:.3f}s[/]"
    )

    return response


# Include routers
app.include_router(health.router)
app.include_router(transcription.router)


# Final ready message
log.info("[bold green]Server ready — http://0.0.0.0:8001[/bold green]")
log.info("[dim]Interactive docs:[/] [underline blue]http://127.0.0.1:8001/docs[/underline blue]")