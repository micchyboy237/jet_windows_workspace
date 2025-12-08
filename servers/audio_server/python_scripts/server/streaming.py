# app/sse/server.py
from __future__ import annotations

import asyncio
from contextlib
from datetime import datetime
from typing import AsyncGenerator, Literal

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anyio

from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
log = logging.getLogger("sse_server")


class SSEEvent(BaseModel):
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None  # milliseconds


async def event_generator(
    request: Request,
    *,
    interval: float = 0.5,
    total_events: int = 50,
) -> AsyncGenerator[str, None]:
    """
    Generic SSE generator — replace or inject your real logic here.
    """
    event_id = 0

    # Send reconnection retry instruction once
    yield f"retry: 3000\n\n"

    try:
        for i in range(total_events):
            if await request.is_disconnected() is False:
                log.info("[yellow]Client disconnected, stopping stream[/yellow]")
                break

            event_id += 1
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "index": i,
                "message": f"Hello from server @ {datetime.utcnow().strftime('%H:%M:%S')}"
            }

            sse_lines = [
                f"id: {event_id}",
                "event: message",
                f"data: {payload}",
                "",  # empty line = end of event
            ]

            yield "\n".join(sse_lines) + "\n"
            log.info(f"[green]Sent event {event_id}/{total_events}[/green]")

            await asyncio.sleep(interval)

        # Final [DONE] message — tells well-behaved clients to stop
        yield "data: [DONE]\n\n"
        log.info("[bold magenta]Stream completed — sent [DONE][/bold magenta]")

    except asyncio.CancelledError:
        log.warning("[red]Stream cancelled by client[/red]")
        raise
    except Exception as exc:
        log.exception(f"[red]Unexpected error in SSE generator: {exc}[/red]")
        yield "event: error\ndata: Internal server error\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (minimal, reusable)
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="FastAPI SSE Demo", version="1.0.0")


@app.get("/stream", response_class=StreamingResponse)
async def stream_sse(request: Request):
    """
    Server-Sent Events endpoint.
    Clients should set Accept: text/event-stream
    """
    async def sse_wrapper():
        async for chunk in event_generator(request):
            yield chunk

    return StreamingResponse(
        sse_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disables nginx buffering
            "Content-Encoding": "identity",
        }
    )


# Optional: health check
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}