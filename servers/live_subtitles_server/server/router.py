from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from server.service import TranscriptionSession
from typing import Dict
import logging

log = logging.getLogger("subtitle_server")

router = APIRouter()

active_sessions: Dict[WebSocket, TranscriptionSession] = {}

@router.websocket("/ws/subtitles")
async def subtitles_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = TranscriptionSession()
    active_sessions[websocket] = session
    log.info("[bold cyan]New client connected[/bold cyan]")

    try:
        while True:
            data = await websocket.receive_bytes()
            subtitles = await session.process_audio_chunk(data)
            for text in subtitles:
                if text.strip():
                    await websocket.send_text(text.strip())
    except WebSocketDisconnect:
        log.info("[bold yellow]Client disconnected[/bold yellow]")
    except Exception as exc:
        log.error(f"[bold red]Error in connection: {exc}[/bold red]")
    finally:
        active_sessions.pop(websocket, None)