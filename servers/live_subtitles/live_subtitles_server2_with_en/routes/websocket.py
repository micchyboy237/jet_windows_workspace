"""
WebSocket route for real-time audio processing.
"""
import asyncio
import json
import logging
import uuid as uuid_module
from fastapi import WebSocket, WebSocketDisconnect
from rich.console import Console
from core.state import (
    get_active_connections,
    get_executor,
)
from core.processing import blocking_process_audio, _get_speaker_labeler

console = Console()
logger = logging.getLogger("uvicorn.error")


def split_message(data: bytes) -> tuple[dict, bytes]:
    """Split raw WebSocket binary message into (header dict, audio bytes)."""
    if b"\x00" not in data:
        raise ValueError("Message does not contain null byte separator")
    header_part, audio_bytes = data.split(b"\x00", 1)
    header = json.loads(header_part.decode("utf-8", errors="replace"))
    return header, audio_bytes


async def safe_send(websocket: WebSocket, payload: dict) -> bool:
    """
    Send a JSON payload over the WebSocket.
    Returns True on success, False if the client has already disconnected.
    """
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except (WebSocketDisconnect, RuntimeError) as exc:
        logger.debug(f"safe_send: client gone ({exc})")
        return False


async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time audio processing."""
    from core.state import save_speaker_state
    
    await websocket.accept()
    client_info = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    client_id = str(uuid_module.uuid4())
    active_connections = get_active_connections()
    active_connections[client_id] = websocket
    executor = get_executor()
    
    console.print(
        f"[success]Client connected[/success] [uuid]{client_id[-6:]}[/uuid]"
        f" from [value]{client_info}[/value]"
        f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
    )
    
    _get_speaker_labeler()
    
    try:
        while True:
            try:
                message: bytes = await websocket.receive_bytes()
            except WebSocketDisconnect:
                break
            except RuntimeError as exc:
                logger.debug(f"receive_bytes RuntimeError (client gone): {exc}")
                break
            
            header_dict: dict = {}
            try:
                header_dict, audio_bytes = split_message(message)
                uuid_ = header_dict.get("uuid", "???")
                segment_id = header_dict.get("segment_id", "???")
                console.rule(style="dim")
                console.print(f"[info]Processing[/info] [uuid]{segment_id}…[/uuid]")
                
                future = asyncio.get_running_loop().run_in_executor(
                    executor,
                    blocking_process_audio,
                    audio_bytes,
                    header_dict,
                )
                response = await future
                
                sent = await safe_send(websocket, response)
                if not sent:
                    logger.info(f"Client gone before result sent uuid={segment_id}…")
                    break
                
                if response["success"]:
                    console.print(
                        f"[success]Processed successfully[/success] [uuid]{segment_id}…[/uuid]"
                    )
                else:
                    console.print(
                        f"[warning]Empty response sent: {response.get('message', '')}[/warning]"
                        f" [uuid]{segment_id}…[/uuid]"
                    )
                console.rule(style="dim")
                
            except Exception as proc_err:
                logger.error(f"Processing error for segment: {proc_err}")
                logger.exception("Full traceback:")
                error_resp = {
                    "uuid": header_dict.get("uuid", "unknown"),
                    "error": str(proc_err),
                    "success": False,
                    "ja_text": "",
                    "en_text": "",
                    "speaker_label": "SPEAKER_UNKNOWN",
                    "speaker_confidence": 0.0,
                    "speakers": [],
                }
                sent = await safe_send(websocket, error_resp)
                if not sent:
                    logger.info("Client gone — could not send error response, exiting.")
                    break
                    
    except Exception as exc:
        logger.error(f"Unexpected WebSocket error: {exc}")
        logger.exception("Full traceback:")
    finally:
        active_connections.pop(client_id, None)
        save_speaker_state()
        console.print(
            f"[warning]Client disconnected[/warning] [uuid]{client_id[-6:]}[/uuid]"
            f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
        )
