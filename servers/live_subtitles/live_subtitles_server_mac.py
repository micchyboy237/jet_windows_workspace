import json
import logging
import uuid
from datetime import datetime
from typing import NotRequired, TypedDict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from rich.logging import RichHandler
from transcribe_jp_funasr import transcribe_japanese

# from translate_jp_en_openai import translate_japanese_to_english
from translate_jp_en_mac import translate_subtitle

# ====================== RICH LOGGING SETUP ======================
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
    ],
)

logger = logging.getLogger("live_subtitles")

for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = True

app = FastAPI(title="Live Japanese Subtitles Server")

# ------------------------------------------------------------------
active_connections: dict[str, WebSocket] = {}
# ------------------------------------------------------------------


class IncomingHeader(TypedDict, total=False):
    uuid: str
    start_sec: float
    end_sec: float
    duration_sec: NotRequired[float]
    sample_rate: int
    format: str
    channels: int
    language: str
    vad_reason: NotRequired[str]
    forced: NotRequired[bool]
    segment_rms: NotRequired[float]
    loudness: NotRequired[float]
    has_sound: NotRequired[bool]
    started_at: str


class TranscriptionResult(TypedDict, total=False):
    text: str


class TranslationResult(TypedDict, total=False):
    text: str


class OutgoingResponse(TypedDict, total=False):
    uuid: str
    transcription_ja: str
    translation_en: str
    start_sec: float
    end_sec: float
    processed_at: str
    processing_time_sec: float


def split_message(data: bytes) -> tuple[dict, bytes]:
    if b"\x00" not in data:
        raise ValueError("Message does not contain null byte separator")
    header_part, audio_bytes = data.split(b"\x00", 1)
    header = json.loads(header_part.decode("utf-8", errors="replace"))
    return header, audio_bytes


async def safe_send(websocket: WebSocket, payload: dict) -> bool:
    """
    Send a JSON payload over the WebSocket.
    Returns True on success, False if the client has already disconnected.
    Raises for unexpected errors.
    """
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except (WebSocketDisconnect, RuntimeError) as exc:
        # Covers: starlette WebSocketDisconnect, uvicorn ClientDisconnected
        # wrapped as RuntimeError, and "close message has been sent".
        logger.debug(f"safe_send: client gone ({exc})")
        return False


@app.websocket("/ws/live-subtitles")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    client_id = str(uuid.uuid4())
    active_connections[client_id] = websocket

    logger.info(
        f"Client connected [bold cyan]{client_id}[/] from [cyan]{client_info}[/]"
    )

    try:
        while True:
            # --- Receive ---
            try:
                message: bytes = await websocket.receive_bytes()
            except WebSocketDisconnect:
                # Clean disconnect from client side — exit loop quietly.
                break
            except RuntimeError as exc:
                # "WebSocket is not connected" or similar after abrupt close.
                logger.debug(f"receive_bytes RuntimeError (client gone): {exc}")
                break

            start_time = datetime.utcnow()
            header_dict: dict = {}

            # --- Process segment ---
            try:
                header_dict, full_audio_bytes = split_message(message)
                header: IncomingHeader = header_dict  # type: ignore[assignment]

                uuid_str: str = header.get("uuid", "unknown")
                sample_rate: int = header.get("sample_rate", 16000)
                audio_kb = len(full_audio_bytes) / 1024

                logger.info(
                    f"Received segment [bold]uuid={uuid_str[:8]}…[/] "
                    f"audio=[cyan]{audio_kb:.1f} KiB[/]"
                )

                # Transcription
                transcribe_result: TranscriptionResult = transcribe_japanese(
                    audio_bytes=full_audio_bytes,
                    sample_rate=sample_rate,
                )
                ja_text = str(transcribe_result.get("text", "")).strip()

                # Translation (only if meaningful text detected)
                trans_en: TranslationResult = {}
                en_text = ""
                if ja_text and len(ja_text.strip()) > 3:
                    # trans_en = translate_japanese_to_english(ja_text=ja_text)
                    # en_text = str(trans_en.get("text", "")).strip()
                    print(f"JA:\n{ja_text}")
                    print("\nEN:")
                    en_text = translate_subtitle(ja_text)

                processing_time = (datetime.utcnow() - start_time).total_seconds()

                # Build response — start with header fields, then overwrite/extend
                response: dict = {
                    **header,
                    "uuid": uuid_str,
                    "client_id": client_id,
                    "transcription_ja": ja_text,
                    "translation_en": en_text,
                    "processed_at": datetime.utcnow().isoformat(),
                    "processing_time_sec": round(processing_time, 3),
                }
                # Merge extra fields from results (excluding 'text' already extracted)
                for k, v in transcribe_result.items():
                    if k != "text":
                        response[k] = v
                for k, v in trans_en.items():
                    if k != "text":
                        response[k] = v

                # --- Send result ---
                sent = await safe_send(websocket, response)
                if not sent:
                    logger.info(
                        f"Client disconnected before result could be sent "
                        f"uuid={uuid_str[:8]}…"
                    )
                    break

                logger.info(
                    f"Sent result for [bold]uuid={uuid_str[:8]}…[/] "
                    f"ja:[green]{len(ja_text)}[/] chars | "
                    f"en:[green]{len(en_text)}[/] chars "
                    f"([dim]{processing_time:.3f}s[/])"
                )
                if en_text:
                    logger.info(
                        f"EN: [blue]{en_text[:120]}{'…' if len(en_text) > 120 else ''}[/]"
                    )

            except Exception as proc_err:
                logger.error(f"Processing error for segment: {proc_err}")
                logger.exception("Full traceback:")

                error_resp = {
                    "uuid": header_dict.get("uuid", "unknown"),
                    "error": str(proc_err),
                    "transcription_ja": "",
                    "translation_en": "",
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
        logger.info(
            f"Client [bold cyan]{client_id}[/] disconnected from [cyan]{client_info}[/]"
        )


if __name__ == "__main__":
    logger.info("🚀 Starting [bold cyan]Live Japanese Subtitles Server[/]")
    logger.info("WebSocket endpoint → [bold]ws://127.0.0.1:8000/ws/live-subtitles[/]")
    logger.info("Press Ctrl+C to stop\n")

    uvicorn.run(
        app="live_subtitles_server_mac:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
