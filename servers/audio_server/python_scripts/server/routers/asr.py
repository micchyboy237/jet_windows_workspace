import asyncio
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import dataclasses
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich import print as rprint
from transformers import AutoTokenizer
from python_scripts.server.services.translator_types import Translator
from python_scripts.server.services.cache_service import (
    load_transcriber_model,
    load_translator_model,
    load_translator_tokenizer,
)
from python_scripts.server.services.translator.batch_translation_service import translate_text, translate_batch_texts
# Remove pydub dependency – use numpy + scipy instead

# Shared constants from existing code
TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"
WHISPER_MODEL_NAME = "kotoba-tech/kotoba-whisper-v2.0-faster"

# Setup logging with enhanced Rich console and tracebacks
console = Console()
install(console=console, show_locals=False)  # Rich tracebacks

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
    ],
    force=True,
)
log = logging.getLogger("live_subtitles_server")

# Add module-level Progress instance for server-wide/client-wide tracking
progress = Progress(
    "[progress.description]{task.description}",
    "[bold blue]{task.fields[client]}",
    BarColumn(),
    MofNCompleteColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    console=console,
)
progress_task_id: Dict[WebSocket, int] = {}

router = APIRouter(tags=["asr", "speech-recognition", "translation"])

executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)


def transcribe_and_translate_chunk(audio_bytes: bytes, loop: asyncio.AbstractEventLoop) -> str:
    """Offloaded heavy work: transcribe JA audio chunk → translate to EN, and return JSON."""

    whisper_model = load_transcriber_model()

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) > 0:
        duration_sec = len(audio_np) / 16000
        log.info(f"[bold green]Transcribing[/] {duration_sec:.3f}s JA audio chunk")

    segments, info = whisper_model.transcribe(
        audio_np,
        language="ja",
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )

    if hasattr(info, "vad_duration_removed") and info.vad_duration_removed is not None:
        removed_sec = info.vad_duration_removed
        log.info(f"VAD filter removed {removed_sec:.3f}s of audio")
    else:
        log.info("VAD filter applied (no duration removed reported)")

    segment_infos: List[dict] = []
    ja_texts: List[str] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            segment_infos.append(dataclasses.asdict(seg))
            ja_texts.append(text)

    if not ja_texts:
        log.info("[yellow]No speech detected – empty transcription/translation[/]")
        return json.dumps({"segments": [], "en_text": "", "en_segment_texts": []})

    log.info(f"Transcribed {len(ja_texts)} segments → translating to EN")
    ja_text = " ".join(ja_texts)

    # The main event loop is running in the FastAPI/Uvicorn thread.
    # We are currently in a worker thread → use run_coroutine_threadsafe to schedule
    # the async translation coroutines on the main loop.

    en_texts_future = asyncio.run_coroutine_threadsafe(translate_batch_texts(ja_texts), loop)
    en_text_future = asyncio.run_coroutine_threadsafe(translate_text(ja_text), loop)

    en_texts: List[str] = en_texts_future.result()
    en_text: str = en_text_future.result()

    # Logging as per edit prompt
    if ja_texts:
        log.info("[bold cyan]JA transcription:[/] %s", ja_text.strip())
        log.info("[bold magenta]EN translation (full):[/] %s", en_text.strip())
        # Log per-segment translations with alignment
        for i, (ja_seg, en_seg) in enumerate(zip(ja_texts, en_texts), 1):
            log.info("[dim]Segment %02d[/] [cyan]JA:[/] %s → [magenta]EN:[/] %s", i, ja_seg.strip(), en_seg.strip())
    else:
        log.info("[yellow]No speech detected – empty transcription/translation[/]")

    log.info("[green]Completed chunk processing – sending %d segments to client[/]", len(ja_texts))

    return json.dumps({
        "segments": segment_infos,
        "en_text": en_text.strip(),
        "en_segment_texts": en_texts,
    })


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, asyncio.Queue] = {}
        self.progress = progress  # Optional: for access if needed

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = asyncio.Queue()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws/live-subtitles")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    log.info(f"New client connected: {websocket.client} – starting progress tracking")
    task_id = progress.add_task(
        "Live subtitles processing",
        total=None,
        client=f"{websocket.client}",
    )
    progress_task_id[websocket] = task_id
    progress.start()
    try:
        while True:
            data = await websocket.receive_bytes()
            # Expect raw PCM s16le, 16kHz, mono bytes
            progress.update(
                progress_task_id[websocket],
                advance=1,
                description="Processing audio chunk",
            )
            log.info(f"Received and queued audio chunk (~{len(data)/32000:.2f}s) for {websocket.client}")
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                executor,
                transcribe_and_translate_chunk,
                data,
                loop,   # ← pass main loop explicitly
            )
            # Put result in per-connection queue for ordered delivery
            await manager.active_connections[websocket].put(future)

            # Drain queue in order
            while not manager.active_connections[websocket].empty():
                fut = await manager.active_connections[websocket].get()
                result_json = await fut
                if result_json.strip():
                    await manager.send_text(result_json, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        log.info(f"Client disconnected: {websocket.client}")
        task_id = progress_task_id.pop(websocket, None)
        if task_id is not None:
            progress.remove_task(task_id)
    except Exception:
        log.exception(f"Unexpected error in WebSocket for {websocket.client}")
        await websocket.close()
