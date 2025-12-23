import asyncio
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from rich.logging import RichHandler
from transformers import AutoTokenizer
from translator_types import Translator  # Assuming this is available from existing code
# Remove pydub dependency – use numpy + scipy instead

# Shared constants from existing code
TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"
WHISPER_MODEL_NAME = "kotoba-tech/kotoba-whisper-v2.0-faster"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=True, show_path=False, markup=True)],
)
log = logging.getLogger("live_subtitles_server")

app = FastAPI(title="Live JA→EN Subtitles Server")

# Global shared resources (loaded once at startup)
whisper_model: Optional[WhisperModel] = None
translator: Optional[Translator] = None
tokenizer: Optional[AutoTokenizer] = None
executor: Optional[ThreadPoolExecutor] = None


@app.on_event("startup")
async def startup_event():
    global whisper_model, translator, tokenizer, executor
    log.info("Loading shared Whisper model on CUDA...")
    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cuda",
        compute_type="float32",
    )
    log.info("Loading shared OPUS-MT ja→en translator...")
    translator = Translator(
        TRANSLATOR_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=4,  # Adjustable based on CPU cores
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    executor = ThreadPoolExecutor(max_workers=4)  # n parallel workers
    log.info("[bold green]Server ready – accepting WebSocket connections[/bold green]")


def transcribe_and_translate_chunk(audio_bytes: bytes) -> str:
    """Offloaded heavy work: transcribe JA audio chunk → translate to EN, and return JSON with segment info."""
    # Convert raw PCM s16le (16kHz, mono) bytes → float32 numpy array in [-1.0, 1.0]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Log VAD-applied duration for visibility (matches existing log pattern)
    if len(audio_np) > 0:
        duration_sec = len(audio_np) / 16000
        log.info(f"Processing audio with duration {duration_sec:.3f}s")

    segments, info = whisper_model.transcribe(
        audio_np,
        language="ja",
        beam_size=5,
        vad_filter=True,  # Helps remove silence in short/live chunks
        word_timestamps=False,  # Not needed for subtitle text
    )

    if hasattr(info, "vad_duration_removed") and info.vad_duration_removed is not None:
        removed_sec = info.vad_duration_removed
        log.info(f"VAD filter removed {removed_sec:.3f}s of audio")
    else:
        log.info("VAD filter applied (no duration removed reported)")

    # Gather each segment for subtitles
    segment_infos: List[Dict[str, Any]] = []
    ja_texts: List[str] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            segment_infos.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })
            ja_texts.append(text)
    ja_text = " ".join(ja_texts)

    if not ja_text:
        return json.dumps({"segments": [], "en_text": ""})

    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(ja_text))
    results = translator.translate_batch([source_tokens])
    en_tokens = results[0].hypotheses[0]
    en_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(en_tokens), skip_special_tokens=True)
    return json.dumps({
        "segments": segment_infos,
        "en_text": en_text.strip(),
    })


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, asyncio.Queue] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = asyncio.Queue()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/live-subtitles")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    log.info(f"New client connected: {websocket.client}")
    try:
        while True:
            data = await websocket.receive_bytes()
            # Expect raw PCM s16le, 16kHz, mono bytes
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                executor, transcribe_and_translate_chunk, data
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
    except Exception as e:
        log.error(f"Error in WebSocket: {e}")
        await websocket.close()
