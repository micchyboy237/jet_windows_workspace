import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

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

from scipy.io import wavfile

def transcribe_and_translate_chunk(audio_bytes: bytes) -> str:
    """Offloaded heavy work: transcribe JA audio chunk → translate to EN."""
    # Save bytes to temporary WAV (faster-whisper expects file path)
    temp_path = Path("temp_chunk.wav")
    # Convert raw PCM s16le (16kHz, mono) bytes → int16 numpy array → write WAV
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(temp_path, 16000, audio_np)

    segments, _ = whisper_model.transcribe(
        str(temp_path),
        language="ja",
        beam_size=5,
        vad_filter=True,  # Helps with short/live chunks
    )
    ja_text = " ".join(s.text.strip() for s in segments if s.text.strip())

    temp_path.unlink()  # Cleanup

    if not ja_text:
        return ""

    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(ja_text))
    results = translator.translate_batch([source_tokens])
    en_tokens = results[0].hypotheses[0]
    en_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(en_tokens), skip_special_tokens=True)
    return en_text


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
                en_text = await fut
                if en_text.strip():
                    await manager.send_text(en_text.strip(), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        log.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        log.error(f"Error in WebSocket: {e}")
        await websocket.close()
