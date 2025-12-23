# main.py
import logging
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse

from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from translator_types import Translator  # Assuming this is from ctranslate2

from utils.audio_utils import resolve_audio_paths  # Existing utility

# Rich logging setup (unchanged from original)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.handlers.RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("transcribe")

# Constants (unchanged)
TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"

# Global models loaded once
whisper_model: Optional[WhisperModel] = None
translator: Optional[Translator] = None
tokenizer: Optional[AutoTokenizer] = None

app = FastAPI(
    title="Japanese Audio → English Translation API",
    description="Batch and streaming transcription + translation using kotoba-whisper-v2.0-faster + OPUS-MT.",
    version="1.0.0",
)


@app.on_event("startup")
async def load_models():
    global whisper_model, translator, tokenizer
    log.info("Loading models on startup...")
    whisper_model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
        num_workers=4,
    )
    translator = Translator(
        TRANSLATOR_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    log.info("All models loaded and ready.")


# Reusable core function (unchanged)
def transcribe_and_translate_file(
    model: WhisperModel,
    translator: Translator,
    tokenizer: AutoTokenizer,
    audio_path: str,
    language: Optional[str] = None,
) -> str:
    log.info(f"Starting transcription + translation: [bold cyan]{audio_path}[/bold cyan]")

    segments_iter, _ = model.transcribe(audio_path, language=language or "ja", beam_size=5, vad_filter=False)

    segments = []
    for s in segments_iter:
        segments.append(dataclasses.asdict(s))

    ja_text = " ".join(segment["text"].strip() for segment in segments if segment["text"].strip())
    if not ja_text:
        log.warning(f"No Japanese text detected in {audio_path}")
        return ""

    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(ja_text))
    results = translator.translate_batch([source_tokens])
    en_tokens = results[0].hypotheses[0]
    en_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(en_tokens), skip_special_tokens=True)
    log.info(f"Completed: [bold green]{audio_path}[/bold green]")
    return en_text


def process_single_file(audio_path: Path, language: str = "ja") -> dict:
    """Wrapper returning dict compatible with original TranslationResult."""
    try:
        en_text = transcribe_and_translate_file(whisper_model, translator, tokenizer, str(audio_path), language)
        return {
            "audio_path": str(audio_path),
            "translation": en_text,
            "success": bool(en_text.strip()),
        }
    except Exception as exc:
        log.error(f"Processing failed for {audio_path}: {exc}")
        return {
            "audio_path": str(audio_path),
            "translation": "",
            "success": False,
        }


# Existing batch async generator (unchanged, only imported here for reuse)
from original_batch import batch_transcribe_and_translate_files_async  # Keep original if preferred, or copy here


@app.post("/transcribe-and-translate-single")
async def transcribe_single(
    file: UploadFile = File(...),
    language: str = Form("ja"),
):
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = process_single_file(temp_path, language)
    temp_path.unlink(missing_ok=True)

    return result


@app.post("/transcribe-and-translate-batch")
async def transcribe_batch_sse(
    files: List[UploadFile] = File(...),
    language: str = Form("ja"),
    save_to_disk: bool = Form(False),
    output_dir: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_dir = Path("temp_batch")
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        for upload in files:
            p = temp_dir / upload.filename
            with open(p, "wb") as f:
                content = await upload.read()
                f.write(content)
            saved_paths.append(p)

        async def event_generator():
            async for result in batch_transcribe_and_translate_files_async(
                audio_paths=[str(p) for p in saved_paths],
                max_workers=4,
                output_dir=output_dir if save_to_disk else None,
                language=language,
            ):
                yield f"data: {result}\n\n"
            yield "data: {\"done\": true}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    finally:
        def cleanup():
            for p in saved_paths:
                p.unlink(missing_ok=True)
            temp_dir.rmdir()
        BackgroundTasks().add_task(cleanup)


@app.websocket("/ws/transcribe-stream")
async def websocket_transcribe_stream(websocket: WebSocket, language: str = "ja"):
    await websocket.accept()
    import tempfile
    import wave
    import numpy as np

    EXPECTED_SAMPLE_RATE = 16000
    CHANNELS = 1

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_wav.name)
    temp_wav.close()

    with wave.open(str(temp_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(EXPECTED_SAMPLE_RATE)

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            with open(temp_path, "ab") as f:
                f.write(data)

            result = process_single_file(temp_path, language)

            if result["success"] and result["translation"].strip():
                await websocket.send_text(f"partial: {result['translation'].strip()}")
            else:
                await websocket.send_text("partial: (processing...)")

    except WebSocketDisconnect:
        log.info("Client disconnected – sending final result")
        final = process_single_file(temp_path, language)
        await websocket.send_text(f"final: {final['translation'].strip() if final['success'] else 'Transcription failed'}")

    except Exception as exc:
        log.error(f"WebSocket error: {exc}")
        await websocket.send_text(f"error: {str(exc)}")

    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/stream-demo", response_class=HTMLResponse)
async def stream_demo_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Live Japanese → English Demo</title></head>
    <body>
        <h1>Live Japanese to English Translation</h1>
        <p>Speak Japanese → see English translation in near real-time.</p>
        <button id="start">Start</button>
        <button id="stop" disabled>Stop</button>
        <pre id="output"></pre>
        <script>
            let ws = null;
            const out = document.getElementById('output');
            document.getElementById('start').onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                const recorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
                ws = new WebSocket(`ws://${location.host}/ws/transcribe-stream?language=ja`);
                ws.onmessage = e => {
                    if (e.data.startsWith('partial:')) out.textContent = e.data.slice(8);
                    if (e.data.startsWith('final:')) out.textContent = '[FINAL] ' + e.data.slice(6);
                };
                recorder.ondataavailable = e => { if (e.data.size && ws.readyState === 1) ws.send(e.data); };
                recorder.start(500);
                document.getElementById('start').disabled = true;
                document.getElementById('stop').disabled = false;
            };
            document.getElementById('stop').onclick = () => {
                ws?.close();
                document.getElementById('start').disabled = false;
                document.getElementById('stop').disabled = true;
            };
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": all(x is not None for x in [whisper_model, translator, tokenizer])}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, workers=1, log_level="info")