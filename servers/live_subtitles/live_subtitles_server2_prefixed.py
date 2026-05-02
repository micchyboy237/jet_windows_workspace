import asyncio
import json
import logging
import shutil
import uuid as uuid_module
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import uvicorn
from audio_context_buffer import AudioContextBuffer
from audio_search import search_audio
from diff_utils import console_diff_highlight, extract_new_ja_text
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from sentence_matcher_ja import fuzzy_match_prefix_texts
from sentence_utils import split_sentences_ja
from transcribe_jp_funasr import TranscriptionResult, transcribe_japanese
# from translate_jp_en_llm import translate_japanese_to_english
# from translate_jp_en_llm_cached import translate_japanese_to_english
from translate_jp_en_llm_prefixed import translate_japanese_to_english

console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "success": "green bold",
            "warning": "yellow",
            "error": "red bold",
            "value": "white bold",
            "time": "magenta bold",
            "number": "bright_white",
            "uuid": "bright_blue",
        }
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("live_subtitles_server2")
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = True


OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
N_SEGMENT_RESULTS = 10
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"
LAST_N_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR
LIVE_AUDIO_BUFFER_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Live Japanese Subtitles Server 2")
active_connections: dict[str, WebSocket] = {}

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe_worker")
context_buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)
prev_end_sec: float | None = None
prev_vad_reason = None


def should_reset_context(header: dict) -> bool:
    """Determine if we should reset the context buffer based on time gap or silence."""
    global prev_vad_reason, prev_end_sec
    current_start_sec = float(header.get("start_sec", 0.0))
    current_end_sec = float(header.get("end_sec", 0.0))
    vad_reason = header.get("vad_reason")
    if prev_end_sec is not None:
        gap = current_start_sec - prev_end_sec
        if gap > 3.0:
            console.print(
                f"[warning]Large time gap detected: {gap:.2f}s > 3.0s → Resetting context[/warning]"
            )
            prev_end_sec = current_end_sec
            prev_vad_reason = vad_reason
            return True
    if context_buffer.segments and prev_vad_reason == "silence":
        console.print("[info]Silence detected via VAD → Resetting context[/info]")
        prev_end_sec = current_end_sec
        prev_vad_reason = vad_reason
        return True
    prev_end_sec = current_end_sec
    prev_vad_reason = vad_reason
    return False


def blocking_process_audio(
    audio_bytes: bytes,
    header: dict
) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work.
    """
    global prev_vad_reason, prev_end_sec

    uuid_ = header.get("uuid")
    if not uuid_:
        console.print("[error]Missing UUID in header[/error]")
        return {"message": "missing uuid", "success": False}

    sample_rate = header.get("sample_rate", 16000)
    full_trans_result = None

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    if should_reset_context(header):
        context_buffer.reset()
    else:
        prev_vad_reason = header["vad_reason"]

    context_audio_int16 = context_buffer.get_context_audio()
    if context_audio_int16.size > 0:
        full_audio_int16 = np.concatenate([context_audio_int16, audio_np])
    else:
        full_audio_int16 = audio_np

    full_audio_bytes = full_audio_int16.tobytes()

    console.print(f"[info]VAD Reason:[/info] [value]{header['vad_reason']}[/value]")
    console.print(f"[info]Context duration:[/info] [time]{context_buffer.get_total_duration():.2f}s[/time]")
    console.print(f"[info]Audio duration:[/info] [time]{header['duration_sec']:.2f}s[/time]")

    full_trans_result = transcribe_japanese(
        audio_bytes=full_audio_bytes,
        sample_rate=sample_rate,
    )
    full_trans_result = full_trans_result.copy()
    full_word_segments = full_trans_result.pop("word_segments")
    full_phrase_segments = full_trans_result.pop("phrase_segments")
    full_metadata = full_trans_result.pop("metadata")

    full_word_segments_text = "".join(s["word"] for s in full_word_segments)
    full_ja_text = full_word_segments_text
    full_ja_sents = split_sentences_ja(full_ja_text)

    history = None
    prev_full_ja_text = None
    prev_full_en_text = None

    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_ja_text = last_meta.get("full_ja_text", "")
        prev_full_en_text = last_meta.get("full_en_text", "")

        # Guard: skip if full transcription is empty after cleaning
        full_ja_clean = full_ja_text.rstrip('.。！？、…・「」『』').rstrip()
        if not full_ja_clean:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "success": False,
                "message": "Empty transcription after cleaning",
            }

        # history = context_buffer.get_context_history()

        # Build history from context buffer segments, mirroring
        # the progressive accumulation in translate_jp_en_llm_prefixed __main__
        history = []
        for _, seg_meta in context_buffer.segments:
            ja = seg_meta.get("full_ja_text", "")
            en = seg_meta.get("full_en_text", "")
            if ja and en:
                history.append({"role": "user", "content": ja})
                history.append({"role": "assistant", "content": en})

        # Step 1: Translate the full current Japanese text
        trans_result = translate_japanese_to_english(
            text=full_ja_text,
            history=history,
        )
        full_en_translated = trans_result["text"].strip()

        # Step 2: Use fuzzy_match_prefix_texts to strip the already-shown
        # prefix and compute only the new/incremental JA and EN parts
        prefix_result = fuzzy_match_prefix_texts({
            "prev_ja": prev_full_ja_text,
            "prev_en": prev_full_en_text,
            "full_ja": full_ja_text,
            "full_en": full_en_translated,
        })

        new_ja_text = prefix_result["new_ja"].strip()
        new_en_text = prefix_result["new_en"].strip()
        is_continuation = prefix_result["is_continuation"]

        console.print(
            f"[info]Prefix match is_continuation:[/info] [value]{is_continuation}[/value]"
        )

        # Step 3: Assemble full_en_text based on whether this is a continuation
        if is_continuation:
            full_en_text = (
                (prev_full_en_text + "\n" + new_en_text).strip()
                if new_en_text else prev_full_en_text
            )
        else:
            # Not a continuation — correction or restart, use full translation
            full_en_text = full_en_translated
            console.print(
                "[success]Not a continuation → using full translation as full_en_text[/success]"
            )

        en_text = new_en_text
        ja_text = new_ja_text

        old_ja_sents = split_sentences_ja(prev_full_ja_text)
        old_en_sents = split_sentences_ja(prev_full_en_text)
        old_ja_text = prev_full_ja_text
        old_en_text = prev_full_en_text
        new_ja_sents = split_sentences_ja(new_ja_text)
        new_en_sents = split_sentences_ja(new_en_text)
        last_sentence_clean = None
        last_sentence_pos = -1

    else:
        # No previous context — translate everything fresh
        ja_sents = full_ja_sents
        ja_text = full_ja_text
        curr_clean = ja_text.rstrip('.。！？、…・「」『』').rstrip()
        if curr_clean:
            # history = context_buffer.get_context_history()


            # No prior segments yet — history starts empty,
            # same as step 1 in translate_jp_en_llm_prefixed __main__
            history = []
            full_trans_en = translate_japanese_to_english(
                text=ja_text,
                history=history,
            )
            new_ja_text = ja_text
            new_ja_sents = full_ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
            new_en_text = en_text
            new_en_sents = split_sentences_ja(new_en_text)
        else:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "success": False,
                "message": "Empty transcription after cleaning",
            }

        old_ja_sents = []
        old_en_sents = []
        old_ja_text = ""
        old_en_text = ""
        last_sentence_clean = None
        last_sentence_pos = -1

    if history:
        console.print(f"[bold magenta]History ({len(history)}):[/bold magenta]")
        console.print(
            json.dumps(history, indent=1, ensure_ascii=False),
            style="bright_blue on grey11",
        )
        console.print("\n")

    if old_ja_sents:
        console.print(f"[success]Old JA ({len(old_ja_sents)} sents):[/success]")
        console.print(f"[bright_white]{old_ja_text}[/bright_white]")

    console.print(f"[success]New JA ({len(new_ja_text)} chars):[/success]")
    console.print(f"[bold cyan]{new_ja_text}[/bold cyan]")

    if old_en_sents:
        console.print(f"[success]Old EN ({len(old_en_sents)} sents):[/success]")
        console.print(f"[bright_white]{old_en_text}[/bright_white]")

    console.print(f"[success]New EN ({len(en_text)} chars):[/success]")
    console.print(f"[bold cyan]{en_text}[/bold cyan]")

    console.print(f"[success]Full JA ({len(full_ja_sents)} sents):[/success]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")

    if en_text.strip():
        console.print("[success]Full EN:[/success]")
        console.print(f"[bold white]{full_en_text}[/bold white]")
    else:
        console.print("[dim italic]No new translation[/dim italic]")

    search_audio(full_audio_bytes, audio_bytes)

    if prev_full_ja_text and full_ja_text != prev_full_ja_text:
        console.print("[info]Diff (previous full JA → current full JA):[/info]")
        console_diff_highlight(
            prev_full_ja_text,
            full_ja_text,
            "Prev JA",
            "Curr JA",
        )

    if prev_full_en_text and full_en_text != prev_full_en_text:
        console.print("[info]Diff (previous full EN → current full EN):[/info]")
        console_diff_highlight(
            prev_full_en_text,
            full_en_text,
            "Prev EN",
            "Curr EN",
        )

    new_en_sents = split_sentences_ja(full_en_text)

    started_at_iso = header.get("started_at")
    if started_at_iso and isinstance(started_at_iso, str):
        iso_str = started_at_iso.replace("Z", "+00:00") if started_at_iso.endswith("Z") else started_at_iso
        try:
            dt = datetime.fromisoformat(iso_str)
            ts_str = dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    segment_dir = LAST_N_SEGMENTS_DIR / f"segments_{ts_str}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)

    audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)

    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)

    wavfile.write(str(segment_dir / "full_sound.wav"), sample_rate, full_audio_int16)

    with open(segment_dir / "ja_sents.json", "w", encoding="utf-8") as f:
        json.dump({
            "old_ja_sents": old_ja_sents,
            "new_ja_sents": new_ja_sents,
        }, f, ensure_ascii=False, indent=2)

    with open(segment_dir / "en_sents.json", "w", encoding="utf-8") as f:
        json.dump({
            "old_en_sents": old_en_sents,
            "new_en_sents": new_en_sents,
        }, f, ensure_ascii=False, indent=2)

    md_results = (
        f"JA: {ja_text}\n"
        f"EN: {en_text}\n"
    )
    with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md_results)

    metadata_out = {
        "uuid": uuid_,
        "duration_sec": header.get("duration_sec"),
        "started_at": header.get("started_at"),
        "transcribed_at": datetime.now().isoformat(),
    }
    with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, ensure_ascii=False, indent=2)

    subdirs = sorted(
        [d for d in LAST_N_SEGMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("segments_")],
        key=lambda d: d.name,
    )
    if len(subdirs) > N_SEGMENT_RESULTS:
        for old in subdirs[:-N_SEGMENT_RESULTS]:
            shutil.rmtree(old, ignore_errors=True)

    context_duration = context_buffer.get_total_duration()
    context_uuid = context_buffer.get_context_uuid() or uuid_

    context_buffer.add_audio_segment(audio_np, {
        "uuid": header["uuid"],
        "forced": header["forced"],
        "vad_reason": header["vad_reason"],
        "start_sec": header["start_sec"],
        "end_sec": header["end_sec"],
        "duration_sec": header["duration_sec"],
        "started_at": header["started_at"],
        "old_ja_sents": old_ja_sents,
        "new_ja_sents": new_ja_sents,
        "old_en_sents": old_en_sents,
        "new_en_sents": new_en_sents,
        "full_ja_text": full_ja_text,
        "full_en_text": full_en_text,
        "ja_text": ja_text,
        "en_text": en_text,
    })

    full_audio_dir = LIVE_AUDIO_BUFFER_DIR
    if full_audio_int16.size > 0:
        wavfile.write(
            str(full_audio_dir / "full_sound.wav"),
            context_buffer.sample_rate,
            full_audio_int16,
        )
    else:
        (full_audio_dir / "full_sound.wav").write_bytes(b"")

    context_summary = {
        "total_duration_sec": round(context_buffer.get_total_duration(), 3),
        "num_chunks": len(context_buffer.segments),
        "max_duration_sec": context_buffer.max_duration_sec,
        "sample_rate": context_buffer.sample_rate,
        "last_updated": datetime.now().isoformat(),
        "context_includes_current_segment": True,
    }
    with open(full_audio_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(context_summary, f, ensure_ascii=False, indent=2)

    full_audio_metadata = context_buffer.get_list_metadata()
    with open(full_audio_dir / "full_audio_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_audio_metadata, f, ensure_ascii=False, indent=2)

    with open(full_audio_dir / "full_transcription.json", "w", encoding="utf-8") as f:
        json.dump(full_trans_result, f, ensure_ascii=False, indent=2)

    with open(full_audio_dir / "full_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, ensure_ascii=False, indent=2)

    with open(full_audio_dir / "full_word_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": "word",
            "count": len(full_word_segments),
            "text": full_word_segments_text,
            "segments": full_word_segments
        }, f, ensure_ascii=False, indent=2)

    with open(full_audio_dir / "full_phrase_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": "phrase",
            "count": len(full_phrase_segments),
            "phrases": [p["phrase"] for p in full_phrase_segments],
            "segments": full_phrase_segments
        }, f, ensure_ascii=False, indent=2)

    with open(full_audio_dir / "full_ja_sents.json", "w", encoding="utf-8") as f:
        json.dump(full_ja_sents, f, ensure_ascii=False, indent=2)

    return {
        "uuid": uuid_,
        "new_duration": header['duration_sec'],
        "context_uuid": context_uuid,
        "context_duration": context_duration,
        "success": bool(ja_text and en_text),
        "transcription_ja": new_ja_text,
        "translation_en": en_text,
        "transcribed_duration_sec": full_metadata["transcribed_duration_sec"],
        "transcribed_duration_pctg": full_metadata["transcribed_duration_pctg"],
        "coverage_label": full_metadata["coverage_label"],
        "old_ja_sents": old_ja_sents,
        "new_ja_sents": new_ja_sents,
        "old_en_sents": old_en_sents,
        "new_en_sents": new_en_sents,
        "phrase_segments": full_phrase_segments,
    }


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


@app.websocket("/ws/live-subtitles")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    client_id = str(uuid_module.uuid4())
    active_connections[client_id] = websocket
    console.print(
        f"[success]Client connected[/success] [uuid]{client_id[-6:]}[/uuid]"
        f" from [value]{client_info}[/value]"
        f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
    )
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
                console.rule(style="dim")
                console.print(f"[info]Processing[/info] [uuid]{uuid_[-6:]}…[/uuid]")

                future = asyncio.get_running_loop().run_in_executor(
                    executor,
                    blocking_process_audio,
                    audio_bytes,
                    header_dict,
                )
                response = await future

                if response["success"]:
                    sent = await safe_send(websocket, response)
                    if not sent:
                        logger.info(f"Client gone before result sent uuid={uuid_[-6:]}…")
                        break
                    console.print(
                        f"[success]Processed successfully[/success] [uuid]{uuid_[-6:]}…[/uuid]"
                    )
                else:
                    console.print(
                        f"[warning]Empty response: {response.get('message', '')}[/warning]"
                        f" [uuid]{uuid_[-6:]}…[/uuid]"
                    )
                console.rule(style="dim")

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
        console.print(
            f"[warning]Client disconnected[/warning] [uuid]{client_id[-6:]}[/uuid]"
            f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
        )

# ====================== NEW: Pydantic Models for REST APIs ======================
class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 encoded PCM int16 audio (optional if file uploaded)")
    sample_rate: int = Field(16000, description="Sample rate of the audio")
    hotwords: Optional[str] = Field(None, description="Hotwords for ASR")

class TranscribeResponse(BaseModel):
    success: bool
    transcription_ja: str
    metadata: Dict[str, Any]
    word_segments: list = []
    phrase_segments: list = []

class TranslateRequest(BaseModel):
    japanese_text: str = Field(..., description="Japanese text to translate")
    history: Optional[list] = Field(default=None, description="Conversation history for context")
    temperature: Optional[float] = Field(0.35, ge=0.0, le=1.0)

class TranslateResponse(BaseModel):
    success: bool
    translation_en: str
    quality: str = "N/A"
    log_prob: Optional[float] = None
    confidence: Optional[float] = None

# ====================== NEW: REST Endpoints ======================

@app.post("/transcribe")
async def transcribe_endpoint(
    audio_file: UploadFile = File(..., description="Japanese audio file (WAV, PCM int16 recommended)"),
    sample_rate: int = Form(16000, description="Sample rate of the audio"),
    hotwords: Optional[str] = Form(None, description="Optional hotwords for better recognition"),
):
    """Transcribe Japanese audio → Japanese text (REST API)"""
    try:
        console.print(f"[info]Received file upload: {audio_file.filename} ({audio_file.content_type})[/info]")
        
        audio_bytes = await audio_file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        console.print(f"[info]Audio size: {len(audio_bytes)/1024:.1f} KB | Sample rate: {sample_rate} Hz[/info]")

        # Call existing transcription function
        result: TranscriptionResult = transcribe_japanese(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            hotwords=hotwords,
        )

        return {
            "success": True,
            "transcription_ja": result.get("text", ""),
            "metadata": result.get("metadata", {}),
            "word_segments": result.get("word_segments", []),
            "phrase_segments": result.get("phrase_segments", []),
        }

    except Exception as e:
        console.print(f"[error]Transcription endpoint failed: {e}[/error]")
        import traceback
        console.print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


@app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest):
    """Translate Japanese text to English only (REST API)."""
    try:
        if not request.japanese_text or not request.japanese_text.strip():
            raise HTTPException(status_code=400, detail="japanese_text is required and cannot be empty")

        result = translate_japanese_to_english(
            text=request.japanese_text.strip(),
            history=request.history,
            temperature=request.temperature or 0.35,
        )

        return {
            "success": True,
            "translation_en": result["text"],
            "quality": result.get("quality", "N/A"),
            "log_prob": result.get("log_prob"),
            "confidence": result.get("confidence"),
        }
    except Exception as e:
        console.print(f"[error]Translation endpoint error: {e}[/error]")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("🚀 Starting [bold cyan]Live Japanese Subtitles Server 2[/]")
    logger.info("WebSocket endpoint → [bold]ws://0.0.0.0:8000/ws/live-subtitles[/]")
    logger.info("New REST endpoints:")
    logger.info("   POST /transcribe")
    logger.info("   POST /translate")
    logger.info("Press Ctrl+C to stop\n")
    uvicorn.run(
        app="live_subtitles_server2:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
