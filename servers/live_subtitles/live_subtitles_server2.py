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
from sentence_matcher_ja import fuzzy_shortest_best_match_contains
from sentence_utils import split_sentences_ja
from transcribe_jp_funasr import TranscriptionResult, transcribe_japanese
from translate_jp_en_llm import translate_japanese_to_english

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

    new_audio_duration_sec = len(audio_np) / sample_rate
    context_duration_sec = context_buffer.get_total_duration()
    max_duration_sec = context_buffer.max_duration_sec
    combined_naive_sec = context_duration_sec + new_audio_duration_sec

    context_audio_int16, actual_context_sec, segments_used = (
        context_buffer.get_context_audio_within_limit(new_audio_duration_sec)
    )

    if combined_naive_sec > max_duration_sec:
        dropped_segments = len(context_buffer.segments) - segments_used
        console.print(
            f"[warning]⚠️  Combined audio ({combined_naive_sec:.2f}s) would exceed "
            f"max_duration_sec ({max_duration_sec:.2f}s). "
            f"Dropped {dropped_segments} oldest segment(s) to stay within limit. "
            f"Using {segments_used} segment(s) = {actual_context_sec:.2f}s context.[/warning]"
        )

    if context_audio_int16.size > 0:
        full_audio_int16 = np.concatenate([context_audio_int16, audio_np])
    else:
        full_audio_int16 = audio_np

    actual_full_duration_sec = len(full_audio_int16) / sample_rate

    # Hard safety guard — should never trigger, but catches bugs immediately.
    if actual_full_duration_sec > max_duration_sec + 1e-3:
        raise RuntimeError(
            f"BUG: full_audio duration {actual_full_duration_sec:.3f}s "
            f"exceeds max_duration_sec {max_duration_sec:.2f}s after trimming. "
            "This should never happen — check get_context_audio_within_limit()."
        )

    full_audio_bytes = full_audio_int16.tobytes()

    console.print(
        f"[info]VAD Reason:[/info] [value]{header['vad_reason']}[/value]"
    )
    console.print(
        f"[info]Context:[/info] [time]{actual_context_sec:.2f}s[/time] used "
        f"/ [time]{context_duration_sec:.2f}s[/time] buffered "
        f"({segments_used}/{len(context_buffer.segments)} segments)"
    )
    console.print(
        f"[info]Full transcription input:[/info] "
        f"[time]{actual_full_duration_sec:.2f}s[/time] / [time]{max_duration_sec:.2f}s[/time] max"
    )
    console.print(
        f"[info]Audio duration:[/info] [time]{header['duration_sec']:.2f}s[/time]"
    )

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

    prev_full_ja_text = None
    prev_full_en_text = None
    unchanged_text = None
    new_ja_text = full_ja_text
    new_ja_start_index = None
    new_ja_similarity = None
    history = None

    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_ja_text = last_meta.get("full_ja_text", "")
        prev_full_en_text = last_meta.get("full_en_text", "")

        new_ja_text_res = extract_new_ja_text(prev_full_ja_text, full_ja_text)
        unchanged_text = new_ja_text_res["unchanged_text"]
        new_ja_text = new_ja_text_res["new_text"]
        new_ja_start_index = new_ja_text_res["start_index"]
        new_ja_similarity = new_ja_text_res["similarity"]

        last_ja_sentence, last_en_sentence, last_utt_id, last_sent_idx = context_buffer.get_last_sentence()

        MATCH_SCORE_CUTOFF = 75
        match_result = fuzzy_shortest_best_match_contains(
            query=new_ja_text,
            texts=full_ja_text,
            score_cutoff=MATCH_SCORE_CUTOFF,
            max_extra_chars=30,
        )

        if match_result["score"] >= MATCH_SCORE_CUTOFF and match_result["start"] != -1:
            console.print("[success bold]✅ Accepted[/success bold]")
            new_text_start = match_result["end"]
            new_text = new_ja_text
        else:
            console.print("[error]❌ Below threshold[/error]")
            console.print(
                f"[warning]Fuzzy match too weak (score={match_result['score']:.1f}).[/warning]"
            )
            console.print(
                f"[warning]Translating the full text.[/warning]"
            )
            new_text = full_ja_text.strip()

        new_clean = new_text.rstrip('.。！？、…・「」『』').rstrip()
        if not new_clean:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "success": False,
                "message": "Same text as previous",
            }

        old_ja_sents = split_sentences_ja(prev_full_ja_text)
        old_ja_text = prev_full_ja_text
        old_en_sents = split_sentences_ja(prev_full_en_text)
        old_en_text = prev_full_en_text
        new_ja_sents = split_sentences_ja(new_text)
        ja_text = "".join(new_ja_sents).strip()

        last_sentence_pos = match_result["start"] if match_result["score"] >= MATCH_SCORE_CUTOFF else -1
        last_sentence_clean = match_result["match"].strip() if match_result["score"] >= MATCH_SCORE_CUTOFF else None

        if ja_text:
            history = context_buffer.get_context_history(max_segments=len(context_buffer.segments))
            print(f"History ({len(history)}) | Segments ({len(context_buffer.segments)})")
            print(json.dumps(history, indent=1, ensure_ascii=False))
            trans_en = translate_japanese_to_english(
                text=ja_text,
                enable_scoring=False,
                history=history,
            )
            en_text = trans_en["text"].strip()
        else:
            en_text = ""

        if prev_full_en_text:
            if new_ja_text_res["start_index"] == 0:
                full_en_text = en_text.strip()
                console.print("[success]Early correction detected → full_en_text reset to clean latest translation (no duplication)[/success]")
            else:
                full_en_text = (prev_full_en_text + "\n" + en_text).strip() if en_text else prev_full_en_text
        else:
            full_en_text = en_text

    else:
        ja_sents = full_ja_sents
        ja_text = full_ja_text
        curr_clean = ja_text.rstrip('.。！？、…・「」『』').rstrip()

        if curr_clean:
            full_trans_en = translate_japanese_to_english(
                text=ja_text,
                enable_scoring=False,
            )
            new_ja_sents = ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
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
        last_sentence_clean = None
        last_sentence_pos = -1

    if history:
        console.print(f"[bold yellow]History ({len(history)}):[/bold yellow]")
        console.print(f"[bold cyan]{history!r}[/bold cyan]")

    if last_sentence_clean:
        console.print(f"[success]Last Sentence (utt_id={last_utt_id[-6:]} | sent_idx={last_sent_idx}):[/success]")
        console.print(f"[bright_white]{last_sentence_clean}[/bright_white]")

    if last_sentence_pos != -1:
        console.print(f"[success]New Text (utt_id={header['uuid'][-6:]} | pos={last_sentence_pos} | start={new_text_start}):[/success]")
        console.print(f"[bright_white]{new_text}[/bright_white]")

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

    if new_ja_text:
        if unchanged_text is not None:
            console.print(f"[success]Unchanged JA ({len(unchanged_text)} chars):[/success]")
            console.print(f"[white]{unchanged_text}[/white]")
        if new_ja_start_index is not None:
            console.print(f"[success]Start index:[/success] [bold cyan]{new_ja_start_index}[/bold cyan]")
        if new_ja_similarity is not None:
            console.print(f"[success]Matched Similarity:[/success] [bold cyan]{new_ja_similarity}[/bold cyan]")

    console.print(f"[success]Full JA ({len(full_ja_sents)} sents):[/success]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")

    if en_text.strip():
        console.print("[success]Full EN:[/success]")
        console.print(f"[bold white]{en_text}[/bold white]")
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
        "matched_pos": last_sentence_pos,
        "matched_sent": last_sentence_clean,
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
        "new_ja_similarity": new_ja_similarity,
        "new_ja_start_index": new_ja_start_index,
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
            enable_scoring=True,   # Enable scoring for REST calls
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
