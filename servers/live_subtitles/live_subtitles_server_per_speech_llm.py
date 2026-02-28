"""
Modern WebSocket live subtitles server (compatible with websockets ≥ 12.0 / 14.0+)
Receives Japanese audio chunks, buffers until end-of-utterance, transcribes & translates.
"""
import asyncio
import base64
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from typing import Optional

from nlp import extract_japanese_nouns
from hotword_manager import HotwordManager
from transcribe_jp_llm import transcribe_japanese, TranscriptionResult
from translate_jp_en_llm import translate_japanese_to_english
# from translate_jp_en_opus import translate_japanese_to_english
from logger import logger

logger.info("Live subtitles server starting...")

# ────────────────────────────────────────────────────────────────────────────────
# Globals / shared resources
# ────────────────────────────────────────────────────────────────────────────────

executor = ThreadPoolExecutor(max_workers=1)
DEFAULT_OUT_DIR: Path | None = Path(__file__).parent / "generated" / Path(__file__).stem
ENABLE_TRANSLATION_SCORING: bool = False


class ConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.current_utterance_id: str | None = None
        self.audio_buffer: bytearray = bytearray()
        self.chunk_count: int = 0
        self.last_context_prompt: str | None = None
        self.last_ja: str | None = None
        self.last_en: str | None = None
        self.utterance_start: datetime | None = None
        self.hotword_manager = HotwordManager()

    def reset_utterance(self):
        self.audio_buffer = bytearray()
        self.chunk_count = 0

    def append_chunk(self, pcm_bytes: bytes, sample_rate: int):
        self.audio_buffer.extend(pcm_bytes)
        self.chunk_count += 1

    def clear_buffer(self):
        self.audio_buffer.clear()
        self.chunk_count = 0

    def get_duration_sec(self, sample_rate: int) -> float:
        return len(self.audio_buffer) / (2 * sample_rate)


connected_states: dict[asyncio.StreamWriter, ConnectionState] = {}


def transcribe_and_translate(
    audio_bytes: bytes,
    sr: int,
    client_id: str,
    segment_num: int,
    end_of_utterance_received_at: datetime,
    rms: float,
    hotwords: str | list[str] | None = None,
    received_at: datetime | None = None,
    context_prompt: str | None = None,
    last_ja: str | None = None,
    last_en: str | None = None,
    out_dir: Path | None = None,
) -> tuple[str, str, float, dict]:
    """Blocking function — run in executor."""
    processing_started_at = datetime.now(timezone.utc)

    # Prepare path if we want to keep the audio
    temp_path = None
    if out_dir:
        ts = processing_started_at.strftime("%Y%m%d_%H%M%S")
        temp_path = out_dir / f"utterance_{client_id}_{segment_num:04d}_{ts}.wav"

    # Core transcription (now in separate module)
    trans_result: TranscriptionResult = transcribe_japanese(
        audio_bytes=audio_bytes,
        sample_rate=sr,
        # hotwords=hotwords,
        # context_prompt=last_ja,
        save_temp_wav=temp_path,
        client_id=client_id,
        segment_num=segment_num,
    )

    ja_text = trans_result["text_ja"]

    history: Optional[list[dict[str, str]]] = None
    # if ja_text:
    #     # Build proper conversation history (user = previous JA, assistant = previous EN)
    #     # Only include if both exist to form a complete previous turn
    #     if last_ja and last_en:
    #         history = [
    #             {"role": "user", "content": last_ja.strip()},
    #             {"role": "assistant", "content": last_en.strip()},
    #         ]

    # Translation
    translation_result = translate_japanese_to_english(
        ja_text,
        max_tokens=768,
        enable_scoring=ENABLE_TRANSLATION_SCORING,
        history=history,
    )
    en_text = translation_result["text"]
    translation_logprob = translation_result["log_prob"]
    translation_confidence = translation_result["confidence"]
    translation_quality = translation_result["quality"]

    processing_finished_at = datetime.now(timezone.utc)
    queue_wait_duration = (processing_started_at - end_of_utterance_received_at).total_seconds()
    processing_duration = (processing_finished_at - processing_started_at).total_seconds()

    meta = {
        **trans_result["metadata"],
        "timestamp_iso": processing_started_at.isoformat(),
        "received_at": received_at.isoformat() if received_at else None,
        "end_of_utterance_received_at": end_of_utterance_received_at.isoformat(),
        "processing_started_at": processing_started_at.isoformat(),
        "processing_finished_at": processing_finished_at.isoformat(),
        "queue_wait_seconds": round(queue_wait_duration, 3),
        "processing_duration_seconds": round(processing_duration, 3),
        "transcription": {
            "text_ja": ja_text,
            "avg_logprob": trans_result["avg_logprob"],
            "confidence": round(trans_result["confidence"], 4) if trans_result["confidence"] is not None else None,
            "quality_label": trans_result["quality_label"],
        },
        "translation": {
            "text_en": en_text,
            "log_prob": round(translation_logprob, 4) if translation_logprob is not None else None,
            "confidence": round(translation_confidence, 4) if translation_confidence is not None else None,
            "quality_label": translation_quality,
        },
        "segments": trans_result["segments"],
        "rms": rms,
        "context": {
            "last_ja": last_ja,
            "last_en": last_en,
            "hotwords": hotwords,
        },
    }

    # Logging
    preview_ja = ja_text[:70] + "..." if len(ja_text) > 70 else ja_text
    preview_en = en_text[:70] + "..." if len(en_text) > 70 else en_text

    # Safe confidence formatting (avoid NoneType crash)
    trans_conf = trans_result.get("confidence")
    if isinstance(trans_conf, (int, float)):
        trans_conf_str = f"{trans_conf:.3f}"
    else:
        trans_conf_str = "N/A"

    logger.info(
        "[transcribe] conf=%s | qual=%s | ja=%s",
        trans_conf_str,
        trans_result.get("quality_label"),
        preview_ja,
    )
    logger.info(
        "[translate] qual=%s | en=%s",
        translation_quality,
        preview_en,
    )

    # Save metadata if permanent storage is enabled
    if out_dir and temp_path:
        meta_path = temp_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved metadata → {meta_path}")

    return ja_text, en_text, trans_result["confidence"], meta


async def handler(websocket):
    """
    Modern websockets handler (websockets.asyncio.server style)
    Receives messages from one client.
    """
    client_id = f"{id(websocket):x}"[:8]
    state = ConnectionState(client_id)
    connected_states[websocket] = state

    logger.info(f"New client connected: {client_id}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type in ("speech_chunk", "complete_utterance"):
                    pcm_b64 = data.get("pcm")
                    if not pcm_b64:
                        logger.warning(f"[{client_id}] Missing pcm")
                        continue

                    pcm_bytes = base64.b64decode(pcm_b64)
                    sr = data.get("sample_rate", 16000)
                    utterance_id = data.get("utterance_id")
                    segment_num = data.get("segment_num", 0)
                    chunk_index = data.get("chunk_index", 0)
                    is_final = data.get("is_final", False)
                    rms = data.get("rms")
                    context_prompt = data.get("context_prompt")

                    if utterance_id != state.current_utterance_id:
                        state.current_utterance_id = utterance_id
                        state.reset_utterance()
                        state.utterance_start = datetime.now(timezone.utc)

                    state.append_chunk(pcm_bytes, sr)
                    state.last_context_prompt = context_prompt

                    logger.info(
                        f"[{client_id}] Processing chunk {chunk_index} for utt {utterance_id} {'(final)' if is_final else '(partial)'}"
                    )

                    await process_utterance(
                        websocket,
                        state,
                        sr,
                        utterance_id,
                        segment_num,
                        context_prompt,
                        state.last_ja,
                        state.last_en,
                        data,
                        rms,
                        is_final=is_final,
                    )

                    if is_final:
                        logger.info(f"[{client_id}] Final chunk received for utt {utterance_id}")
                        state.reset_utterance()

                else:
                    logger.warning(f"[{client_id}] Unknown message type: {msg_type}")

            except json.JSONDecodeError:
                logger.warning(f"[{client_id}] Invalid JSON received")
            except Exception as e:
                logger.exception(f"[{client_id}] Message handler error: {e}")

    except Exception as e:
        logger.exception(f"[{client_id}] Connection error: {e}")
    finally:
        connected_states.pop(websocket, None)
        logger.info(f"[{client_id}] Disconnected")


async def process_utterance(
    websocket,
    state: ConnectionState,
    sample_rate: int,
    utterance_id: str,
    segment_num: int,
    context_prompt: str | None,
    last_ja: str | None,
    last_en: str | None,
    data: dict,
    rms: float,
    is_final: bool = False,
) -> None:
    if len(state.audio_buffer) < 1000:
        logger.warning(f"[{state.client_id}] Empty or too small utterance {utterance_id}")
        return

    hotwords = state.hotword_manager.get_hotwords()

    loop = asyncio.get_running_loop()

    ja, en, conf, meta = await loop.run_in_executor(
        executor,
        transcribe_and_translate,
        bytes(state.audio_buffer),
        sample_rate,
        state.client_id,
        segment_num,
        datetime.now(timezone.utc),  # end_of_utterance_received_at
        rms,
        hotwords,
        None,
        context_prompt,
        last_ja,
        last_en,
        DEFAULT_OUT_DIR,
    )

    segment_idx = data.get("segment_idx", 0)
    segment_type = data.get("segment_type", "speech")
    avg_vad_confidence = data.get("avg_vad_confidence", 0.0)
    rms = data.get("rms")
    client_duration = data.get("duration_sec")

    log_prefix = "FINAL" if is_final else "PARTIAL"
    duration_sec = client_duration if client_duration is not None else state.get_duration_sec(sample_rate)


    logger.info(
        f"[{state.client_id}] {log_prefix} utt {utterance_id}\n"
        f"duration: {duration_sec:.2f}\n"
        f"last ja:  {last_ja!r}\n"
        f"last en:  {last_en!r}\n"
        f"hotwords: {hotwords!r}\n"
        f"ja:       {ja!r}\n"
        f"en:       {en!r}\n"
        f"vad:      {avg_vad_confidence}\n"
        f"tr_conf:  {meta['translation']['confidence']}\n"
        f"tl_conf:  {meta['transcription']['confidence']}\n"
        "------------"
    )


    payload = {
        "type": "final_subtitle" if is_final else "partial_subtitle",
        "utterance_id": utterance_id,
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "segment_type": segment_type,
        "avg_vad_confidence": avg_vad_confidence,
        "rms": rms,
        "transcription_ja": meta["transcription"]["text_ja"],
        "translation_en": meta["translation"]["text_en"],
        "duration_sec": round(duration_sec, 3),
        "transcription_confidence": meta["transcription"]["confidence"],
        "transcription_quality": meta["transcription"]["quality_label"],
        "translation_confidence": meta["translation"].get("confidence"),
        "translation_quality": meta["translation"]["quality_label"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
    }

    state.last_ja = ja
    state.last_en = en

    # --- Hotword update logic (only on final utterances) ---
    if is_final and ja:
        nouns = extract_japanese_nouns(
            ja,
            deduplicate=True,
        )

        if nouns:
            state.hotword_manager.update_from_tokens(nouns)
            state.hotword_manager.decay()

            updated_hotwords = state.hotword_manager.get_hotwords()

            logger.debug(
                f"[{state.client_id}] Hotwords updated ({len(updated_hotwords)}) → "
                f"{updated_hotwords}"
            )

    await websocket.send(json.dumps(payload, ensure_ascii=False))
    logger.info(f"[{state.client_id}] Sent {'final' if is_final else 'partial'} subtitle for utt {utterance_id}")



def pcm_bytes_to_waveform(pcm: bytes | bytearray, *, dtype=np.int16) -> np.ndarray:
    if len(pcm) == 0:
        raise ValueError("Empty PCM buffer")
    waveform = np.frombuffer(pcm, dtype=dtype)
    if waveform.size == 0:
        raise ValueError("Empty PCM buffer after conversion")
    waveform = waveform.astype(np.float32) / 32768.0
    return waveform


async def main():
    from websockets.asyncio.server import serve

    async with serve(
        handler,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=None,
    ) as server:
        logger.info("WebSocket server listening on ws://0.0.0.0:8765")
        await server.serve_forever()


if __name__ == "__main__":
    import os
    import shutil

    out_dir_str = os.getenv("UTTERANCE_OUT_DIR")
    if out_dir_str:
        shutil.rmtree(out_dir_str, ignore_errors=True)
        DEFAULT_OUT_DIR = Path(out_dir_str).resolve()
        logger.info(f"Permanent utterance storage enabled: {DEFAULT_OUT_DIR}")
    else:
        logger.info(
            "Using temporary files for utterances "
            "(set UTTERANCE_OUT_DIR env var to enable permanent storage)"
        )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")