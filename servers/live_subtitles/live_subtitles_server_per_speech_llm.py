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

# ────────────────────────────────────────────────────────────────────────────────
# Imports from refactored modules
# ────────────────────────────────────────────────────────────────────────────────
from transcribe_jp_llm import (
    transcribe_japanese_llm_from_bytes,
    TranscriptionResult,
)
from translate_jp_en_llm import translate_japanese_to_english_structured
from logger import logger

logger.info("Live subtitles server starting...")

# ────────────────────────────────────────────────────────────────────────────────
# Globals / shared resources
# ────────────────────────────────────────────────────────────────────────────────

executor = ThreadPoolExecutor(max_workers=1)
DEFAULT_OUT_DIR: Path | None = None
ENABLE_TRANSLATION_SCORING: bool = False


class ConnectionState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.current_utterance_id: str | None = None
        self.audio_buffer: bytearray = bytearray()
        self.chunk_count: int = 0
        self.last_context_prompt: str | None = None
        self.utterance_start: datetime | None = None

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
    utterance_id: str,
    segment_num: int,
    end_of_utterance_received_at: datetime,
    normalized_rms: float,
    received_at: datetime | None = None,
    context_prompt: str | None = None,
    out_dir: Path | None = None,
) -> tuple[str, str, float, dict]:

    processing_started_at = datetime.now(timezone.utc)

    trans_result: TranscriptionResult = transcribe_japanese_llm_from_bytes(
        audio_bytes=audio_bytes,
        sample_rate=sr,
        client_id=client_id,
        utterance_id=utterance_id,
        segment_num=segment_num,
    )

    ja_text = trans_result.text_ja

    en_text, translation_logprob, translation_confidence, translation_quality = translate_japanese_to_english_structured(
        ja_text,
        max_decoding_length=512,
        min_tokens_for_confidence=3,
        enable_scoring=ENABLE_TRANSLATION_SCORING,
    )


    processing_finished_at = datetime.now(timezone.utc)

    meta = {
        **trans_result.metadata,
        "timestamp_iso": processing_started_at.isoformat(),
        "transcription": {
            "text_ja": ja_text,
            "confidence": None,
            "quality_label": "N/A",
        },
        "translation": {
            "text_en": en_text,
            "log_prob": round(translation_logprob, 4) if translation_logprob else None,
            "confidence": round(translation_confidence, 4) if translation_confidence else None,
            "quality_label": translation_quality,
        },
        "segments": trans_result.segments,
        "normalized_rms": normalized_rms,
    }

    return ja_text, en_text, 0.0, meta


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
                    normalized_rms = data.get("normalized_rms")
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
                        data,
                        normalized_rms,
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
    data: dict,
    normalized_rms: float,
    is_final: bool = False,
) -> None:
    if len(state.audio_buffer) < 1000:
        logger.warning(f"[{state.client_id}] Empty or too small utterance {utterance_id}")
        return

    loop = asyncio.get_running_loop()

    ja, en, conf, meta = await loop.run_in_executor(
        executor,
        transcribe_and_translate,
        bytes(state.audio_buffer),
        sample_rate,
        state.client_id,
        utterance_id,
        segment_num,
        datetime.now(timezone.utc),  # end_of_utterance_received_at
        normalized_rms,
        None,
        context_prompt,
        DEFAULT_OUT_DIR,
    )

    segment_idx = data.get("segment_idx", 0)
    segment_type = data.get("segment_type", "speech")
    client_duration = data.get("duration_sec")
    avg_vad_confidence = data.get("avg_vad_confidence", 0.0)
    normalized_rms = data.get("normalized_rms")

    duration_sec = client_duration if client_duration is not None else state.get_duration_sec(sample_rate)

    payload = {
        "type": "final_subtitle" if is_final else "partial_subtitle",
        "utterance_id": utterance_id,
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "segment_type": segment_type,
        "avg_vad_confidence": avg_vad_confidence,
        "normalized_rms": normalized_rms,
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