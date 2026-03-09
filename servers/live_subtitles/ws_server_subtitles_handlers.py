# ws_server_subtitles_handlers.py

"""
Modern WebSocket handler for live Japanese subtitles server.
Receives speech chunks / complete utterances → buffers per utterance → triggers fast & slow processing.
Compatible with client messages containing 'speech_chunk' and 'complete_utterance'.
"""
import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Literal

from logger import logger  # assuming centralized logger

from processing.fast_processor import process_fast_llm
from processing.slow_processor import process_slow

from websockets.server import WebSocketServerProtocol

connected_states: Dict[WebSocketServerProtocol, 'ConnectionState'] = {}


class ConnectionState:
    """Per-client connection state"""
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.current_utterance_id: Optional[str] = None
        self.audio_buffer: bytearray = bytearray()
        self.chunk_count: int = 0
        self.prev_utterance_pcm: Optional[bytes] = None           # ← NEW
        self.last_context_prompt: Optional[str] = None
        self.last_ja: Optional[str] = None
        self.last_en: Optional[str] = None
        self.utterance_start: Optional[datetime] = None
        # self.hotword_manager = HotwordManager()  # ← enable when implemented

    def reset_utterance(self) -> None:
        self.audio_buffer = bytearray()
        self.chunk_count = 0
        # Note: we KEEP prev_utterance_pcm across utterances

    def append_chunk(self, pcm_bytes: bytes, sample_rate: int) -> None:
        self.audio_buffer.extend(pcm_bytes)
        self.chunk_count += 1

    def get_duration_sec(self, sample_rate: int) -> float:
        if sample_rate <= 0:
            return 0.0
        return len(self.audio_buffer) / (2 * sample_rate)  # int16 = 2 bytes per sample


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
                    await handle_speech_message(websocket, state, data)
                elif msg_type == "speaker_diarization":
                    await handle_speaker_diarization(websocket, state, data)
                elif msg_type == "emotion_classification":
                    await handle_emotion_classification(websocket, state, data)
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

async def handle_speech_message(
    websocket: WebSocketServerProtocol,
    state: ConnectionState,
    data: dict
) -> None:
    """Handle speech_chunk / complete_utterance messages."""

    client_id = state.client_id

    pcm_b64 = data.get("pcm")
    if not pcm_b64:
        logger.warning(f"[{client_id}] Missing pcm")
        return

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
        f"[{client_id}] Processing chunk {chunk_index} for utt {utterance_id} "
        f"{'(final)' if is_final else '(partial)'}"
    )

    await process_utterance(
        websocket,
        state,
        sr,
        utterance_id,
        segment_num,
        data["segment_type"],
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


async def handle_speaker_diarization(
    websocket: WebSocketServerProtocol,
    state: ConnectionState,
    data: dict
) -> None:
    """Handle standalone speaker diarization request."""

    client_id = state.client_id

    curr_pcm_b64 = data.get("curr_pcm")
    prev_pcm_b64 = data.get("prev_pcm")
    sample_rate = data.get("sample_rate", 16000)

    if not curr_pcm_b64:
        logger.warning(f"[{client_id}] speaker_diarization missing curr_pcm")
        return

    curr_pcm = base64.b64decode(curr_pcm_b64)
    prev_pcm = base64.b64decode(prev_pcm_b64) if prev_pcm_b64 else None

    from live_subtitles_server import executor_slow

    loop = asyncio.get_running_loop()

    speaker_res, _ = await loop.run_in_executor(
        executor_slow,
        process_slow,
        curr_pcm,
        prev_pcm,
        sample_rate,
        data.get("utterance_id", "external"),
        data.get("segment_idx", 0),
        data.get("segment_num", 0),
    )

    await websocket.send(json.dumps({
        "type": "speaker_update",
        **speaker_res
    }, ensure_ascii=False))


async def handle_emotion_classification(
    websocket: WebSocketServerProtocol,
    state: ConnectionState,
    data: dict
) -> None:
    """Handle standalone emotion classification request."""

    client_id = state.client_id

    pcm_b64 = data.get("pcm")
    sample_rate = data.get("sample_rate", 16000)

    if not pcm_b64:
        logger.warning(f"[{client_id}] emotion_classification missing pcm")
        return

    pcm = base64.b64decode(pcm_b64)

    from live_subtitles_server import executor_slow

    loop = asyncio.get_running_loop()

    _, emotion_res = await loop.run_in_executor(
        executor_slow,
        process_slow,
        pcm,
        None,
        sample_rate,
        data.get("utterance_id", "external"),
        data.get("segment_idx", 0),
        data.get("segment_num", 0),
    )

    await websocket.send(json.dumps({
        "type": "emotion_classification_update",
        **emotion_res
    }, ensure_ascii=False))



async def process_utterance(
    websocket: WebSocketServerProtocol,
    state: ConnectionState,
    sample_rate: int,
    utterance_id: str,
    segment_num: int,
    segment_type: Literal["speech", "non-speech"],
    context_prompt: Optional[str],
    last_ja: Optional[str],
    last_en: Optional[str],
    incoming_data: dict,
    rms: float,
    is_final: bool = True,
) -> None:
    """Trigger fast transcription+translation, then (if final) slow diarization+emotion"""
    if len(state.audio_buffer) < 800:  # ~25 ms @ 16kHz
        logger.warning(f"[{state.client_id}] Utterance {utterance_id} too short — ignoring")
        return
    
    audio_bytes = bytes(state.audio_buffer)

    if not audio_bytes or audio_bytes == b'\x00' * len(audio_bytes):
        logger.warning(
            f"[{state.client_id}] Utterance {utterance_id} is empty or pure silence "
            f"({len(audio_bytes)} bytes) — skipping processing"
        )
        if is_final:
            state.reset_utterance()
        return

    
    from live_subtitles_server import executor_fast, executor_slow, DEFAULT_OUT_DIR

    loop = asyncio.get_running_loop()

    # ── Fast path: transcription + translation ───────────────────────────────
    fast_result = await loop.run_in_executor(
        executor_fast,
        process_fast_llm,
        audio_bytes,
        sample_rate,
        state.client_id,
        segment_num,
        datetime.now(timezone.utc),  # end_of_utterance_received_at
        rms,
        None,                        # hotwords — add when implemented
        None,                        # received_at
        context_prompt,
        last_ja,
        last_en,
        DEFAULT_OUT_DIR,
    )

    ja = fast_result["transcription_ja"]
    en = fast_result["translation_en"]
    meta = fast_result["meta"]

    # Extract fields from client message (with fallbacks)
    segment_idx = incoming_data.get("segment_idx", 0)
    chunk_index = incoming_data.get("chunk_index", 0)
    segment_type = incoming_data.get("segment_type", "speech")
    avg_vad_confidence = incoming_data.get("avg_vad_confidence", 0.0)
    duration_sec = incoming_data.get("duration_sec", state.get_duration_sec(sample_rate))
    overlap_sec = incoming_data.get("overlap_sec", 0.0)
    start_time = incoming_data.get("start_time")

    log_prefix = "FINAL" if is_final else "PARTIAL"
    logger.info(
        f"\n[{state.client_id}] {log_prefix} utt:{utterance_id}\n"
        f"chunk: {chunk_index} | dur:{duration_sec:.2f}s\n"
        f"ja:{ja[:60]!r}\nen:{en[:60]!r}"
    )

    payload = {
        "type": "final_subtitle" if is_final else "partial_subtitle",
        "utterance_id": utterance_id,
        "is_final": is_final,
        "segment_idx": segment_idx,
        "segment_num": segment_num,
        "chunk_index": chunk_index,
        "segment_type": segment_type,
        "avg_vad_confidence": avg_vad_confidence,
        "rms": rms,
        "transcription_ja": ja,
        "translation_en": en,
        "duration_sec": round(duration_sec, 3),
        "overlap_sec": overlap_sec,
        "start_time": start_time,
        "transcription_confidence": meta["transcription"].get("confidence"),
        "transcription_quality": meta["transcription"].get("quality_label", "N/A"),
        "translation_confidence": meta["translation"].get("confidence"),
        "translation_quality": meta["translation"].get("quality", "N/A"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        # consider adding: "has_prev_pcm": state.prev_utterance_pcm is not None
    }

    state.last_ja = ja
    state.last_en = en

    await websocket.send(json.dumps(payload, ensure_ascii=False))
    logger.debug(f"[{state.client_id}] Sent {payload['type']} for utt {utterance_id}")

    # ── Slow path ─────────────────────────────────
    # Only run slow processing (speaker + emotion) on final utterances
    # if is_final:  
    #     async def run_slow_processing() -> None:
    #         try:
    #             prev_pcm_to_use = state.prev_utterance_pcm
    #             # After using it → update for next utterance
    #             state.prev_utterance_pcm = bytes(state.audio_buffer)

    #             speaker_res, emotion_res = await loop.run_in_executor(
    #                 executor_slow,
    #                 process_slow,
    #                 bytes(state.audio_buffer),
    #                 prev_pcm_to_use,     # ← now using previous utterance if available
    #                 sample_rate,
    #                 utterance_id,
    #                 segment_idx,
    #                 segment_num,
    #             )

    #             # Send speaker update
    #             await websocket.send(json.dumps({
    #                 "type": "speaker_update",
    #                 "utterance_id": utterance_id,
    #                 "segment_idx": segment_idx,
    #                 "segment_num": segment_num,
    #                 "segment_type": "speech",
    #                 "chunk_index": chunk_index,
    #                 "is_final": is_final,
    #                 **speaker_res
    #             }, ensure_ascii=False))

    #             # Send emotion update
    #             await websocket.send(json.dumps({
    #                 "type": "emotion_classification_update",
    #                 "utterance_id": utterance_id,
    #                 "segment_idx": segment_idx,
    #                 "segment_num": segment_num,
    #                 "segment_type": "speech",
    #                 "chunk_index": chunk_index,
    #                 "is_final": is_final,
    #                 **emotion_res
    #             }, ensure_ascii=False))

    #             logger.info(f"[{state.client_id}] Slow results sent for utt {utterance_id}")

    #         except Exception as e:
    #             logger.exception(f"[{state.client_id}] Slow processing failed for utt {utterance_id}")

    #     asyncio.create_task(run_slow_processing())
