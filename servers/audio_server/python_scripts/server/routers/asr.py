from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Literal

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as TranscriptionSegment
from rich.logging import RichHandler

from python_scripts.server.services.cache_service import load_model

router = APIRouter(prefix="/asr", tags=["asr", "speech-recognition", "translation"])

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("asr_router")


ResultType = Literal["partial", "final"]


async def streaming_asr_inference(
    audio_chunks: AsyncGenerator[bytes, None],
) -> AsyncGenerator[dict[str, object], None]:
    """
    Perform streaming Japanese → English translation using faster-whisper.

    Processes incoming raw mono 16kHz PCM16 audio chunks:
      - Buffers until a minimum duration is reached.
      - Runs translation with VAD-aware segmentation.
      - Yields partial hypotheses and final translated segments.
      - Maintains overlap context from the last detected speech segment.

    Args:
        audio_chunks: Async generator yielding raw PCM16 bytes.

    Yields:
        JSON-serializable dicts with keys depending on result type:
          - partial results: {"partial": str, "final": False}
          - final results:   {"english": str, "final": True, "start": float, "end": float}
    """
    buffer: list[npt.NDArray[np.float32]] = []
    sample_rate: int = 16000
    chunk_duration_sec: float = 5.0
    min_buffer_sec: float = 0.250

    chunk_count: int = 0

    model: WhisperModel = await load_model()

    async for chunk_bytes in audio_chunks:
        # Convert int16 bytes → float32 in range [-1.0, 1.0]
        audio_np: npt.NDArray[np.float32] = (
            np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        )

        buffer.append(audio_np)
        current_duration: float = len(buffer) * chunk_duration_sec
        chunk_count += 1

        log.debug(
            f"[bold cyan]Received chunk {chunk_count}[/] – buffer duration: {current_duration:.2f}s"
        )

        if current_duration < min_buffer_sec:
            continue

        # Concatenate for inference
        full_audio: npt.NDArray[np.float32] = np.concatenate(buffer)

        log.info(
            f"[bold green]Running inference on {current_duration:.2f}s buffer ({len(full_audio)} samples)[/]"
        )

        segments: tuple[list[TranscriptionSegment], object]
        segments, _ = model.transcribe(
            full_audio,
            language="ja",
            task="translate",
            beam_size=5,
            temperature=0.0,
            without_timestamps=False,
            vad_filter=True,  # Explicitly enable VAD for better silence handling
            log_progress=True,
        )

        segments_list: list[TranscriptionSegment] = list(segments)

        if segments_list:
            for seg in segments_list:
                cleaned_text: str = seg.text.strip()

                # Note: faster-whisper transcribe() yields only final segments.
                # True token-level partials would require a custom streaming decoder.
                yield {"partial": cleaned_text, "final": False}

                yield {
                    "english": cleaned_text,
                    "final": True,
                    "start": seg.start,
                    "end": seg.end,
                }

            # Keep overlap from last speech segment for context
            last_end: float = segments_list[-1].end
            overlap_samples: int = int(last_end * sample_rate)

            if 0 < overlap_samples < len(full_audio):
                buffer = [full_audio[overlap_samples:]]
                log.debug(
                    f"[dim]Kept overlap of {len(buffer[0])/sample_rate:.2f}s for context[/]"
                )
            else:
                buffer = []
                log.debug("[dim]Buffer cleared after processing[/]")
        else:
            # No speech detected → clear buffer and send empty partial
            buffer = []
            log.debug("[dim]No speech — buffer cleared[/]")
            yield {"partial": "", "final": False}


@router.websocket("/live-jp-en")
async def live_japanese_to_english_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for live streaming Japanese → English translation.

    Client sends raw 16kHz mono PCM16 audio chunks.
    Server responds with JSON objects containing partial/final translations.
    """
    await websocket.accept()

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    log.info("[bold blue]WebSocket connection accepted – starting audio receive/process tasks[/]")

    async def receive_audio() -> None:
        """Receive audio chunks from the client."""
        received_count: int = 0
        while True:
            try:
                audio_bytes: bytes = await websocket.receive_bytes()
                received_count += 1
                log.debug(
                    f"[cyan]← Client chunk {received_count} ({len(audio_bytes)} bytes)[/]"
                )
                await audio_queue.put(audio_bytes)
            except WebSocketDisconnect:
                log.info("[yellow]Client disconnected during receive[/]")
                break

    async def process_and_send() -> None:
        """Consume queued audio and stream ASR results back to the client."""
        sent_count: int = 0

        async def audio_generator() -> AsyncGenerator[bytes, None]:
            while True:
                chunk = await audio_queue.get()
                yield chunk

        async for result in streaming_asr_inference(audio_generator()):
            sent_count += 1
            final_str: str = "[bold green]FINAL[/]" if result.get("final") else "[dim]partial[/]"
            text_preview: str = (
                result.get("english", result.get("partial", ""))[:60] + "..."
                if result.get("english") or result.get("partial")
                else ""
            )
            log.debug(f"[magenta]→ Result {sent_count} {final_str}: {text_preview}[/]")
            await websocket.send_json(result)

    try:
        await asyncio.gather(receive_audio(), process_and_send())
    except WebSocketDisconnect:
        log.info("[yellow]WebSocket disconnected – cleanup complete[/]")