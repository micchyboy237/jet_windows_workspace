import asyncio
import websockets
from typing import AsyncIterator
from protocol import AudioChunk, TextResult
from rich.console import Console

console = Console()


async def fake_transcribe_japanese_to_english(chunk: bytes) -> str:
    # Placeholder: swap in Whisper, faster-whisper, Deepgram, etc.
    return f"TRANSCRIBED(ja)->en: {len(chunk)} bytes"


async def process_stream(websocket) -> None:
    console.log("[server] connection accepted")

    async for message in websocket:
        if not isinstance(message, (bytes, bytearray)):
            continue  # ignore non-bytes frames

        # Convert raw bytes to text (placeholder)
        text = await fake_transcribe_japanese_to_english(message)

        payload: TextResult = {"type": "text", "text": text}
        await websocket.send(payload["text"].encode("utf-8"))


async def main() -> None:
    async with websockets.serve(process_stream, "0.0.0.0", 8765):
        console.log("[server] listening on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
