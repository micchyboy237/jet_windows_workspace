import asyncio
import websockets
from protocol import TextResult
from rich.console import Console

console = Console()


class SpeechToText:
    async def transcribe(self, pcm_bytes: bytes) -> str:
        # TODO: swap real model
        return f"(final ja->en) {len(pcm_bytes)} bytes"


async def process_stream(ws):
    stt = SpeechToText()
    buffer: list[bytes] = []
    console.log("[server] connected")

    async for raw in ws:
        msg = raw.decode("utf-8", errors="ignore")

        if msg == "start":
            buffer.clear()
            continue

        if msg == "end":
            pcm = b"".join(buffer)
            buffer.clear()
            text = await stt.transcribe(pcm)
            payload: TextResult = {"type": "text", "text": text}
            await ws.send(payload["text"])
            continue

        # else: speech
        buffer.append(raw)


async def main():
    async with websockets.serve(process_stream, "0.0.0.0", 8765):
        console.log("server ready")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
