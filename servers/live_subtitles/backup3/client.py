import asyncio
import sounddevice as sd
import websockets
from rich.console import Console

console = Console()


async def audio_chunks(blocksize: int, samplerate: int):
    # Generator that yields raw PCM frames
    def callback(indata, frames, time, status):
        if status:
            console.log(status)
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        while True:
            chunk = await queue.get()
            yield chunk


async def run_client(uri: str = "ws://localhost:8765"):
    console.log("[client] connecting...")
    async with websockets.connect(uri) as ws:
        console.log("[client] connected")

        producer = audio_chunks(blocksize=2048, samplerate=16000)

        async for chunk in producer:
            await ws.send(chunk)
            # Non-blocking check for subtitles
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                console.log(f"[subtitle] {msg}")
            except asyncio.TimeoutError:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        console.log("client stopped")
