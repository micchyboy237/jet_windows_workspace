import asyncio
import websockets
import sounddevice as sd
import numpy as np

# Audio settings – must match server expectations
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_SECONDS = 2.0  # Send 2-second chunks for low latency


async def live_subtitles_client(uri: str = "ws://localhost:8000/ws/live-subtitles"):
    async with websockets.connect(uri) as ws:
        print("Connected to live subtitles server. Speak Japanese – translations will appear.")

        def audio_callback(indata: np.ndarray, frames: int, time, status):
            """Callback called by sounddevice for each audio block."""
            if status:
                print(status)
            # Convert to bytes (little-endian int16)
            audio_bytes = indata.tobytes()
            # Fire-and-forget send
            asyncio.create_task(ws.send(audio_bytes))

        # Start blocking input stream (runs in background thread)
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
            callback=audio_callback,
        ):
            while True:
                try:
                    message = await ws.recv()
                    print(f"\n[EN Subtitle]: {message}")
                except websockets.ConnectionClosed:
                    print("\nServer disconnected.")
                    break


if __name__ == "__main__":
    # Run with: python client_example.py
    # Install dependencies: pip install websockets sounddevice numpy
    try:
        asyncio.run(live_subtitles_client())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
