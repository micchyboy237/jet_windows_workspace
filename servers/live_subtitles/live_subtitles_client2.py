# ====================== CLIENT ======================
import asyncio
import json
import os
import queue

import numpy as np
import sounddevice as sd
import websockets

WS_URI = os.getenv("LOCAL_WS_LIVE_SUBTITLES_URL", "ws://127.0.0.1:8765")

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION)

audio_queue: queue.Queue = queue.Queue()


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """Called by sounddevice in a separate thread"""
    if status:
        print(status)
    # indata shape = (BLOCKSIZE, 1) → flatten to 1D float32
    audio_queue.put(indata.copy().flatten().astype(np.float32))


async def audio_sender(websocket):
    """Send microphone chunks as fast as they arrive"""
    print("🎤 Microphone live – speak Japanese now!")
    while True:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            await websocket.send(chunk.tobytes())  # raw float32 bytes
        else:
            await asyncio.sleep(0.001)  # tiny sleep to yield


async def subtitle_receiver(websocket):
    """Print live subtitles as they arrive"""
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "segments_update":
                segments = data.get("segments", [])
                print(f"\n🔄 Received {len(segments)} segment update(s):")
                for seg in segments:
                    uid = seg["uuid"][:8]
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    jp = seg.get("jp", "")
                    en = seg.get("en", "")

                    print(f"[{start:6.2f} → {end:6.2f}]")
                    print(f"  JP: {jp}")
                    print(f"  EN: {en}")
                    print("Status: Update")
                    print("-" * 60)
            elif data.get("type") == "segment":
                start = data["start"]
                end = data["end"]
                jp = data.get("jp", "")
                en = data.get("en", "")

                print(f"[{start:6.2f} → {end:6.2f}]")
                print(f"  JP: {jp}")
                print(f"  EN: {en}")
                print("Status: Final")
                print("-" * 60)
            else:
                print("Unknown payload:", data)
    except Exception as e:
        print(f"Receiver closed: {e}")


async def main():
    try:
        async with websockets.connect(WS_URI) as ws:
            # Start microphone
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCKSIZE,
                callback=audio_callback,
                latency="low",
            )
            with stream:
                await asyncio.gather(audio_sender(ws), subtitle_receiver(ws))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
