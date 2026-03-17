# live_subtitles_server2.py
# Run with: python live_subtitles_server2.py

import asyncio
import json

import websockets


# Placeholder — replace with real ASR (FasterWhisper / WhisperX / etc.)
def fake_transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    sec = len(audio_bytes) / (sample_rate * 4)  # float32 = 4 bytes
    return f"これはテスト音声です ({sec:.1f}秒) [fake]"


# Placeholder translation
def fake_translate_ja_to_en(text: str) -> str:
    return f"[EN] This is test audio ({text.split('(')[1] if '(' in text else '??'})"


connected_clients = set()


async def process_audio(websocket):
    connected_clients.add(websocket)
    print(f"[SERVER] Client connected — total {len(connected_clients)}")

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                await websocket.send(json.dumps({"error": "binary message required"}))
                continue

            parts = message.split(b"\x00", 1)
            if len(parts) != 2:
                continue

            try:
                header = json.loads(parts[0].decode("utf-8"))
                audio = parts[1]

                uuid_ = header.get("uuid")
                if not uuid_:
                    continue

                ja = fake_transcribe(audio, header.get("sample_rate", 16000))
                en = fake_translate_ja_to_en(ja)

                response_header = {
                    "uuid": uuid_,
                    "transcription_ja": ja,
                    "translation_en": en,
                    "success": True,
                }
                # Echo header only (no need to send audio back)
                await websocket.send(json.dumps(response_header).encode("utf-8"))

                print(f"[SERVER] Processed {uuid_[:8]}… → {ja[:40]}…")

            except json.JSONDecodeError:
                print("[SERVER] Invalid JSON header")
            except Exception as e:
                print(f"[SERVER] Processing error: {e}")

    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[SERVER] Client disconnected — total {len(connected_clients)}")


async def main():
    async with websockets.serve(
        process_audio,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
    ) as server:
        print("[SERVER] Listening on ws://0.0.0.0:8765")
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
