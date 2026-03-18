# servers\live_subtitles\live_subtitles_server2.py
import asyncio
import json
import numpy as np
import websockets
from concurrent.futures import ThreadPoolExecutor
from transcribe_jp_funasr import transcribe_japanese, TranscriptionResult
from translate_jp_en_llm import translate_japanese_to_english, TranslationResult
# from transcribe_jp_funasr_nano import transcribe_japanese, TranscriptionResult
# from translate_jp_en_sarashin import translate_japanese_to_english, TranslationResult
from audio_context_buffer import AudioContextBuffer
from utils import split_sentences_ja

connected_clients = set()

# Adjust max_workers depending on your CPU cores + GPU contention
# SenseVoiceSmall + small LLM usually do well with 2–6 workers
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe_worker")

context_buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)

def blocking_process_audio(
    audio_bytes: bytes,
    header: dict
) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work
    """
    uuid_ = header.get("uuid")
    if not uuid_:
        return {"error": "missing uuid", "success": False}

    sample_rate = header.get("sample_rate", 16000)
   
    context_audio = context_buffer.get_context_audio()
    if context_audio.size == 0:
        context_audio_bytes = b""
    else:
        context_audio_bytes = context_audio.tobytes()

    try:
        if len(context_audio_bytes) == 0:
            print("[empty context]")
        else:
            full_trans_result: TranscriptionResult = transcribe_japanese(
                audio_bytes=context_audio_bytes,
                sample_rate=sample_rate,
            )
            ja_text_with_context = full_trans_result.get("text_ja", "").strip()
            print(f"RAW JA\n{ja_text_with_context if ja_text_with_context else '[empty transcription]'}")
            ja_sents = split_sentences_ja(ja_text_with_context)
            ja_sents_str = "\n".join(ja_sents).strip()
            print(f"FULL JA (sents={len(ja_sents)})\n{ja_sents_str if ja_sents_str else '[empty transcription]'}")

            # full_trans_en: TranslationResult = translate_japanese_to_english(
            #     ja_text=ja_sents_str,
            #     enable_scoring=False,
            #     history=None,
            # )
            # en_text_with_context = full_trans_en["text"].strip()
            # print(f"FULL EN:\n{en_text_with_context if en_text_with_context else '[empty translation]'}")
        
        trans_result: TranscriptionResult = transcribe_japanese(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )
        ja_text = trans_result.get("text_ja", "").strip()
        print(f"JA: {ja_text if ja_text else '[empty transcription]'}")

        if ja_text:
            trans_en: TranslationResult = translate_japanese_to_english(
                ja_text=ja_text,
                # enable_scoring=False,
                # history=None,
            )
            en_text = trans_en["text"].strip()
        else:
            en_text = ""

        print(f"EN: {en_text if en_text else '[empty translation]'}")

        return {
            "uuid": uuid_,
            "transcription_ja": ja_text,
            "translation_en": en_text,
            "success": bool(ja_text or en_text),
        }

    except Exception as e:
        print(f"[WORKER] Processing error for {uuid_[:8]}… : {type(e).__name__}: {e}")
        return {
            "uuid": uuid_,
            "error": str(e),
            "success": False,
        }


async def process_audio(websocket):
    connected_clients.add(websocket)
    print(f"[SERVER] Client connected — total {len(connected_clients)}")

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                await websocket.send(json.dumps({"error": "binary message required"}).encode())
                continue

            parts = message.split(b"\x00", 1)
            if len(parts) != 2:
                continue

            try:
                header = json.loads(parts[0].decode("utf-8"))
                audio_bytes = parts[1]
            except json.JSONDecodeError:
                print("[SERVER] Invalid JSON header")
                await websocket.send(json.dumps({"error": "invalid json header"}).encode())
                continue

            # Convert raw bytes → numpy float32 array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # Now safe to add to context buffer
            context_buffer.add_audio_segment(header["start_sec"], audio_np)

            # Offload heavy work to thread pool
            future = asyncio.get_running_loop().run_in_executor(
                executor,
                blocking_process_audio,
                audio_bytes,
                header
            )

            # While transcription/translation runs in background,
            # websocket can continue receiving new messages
            try:
                response = await future

                await websocket.send(json.dumps(response).encode("utf-8"))
                uuid_ = header.get("uuid", "???")
                ja_preview = response.get("transcription_ja", "")[:50]
                print(f"[SERVER] Processed {uuid_[:8]}… → {ja_preview}…")
            except Exception as e:
                print(f"[SERVER] Error sending response: {e}")
                await websocket.send(
                    json.dumps({"error": str(e), "success": False}).encode()
                )

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[SERVER] Unexpected websocket error: {e}")
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
        max_size=2**23,
    ) as server:
        print("[SERVER] Listening on ws://0.0.0.0:8765")
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down…")
    finally:
        executor.shutdown(wait=True)
        print("[SERVER] ThreadPoolExecutor shut down")
