# servers\live_subtitles\live_subtitles_server2.py

import asyncio
import json
import numpy as np
from datetime import datetime
import scipy.io.wavfile as wavfile
import websockets
from concurrent.futures import ThreadPoolExecutor
from transcribe_jp_funasr import transcribe_japanese, TranscriptionResult
from translate_jp_en_llm import translate_japanese_to_english, TranslationResult
# from transcribe_jp_funasr_nano import transcribe_japanese, TranscriptionResult
# from translate_jp_en_sarashin import translate_japanese_to_english, TranslationResult
from audio_context_buffer import AudioContextBuffer
from audio_search import search_audio
from utils import split_sentences_ja

import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

N_SEGMENT_RESULTS = 10
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"
LAST_N_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

LIVE_AUDIO_CONTEXT_DIR = OUTPUT_DIR / "audio_context"
LIVE_AUDIO_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)


def _save_subtitles_srt(segments: list, srt_path: Path) -> None:
    """Generate simple SRT from word segments (one line per segment)."""
    if not segments:
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n[No transcription]\n\n",
            encoding="utf-8",
        )
        return

    def _ms_to_srt_time(ms):
        ms = ms or 0
        hours = ms // 3_600_000
        minutes = (ms % 3_600_000) // 60_000
        seconds = (ms % 60_000) // 1_000
        millis = ms % 1_000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_str = _ms_to_srt_time(seg.get("start_ms"))
            end_str = _ms_to_srt_time(seg.get("end_ms"))
            word = seg.get("word") or "[no speech]"
            f.write(f"{i}\n{start_str} --> {end_str}\n{word}\n\n")


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
        # early return before any saving
        return {"error": "missing uuid", "success": False}

    sample_rate = header.get("sample_rate", 16000)
    full_trans_result = None
    context_audio = context_buffer.get_context_audio()
    if context_audio.size == 0:
        context_audio_bytes = b""
    else:
        context_audio_bytes = context_audio.tobytes()

    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        search_audio(
            context_audio,
            audio_np,
        )
        print(f"Context duration: {context_buffer.get_total_duration():.2f}s")
        print(f"Audio duration: {header['duration_sec']:.2f}s")
        full_trans_result = transcribe_japanese(
            audio_bytes=context_audio_bytes,
            sample_rate=sample_rate,
        )
        full_word_segments = full_trans_result["segments"]
        full_metadata = full_trans_result["metadata"]
        print(f"FIRST 3 JA SEGMENTS\n{full_word_segments[:3]!r}")
        print(f"LAST 3 JA SEGMENTS\n{full_word_segments[-3:]!r}")
        full_word_segments_text = "".join([s["word"] for s in full_word_segments])
        ja_text_with_context = full_trans_result["text_ja"].strip()
        print(f"TOTAL JA SEGMENTS: {len(full_word_segments)}")
        print(f"FULL JA WORDS\n{full_word_segments_text}")
        full_ja_sents = split_sentences_ja(ja_text_with_context)
        full_ja_sents_str = "\n".join(full_ja_sents).strip()
        print(f"FULL JA (sents={len(full_ja_sents)})\n{full_ja_sents_str}")
        print(f"FULL METADATA\n{full_metadata!r}")

        # full_trans_en = translate_japanese_to_english(
        #     ja_text=full_ja_sents_str,
        #     enable_scoring=False,
        #     history=None,
        # )
        # en_text_with_context = full_trans_en["text"].strip()
        # print(f"FULL EN:\n{en_text_with_context if en_text_with_context else '[empty translation]'}")
        
        trans_result = transcribe_japanese(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )
        metadata = trans_result["metadata"]
        ja_text = trans_result.get("text_ja", "").strip()
        ja_sents = split_sentences_ja(ja_text)
        ja_sents_str = "\n".join(ja_sents).strip()
        if ja_sents_str:
            print(f"JA (sents={len(ja_sents)})\n{ja_sents_str}")
            print(f"METADATA\n{metadata!r}")
        else:
            print("JA: [empty transcription]")

        if ja_sents_str:
            trans_en = translate_japanese_to_english(
                ja_text=ja_sents_str,
                # enable_scoring=False,
                # history=None,
            )
            en_text = trans_en["text"].strip()
        else:
            en_text = ""

        print(f"EN: {en_text if en_text else '[empty translation]'}")

        # === SAVE LAST-N SEGMENT RESULTS (per incoming chunk) ===
        try:
            started_at_iso = header.get("started_at")
            if started_at_iso and isinstance(started_at_iso, str):
                iso_str = started_at_iso.replace("Z", "+00:00") if started_at_iso.endswith("Z") else started_at_iso
                try:
                    dt = datetime.fromisoformat(iso_str)
                    ts_str = dt.strftime("%Y%m%d_%H%M%S")
                except Exception:
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            segment_dir = LAST_N_SEGMENTS_DIR / f"segments_{ts_str}"
            segment_dir.mkdir(parents=True, exist_ok=True)

            # header.json
            with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
                json.dump(header, f, ensure_ascii=False, indent=2)

            # sound.wav (raw incoming chunk)
            audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)

            # transcription.json (full TranscriptionResult for this segment)
            with open(segment_dir / "transcription.json", "w", encoding="utf-8") as f:
                json.dump(trans_result, f, ensure_ascii=False, indent=2)

            # translation.json
            translation_data = {"text": en_text}
            with open(segment_dir / "translation.json", "w", encoding="utf-8") as f:
                json.dump(translation_data, f, ensure_ascii=False, indent=2)

            # metadata.json (combined helpful info)
            metadata_out = {
                "transcription": trans_result.get("metadata", {}),
                "uuid": uuid_,
                "duration_sec": header.get("duration_sec"),
                "started_at": header.get("started_at"),
                "transcribed_at": datetime.now().isoformat(),
            }
            with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata_out, f, ensure_ascii=False, indent=2)

            # subtitles.srt (timed Japanese subtitles from segments)
            _save_subtitles_srt(trans_result.get("segments", []), segment_dir / "subtitles.srt")

            print(f"[SAVE SEGMENT] Saved to {segment_dir}")

            # keep only last N segment dirs (by name sort = chronological)
            subdirs = sorted(
                [d for d in LAST_N_SEGMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("segments_")],
                key=lambda d: d.name,
            )
            if len(subdirs) > N_SEGMENT_RESULTS:
                for old in subdirs[:-N_SEGMENT_RESULTS]:
                    shutil.rmtree(old, ignore_errors=True)
                    print(f"[CLEAN] Removed old segment: {old.name}")
        except Exception as save_err:
            print(f"[SAVE SEGMENT] Non-fatal error: {save_err}")

        # === SAVE UPDATING LIVE AUDIO CONTEXT BUFFER DATA ===
        try:
            context_dir = LIVE_AUDIO_CONTEXT_DIR
            # full sound.wav (current buffer contents, int16 PCM)
            if context_audio.size > 0:
                wavfile.write(
                    str(context_dir / "full_sound.wav"),
                    context_buffer.sample_rate,
                    context_audio,
                )
            else:
                (context_dir / "full_sound.wav").write_bytes(b"")

            # metadata.json (buffer stats + helpful insights)
            context_metadata = {
                "total_duration_sec": round(context_buffer.get_total_duration(), 3),
                "num_chunks": len(context_buffer.segments),
                "max_duration_sec": context_buffer.max_duration_sec,
                "sample_rate": context_buffer.sample_rate,
                "last_updated": datetime.now().isoformat(),
                "context_includes_current_segment": True,
            }
            with open(context_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(context_metadata, f, ensure_ascii=False, indent=2)

            # full transcription.json (transcription of the entire context buffer)
            if full_trans_result is not None:
                with open(context_dir / "full_transcription.json", "w", encoding="utf-8") as f:
                    json.dump(full_trans_result, f, ensure_ascii=False, indent=2)
            else:
                (context_dir / "full_transcription.json").write_text(
                    '{"text_ja": "", "segments": [], "metadata": {}}',
                    encoding="utf-8",
                )

            print(f"[SAVE CONTEXT] Updated live buffer data in {context_dir}")
        except Exception as ctx_err:
            print(f"[SAVE CONTEXT] Non-fatal error: {ctx_err}")

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
                    # error response (no saving happened)
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
