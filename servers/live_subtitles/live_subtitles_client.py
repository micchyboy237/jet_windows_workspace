# JetScripts/audio/run_record_mic_speech_detection.py

import argparse
import asyncio
import base64
import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import websockets
from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.record_mic_speech_detection import record_from_mic
from jet.audio.speech.silero.speech_types import SpeechSegment
from jet.audio.speech.wav_utils import save_wav_file
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WS_URI = os.getenv("LOCAL_WS_IP", "ws://127.0.0.1:8765")
SEND_TO_WEBSOCKET = True  # Can be overridden by CLI
SAVE_LOCALLY = True

segment_dirs: dict[
    int, Path
] = {}  # segment_num → segment_dir (for saving subtitle later)

SRT_PATH = OUTPUT_DIR / "full_subtitles.srt"
srt_lock: asyncio.Lock | None = None
total_srt_duration: float = 0.0


def save_segment_data(speech_seg: SpeechSegment, seg_audio_np: np.ndarray):
    segment_root = Path(OUTPUT_DIR) / "segments"
    segment_root.mkdir(parents=True, exist_ok=True)

    # Find next available segment directory name (segment_001, segment_002, ...)
    existing = sorted(segment_root.glob("segment_*"))
    used_numbers = set()
    for seg in existing:
        try:
            used_numbers.add(int(seg.name.split("_")[1]))
        except Exception:
            continue

    # Pick smallest unused positive integer for segment id/dir
    seg_number = 1
    while seg_number in used_numbers:
        seg_number += 1

    seg_dir = segment_root / f"segment_{seg_number:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    wav_path = seg_dir / "sound.wav"
    metadata_path = seg_dir / "metadata.json"

    seg_sound_file = save_wav_file(wav_path, seg_audio_np)
    metadata_path.write_text(json.dumps(speech_seg, indent=2), encoding="utf-8")

    logger.success(f"Segment {seg_number} data saved to:")
    logger.success(seg_sound_file, bright=True)
    logger.success(metadata_path, bright=True)

    # Remember this directory for later subtitle saving
    if SAVE_LOCALLY:
        segment_dirs[seg_number] = seg_dir

    return seg_number, seg_dir  # return for potential use


def audio_float32_to_pcm_base64(audio: np.ndarray) -> str:
    """Convert normalized float32 [-1,1] → int16 PCM → base64"""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    int16 = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
    pcm_bytes = int16.tobytes()
    return base64.b64encode(pcm_bytes).decode("ascii")


async def send_segment(
    websocket, speech_seg: SpeechSegment, audio_np: np.ndarray, seg_idx: int
):
    duration_sec = len(audio_np) / SAMPLE_RATE
    payload = {
        "type": "complete_utterance",
        "pcm": audio_float32_to_pcm_base64(audio_np),
        "sample_rate": int(SAMPLE_RATE),
        "duration_sec": round(duration_sec, 3),
        "segment_num": seg_idx,
        # optional extras
        "avg_vad_confidence": speech_seg.get("confidence", None),
    }
    try:
        await websocket.send(json.dumps(payload, ensure_ascii=False))
        logger.info(f"[WS] Sent segment {seg_idx}  | {duration_sec:.2f}s")
    except Exception as e:
        logger.error(f"[WS] Failed to send segment {seg_idx}: {e}")


async def receive_subtitles(websocket, subtitles_list: list[dict]):
    """Background task: listen for final_subtitle messages from server"""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "final_subtitle":
                    logger.success(
                        f"[WS] Received final_subtitle | utterance {data.get('utterance_id')}"
                    )
                    subtitles_list.append(data)

                    # --- Incremental SRT append (non-blocking) ---
                    global total_srt_duration
                    duration = data.get("duration_sec", 0.0)

                    start_sec = total_srt_duration
                    end_sec = start_sec + duration
                    total_srt_duration += duration

                    def format_srt_time(seconds: float) -> str:
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        millis = int((seconds - int(seconds)) * 1000)
                        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

                    index = len(subtitles_list)
                    text = data.get("translation_en", "").strip()

                    if text and srt_lock:
                        async with srt_lock:
                            block = (
                                f"{index}\n"
                                f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}\n"
                                f"{text}\n\n"
                            )

                            await asyncio.to_thread(
                                lambda: SRT_PATH.open("a", encoding="utf-8").write(
                                    block
                                )
                            )

                    # Optional: pretty-print received subtitle
                    ja = data.get("transcription_ja", "").strip()
                    en = data.get("translation_en", "").strip()
                    logger.info(f"  JA: {ja[:80]}{'...' if len(ja) > 80 else ''}")
                    logger.info(f"  EN: {en[:80]}{'...' if len(en) > 80 else ''}")

                    # Try to save per-segment subtitle file
                    seg_num = data.get("segment_num")
                    if seg_num is not None and seg_num in segment_dirs:
                        seg_dir = segment_dirs[seg_num]
                        subtitle_path = seg_dir / "subtitle.json"
                        with subtitle_path.open("w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        logger.success(f"Saved per-segment subtitle → {subtitle_path}")
                    elif seg_num is not None:
                        logger.warning(
                            f"No matching segment folder found for segment_num={seg_num}"
                        )

                else:
                    logger.debug(
                        f"[WS] Received unknown message type: {data.get('type')}"
                    )
            except json.JSONDecodeError:
                logger.warning("[WS] Invalid JSON received from server")
            except Exception as e:
                logger.error(f"[WS] Error processing incoming message: {e}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("[WS] Server closed connection")
    except Exception as e:
        logger.error(f"[WS] Receiver task error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ws",
        action="store_true",
        default=SEND_TO_WEBSOCKET,
        help="Send segments to live subtitles websocket",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable local file saving"
    )
    args = parser.parse_args()

    # Update module-level flags (no global keyword needed)
    SEND_TO_WEBSOCKET = args.ws
    SAVE_LOCALLY = not args.no_save

    duration_seconds = None
    trim_silent = True
    quit_on_silence = False

    async def main():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        global srt_lock, total_srt_duration
        srt_lock = asyncio.Lock()
        total_srt_duration = 0.0

        SRT_PATH.write_text("", encoding="utf-8")  # reset file at start

        def mic_worker():
            for item in record_from_mic(
                duration_seconds,
                trim_silent=trim_silent,
                quit_on_silence=quit_on_silence,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(item), loop)

        mic_thread = threading.Thread(target=mic_worker, daemon=True)
        mic_thread.start()

        websocket = None
        segment_counter = 0
        received_subtitles: list[dict] = []

        # Clear any old mapping
        segment_dirs.clear()

        if SEND_TO_WEBSOCKET:
            try:
                websocket = await websockets.connect(WS_URI)
                logger.success(f"Connected to live subtitles server → {WS_URI}")
            except Exception as e:
                logger.error(f"Cannot connect to WebSocket: {e}")
                websocket = None

        # Start receiver task if connected
        receiver_task = None
        if websocket:
            receiver_task = asyncio.create_task(
                receive_subtitles(websocket, received_subtitles)
            )

        segments: list[dict] = []

        try:
            while True:
                speech_seg, seg_audio_np, full_audio_np = await queue.get()

                speech_seg_copy = speech_seg.copy()
                for key in ["start", "end"]:
                    if key in speech_seg_copy:
                        speech_seg_copy[key] = round(
                            speech_seg_copy[key] / SAMPLE_RATE, 3
                        )

                segment_counter += 1

                if SAVE_LOCALLY:
                    seg_number, _ = save_segment_data(speech_seg_copy, seg_audio_np)
                segments.append(speech_seg_copy)
                save_file(segments, OUTPUT_DIR / "all_segments.json", verbose=False)

                output_file = f"{OUTPUT_DIR}/full_recording.wav"
                save_wav_file(output_file, full_audio_np)

                if SEND_TO_WEBSOCKET and websocket:
                    # Send sequentially (safe and ordered)
                    await send_segment(
                        websocket, speech_seg_copy, seg_audio_np, segment_counter
                    )

        except KeyboardInterrupt:
            logger.info("Recording stopped by user")

        finally:
            # Give receiver some time to get pending responses
            if receiver_task:
                await asyncio.sleep(2.0)
                receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    pass

            if websocket:
                try:
                    await websocket.close()
                    logger.info("WebSocket connection closed")
                except Exception:
                    pass

            # Save collected subtitles
            if received_subtitles:
                json_path = OUTPUT_DIR / "subtitles_raw.json"

                save_file(received_subtitles, json_path, verbose=True)
                logger.success(f"Collected {len(received_subtitles)} subtitle entries")
            else:
                logger.info("No subtitles received from server")

    asyncio.run(main())
