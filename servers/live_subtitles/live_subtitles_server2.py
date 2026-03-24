# servers\live_subtitles\live_subtitles_server2.py

import asyncio
import json
from rich.console import Console
from rich.theme import Theme

console = Console(theme=Theme({
    "info": "cyan",
    "success": "green bold",
    "warning": "yellow",
    "error": "red bold",

    # value styles
    "value": "white bold",
    "time": "magenta bold",     # great for durations
    "number": "bright_white",
    "uuid": "bright_blue",
}))

import numpy as np
from datetime import datetime
import scipy.io.wavfile as wavfile
import websockets
from concurrent.futures import ThreadPoolExecutor
from transcribe_jp_funasr import transcribe_japanese, TranscriptionResult
from translate_jp_en_llm import translate_japanese_to_english
# from transcribe_jp_funasr_nano import transcribe_japanese, TranscriptionResult
# from translate_jp_en_sarashin import translate_japanese_to_english
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

LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR
LIVE_AUDIO_BUFFER_DIR.mkdir(parents=True, exist_ok=True)


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

context_buffer = AudioContextBuffer(max_duration_sec=60.0, sample_rate=16000)

def blocking_process_audio(  # ← unchanged signature
    audio_bytes: bytes,
    header: dict
) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work
    """
    uuid_ = header.get("uuid")
    if not uuid_:
        console.print("[error]Missing UUID in header[/error]")
        return {"error": "missing uuid", "success": False}

    sample_rate = header.get("sample_rate", 16000)
    full_trans_result = None

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    # Both are now guaranteed int16
    context_audio_int16 = context_buffer.get_context_audio()

    if context_audio_int16.size > 0:
        full_audio_int16 = np.concatenate([context_audio_int16, audio_np])
    else:
        full_audio_int16 = audio_np

    full_audio_bytes = full_audio_int16.tobytes()

    search_audio(
        full_audio_bytes,
        audio_bytes,
    )

    console.print(
        f"[info]Context duration:[/info] [time]{context_buffer.get_total_duration():.2f}s[/time]"
    )
    console.print(
        f"[info]Audio duration:[/info] [time]{header['duration_sec']:.2f}s[/time]"
    )

    full_trans_result = transcribe_japanese(
        audio_bytes=full_audio_bytes,
        sample_rate=sample_rate,
    )
    full_trans_result = full_trans_result.copy()
    full_word_segments = full_trans_result.pop("segments")
    full_metadata = full_trans_result.pop("metadata")

    full_word_segments_text = "".join(s["word"] for s in full_word_segments)
    # full_ja_text = full_trans_result["text_ja"].strip()
    full_ja_text = full_word_segments_text

    full_ja_sents = split_sentences_ja(full_ja_text)
    full_ja_sents_str = "".join(full_ja_sents).strip()
    full_ja_text = full_ja_sents_str


    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_en_text = last_meta.get("full_en_text", "")

        last_sentence, last_utt_id, last_sent_idx = context_buffer.get_last_sentence()
        prev_ja_text = last_meta["full_ja_text"]
        prev_ja_sents = split_sentences_ja(prev_ja_text)
        # Find the index where new sentences begin (i.e. not in prev_ja_sents)
        start_index = 0
        last_sentence_pos = -1
        last_sentence_clean = last_sentence.rstrip('。！？、…・「」『』').rstrip()

        for i, curr in enumerate(full_ja_sents):
            if i >= len(prev_ja_sents):
                break

            prev = prev_ja_sents[i]

            # Compare without final punctuation
            curr_clean = curr.rstrip('。！？、…・「」『』').rstrip()
            prev_clean = prev.rstrip('。！？、…・「」『』').rstrip()

            if curr_clean == prev_clean or (last_sentence_clean and last_sentence_clean in curr_clean):
                start_index += 1
            else:
                break

        if last_sentence_clean in full_ja_text:
            last_sentence_pos = full_ja_text.find(last_sentence_clean)

        if last_sentence_pos != -1:
            # Show the last known sentence + what comes after it (new continuation)
            new_text_start = last_sentence_pos + len(last_sentence_clean)
            old_text = full_ja_text[:new_text_start].strip()
            new_text = full_ja_text[new_text_start:].strip()

            new_clean = new_text.rstrip('。！？、…・「」『』').rstrip()

            if not new_clean:
                return {
                    "uuid": uuid_,
                    "transcription_ja": "",
                    "translation_en": "",
                    "success": False,
                }

            old_sents = split_sentences_ja(old_text)
            new_sents = split_sentences_ja(new_text)

            # Length statistics
            console.print(
                f"[info]Diff Sentence Lengths (Last Sentence):[/info]   "
                f"old sentences = [cyan]{len(old_sents):2d}[/cyan]  "
                f"new sentences = [cyan]{len(new_sents):2d}[/cyan]  "
                f"Δ = [bright_blue]{len(new_sents) - len(old_sents):+2d}[/bright_blue]"
            )

            ja_sents = new_sents
            ja_sents_str = "".join(ja_sents).strip()
            ja_text = ja_sents_str
            if ja_text:
                trans_en = translate_japanese_to_english(
                    ja_text=ja_text,
                    enable_scoring=False,
                    history=None,
                )
                en_text = trans_en["text"].strip()
            else:
                en_text = ""

            # ✅ Reconstruct full_en_text to keep context consistent
            if prev_full_en_text:
                full_en_text = (prev_full_en_text + "\n" + en_text).strip() if en_text else prev_full_en_text
            else:
                full_en_text = en_text

        else:
            old_sents = prev_ja_sents[:start_index]
            new_sents = full_ja_sents[start_index:]
            # Length statistics
            console.print(
                f"[info]Diff Sentence Lengths:[/info]   "
                f"start index = [cyan]{start_index}[/cyan]  "
                f"old sentences = [cyan]{len(old_sents):2d}[/cyan]  "
                f"new sentences = [cyan]{len(new_sents):2d}[/cyan]  "
                f"Δ = [bright_blue]{len(new_sents) - len(old_sents):+2d}[/bright_blue]"
            )

            ja_sents = new_sents
            ja_sents_str = "".join(ja_sents).strip()
            ja_text = ja_sents_str
            if ja_text:
                trans_en = translate_japanese_to_english(
                    ja_text=ja_text,
                    enable_scoring=False,
                    history=None,
                )
                en_text = trans_en["text"].strip()
            else:
                en_text = ""

            # ✅ Reconstruct full_en_text to keep context consistent
            if prev_full_en_text:
                full_en_text = (prev_full_en_text + "\n" + en_text).strip() if en_text else prev_full_en_text
            else:
                full_en_text = en_text

    else:
        ja_sents = full_ja_sents
        ja_sents_str = full_ja_sents_str
        ja_text = full_ja_text

        curr_clean = ja_text.rstrip('。！？、…・「」『』').rstrip()
        if curr_clean:
            full_trans_en = translate_japanese_to_english(
                ja_text=ja_text,
                enable_scoring=False,
                history=None,
            )

            new_sents = ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
        else:
            # ja_text = ""
            # new_sents = []
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "success": False,
            }

        old_sents = []
        last_sentence = None
        last_sentence_pos = -1

    # ── Rich styled output ────────────────────────────────────────
    if last_sentence:
        console.print(f"[success]Last Sentence (utt_id={last_utt_id[-6:]} | sent_idx={last_sent_idx}):[/success]")
        console.print(f"[bright_white]{last_sentence}[/bright_white]")
    if last_sentence_pos != -1:
        console.print(f"[success]New Text (pos={last_sentence_pos} | start={new_text_start}):[/success]")
        console.print(f"[bright_white]{new_text}[/bright_white]")

    if old_sents:
        old_ja_text = "".join(old_sents).strip()
        console.print(f"[success]Old JA ({len(old_sents)} sent):[/success]")
        console.print(f"[bright_white]{old_ja_text}[/bright_white]")
    if ja_text.strip():
        console.print(f"[success]New JA ({len(new_sents)} sent):[/success]")
        console.print(f"[bright_white]{ja_text}[/bright_white]")
    else:
        console.print("[dim]No new Japanese text[/dim]")

    if en_text.strip():
        console.print("[success]EN:[/success]")
        console.print(f"[white]{en_text}[/white]")
    else:
        console.print("[dim italic]No new translation[/dim italic]")

    # === SAVE LAST-N SEGMENT RESULTS (per incoming chunk) ===
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

    # Save transcription and translation results as markdown
    md_results = (
        f"JA: {ja_text}\n"
        f"EN: {en_text}\n"
    )
    with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md_results)

    # metadata.json (combined helpful info)
    metadata_out = {
        # "transcription": trans_result.get("metadata", {}),
        "uuid": uuid_,
        "duration_sec": header.get("duration_sec"),
        "started_at": header.get("started_at"),
        "transcribed_at": datetime.now().isoformat(),
    }
    with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, ensure_ascii=False, indent=2)

    # subtitles.srt (timed Japanese subtitles from segments)
    # _save_subtitles_srt(trans_result.get("segments", []), segment_dir / "subtitles.srt")

    # keep only last N segment dirs (by name sort = chronological)
    subdirs = sorted(
        [d for d in LAST_N_SEGMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("segments_")],
        key=lambda d: d.name,
    )
    if len(subdirs) > N_SEGMENT_RESULTS:
        for old in subdirs[:-N_SEGMENT_RESULTS]:
            shutil.rmtree(old, ignore_errors=True)
            # print(f"[CLEAN] Removed old segment: {old.name}")

    # === SAVE UPDATING LIVE AUDIO BUFFER DATA ===

    # Now safe to add to context buffer
    context_buffer.add_audio_segment(audio_np, {
        "uuid": header["uuid"],
        "forced": header["forced"],
        "vad_reason": header["vad_reason"],
        "start_sec": header["start_sec"],
        "end_sec": header["end_sec"],
        "duration_sec": header["duration_sec"],
        "started_at": header["started_at"],
        "old_sents": old_sents,
        "new_sents": new_sents,
        "full_ja_text": full_ja_text,
        "full_en_text": full_en_text,
        "ja_text": ja_text,
        "en_text": en_text,
    })

    full_audio_dir = LIVE_AUDIO_BUFFER_DIR
    # full sound.wav (current buffer contents, int16 PCM)
    if full_audio_int16.size > 0:
        wavfile.write(
            str(full_audio_dir / "full_sound.wav"),
            context_buffer.sample_rate,
            full_audio_int16,
        )
    else:
        (full_audio_dir / "full_sound.wav").write_bytes(b"")

    # summary.json (buffer stats + helpful insights)
    context_summary = {
        "total_duration_sec": round(context_buffer.get_total_duration(), 3),
        "num_chunks": len(context_buffer.segments),
        "max_duration_sec": context_buffer.max_duration_sec,
        "sample_rate": context_buffer.sample_rate,
        "last_updated": datetime.now().isoformat(),
        "context_includes_current_segment": True,
    }
    with open(full_audio_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(context_summary, f, ensure_ascii=False, indent=2)

    # Save full audio buffer metadata
    full_audio_metadata = context_buffer.get_list_metadata()
    with open(full_audio_dir / "full_audio_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_audio_metadata, f, ensure_ascii=False, indent=2)

    # full transcription.json (transcription of the entire context buffer)
    with open(full_audio_dir / "full_transcription.json", "w", encoding="utf-8") as f:
        json.dump(full_trans_result, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_word_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "text": full_word_segments_text,
            "words": full_word_segments
        }, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_ja_sents.json", "w", encoding="utf-8") as f:
        json.dump(full_ja_sents, f, ensure_ascii=False, indent=2)

    # Reset if vad reason is min silence
    if not header["forced"]:
        context_buffer.reset()

    return {
        "uuid": uuid_,
        "transcription_ja": ja_text,
        "translation_en": en_text,
        "success": bool(ja_text or en_text),
    }


async def process_audio(websocket):
    connected_clients.add(websocket)
    console.print(
        f"[success]Client connected[/success] — total [bright_blue]{len(connected_clients)}[/bright_blue]"
    )

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                await websocket.send(json.dumps({"error": "binary message required"}).encode())
                console.print("[warning]Received non-binary message[/warning]")
                continue

            parts = message.split(b"\x00", 1)
            if len(parts) != 2:
                console.print("[warning]Invalid message format (missing delimiter)[/warning]")
                continue

            try:
                header = json.loads(parts[0].decode("utf-8"))
                audio_bytes = parts[1]
            except json.JSONDecodeError:
                console.print("[error]Invalid JSON header[/error]")
                await websocket.send(json.dumps({"error": "invalid json header"}).encode())
                continue

            # Offload heavy work to thread pool
            future = asyncio.get_running_loop().run_in_executor(
                executor,
                blocking_process_audio,
                audio_bytes,
                header
            )

            uuid_ = header.get("uuid", "???")

            console.rule(style="dim")

            console.print(f"[info]Processing[/info] [uuid]{uuid_[-6:]}…[/uuid]")

            # While transcription/translation runs in background,
            # websocket can continue receiving new messages
            response = await future

            if response["success"]:
                await websocket.send(json.dumps(response).encode("utf-8"))
                console.print(f"[success]Processed[/success] [uuid]{uuid_[-6:]}…[/uuid]")
            else:
                console.print(f"[warning]Processed empty audio[/warning] [uuid]{uuid_[-6:]}…[/uuid]")

            console.rule(style="dim")

    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        console.print(
            f"[warning]Client disconnected[/warning] — total [bright_blue]{len(connected_clients)}[/bright_blue]"
        )


async def main():
    async with websockets.serve(
        process_audio,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=2**23,
    ) as server:
        console.print("[success bold]Server listening on[/success bold] [bright_cyan]ws://0.0.0.0:8765[/bright_cyan]")
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[warning]Shutting down…[/warning]")
        pass
    finally:
        executor.shutdown(wait=True)
        console.print("[dim]ThreadPoolExecutor shut down[/dim]")
