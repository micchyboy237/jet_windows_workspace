# servers\live_subtitles\live_subtitles_server2.py

import asyncio
import json
import difflib
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
from sentence_utils import split_sentences_ja
from sentence_matcher_ja import fuzzy_shortest_best_match
from diff_utils import console_diff_highlight, extract_new_ja_text

import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

N_SEGMENT_RESULTS = 10
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"
LAST_N_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR
LIVE_AUDIO_BUFFER_DIR.mkdir(parents=True, exist_ok=True)


connected_clients = set()

# Adjust max_workers depending on your CPU cores + GPU contention
# SenseVoiceSmall + small LLM usually do well with 2–6 workers
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe_worker")

context_buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)

# ====================== GLOBAL STATE FOR GAP DETECTION ======================
prev_start_sec: float = 0.0          # Tracks the start_sec of the last processed segment
prev_vad_reason = None


def should_reset_context(header: dict) -> bool:
    """Determine if we should reset the context buffer based on time gap or silence."""
    global prev_vad_reason, prev_start_sec

    current_start_sec = float(header.get("start_sec", 0.0))
    vad_reason = header.get("vad_reason")

    # Case 1: Large time gap (> 5.0 seconds)
    # gap = current_start_sec - prev_start_sec
    # if gap > 5.0:
    #     console.print(
    #         f"[warning]Large time gap detected: {gap:.2f}s > 5.0s → Resetting context[/warning]"
    #     )
    #     prev_start_sec = current_start_sec
    #     return True

    # Case 2: Silence from VAD
    if context_buffer.segments and prev_vad_reason == "silence":
        console.print("[info]Silence detected via VAD → Resetting context[/info]")
        prev_start_sec = current_start_sec
        prev_vad_reason = vad_reason
        return True

    # No reset needed - just update previous start time
    prev_start_sec = current_start_sec
    prev_vad_reason = vad_reason
    return False


def blocking_process_audio(
    audio_bytes: bytes,
    header: dict
) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work
    """
    global prev_vad_reason, prev_start_sec

    uuid_ = header.get("uuid")
    if not uuid_:
        console.print("[error]Missing UUID in header[/error]")
        return {"message": "missing uuid", "success": False}
    sample_rate = header.get("sample_rate", 16000)
    full_trans_result = None
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # === NEW: Reset decision BEFORE any transcription ===
    # Resets on either: 1) silence (VAD), or 2) time gap > 5.0 seconds
    if should_reset_context(header):
        context_buffer.reset()
    else:
        prev_vad_reason = header["vad_reason"]

    context_audio_int16 = context_buffer.get_context_audio()
    if context_audio_int16.size > 0:
        full_audio_int16 = np.concatenate([context_audio_int16, audio_np])
    else:
        full_audio_int16 = audio_np
    full_audio_bytes = full_audio_int16.tobytes()
    console.print(
        f"[info]VAD Reason:[/info] [value]{header["vad_reason"]}[/value]"
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
    full_word_segments = full_trans_result.pop("word_segments")
    full_phrase_segments = full_trans_result.pop("phrase_segments")
    full_metadata = full_trans_result.pop("metadata")
    full_word_segments_text = "".join(s["word"] for s in full_word_segments)
    full_ja_text = full_word_segments_text
    full_ja_sents = split_sentences_ja(full_ja_text)
    # full_ja_sents_str = "".join(full_ja_sents).strip()
    # full_ja_text = full_ja_sents_str

    prev_full_ja_text = None
    prev_full_en_text = None

    unchanged_text = None
    new_ja_text = full_ja_text
    new_ja_start_index = None
    new_ja_similarity = None
    history = None

    # === Decision logic (exactly as before; now sees the *post-reset* buffer state) ===
    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_ja_text = last_meta.get("full_ja_text", "")
        prev_full_en_text = last_meta.get("full_en_text", "")

        new_ja_text_res = extract_new_ja_text(prev_full_ja_text, full_ja_text)
        unchanged_text = new_ja_text_res["unchanged_text"]
        new_ja_text = new_ja_text_res["new_text"]
        new_ja_start_index = new_ja_text_res["start_index"]
        new_ja_similarity = new_ja_text_res["similarity"]
    
        # if not new_ja_similarity == 1.0:
        #     console.print("[error]No change in previous speech[/error]")
        #     return {"message": "no change", "success": False}
        

        last_sentence, last_utt_id, last_sent_idx = context_buffer.get_last_sentence()

        MATCH_SCORE_CUTOFF = 75
        match_result = fuzzy_shortest_best_match(
            # query=last_sentence or "",
            query=new_ja_text,
            texts=full_ja_text,
            score_cutoff=MATCH_SCORE_CUTOFF,
            max_extra_chars=30,
        )
        # # === Fuzzy result logs (mirroring sentence_matcher_ja main output) ===
        # console_diff_highlight(
        #     last_sentence or '',
        #     match_result['match'],
        #     "Last JA Sent",
        #     "Match",
        # )
        # console.print(f"[info]Score:[/info] [number]{match_result['score']:.1f}[/number]")
        # console.print(f"[info]Slice:[/info] [bright_white][{match_result['start']}:{match_result['end']}][/bright_white]")
        # console.print(f"[info]Length:[/info] [number]{match_result['end'] - match_result['start']}[/number]")

        # # Highlight the matched slice inside the full text
        # highlighted = (
        #     match_result["text"][: match_result["start"]]
        #     + f"\033[1;33m{match_result['text'][match_result['start'] : match_result['end']]}\033[0m"
        #     + match_result["text"][match_result["end"] :]
        # )
        # print("\nHighlighted in text:")
        # print(highlighted)

        if match_result["score"] >= MATCH_SCORE_CUTOFF and match_result["start"] != -1:
            console.print("[success bold]✅ Accepted[/success bold]")
            new_text_start = match_result["end"]

            # Just the added parts
            # new_text = full_ja_text[new_text_start:].strip()

            # Including full context
            # new_text = full_ja_text.strip()

            # Just the new JA text
            new_text = new_ja_text
        else:
            console.print("[error]❌ Below threshold[/error]")
            console.print(
                f"[warning]Fuzzy match too weak (score={match_result['score']:.1f}).[/warning]"
            )
            console.print(
                f"[warning]Translating the full text.[/warning]"
            )
            new_text = full_ja_text.strip()  # ← ONLY current incremental text

        new_clean = new_text.rstrip('.。！？、…・「」『』').rstrip()
        if not new_clean:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "success": False,
                "message": "Same text as previous",
            }

        old_sents = split_sentences_ja(prev_full_ja_text)  # no longer needed for translation
        old_ja_text = prev_full_ja_text

        # Always translate only the incremental new text
        new_sents = split_sentences_ja(new_text)
        ja_text = "".join(new_sents).strip()
        # new_sents = split_sentences_ja(new_ja_text)
        # ja_text = "".join(new_ja_text).strip()

        last_sentence_pos = match_result["start"] if match_result["score"] >= MATCH_SCORE_CUTOFF else -1
        last_sentence_clean = match_result["match"].strip() if match_result["score"] >= MATCH_SCORE_CUTOFF else None

        if ja_text:
            history = context_buffer.get_context_history()
            trans_en = translate_japanese_to_english(
                ja_text=ja_text,
                enable_scoring=False,
                history=history,
            )
            en_text = trans_en["text"].strip()
        else:
            en_text = ""
        if prev_full_en_text:
            full_en_text = (prev_full_en_text + "\n" + en_text).strip() if en_text else prev_full_en_text
        else:
            full_en_text = en_text
    else:
        ja_sents = full_ja_sents
        ja_text = full_ja_text
        curr_clean = ja_text.rstrip('.。！？、…・「」『』').rstrip()
        if curr_clean:
            history = context_buffer.get_context_history()
            full_trans_en = translate_japanese_to_english(
                ja_text=ja_text,
                enable_scoring=False,
                history=history,
            )
            new_sents = ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
        else:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "success": False,
                "message": "Empty transcription after cleaning",
            }
        old_sents = []
        last_sentence_clean = None
        last_sentence_pos = -1

    # === SHOW DIFF CHANGES ===
    if prev_full_ja_text and full_ja_text != prev_full_ja_text:
        console.print("[info]Diff (previous full JA → current full JA):[/info]")
        console_diff_highlight(
            prev_full_ja_text,
            full_ja_text,
            "Prev",
            "Curr",
        )

    # === Everything below this point is 100% unchanged from original ===
    # (prints, file saving, context_buffer.add_audio_segment, return)

    if history:
        console.print(f"[bold yellow]History ({len(history)}):[/bold yellow]")
        console.print(f"[bold cyan]{history!r}[/bold cyan]")

    if last_sentence_clean:
        console.print(f"[success]Last Sentence (utt_id={last_utt_id[-6:]} | sent_idx={last_sent_idx}):[/success]")
        console.print(f"[bright_white]{last_sentence_clean}[/bright_white]")
    if last_sentence_pos != -1:
        console.print(f"[success]New Text (utt_id={header["uuid"][-6:]} | pos={last_sentence_pos} | start={new_text_start}):[/success]")
        console.print(f"[bright_white]{new_text}[/bright_white]")
    if old_sents:
        console.print(f"[success]Old JA ({len(old_sents)} sents):[/success]")
        console.print(f"[bright_white]{old_ja_text}[/bright_white]")

    if new_ja_text:
        if unchanged_text is not None:
            console.print(f"[success]Unchanged JA ({len(unchanged_text)} chars):[/success]")
            console.print(f"[white]{unchanged_text}[/white]")
        if new_ja_start_index is not None:
            console.print(f"[success]Start index:[/success] [bold cyan]{new_ja_start_index}[/bold cyan]")
        if new_ja_similarity is not None:
            console.print(f"[success]Matched Similarity:[/success] [bold cyan]{new_ja_similarity}[/bold cyan]")
        console.print(f"[success]New JA ({len(new_ja_text)} chars):[/success]")
        console.print(f"[bold cyan]{new_ja_text}[/bold cyan]")

    console.print(f"[success]Full JA ({len(full_ja_sents)} sents):[/success]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")
    if en_text.strip():
        console.print("[success]Full EN:[/success]")
        console.print(f"[bold white]{en_text}[/bold white]")
    else:
        console.print("[dim italic]No new translation[/dim italic]")

    search_audio(
        full_audio_bytes,
        audio_bytes,
    )

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

    # Save new audio
    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)
    audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)

    # Save audio with context
    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)
    wavfile.write(str(segment_dir / "full_sound.wav"), sample_rate, full_audio_int16)

    md_results = (
        f"JA: {ja_text}\n"
        f"EN: {en_text}\n"
    )
    with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md_results)
    metadata_out = {
        "uuid": uuid_,
        "duration_sec": header.get("duration_sec"),
        "started_at": header.get("started_at"),
        "transcribed_at": datetime.now().isoformat(),
    }
    with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, ensure_ascii=False, indent=2)
    subdirs = sorted(
        [d for d in LAST_N_SEGMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("segments_")],
        key=lambda d: d.name,
    )
    if len(subdirs) > N_SEGMENT_RESULTS:
        for old in subdirs[:-N_SEGMENT_RESULTS]:
            shutil.rmtree(old, ignore_errors=True)

    context_duration = context_buffer.get_total_duration()
    context_uuid = context_buffer.get_context_uuid() or uuid_

    context_buffer.add_audio_segment(audio_np, {
        "uuid": header["uuid"],
        "forced": header["forced"],
        "vad_reason": header["vad_reason"],
        "start_sec": header["start_sec"],
        "end_sec": header["end_sec"],
        "duration_sec": header["duration_sec"],
        "started_at": header["started_at"],
        "matched_pos": last_sentence_pos,
        "matched_sent": last_sentence_clean,
        "old_sents": old_sents,
        "new_sents": new_sents,
        "full_ja_text": full_ja_text,
        "full_en_text": full_en_text,
        "ja_text": ja_text,
        "en_text": en_text,
    })
    full_audio_dir = LIVE_AUDIO_BUFFER_DIR
    if full_audio_int16.size > 0:
        wavfile.write(
            str(full_audio_dir / "full_sound.wav"),
            context_buffer.sample_rate,
            full_audio_int16,
        )
    else:
        (full_audio_dir / "full_sound.wav").write_bytes(b"")
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
    full_audio_metadata = context_buffer.get_list_metadata()
    with open(full_audio_dir / "full_audio_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_audio_metadata, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_transcription.json", "w", encoding="utf-8") as f:
        json.dump(full_trans_result, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_word_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": "word",
            "count": len(full_word_segments),
            "text": full_word_segments_text,
            "segments": full_word_segments
        }, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_phrase_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": "phrase",
            "count": len(full_phrase_segments),
            "phrases": [p["phrase"] for p in full_phrase_segments],
            "segments": full_phrase_segments
        }, f, ensure_ascii=False, indent=2)
    with open(full_audio_dir / "full_ja_sents.json", "w", encoding="utf-8") as f:
        json.dump(full_ja_sents, f, ensure_ascii=False, indent=2)
    return {
        "uuid": uuid_,
        "success": bool(ja_text and en_text),
        "context_duration": context_duration,
        "context_uuid": context_uuid,
        "new_ja_similarity": new_ja_similarity,
        "new_ja_start_index": new_ja_start_index,
        "transcription_ja": new_ja_text,
        "translation_en": en_text,
        "transcribed_duration_sec": full_metadata["transcribed_duration_sec"],
        "transcribed_duration_pctg": full_metadata["transcribed_duration_pctg"],
        "coverage_label": full_metadata["coverage_label"],
        "phrase_segments": full_phrase_segments,
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
                console.print(f"[success]Processed successfully[/success] [uuid]{uuid_[-6:]}…[/uuid]")
            else:
                console.print(f"[warning]Empty response message: {response["message"]}[/warning] [uuid]{uuid_[-6:]}…[/uuid]")

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
