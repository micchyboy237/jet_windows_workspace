import asyncio
import json
import logging
import shutil
import time
import torch
import uuid as uuid_module
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import scipy.io.wavfile as wavfile
import uvicorn
from audio_context_buffer import AudioContextBuffer
from diff_utils import console_diff_highlight, extract_new_ja_text
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from sentence_matcher_ja import fuzzy_shortest_best_match_contains, fuzzy_match_prefix_texts
from sentence_utils import split_sentences_ja
from transcribe_jp_funasr import TranscriptionResult, transcribe_japanese
from translate_jp_en_llm_prefixed import translate_japanese_to_english
from speaker_labeler import SpeakerLabeler
from pyannote.core import Segment
from pyannote.audio import Inference, Model

console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "success": "green bold",
            "warning": "yellow",
            "error": "red bold",
            "value": "white bold",
            "time": "magenta bold",
            "number": "bright_white",
            "uuid": "bright_blue",
            "speaker": "bright_green",
        }
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("live_subtitles_server2_speakers")
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = True


OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
N_SEGMENT_RESULTS = 10
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"
LAST_N_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR
LIVE_AUDIO_BUFFER_DIR.mkdir(parents=True, exist_ok=True)

# Speaker state persistence
SPEAKER_STATE_PATH = OUTPUT_DIR / "speaker_state.json"

app = FastAPI(title="Live Japanese Subtitles Server 2")
active_connections: dict[str, WebSocket] = {}

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe_worker")
context_buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)
prev_end_sec: float | None = None
prev_vad_reason = None

# ─── Speaker Labeler Setup ───────────────────────────────────────────────────
# Lazy initialization to avoid loading model at import time
_speaker_labeler: Optional[SpeakerLabeler] = None
_current_speakers: Dict[str, Dict] = {}  # Track multiple active speakers
_last_speaker_change_time: float = 0.0


def _get_speaker_labeler() -> SpeakerLabeler:
    """Get or initialize the speaker labeler singleton.
    
    Lazy initialization defers model loading until first use.
    Uses the new SpeakerLabeler with segmentation model.
    """
    global _speaker_labeler
    
    if _speaker_labeler is not None:
        return _speaker_labeler
    
    console.print("[info]Loading speaker segmentation model...[/info]")
    console.print("[info]This may take a minute on first run (downloading pyannote/segmentation-3.0)[/info]")
    
    try:
        # Initialize the new SpeakerLabeler with segmentation model
        _speaker_labeler = SpeakerLabeler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            min_duration_on=0.1,      # Minimum speech segment duration
            min_duration_off=0.3,     # Minimum non-speech gap
            max_speakers_per_chunk=3,  # Max speakers per 10s chunk
            max_speakers_per_frame=2,  # Max overlapping speakers
            chunk_duration=10.0,       # Process in 10-second chunks
            overlap_threshold=0.5,     # Overlap detection threshold
        )
        console.print("[success]SpeakerLabeler initialized with pyannote/segmentation-3.0[/success]")
        
        # Try to restore previous state if available
        if SPEAKER_STATE_PATH.exists():
            try:
                with open(SPEAKER_STATE_PATH, 'r') as f:
                    state = json.load(f)
                # Restore speaker references and segments
                if "speaker_references" in state:
                    # Convert dict back to Segment objects
                    restored_refs = {}
                    for k, segs in state["speaker_references"].items():
                        restored_refs[k] = [Segment(s["start"], s["end"]) for s in segs]
                    _speaker_labeler.speaker_references = restored_refs
                if "all_segments" in state:
                    _speaker_labeler.all_segments = state["all_segments"]
                console.print(
                    f"[success]Restored speaker state: "
                    f"{len(_speaker_labeler.speaker_references)} speaker(s), "
                    f"{len(_speaker_labeler.all_segments)} segments processed[/success]"
                )
            except Exception as e:
                console.print(f"[warning]Could not restore speaker state: {e}[/warning]")
                
    except Exception as e:
        console.print(f"[error]Failed to initialize speaker labeler: {e}[/error]")
        raise
    
    return _speaker_labeler


def save_speaker_state():
    """Persist the current speaker labeler state to disk."""
    if _speaker_labeler is None:
        return
    try:
        state = {
            "speaker_references": {
                k: [{"start": s.start, "end": s.end} for s in segs]
                for k, segs in _speaker_labeler.speaker_references.items()
            },
            "all_segments": _speaker_labeler.all_segments,
        }
        with open(SPEAKER_STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        console.print(f"[warning]Could not save speaker state: {e}[/warning]")


def label_speakers_for_segment(
    waveform: np.ndarray,
    sample_rate: int,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> List[Dict]:
    """Label speakers in an audio segment using the segmentation model.
    
    Unlike the old version which returns a single speaker, this returns
    a list of speakers detected in the segment, including overlapping speech.
    
    Parameters
    ----------
    waveform : np.ndarray
        Audio waveform as int16 numpy array.
    sample_rate : int
        Sample rate of the audio.
    start_time : float
        Start time offset for this segment.
    end_time : float, optional
        End time offset. If None, calculated from waveform duration.
    
    Returns
    -------
    List[Dict]
        List of speaker segments with keys:
        - speaker: Speaker label
        - start: Start time in seconds
        - end: End time in seconds
        - duration: Segment duration
        - is_overlapped: Whether this is overlapping speech
        - confidence: Confidence score (0-1)
    """
    global _current_speakers, _last_speaker_change_time
    
    if waveform.size == 0:
        return []
    
    if end_time is None:
        end_time = start_time + len(waveform) / sample_rate
    
    labeler = _get_speaker_labeler()
    
    # Convert to float32 for model (normalize int16 to [-1, 1])
    waveform_float = waveform.astype(np.float32) / 32768.0
    
    # Process the segment directly using process_chunk
    results = labeler.process_chunk(
        waveform=waveform_float,
        sample_rate=sample_rate,
        chunk_start_time=start_time,
    )
    
    # Update current speakers tracking
    current_time = time.time()
    speaker_segments = results.get("speaker_segments", [])
    
    # Update global current speakers based on latest segments
    if speaker_segments:
        latest_speakers = set()
        for seg in speaker_segments:
            latest_speakers.add(seg["speaker"])
        
        # Compare with previous speakers
        old_speakers = set(_current_speakers.keys())
        if latest_speakers != old_speakers:
            console.print(
                f"[speaker]🔊 Speaker change: {old_speakers} → {latest_speakers}[/speaker]"
            )
            _last_speaker_change_time = current_time
        
        # Update current speakers info
        _current_speakers = {
            seg["speaker"]: {
                "last_seen": current_time,
                "is_overlapped": seg["is_overlapped"],
            }
            for seg in speaker_segments
        }
    
    # Save state periodically
    if labeler.all_segments and len(labeler.all_segments) % 10 == 0:
        save_speaker_state()
    
    return speaker_segments


def get_speaker_diarization() -> Dict:
    """Get current speaker diarization summary including multiple speakers."""
    labeler = _speaker_labeler
    if labeler is None:
        return {
            "current_speakers": [],
            "known_speakers": [],
            "speaker_count": 0,
            "speakers_info": {},
            "total_speaker_segments": 0,
            "speaker_timeline": [],
        }
    
    return {
        "current_speakers": [
            {
                "speaker": speaker,
                "is_overlapped": info.get("is_overlapped", False),
                "last_seen": info.get("last_seen"),
            }
            for speaker, info in _current_speakers.items()
        ],
        "known_speakers": labeler.get_unique_speakers(),
        "speaker_count": len(labeler.speaker_references),
        "speakers_info": {
            speaker: {
                "segment_count": len(segments),
                "total_duration_seconds": sum(s.duration for s in segments),
            }
            for speaker, segments in labeler.speaker_references.items()
        },
        "total_speaker_segments": len(labeler.all_segments),
        "speaker_timeline": labeler.get_speaker_timeline()[-20:],  # Last 20 segments
    }


def should_reset_context(header: dict) -> bool:
    """Determine if we should reset the context buffer based on time gap or silence."""
    return True


def blocking_process_audio(
    audio_bytes: bytes,
    header: dict
) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work.
    Updated to handle multiple speakers.
    """
    global prev_vad_reason, prev_end_sec
    
    uuid_ = header.get("uuid")
    if not uuid_:
        console.print("[error]Missing UUID in header[/error]")
        return {"message": "missing uuid", "success": False}
    
    sample_rate = header.get("sample_rate", 16000)
    full_trans_result = None
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    
    if should_reset_context(header):
        context_buffer.reset()
    else:
        prev_vad_reason = header["vad_reason"]
    
    new_audio_duration_sec = len(audio_np) / sample_rate
    context_duration_sec = context_buffer.get_total_duration()
    max_duration_sec = context_buffer.max_duration_sec
    combined_naive_sec = context_duration_sec + new_audio_duration_sec
    
    context_audio_int16, actual_context_sec, segments_used = (
        context_buffer.get_context_audio_within_limit(new_audio_duration_sec)
    )
    
    if combined_naive_sec > max_duration_sec:
        dropped_segments = len(context_buffer.segments) - segments_used
        console.print(
            f"[warning]⚠️  Combined audio ({combined_naive_sec:.2f}s) would exceed "
            f"max_duration_sec ({max_duration_sec:.2f}s). "
            f"Dropped {dropped_segments} oldest segment(s). "
            f"Using {segments_used} segment(s) = {actual_context_sec:.2f}s context.[/warning]"
        )
    
    if context_audio_int16.size > 0:
        full_audio_int16 = np.concatenate([context_audio_int16, audio_np])
    else:
        full_audio_int16 = audio_np
    
    actual_full_duration_sec = len(full_audio_int16) / sample_rate
    if actual_full_duration_sec > max_duration_sec + 1e-3:
        raise RuntimeError(
            f"BUG: full_audio duration {actual_full_duration_sec:.3f}s "
            f"exceeds max_duration_sec {max_duration_sec:.2f}s after trimming."
        )
    
    full_audio_bytes = full_audio_int16.tobytes()
    
    console.print(
        f"[info]VAD Reason:[/info] [value]{header['vad_reason']}[/value]"
    )
    console.print(
        f"[info]Context Duration:[/info] [time]{actual_context_sec:.2f}s[/time] used "
        f"/ [time]{context_duration_sec:.2f}s[/time] buffered "
        f"({segments_used}/{len(context_buffer.segments)} segments)"
    )
    console.print(
        f"[info]New Audio Duration:[/info] [time]{header['duration_sec']:.2f}s[/time]"
    )
    console.print(
        f"[info]Full Duration:[/info] "
        f"[time]{actual_full_duration_sec:.2f}s[/time] / [time]{max_duration_sec:.2f}s[/time] max"
    )
    
    segment_start_time = header.get("start_sec", time.time())
    segment_end_time = header.get("end_sec", segment_start_time + new_audio_duration_sec)
    
    # Get speakers in this segment (returns list of speaker segments)
    speaker_segments = label_speakers_for_segment(
        waveform=audio_np,
        sample_rate=sample_rate,
        start_time=segment_start_time,
        end_time=segment_end_time,
    )
    
    # Display speaker info
    if speaker_segments:
        speaker_summary = ", ".join([
            f"{s['speaker']} ({s['start']:.1f}s-{s['end']:.1f}s)" 
            for s in speaker_segments
        ])
        console.print(f"[speaker]Speakers detected: {speaker_summary}[/speaker]")
    else:
        console.print("[dim]No speakers detected in segment[/dim]")
    
    # Continue with transcription
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
    
    prev_full_ja_text = None
    prev_full_en_text = None
    unchanged_text = None
    new_ja_text = full_ja_text
    new_ja_start_index = None
    new_ja_similarity = None
    history = None
    
    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_ja_text = last_meta.get("full_ja_text", "")
        prev_full_en_text = last_meta.get("full_en_text", "")
        new_ja_text_res = extract_new_ja_text(prev_full_ja_text, full_ja_text)
        unchanged_text = new_ja_text_res["unchanged_text"]
        new_ja_text = new_ja_text_res["new_text"]
        new_ja_start_index = new_ja_text_res["start_index"]
        new_ja_similarity = new_ja_text_res["similarity"]
        last_ja_sentence, last_en_sentence, last_utt_id, last_sent_idx = context_buffer.get_last_sentence()
        
        MATCH_SCORE_CUTOFF = 75
        match_result = fuzzy_shortest_best_match_contains(
            query=new_ja_text,
            texts=full_ja_text,
            score_cutoff=MATCH_SCORE_CUTOFF,
            max_extra_chars=30,
        )
        if match_result["score"] >= MATCH_SCORE_CUTOFF and match_result["start"] != -1:
            console.print("[success bold]✅ Accepted[/success bold]")
            new_text_start = match_result["end"]
            new_text = new_ja_text
        else:
            console.print("[error]❌ Below threshold[/error]")
            console.print(
                f"[warning]Fuzzy match too weak (score={match_result['score']:.1f}).[/warning]"
            )
            console.print(f"[warning]Translating the full text.[/warning]")
            new_text = full_ja_text.strip()
        
        new_clean = new_text.rstrip('.。！？、…・「」『』').rstrip()
        if not new_clean:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "speaker_segments": speaker_segments,
                "success": False,
                "message": "Same text as previous",
            }
        
        old_ja_sents = split_sentences_ja(prev_full_ja_text)
        old_ja_text = prev_full_ja_text
        old_en_sents = split_sentences_ja(prev_full_en_text)
        old_en_text = prev_full_en_text
        new_ja_sents = split_sentences_ja(new_text)
        ja_text = "".join(new_ja_sents).strip()
        last_sentence_pos = match_result["start"] if match_result["score"] >= MATCH_SCORE_CUTOFF else -1
        last_sentence_clean = match_result["match"].strip() if match_result["score"] >= MATCH_SCORE_CUTOFF else None
        
        if ja_text:
            hist_result = context_buffer.get_context_history_by_duration(
                max_duration_sec=context_buffer.max_duration_sec,
                reserved_duration_sec=new_audio_duration_sec,
            )
            history = hist_result["history"]
            included_indices = hist_result["included_indices"]
            history_pairs = len(history) // 2
            max_dur = context_buffer.max_duration_sec
            inc_dur = hist_result["included_duration_sec"]
            total_history_dur = inc_dur + new_audio_duration_sec
            over_budget = total_history_dur > max_dur
            
            console.print(
                f"[info]History:[/info] "
                f"[number]{history_pairs}[/number] pairs "
                f"([number]{hist_result['included_segments']}[/number] included / "
                f"[number]{hist_result['excluded_segments']}[/number] excluded / "
                f"[number]{hist_result['total_segments']}[/number] total segments)"
            )
            console.print(
                f"[info]Combined Duration:[/info] "
                f"[time]{inc_dur:.2f}s[/time] history"
                f"  +  [time]{new_audio_duration_sec:.2f}s[/time] new"
                f"  =  [{'error' if over_budget else 'time'}]{total_history_dur:.2f}s[/{'error' if over_budget else 'time'}] total"
                f"  /  [time]{max_dur:.2f}s[/time] max"
                + (" [error]⚠ EXCEEDS BUDGET[/error]" if over_budget else "")
            )
            
            for i, (_, meta) in enumerate(list(context_buffer.segments)):
                seg_en = (meta.get("en_text") or "").strip()[:60]
                seg_dur = float(meta.get("duration_sec") or 0.0)
                has_text = bool((meta.get("en_text") or "").strip())
                tag = "success" if i in included_indices else "dim"
                text_tag = "white" if has_text else "yellow"
                console.print(
                    f"  [{tag}]seg[{i}][/{tag}] "
                    f"[time]{meta.get('duration_sec', 0):.2f}s[/time] "
                    f"[uuid]{meta.get('uuid', '')[-6:]}[/uuid] "
                    f"[{text_tag}]{seg_en}{'…' if seg_en else '(no text)'}[/{text_tag}]"
                    + (f"  [dim]+{seg_dur:.2f}s[/dim]" if i in included_indices else "")
                )
            
            trans_en = translate_japanese_to_english(
                text=ja_text,
                history=history,
            )
            en_text = trans_en["text"].strip()
        else:
            en_text = ""
        
        if prev_full_en_text:
            if new_ja_text_res["start_index"] == 0:
                full_en_text = en_text.strip()
                console.print("[success]Early correction detected → full_en_text reset to clean latest translation (no duplication)[/success]")
            else:
                full_en_text = (prev_full_en_text + "\n" + en_text).strip() if en_text else prev_full_en_text
        else:
            full_en_text = en_text
    else:
        ja_sents = full_ja_sents
        ja_text = full_ja_text
        curr_clean = ja_text.rstrip('.。！？、…・「」『』').rstrip()
        if curr_clean:
            full_trans_en = translate_japanese_to_english(text=ja_text)
            new_ja_sents = ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
        else:
            return {
                "uuid": uuid_,
                "transcription_ja": "",
                "translation_en": "",
                "speaker_segments": speaker_segments,
                "success": False,
                "message": "Empty transcription after cleaning",
            }
        old_ja_sents = []
        old_en_sents = []
        last_sentence_clean = None
        last_sentence_pos = -1

    if history:
        console.print(f"[bold yellow]History ({len(history)}):[/bold yellow]")
        console.print(f"[bold cyan]{history!r}[/bold cyan]")

    if last_sentence_clean:
        console.print(f"[success]Last Sentence (utt_id={last_utt_id[-6:]} | sent_idx={last_sent_idx}):[/success]")
        console.print(f"[bright_white]{last_sentence_clean}[/bright_white]")

    if last_sentence_pos != -1:
        console.print(f"[success]New Text (utt_id={header['uuid'][-6:]} | pos={last_sentence_pos} | start={new_text_start}):[/success]")
        console.print(f"[bright_white]{new_text}[/bright_white]")

    if old_ja_sents:
        console.print(f"[success]Old JA ({len(old_ja_sents)} sents):[/success]")
        console.print(f"[bright_white]{old_ja_text}[/bright_white]")

    console.print(f"[success]New JA ({len(new_ja_text)} chars):[/success]")
    console.print(f"[bold cyan]{new_ja_text}[/bold cyan]")

    if old_en_sents:
        console.print(f"[success]Old EN ({len(old_en_sents)} sents):[/success]")
        console.print(f"[bright_white]{old_en_text}[/bright_white]")

    console.print(f"[success]New EN ({len(en_text)} chars):[/success]")
    console.print(f"[bold cyan]{en_text}[/bold cyan]")

    if new_ja_text:
        if unchanged_text is not None:
            console.print(f"[success]Unchanged JA ({len(unchanged_text)} chars):[/success]")
            console.print(f"[white]{unchanged_text}[/white]")
        if new_ja_start_index is not None:
            console.print(f"[success]Start index:[/success] [bold cyan]{new_ja_start_index}[/bold cyan]")
        if new_ja_similarity is not None:
            console.print(f"[success]Matched Similarity:[/success] [bold cyan]{new_ja_similarity}[/bold cyan]")

    console.print(f"[success]Full JA ({len(full_ja_sents)} sents):[/success]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")

    if en_text.strip():
        console.print("[success]Full EN:[/success]")
        console.print(f"[bold white]{en_text}[/bold white]")
    else:
        console.print("[dim italic]No new translation[/dim italic]")

    # Log previous and current diffs
    if prev_full_ja_text and full_ja_text != prev_full_ja_text:
        console.print("[info]Diff (previous full JA → current full JA):[/info]")
        console_diff_highlight(
            prev_full_ja_text,
            full_ja_text,
            "Prev JA",
            "Curr JA",
        )

    if prev_full_en_text and full_en_text != prev_full_en_text:
        console.print("[info]Diff (previous full EN → current full EN):[/info]")
        console_diff_highlight(
            prev_full_en_text,
            full_en_text,
            "Prev EN",
            "Curr EN",
        )

    prefix_result = fuzzy_match_prefix_texts({
        "prev_ja": prev_full_ja_text,
        "prev_en": prev_full_en_text,
        "full_ja": full_ja_text,
        "full_en": full_en_text,
    })
    console.print(
        f"[info]Prefix match is_continuation:[/info] [value]{prefix_result['is_continuation']}[/value]"
    )

    ja_text = prefix_result["new_ja"]
    en_text = prefix_result["new_en"]

    new_en_sents = split_sentences_ja(full_en_text)

    started_at_iso = header.get("started_at")
    if started_at_iso and isinstance(started_at_iso, str):
        iso_str = started_at_iso.replace("Z", "+00:00") if started_at_iso.endswith("Z") else started_at_iso
        try:
            dt = datetime.fromisoformat(iso_str)
            ts_str = dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts_str = datetime.now().strftime("%Y%m%d_H%M%S")
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    segment_dir = LAST_N_SEGMENTS_DIR / f"segments_{ts_str}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)

    audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)

    wavfile.write(str(segment_dir / "full_sound.wav"), sample_rate, full_audio_int16)

    with open(segment_dir / "ja_sents.json", "w", encoding="utf-8") as f:
        json.dump({
            "old_ja_sents": old_ja_sents,
            "new_ja_sents": new_ja_sents,
        }, f, ensure_ascii=False, indent=2)

    with open(segment_dir / "en_sents.json", "w", encoding="utf-8") as f:
        json.dump({
            "old_en_sents": old_en_sents,
            "new_en_sents": new_en_sents,
        }, f, ensure_ascii=False, indent=2)

    # Save speaker info with the segment
    with open(segment_dir / "speaker_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "speaker_segments": speaker_segments,
            "speaker_count": len(set(s["speaker"] for s in speaker_segments)) if speaker_segments else 0,
            "diarization": get_speaker_diarization(),
        }, f, ensure_ascii=False, indent=2)

    # Build speaker summary for markdown
    if speaker_segments:
        speaker_summary_lines = []
        for seg in speaker_segments:
            overlap_marker = " (overlap)" if seg.get("is_overlapped", False) else ""
            speaker_summary_lines.append(
                f"  - **{seg['speaker']}**{overlap_marker}: "
                f"{seg['start']:.2f}s - {seg['end']:.2f}s "
                f"(duration: {seg['duration']:.2f}s)"
            )
        speaker_section = "**Speakers Detected:**\n" + "\n".join(speaker_summary_lines)
    else:
        speaker_section = "**Speakers Detected:** None"

    md_results = (
        f"{speaker_section}\n\n"
        f"**JA:** {ja_text}\n\n"
        f"**EN:** {en_text}\n"
    )
    with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md_results)

    metadata_out = {
        "uuid": uuid_,
        "duration_sec": header.get("duration_sec"),
        "started_at": header.get("started_at"),
        "transcribed_at": datetime.now().isoformat(),
        "speaker_segments": speaker_segments,
        "speaker_count": len(set(s["speaker"] for s in speaker_segments)) if speaker_segments else 0,
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
        "old_ja_sents": old_ja_sents,
        "new_ja_sents": new_ja_sents,
        "old_en_sents": old_en_sents,
        "new_en_sents": new_en_sents,
        "full_ja_text": full_ja_text,
        "full_en_text": full_en_text,
        "ja_text": ja_text,
        "en_text": en_text,
        "speaker_segments": speaker_segments,
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
        "current_speakers": _current_speakers,
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

    # Save speaker state periodically
    if _speaker_labeler and len(_speaker_labeler.all_segments) % 5 == 0:
        save_speaker_state()
    
    return {
        "uuid": uuid_,
        "new_duration": header['duration_sec'],
        "context_uuid": context_buffer.get_context_uuid() or uuid_,
        "context_duration": context_buffer.get_total_duration(),
        "success": bool(ja_text and en_text),
        "new_ja_similarity": new_ja_similarity,
        "new_ja_start_index": new_ja_start_index,
        "transcription_ja": new_ja_text,
        "translation_en": en_text,
        "transcribed_duration_sec": full_metadata["transcribed_duration_sec"],
        "transcribed_duration_pctg": full_metadata["transcribed_duration_pctg"],
        "coverage_label": full_metadata["coverage_label"],
        "speaker_segments": speaker_segments,
        "diarization": get_speaker_diarization(),
        "old_ja_sents": old_ja_sents,
        "new_ja_sents": new_ja_sents,
        "old_en_sents": old_en_sents,
        "new_en_sents": new_en_sents,
        "phrase_segments": full_phrase_segments,
    }


def split_message(data: bytes) -> tuple[dict, bytes]:
    """Split raw WebSocket binary message into (header dict, audio bytes)."""
    if b"\x00" not in data:
        raise ValueError("Message does not contain null byte separator")
    header_part, audio_bytes = data.split(b"\x00", 1)
    header = json.loads(header_part.decode("utf-8", errors="replace"))
    return header, audio_bytes


async def safe_send(websocket: WebSocket, payload: dict) -> bool:
    """
    Send a JSON payload over the WebSocket.
    Returns True on success, False if the client has already disconnected.
    """
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except (WebSocketDisconnect, RuntimeError) as exc:
        logger.debug(f"safe_send: client gone ({exc})")
        return False


@app.websocket("/ws/live-subtitles")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    client_id = str(uuid_module.uuid4())
    active_connections[client_id] = websocket
    console.print(
        f"[success]Client connected[/success] [uuid]{client_id[-6:]}[/uuid]"
        f" from [value]{client_info}[/value]"
        f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
    )
    
    # Ensure speaker labeler is initialized when first client connects
    _get_speaker_labeler()
    
    try:
        while True:
            try:
                message: bytes = await websocket.receive_bytes()
            except WebSocketDisconnect:
                break
            except RuntimeError as exc:
                logger.debug(f"receive_bytes RuntimeError (client gone): {exc}")
                break

            header_dict: dict = {}
            try:
                header_dict, audio_bytes = split_message(message)
                uuid_ = header_dict.get("uuid", "???")
                console.rule(style="dim")
                console.print(f"[info]Processing[/info] [uuid]{uuid_[-6:]}…[/uuid]")

                future = asyncio.get_running_loop().run_in_executor(
                    executor,
                    blocking_process_audio,
                    audio_bytes,
                    header_dict,
                )
                response = await future

                sent = await safe_send(websocket, response)
                if not sent:
                    logger.info(f"Client gone before result sent uuid={uuid_[-6:]}…")
                    break

                if response["success"]:
                    console.print(
                        f"[success]Processed successfully[/success] [uuid]{uuid_[-6:]}…[/uuid]"
                    )
                else:
                    console.print(
                        f"[warning]Empty response sent: {response.get('message', '')}[/warning]"
                        f" [uuid]{uuid_[-6:]}…[/uuid]"
                    )

                console.rule(style="dim")

            except Exception as proc_err:
                logger.error(f"Processing error for segment: {proc_err}")
                logger.exception("Full traceback:")
                error_resp = {
                    "uuid": header_dict.get("uuid", "unknown"),
                    "error": str(proc_err),
                    "success": False,
                    "transcription_ja": "",
                    "translation_en": "",
                    "speaker_segments": [],
                    "diarization": get_speaker_diarization(),
                }
                sent = await safe_send(websocket, error_resp)
                if not sent:
                    logger.info("Client gone — could not send error response, exiting.")
                    break

    except Exception as exc:
        logger.error(f"Unexpected WebSocket error: {exc}")
        logger.exception("Full traceback:")
    finally:
        active_connections.pop(client_id, None)
        # Save speaker state on disconnect
        save_speaker_state()
        console.print(
            f"[warning]Client disconnected[/warning] [uuid]{client_id[-6:]}[/uuid]"
            f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
        )


# ====================== REST Endpoints for Speaker Info ======================

@app.get("/speakers")
async def get_speakers():
    """Get current speaker diarization information."""
    return get_speaker_diarization()


@app.post("/speakers/reset")
async def reset_speakers():
    """Reset speaker labeler state."""
    global _current_speakers, _last_speaker_change_time
    labeler = _speaker_labeler
    if labeler:
        labeler.reset()
    _current_speakers = {}
    _last_speaker_change_time = 0.0
    save_speaker_state()
    return {"success": True, "message": "Speaker state reset"}


@app.post("/speakers/merge")
async def merge_speakers(label1: str = Form(...), label2: str = Form(...)):
    """Merge two speaker labels into one."""
    labeler = _speaker_labeler
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    result = labeler.merge_speakers(label1, label2)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Could not merge {label1} and {label2}")
    
    save_speaker_state()
    return {"success": True, "merged_label": result}


# ====================== Pydantic Models for REST APIs ======================
class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 encoded PCM int16 audio (optional if file uploaded)")
    sample_rate: int = Field(16000, description="Sample rate of the audio")
    hotwords: Optional[str] = Field(None, description="Hotwords for ASR")

class TranscribeResponse(BaseModel):
    success: bool
    transcription_ja: str
    speaker_label: str = "SPEAKER_UNKNOWN"
    speaker_confidence: float = 0.0
    speaker_segments: List[Dict] = []
    metadata: Dict[str, Any]
    word_segments: list = []
    phrase_segments: list = []

class TranslateRequest(BaseModel):
    japanese_text: str = Field(..., description="Japanese text to translate")
    history: Optional[list] = Field(default=None, description="Conversation history for context")
    temperature: Optional[float] = Field(0.35, ge=0.0, le=1.0)

class TranslateResponse(BaseModel):
    success: bool
    translation_en: str
    quality: str = "N/A"
    log_prob: Optional[float] = None
    confidence: Optional[float] = None


# ====================== REST Endpoints ======================

@app.post("/transcribe")
async def transcribe_endpoint(
    audio_file: UploadFile = File(..., description="Japanese audio file (WAV, PCM int16 recommended)"),
    sample_rate: int = Form(16000, description="Sample rate of the audio"),
    hotwords: Optional[str] = Form(None, description="Optional hotwords for better recognition"),
):
    """Transcribe Japanese audio → Japanese text (REST API)"""
    try:
        console.print(f"[info]Received file upload: {audio_file.filename} ({audio_file.content_type})[/info]")
        
        audio_bytes = await audio_file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        console.print(f"[info]Audio size: {len(audio_bytes)/1024:.1f} KB | Sample rate: {sample_rate} Hz[/info]")

        # Call existing transcription function
        result: TranscriptionResult = transcribe_japanese(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            hotwords=hotwords,
        )
        
        # Label speakers for this audio
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        speaker_segments = label_speakers_for_segment(
            waveform=audio_np,
            sample_rate=sample_rate,
            start_time=time.time(),
        )
        
        # Get primary speaker (first one) for backward compatibility
        primary_speaker = speaker_segments[0]["speaker"] if speaker_segments else "SPEAKER_UNKNOWN"
        primary_confidence = 0.8 if speaker_segments else 0.0

        return {
            "success": True,
            "transcription_ja": result.get("text", ""),
            "speaker_label": primary_speaker,
            "speaker_confidence": primary_confidence,
            "speaker_segments": speaker_segments,
            "metadata": result.get("metadata", {}),
            "word_segments": result.get("word_segments", []),
            "phrase_segments": result.get("phrase_segments", []),
        }

    except Exception as e:
        console.print(f"[error]Transcription endpoint failed: {e}[/error]")
        import traceback
        console.print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


@app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest):
    """Translate Japanese text to English only (REST API)."""
    try:
        if not request.japanese_text or not request.japanese_text.strip():
            raise HTTPException(status_code=400, detail="japanese_text is required and cannot be empty")

        result = translate_japanese_to_english(
            text=request.japanese_text.strip(),
            history=request.history,
            temperature=request.temperature or 0.35,
        )

        return {
            "success": True,
            "translation_en": result["text"],
            "quality": result.get("quality", "N/A"),
            "log_prob": result.get("log_prob"),
            "confidence": result.get("confidence"),
        }
    except Exception as e:
        console.print(f"[error]Translation endpoint error: {e}[/error]")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("🚀 Starting [bold cyan]Live Japanese Subtitles Server 2[/]")
    logger.info("WebSocket endpoint → [bold]ws://0.0.0.0:8000/ws/live-subtitles[/]")
    logger.info("REST endpoints:")
    logger.info("   POST /transcribe")
    logger.info("   POST /translate")
    logger.info("   GET  /speakers")
    logger.info("   POST /speakers/reset")
    logger.info("   POST /speakers/merge")
    logger.info("Press Ctrl+C to stop\n")
    uvicorn.run(
        app="live_subtitles_server2_speakers:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
