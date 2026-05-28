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
from typing import Any, Dict, Optional
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
from transcribe_funasr import TranscriptionResult, transcribe_audio
from translate_jp_en_llm_prefixed import translate_japanese_to_english
from segment_speaker_labeler import SegmentSpeakerLabeler
from pyannote.audio import Inference, Model
from live_subtitles_server_utils import (
    get_next_segment_number,
    load_segment_counter,
    prepare_segment_directory,
    save_segment_counter,
)
from audio_language_detector import AudioLanguageDetector

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

logger = logging.getLogger(__name__)
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = True

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
N_SEGMENT_RESULTS = 20
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"
LAST_N_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR
LIVE_AUDIO_BUFFER_DIR.mkdir(parents=True, exist_ok=True)
SPEAKER_STATE_PATH = OUTPUT_DIR / "speaker_state.json"
_SEGMENT_INDEX_PATH = LAST_N_SEGMENTS_DIR / "_segment_index.json"

app = FastAPI(title="Live Japanese Subtitles Server 2")
active_connections: dict[str, WebSocket] = {}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe_worker")
context_buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)
prev_end_sec: float | None = None
prev_vad_reason = None
_speaker_labeler: Optional[SegmentSpeakerLabeler] = None
_embedding_model: Optional[Model] = None
_embedding_inference: Optional[Inference] = None
_current_speaker: Optional[str] = None
_last_speaker_change_time: float = 0.0

console.print("Initializing AudioLanguageDetector...")
audio_language_detector = AudioLanguageDetector()
console.print("Detector initialized successfully!\n")


def _get_speaker_labeler() -> SegmentSpeakerLabeler:
    """Get or initialize the speaker labeler singleton.
    Lazy initialization defers model loading until first use.
    """
    global _speaker_labeler, _embedding_model, _embedding_inference
    if _speaker_labeler is not None:
        return _speaker_labeler

    console.print("[info]Loading speaker embedding model...[/info]")
    try:
        _embedding_model = Model.from_pretrained("pyannote/embedding")
        _embedding_inference = Inference(_embedding_model, window="whole")
        
        if SPEAKER_STATE_PATH.exists():
            try:
                with open(SPEAKER_STATE_PATH, 'r') as f:
                    state = json.load(f)
                _speaker_labeler = SegmentSpeakerLabeler.from_dict(
                    state,
                    embedding_model=_embedding_inference,
                )
                console.print(
                    f"[success]Restored speaker state: "
                    f"{_speaker_labeler.speaker_count} speaker(s), "
                    f"{_speaker_labeler.total_segments_processed} segments processed[/success]"
                )
                return _speaker_labeler
            except Exception as e:
                console.print(f"[warning]Could not restore speaker state: {e}[/warning]")
        
        _speaker_labeler = SegmentSpeakerLabeler(
            embedding_model=_embedding_inference,
            debug=True,
        )
        console.print("[success]Speaker labeler initialized[/success]")
    except Exception as e:
        console.print(f"[error]Failed to initialize speaker labeler: {e}[/error]")
        raise
    return _speaker_labeler


def save_speaker_state():
    """Persist the current speaker labeler state to disk."""
    if _speaker_labeler is None:
        return
    try:
        state = _speaker_labeler.to_dict()
        with open(SPEAKER_STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        console.print(f"[warning]Could not save speaker state: {e}[/warning]")


def label_speakers_for_segment(
    waveform: np.ndarray,
    sample_rate: int,
    timestamp: Optional[float] = None,
    return_multiple: bool = True,
) -> tuple[list[dict], str, float, Dict]:
    """Label speakers for an audio segment using the progressive labeler.
    
    Now returns a list of possible speakers for segments that may contain
    multiple speakers (especially longer segments 2-20s).
    
    Parameters
    ----------
    waveform : np.ndarray
        Audio waveform as int16 numpy array.
    sample_rate : int
        Sample rate of the audio.
    timestamp : float, optional
        Timestamp for this segment. If None, uses current time.
    return_multiple : bool
        If True, uses label_segments() to return multiple speaker results.
        If False, falls back to single-speaker label_segment().
    
    Returns
    -------
    tuple[list[dict], str, float, Dict]
        - List of speaker results (each with label, confidence, match_type, etc.)
        - Primary speaker label (for backward compatibility)
        - Primary confidence score
        - Additional metadata
    """
    global _current_speaker, _last_speaker_change_time
    
    if waveform.size == 0:
        empty_result = [{
            "label": "SPEAKER_UNKNOWN",
            "confidence": 0.0,
            "match_type": "empty_waveform",
            "is_primary": True,
            "is_new_speaker": False,
        }]
        return empty_result, "SPEAKER_UNKNOWN", 0.0, {"error": "empty_waveform"}
    
    if timestamp is None:
        timestamp = time.time()
    
    labeler = _get_speaker_labeler()
    
    waveform_float = waveform.astype(np.float32) / 32768.0
    waveform_tensor = torch.from_numpy(waveform_float)
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    
    context = {
        "previous_speaker": _current_speaker,
        "time_since_last_change": (
            timestamp - _last_speaker_change_time
            if _last_speaker_change_time > 0
            else float('inf')
        ),
        "segment_duration": len(waveform) / sample_rate,
    }
    
    if return_multiple:
        speaker_results = labeler.label_segments(
            waveform=waveform_tensor,
            sample_rate=sample_rate,
            timestamp=timestamp,
            context=context,
        )
        primary = speaker_results[0] if speaker_results else {
            "label": "SPEAKER_UNKNOWN",
            "confidence": 0.0,
            "match_type": "unknown",
            "is_primary": True,
            "is_new_speaker": False,
        }
        primary_label = primary["label"]
        primary_confidence = primary["confidence"]
        metadata = {
            "match_type": primary.get("match_type", "unknown"),
            "speaker_list": speaker_results,
            "total_speakers": len(speaker_results),
        }
    else:
        label, confidence, metadata = labeler.label_segment(
            waveform=waveform_tensor,
            sample_rate=sample_rate,
            timestamp=timestamp,
            context=context,
        )
        primary_label = label
        primary_confidence = confidence
        speaker_results = [{
            "label": label,
            "confidence": confidence,
            "match_type": metadata.get("match_type", "unknown"),
            "is_primary": True,
            "is_new_speaker": metadata.get("is_new_speaker", False),
        }]
        metadata["speaker_list"] = speaker_results
    
    if primary_label and primary_label != _current_speaker:
        console.print(
            f"[speaker]🔊 Speaker change: {_current_speaker} → {primary_label} "
            f"(confidence: {primary_confidence:.3f})[/speaker]"
        )
        _current_speaker = primary_label
        _last_speaker_change_time = timestamp
    
    if labeler.total_segments_processed % 10 == 0:
        save_speaker_state()
        # if labeler.total_segments_processed % 20 == 0 and labeler.speaker_count > 1:
        #     consol_result = labeler.consolidate_speakers(dry_run=False)
        #     if consol_result["merges_performed"]:
        #         console.print(
        #             f"[info]Auto-consolidation: merged {len(consol_result['merges_performed'])} "
        #             f"speaker pairs ({consol_result['speakers_before']} → "
        #             f"{consol_result['speakers_after']} speakers)[/info]"
        #         )
    
    return speaker_results, primary_label, primary_confidence, metadata


def get_speaker_diarization() -> Dict:
    """Get current speaker diarization summary with speaker list support."""
    labeler = _speaker_labeler
    if labeler is None:
        return {
            "current_speaker": None,
            "known_speakers": [],
            "speaker_count": 0,
            "speakers_info": {},
            "total_segments_processed": 0,
        }
    
    all_info = labeler.get_all_speakers_info()
    sorted_speakers = sorted(
        all_info.items(),
        key=lambda x: x[1].get("last_seen", 0),
        reverse=True,
    )
    
    return {
        "current_speaker": _current_speaker,
        "known_speakers": labeler.known_speakers,
        "speaker_count": labeler.speaker_count,
        "speakers_info": dict(sorted_speakers),
        "total_segments_processed": labeler.total_segments_processed,
    }


def should_reset_context(header: dict) -> bool:
    """Determine if we should reset the context buffer based on time gap or silence."""
    return True


def should_label_speaker(text: str, min_chars: int = 5) -> bool:
    """
    Determine if speaker labeling should be performed based on text content.

    Parameters
    ----------
    text : str
        Transcribed text (any language)
    min_chars : int
        Minimum number of meaningful characters required

    Returns
    -------
    bool
        True if speaker labeling should proceed
    """
    clean_text = text.strip()

    # Count any alphanumeric or CJK character as meaningful
    meaningful_chars = sum(
        1 for c in clean_text
        if c.isalnum()                          # Latin, digits, etc.
        or '\u3040' <= c <= '\u309f'            # Hiragana
        or '\u30a0' <= c <= '\u30ff'            # Katakana
        or '\u4e00' <= c <= '\u9fff'            # CJK Unified Ideographs
        or '\u3400' <= c <= '\u4dbf'            # CJK Extension A
        or '\uac00' <= c <= '\ud7af'            # Hangul syllables
        or '\u0600' <= c <= '\u06ff'            # Arabic
        or '\u0900' <= c <= '\u097f'            # Devanagari
        or '\u0400' <= c <= '\u04ff'            # Cyrillic
        or '\u0370' <= c <= '\u03ff'            # Greek
        or '\u0e00' <= c <= '\u0e7f'            # Thai
    )

    return meaningful_chars >= min_chars


def blocking_process_audio(
    audio_bytes: bytes,
    header: dict
) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work.
    """
    global prev_vad_reason, prev_end_sec
    uuid_ = header.get("uuid")
    if not uuid_:
        console.print("[error]Missing UUID in header[/error]")
        return {"message": "missing uuid", "success": False}
    
    sample_rate = header.get("sample_rate", 16000)
    language = header.get("language", "auto")
    full_trans_result = None
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Language detection
    if not language or language == "auto":
        console.print("[info]Detecting language with AudioLanguageDetector...[/info]")
        try:
            audio_tensor = torch.from_numpy(audio_np).float() / 32768.0
            audio_tensor = audio_tensor.unsqueeze(0)
            detected_lang = audio_language_detector.detect_from_bytes(
                audio_tensor, sample_rate=sample_rate
            )
            console.print(f"[success]Detected language: {detected_lang}[/success]")
            language = detected_lang
        except Exception as e:
            console.print(f"[error]Language detection failed: {e}. Falling back to 'ja'[/error]")
            language = "ja"
    
    console.print(f"[info]Transcribing with language: {language}[/info]")
    
    # Context buffer management
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
            f"Dropped {dropped_segments} oldest segment(s) to stay within limit. "
            f"Using {segments_used} segment(s) = {actual_context_sec:.2f}s context.[/warning]"
        )
    
    # Build full audio
    if context_audio_int16.size > 0:
        full_audio_int16 = np.concatenate([context_audio_int16, audio_np])
    else:
        full_audio_int16 = audio_np
    
    actual_full_duration_sec = len(full_audio_int16) / sample_rate
    if actual_full_duration_sec > max_duration_sec + 1e-3:
        raise RuntimeError(
            f"BUG: full_audio duration {actual_full_duration_sec:.3f}s "
            f"exceeds max_duration_sec {max_duration_sec:.2f}s after trimming. "
            "This should never happen — check get_context_audio_within_limit()."
        )
    
    full_audio_bytes = full_audio_int16.tobytes()
    
    # Log audio info
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
    
    # Transcribe
    full_trans_result = transcribe_audio(
        audio_bytes=full_audio_bytes,
        language=language,
        sample_rate=sample_rate,
    )
    full_trans_result = full_trans_result.copy()
    full_word_segments = full_trans_result.pop("word_segments")
    full_phrase_segments = full_trans_result.pop("phrase_segments")
    full_metadata = full_trans_result.pop("metadata")
    
    if header.get("language", "auto") == "auto":
        full_metadata["language"] = language
    
    # === FIX: Proper word joining based on language ===
    # Determine if the language uses spaces between words
    SPACELESS_LANGUAGES = {"ja", "jpn", "zh", "chi", "zho", "ko", "kor", "th", "tha"}
    is_spaceless = language in SPACELESS_LANGUAGES
    
    if is_spaceless:
        full_word_segments_text = "".join(s["word"] for s in full_word_segments)
    else:
        # For space-separated languages, join with spaces
        full_word_segments_text = " ".join(s["word"] for s in full_word_segments)
    
    # === ADD LOGGING: Log transcribed text ===
    console.print("[bold green]📝 Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{full_word_segments_text}[/bright_white]")
    console.print(f"[dim]Language: {language} | Words: {len(full_word_segments)}[/dim]")
    
    # === NON-JAPANESE LANGUAGE PATH (FIXED) ===
    if language != "ja" and language != "jpn":
        # For non-Japanese languages, use the transcribed text directly as en_text
        en_text = full_word_segments_text.strip()
        
        # === ADD LOGGING: Log transcribed text ===
        console.print("[bold green]📝 Non-Japanese Transcribed Text:[/bold green]")
        console.print(f"[bright_white]{en_text}[/bright_white]")
        console.print(f"[dim]Language: {language} | Words: {len(full_word_segments)}[/dim]")
        
        # Speaker labeling
        text_has_sufficient_content = should_label_speaker(full_word_segments_text, min_chars=5)
        speaker_results = []
        primary_label = None
        primary_confidence = 0.0
        speaker_metadata = {"match_type": "skipped_no_text"}
        
        if text_has_sufficient_content:
            segment_timestamp = header.get("start_sec", time.time())
            segment_duration = header.get("duration_sec", len(audio_np) / sample_rate)
            use_multiple = segment_duration >= 3.0
            speaker_results, primary_label, primary_confidence, speaker_metadata = (
                label_speakers_for_segment(
                    waveform=audio_np,
                    sample_rate=sample_rate,
                    timestamp=segment_timestamp,
                    return_multiple=use_multiple,
                )
            )
            if len(speaker_results) > 1:
                speakers_str = ", ".join(
                    f"{r['label']}({r['confidence']:.2f})" for r in speaker_results[:3]
                )
                console.print(
                    f"[speaker]Speakers: [{speakers_str}] "
                    f"(primary: {primary_label}, type: {speaker_metadata.get('match_type', 'unknown')})[/speaker]"
                )
            else:
                console.print(
                    f"[speaker]Speaker: {primary_label} "
                    f"(confidence: {primary_confidence:.3f}, "
                    f"type: {speaker_metadata.get('match_type', 'unknown')})[/speaker]"
                )
        else:
            console.print(
                f"[warning]Skipping speaker labeling - insufficient text content "
                f"(text: '{full_word_segments_text[:50]}{'...' if len(full_word_segments_text) > 50 else ''}', "
                f"length: {len(full_word_segments_text)} chars)[/warning]"
            )
        
        # === FIX: Save segment files (was missing from non-Japanese path) ===
        segment_num = get_next_segment_number()
        segment_dir = prepare_segment_directory(
            segment_num,
            segments_dir=LAST_N_SEGMENTS_DIR,
            segment_index_path=_SEGMENT_INDEX_PATH,
            n_results=N_SEGMENT_RESULTS,
        )
        segment_dir_name = f"segment_{segment_num:03d}"
        console.print(
            f"[info]Segment directory:[/info] [uuid]{segment_dir_name}[/uuid] "
            f"(#{segment_num}, keeping last {N_SEGMENT_RESULTS})"
        )
        
        # Save header
        with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
            json.dump(header, f, ensure_ascii=False, indent=2)
        
        # Save audio files
        audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)
        wavfile.write(str(segment_dir / "full_sound.wav"), sample_rate, full_audio_int16)
        
        # Save sentence files (empty for JA, use EN text for EN)
        with open(segment_dir / "ja_sents.json", "w", encoding="utf-8") as f:
            json.dump({
                "old_ja_sents": [],
                "new_ja_sents": [],
            }, f, ensure_ascii=False, indent=2)
        
        with open(segment_dir / "en_sents.json", "w", encoding="utf-8") as f:
            json.dump({
                "old_en_sents": [],
                "new_en_sents": [en_text] if en_text else [],
            }, f, ensure_ascii=False, indent=2)
        
        # Save speaker info
        with open(segment_dir / "speaker_info.json", "w", encoding="utf-8") as f:
            json.dump({
                "speaker_label": primary_label,
                "speaker_confidence": primary_confidence,
                "speaker_metadata": speaker_metadata,
                "speakers": speaker_results,
                "diarization": get_speaker_diarization(),
            }, f, ensure_ascii=False, indent=2)
        
        # Save results markdown
        if len(speaker_results) > 1:
            speaker_lines = []
            for r in speaker_results[:5]:
                speaker_lines.append(
                    f"- {r['label']} ({r['confidence']:.3f}, {r['match_type']})"
                )
            speaker_md = "\n".join(speaker_lines)
            md_results = (
                f"**Segment:** {segment_dir_name} (#{segment_num})\n\n"
                f"**Language:** {language}\n\n"
                f"**Speakers:**\n{speaker_md}\n\n"
                f"**Primary:** {primary_label} (confidence: {primary_confidence:.3f})\n\n"
                f"**Transcribed Text:**\n{en_text}\n"
            )
        else:
            md_results = (
                f"**Segment:** {segment_dir_name} (#{segment_num})\n\n"
                f"**Language:** {language}\n\n"
                f"**Speaker:** {primary_label} (confidence: {primary_confidence:.3f})\n\n"
                f"**Transcribed Text:**\n{en_text}\n"
            )
        
        with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
            f.write(md_results)
        
        # Save metadata
        metadata_out = {
            "uuid": uuid_,
            "segment_number": segment_num,
            "segment_dir": segment_dir_name,
            "duration_sec": header.get("duration_sec"),
            "started_at": header.get("started_at"),
            "transcribed_at": datetime.now().isoformat(),
            "language": language,
            "speaker_label": primary_label,
            "speaker_confidence": primary_confidence,
            "speakers": speaker_results,
            "speaker_count": len(speaker_results),
        }
        
        with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_out, f, ensure_ascii=False, indent=2)
        
        # Save full audio files (for the live buffer, not segment-specific)
        full_audio_dir = LIVE_AUDIO_BUFFER_DIR
        if full_audio_int16.size > 0:
            wavfile.write(
                str(full_audio_dir / "full_sound.wav"),
                context_buffer.sample_rate,
                full_audio_int16,
            )
        
        context_summary = {
            "total_duration_sec": round(context_buffer.get_total_duration(), 3),
            "num_chunks": len(context_buffer.segments),
            "max_duration_sec": context_buffer.max_duration_sec,
            "sample_rate": context_buffer.sample_rate,
            "last_updated": datetime.now().isoformat(),
            "context_includes_current_segment": True,
            "current_speaker": _current_speaker,
            "speaker_count": _speaker_labeler.speaker_count if _speaker_labeler else 0,
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
        
        # Store in context buffer
        context_buffer.add_audio_segment(audio_np, {
            "uuid": header["uuid"],
            "forced": header["forced"],
            "vad_reason": header["vad_reason"],
            "start_sec": header["start_sec"],
            "end_sec": header["end_sec"],
            "duration_sec": header["duration_sec"],
            "started_at": header["started_at"],
            "full_ja_text": "",
            "full_en_text": en_text,
            "ja_text": "",
            "en_text": en_text,
            "speaker_label": primary_label,
            "speaker_confidence": primary_confidence,
            "speakers": speaker_results,
        })
        
        # Save speaker state periodically
        if _speaker_labeler and _speaker_labeler.total_segments_processed % 5 == 0:
            save_speaker_state()
        
        # === ADD LOGGING: Final response summary ===
        console.print("[bold green]✅ Non-Japanese Response Summary:[/bold green]")
        console.print(f"  UUID: [uuid]{uuid_[-6:]}[/uuid]")
        console.print(f"  Language: [value]{language}[/value]")
        console.print(f"  Text length: [number]{len(en_text)}[/number] chars")
        console.print(f"  Speaker: [speaker]{primary_label}[/speaker]")
        console.print(f"  Segment: [uuid]{segment_dir_name}[/uuid] (#{segment_num})")
        
        return {
            "uuid": uuid_,
            "new_duration": header["duration_sec"],
            "context_uuid": context_buffer.get_context_uuid() or uuid_,
            "context_duration": context_buffer.get_total_duration(),
            "success": bool(en_text),
            "ja_text": "",
            "en_text": en_text,
            "language": full_metadata["language"],
            "event": full_metadata["event"],
            "emo": full_metadata["emo"],
            "transcribed_duration_sec": full_metadata["transcribed_duration_sec"],
            "transcribed_duration_pctg": full_metadata["transcribed_duration_pctg"],
            "coverage_label": full_metadata["coverage_label"],
            "speaker_label": primary_label,
            "speaker_confidence": primary_confidence,
            "speaker_match_type": speaker_metadata.get("match_type", "skipped"),
            "speakers": speaker_results,
            "diarization": get_speaker_diarization() if text_has_sufficient_content else {
                "current_speaker": _current_speaker,
                "known_speakers": [],
                "speaker_count": 0,
                "speakers_info": {},
                "total_segments_processed": 0,
                "note": "Speaker labeling skipped - insufficient text content",
            },
            "speaker_labeling_performed": text_has_sufficient_content,
            "old_ja_sents": [],
            "new_ja_sents": [],
            "old_en_sents": [],
            "new_en_sents": [en_text] if en_text else [],
            "phrase_segments": full_phrase_segments,
            "new_ja_similarity": None,
            "new_ja_start_index": None,
            "segment_number": segment_num,
            "segment_dir": segment_dir_name,
        }

    # === JAPANESE PATH (existing code with added logging) ===
    full_ja_text = full_word_segments_text
    full_ja_sents = split_sentences_ja(full_ja_text)
    
    # === ADD LOGGING: Log Japanese transcribed text ===
    console.print("[bold green]📝 Japanese Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")
    console.print(f"[dim]Sentences: {len(full_ja_sents)}[/dim]")
    
    # Speaker labeling
    text_has_sufficient_content = should_label_speaker(full_word_segments_text, min_chars=5)
    speaker_results = []
    primary_label = None
    primary_confidence = 0.0
    speaker_metadata = {"match_type": "skipped_no_text"}
    
    if text_has_sufficient_content:
        segment_timestamp = header.get("start_sec", time.time())
        segment_duration = header.get("duration_sec", len(audio_np) / sample_rate)
        use_multiple = segment_duration >= 3.0
        speaker_results, primary_label, primary_confidence, speaker_metadata = (
            label_speakers_for_segment(
                waveform=audio_np,
                sample_rate=sample_rate,
                timestamp=segment_timestamp,
                return_multiple=use_multiple,
            )
        )
        if len(speaker_results) > 1:
            speakers_str = ", ".join(
                f"{r['label']}({r['confidence']:.2f})" for r in speaker_results[:3]
            )
            console.print(
                f"[speaker]Speakers: [{speakers_str}] "
                f"(primary: {primary_label}, type: {speaker_metadata.get('match_type', 'unknown')})[/speaker]"
            )
        else:
            console.print(
                f"[speaker]Speaker: {primary_label} "
                f"(confidence: {primary_confidence:.3f}, "
                f"type: {speaker_metadata.get('match_type', 'unknown')})[/speaker]"
            )
    else:
        console.print(
            f"[warning]Skipping speaker labeling - insufficient text content "
            f"(text: '{full_word_segments_text[:50]}{'...' if len(full_word_segments_text) > 50 else ''}', "
            f"length: {len(full_word_segments_text)} chars)[/warning]"
        )
    
    # Existing Japanese processing continues...
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
            console.print(
                f"[warning]Translating the full text.[/warning]"
            )
            new_text = full_ja_text.strip()
        
        new_clean = new_text.rstrip('.。！？、…・「」『』').rstrip()
        if not new_clean:
            return {
                "uuid": uuid_,
                "ja_text": "",
                "en_text": "",
                "speaker_label": primary_label,
                "speaker_confidence": primary_confidence,
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
            
            # === ADD LOGGING: Log translation result ===
            console.print("[bold cyan]🌐 English Translation:[/bold cyan]")
            console.print(f"[bright_white]{en_text}[/bright_white]")
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
            full_trans_en = translate_japanese_to_english(
                text=ja_text,
            )
            new_ja_sents = ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
            
            # === ADD LOGGING: Log translation result ===
            console.print("[bold cyan]🌐 English Translation (First Segment):[/bold cyan]")
            console.print(f"[bright_white]{en_text}[/bright_white]")
        else:
            return {
                "uuid": uuid_,
                "ja_text": "",
                "en_text": "",
                "speaker_label": primary_label,
                "speaker_confidence": primary_confidence,
                "success": False,
                "message": "Empty transcription after cleaning",
            }
        old_ja_sents = []
        old_en_sents = []
        last_sentence_clean = None
        last_sentence_pos = -1
    
    # === ADD LOGGING: Summary of all texts ===
    console.print("[bold yellow]📊 Text Processing Summary:[/bold yellow]")
    console.print(f"  Original JA text: [bright_white]{full_ja_text[:100]}{'...' if len(full_ja_text) > 100 else ''}[/bright_white]")
    console.print(f"  New JA text: [bright_white]{new_ja_text[:100]}{'...' if len(new_ja_text) > 100 else ''}[/bright_white]")
    console.print(f"  EN text: [bright_white]{en_text[:100]}{'...' if len(en_text) > 100 else ''}[/bright_white]")
    
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
    
    segment_num = get_next_segment_number()
    segment_dir = prepare_segment_directory(
        segment_num,
        segments_dir=LAST_N_SEGMENTS_DIR,
        segment_index_path=_SEGMENT_INDEX_PATH,
        n_results=N_SEGMENT_RESULTS,
    )
    segment_dir_name = f"segment_{segment_num:03d}"
    console.print(
        f"[info]Segment directory:[/info] [uuid]{segment_dir_name}[/uuid] "
        f"(#{segment_num}, keeping last {N_SEGMENT_RESULTS})"
    )
    
    # Save files (existing code...)
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
    
    with open(segment_dir / "speaker_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "speaker_label": primary_label,
            "speaker_confidence": primary_confidence,
            "speaker_metadata": speaker_metadata,
            "speakers": speaker_results,
            "diarization": get_speaker_diarization(),
        }, f, ensure_ascii=False, indent=2)
    
    if len(speaker_results) > 1:
        speaker_lines = []
        for r in speaker_results[:5]:
            speaker_lines.append(
                f"- {r['label']} ({r['confidence']:.3f}, {r['match_type']})"
            )
        speaker_md = "\n".join(speaker_lines)
        md_results = (
            f"**Segment:** {segment_dir_name} (#{segment_num})\n\n"
            f"**Speakers:**\n{speaker_md}\n\n"
            f"**Primary:** {primary_label} (confidence: {primary_confidence:.3f})\n\n"
            f"JA: {ja_text}\n\n"
            f"EN: {en_text}\n"
        )
    else:
        md_results = (
            f"**Segment:** {segment_dir_name} (#{segment_num})\n\n"
            f"**Speaker:** {primary_label} (confidence: {primary_confidence:.3f})\n\n"
            f"JA: {ja_text}\n\n"
            f"EN: {en_text}\n"
        )
    
    with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md_results)
    
    metadata_out = {
        "uuid": uuid_,
        "segment_number": segment_num,
        "segment_dir": segment_dir_name,
        "duration_sec": header.get("duration_sec"),
        "started_at": header.get("started_at"),
        "transcribed_at": datetime.now().isoformat(),
        "speaker_label": primary_label,
        "speaker_confidence": primary_confidence,
        "speakers": speaker_results,
        "speaker_count": len(speaker_results),
    }
    
    with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, ensure_ascii=False, indent=2)
    
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
        "speaker_label": primary_label,
        "speaker_confidence": primary_confidence,
        "speakers": speaker_results,
    })
    
    # Save audio files (existing code...)
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
        "current_speaker": _current_speaker,
        "speaker_count": _speaker_labeler.speaker_count if _speaker_labeler else 0,
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
    
    if _speaker_labeler and _speaker_labeler.total_segments_processed % 5 == 0:
        save_speaker_state()
    
    # === ADD LOGGING: Final response summary ===
    console.print("[bold green]✅ Japanese Response Summary:[/bold green]")
    console.print(f"  UUID: [uuid]{uuid_[-6:]}[/uuid]")
    console.print(f"  Language: [value]{language}[/value]")
    console.print(f"  JA text: [number]{len(ja_text)}[/number] chars")
    console.print(f"  EN text: [number]{len(en_text)}[/number] chars")
    console.print(f"  Speaker: [speaker]{primary_label}[/speaker]")
    
    response = {
        "uuid": uuid_,
        "new_duration": header['duration_sec'],
        "context_uuid": context_uuid,
        "context_duration": context_duration,
        "success": bool(ja_text and en_text),
        "new_ja_similarity": new_ja_similarity,
        "new_ja_start_index": new_ja_start_index,
        "ja_text": new_ja_text,
        "en_text": en_text,
        "language": full_metadata["language"],
        "event": full_metadata["event"],
        "emo": full_metadata["emo"],
        "transcribed_duration_sec": full_metadata["transcribed_duration_sec"],
        "transcribed_duration_pctg": full_metadata["transcribed_duration_pctg"],
        "coverage_label": full_metadata["coverage_label"],
        "speaker_label": primary_label,
        "speaker_confidence": primary_confidence,
        "speaker_match_type": speaker_metadata.get("match_type", "skipped"),
        "speakers": speaker_results,
        "diarization": get_speaker_diarization() if text_has_sufficient_content else {
            "current_speaker": _current_speaker,
            "known_speakers": [],
            "speaker_count": 0,
            "speakers_info": {},
            "total_segments_processed": 0,
            "note": "Speaker labeling skipped - insufficient text content",
        },
        "speaker_labeling_performed": text_has_sufficient_content,
        "old_ja_sents": old_ja_sents,
        "new_ja_sents": new_ja_sents,
        "old_en_sents": old_en_sents,
        "new_en_sents": new_en_sents,
        "phrase_segments": full_phrase_segments,
        "segment_number": segment_num,
        "segment_dir": segment_dir_name,
    }
    
    return response


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
                    "ja_text": "",
                    "en_text": "",
                    "speaker_label": "SPEAKER_UNKNOWN",
                    "speaker_confidence": 0.0,
                    "speakers": [],
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
        save_speaker_state()
        console.print(
            f"[warning]Client disconnected[/warning] [uuid]{client_id[-6:]}[/uuid]"
            f" — total [bright_blue]{len(active_connections)}[/bright_blue]"
        )


@app.get("/speakers")
async def get_speakers():
    """Get current speaker diarization information."""
    return get_speaker_diarization()


@app.post("/speakers/reset")
async def reset_speakers():
    """Reset speaker labeler state - fully clears all speaker tracking.
    
    Fixed: Also resets the global _current_speaker, clears context buffer
    speaker metadata, and forces next labeling to ignore previous speaker.
    """
    global _current_speaker, _last_speaker_change_time
    
    # Reset the labeler
    labeler = _speaker_labeler
    if labeler:
        labeler.reset()
    
    # Reset global state
    _current_speaker = None
    _last_speaker_change_time = 0.0
    
    # Clear speaker labels from context buffer segments
    if context_buffer.segments:
        for segment_audio, metadata in context_buffer.segments:
            metadata["speaker_label"] = None
            metadata["speaker_confidence"] = 0.0
            metadata["speakers"] = []
    
    # Delete persisted state file
    if SPEAKER_STATE_PATH.exists():
        SPEAKER_STATE_PATH.unlink()
    
    save_speaker_state()
    
    console.print("[warning]🔄 Speaker state fully reset: labeler + global state + context buffer[/warning]")
    
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


@app.post("/speakers/consolidate")
async def consolidate_speakers_endpoint(
    threshold: float = Form(0.85),
    dry_run: bool = Form(False),
):
    """Consolidate similar speakers by merging those above similarity threshold.
    
    Parameters
    ----------
    threshold : float
        Similarity threshold above which speakers are merged (0.0 to 1.0).
    dry_run : bool
        If true, returns proposed merges without executing them.
    """
    labeler = _speaker_labeler
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    result = labeler.consolidate_speakers(threshold=threshold, dry_run=dry_run)
    if not dry_run:
        save_speaker_state()
    return {
        "success": True,
        **result,
    }


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 encoded PCM int16 audio (optional if file uploaded)")
    sample_rate: int = Field(16000, description="Sample rate of the audio")
    hotwords: Optional[str] = Field(None, description="Hotwords for ASR")


class TranscribeResponse(BaseModel):
    success: bool
    transcription: str
    speaker_label: str = "SPEAKER_UNKNOWN"
    speaker_confidence: float = 0.0
    metadata: Dict[str, Any]
    word_segments: list = []
    phrase_segments: list = []


class TranslateRequest(BaseModel):
    japanese_text: str = Field(..., description="Japanese text to translate")
    history: Optional[list] = Field(default=None, description="Conversation history for context")
    temperature: Optional[float] = Field(0.35, ge=0.0, le=1.0)


class TranslateResponse(BaseModel):
    success: bool
    en_text: str
    quality: str = "N/A"
    log_prob: Optional[float] = None
    confidence: Optional[float] = None


@app.post("/transcribe")
async def transcribe_endpoint(
    audio_file: UploadFile = File(..., description="Japanese audio file (WAV, PCM int16 recommended)"),
    sample_rate: int = Form(16000, description="Sample rate of the audio"),
    hotwords: Optional[str] = Form(None, description="Optional hotwords for better recognition"),
    language: str = Form(
        "auto",
        description="Language code (e.g., 'ja' for Japanese, 'en' for English, 'auto' for auto-detect)",
    ),
):
    """Transcribe Japanese or English audio → text (REST API)"""
    try:
        console.print(f"[info]Received file upload: {audio_file.filename} ({audio_file.content_type})[/info]")
        audio_bytes = await audio_file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        console.print(f"[info]Audio size: {len(audio_bytes)/1024:.1f} KB | Sample rate: {sample_rate} Hz[/info]")
        result: TranscriptionResult = transcribe_audio(
            audio_bytes=audio_bytes,
            language=language,
            sample_rate=sample_rate,
            hotwords=hotwords,
        )
   
        return {
            "success": True,
            "transcription": result.get("text", ""),
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
            "en_text": result["text"],
            "quality": result.get("quality", "N/A"),
            "log_prob": result.get("log_prob"),
            "confidence": result.get("confidence"),
        }
    except Exception as e:
        console.print(f"[error]Translation endpoint error: {e}[/error]")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Initialize segment counter from disk
    _segment_counter = load_segment_counter(_SEGMENT_INDEX_PATH)
    console.print(
        f"[info]Segment counter initialized: {_segment_counter} "
        f"(next will be segment_{_segment_counter + 1:03d})[/info]"
    )
    
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
        app="live_subtitles_server2_with_en:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
