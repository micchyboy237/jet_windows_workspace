"""
Core audio processing logic extracted from the monolithic server file.

Contains the heavy CPU/GPU work that runs in the thread pool.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from rich.console import Console
from services.diff_utils import console_diff_highlight, extract_new_ja_text
from services.sentence_matcher_ja import fuzzy_shortest_best_match_contains, fuzzy_match_prefix_texts
from services.sentence_utils import split_sentences_ja
from services.transcribe_funasr import transcribe_audio
from services.translate_jp_en_llm_prefixed import translate_japanese_to_english
from services.live_subtitles_server_utils import (
    get_next_segment_number,
    prepare_segment_directory,
)
from core.state import (
    get_context_buffer,
    get_current_speaker,
    set_current_speaker,
    get_last_speaker_change_time,
    set_last_speaker_change_time,
    get_speaker_labeler,
    save_speaker_state,
    get_audio_language_detector,
    get_last_n_segments_dir,
    get_live_audio_buffer_dir,
    get_segment_index_path,
    get_n_segment_results,
)

console = Console()

# Spaceless languages (no spaces between words)
SPACELESS_LANGUAGES = {"ja", "jpn", "zh", "chi", "zho", "ko", "kor", "th", "tha"}


def _get_speaker_labeler():
    """Get or initialize the speaker labeler singleton."""
    from core.state import (
        get_speaker_labeler,
        set_speaker_labeler,
        get_embedding_inference,
        set_embedding_inference,
        get_speaker_state_path,
    )
    from pyannote.audio import Inference, Model
    from services.segment_speaker_labeler import SegmentSpeakerLabeler
    
    labeler = get_speaker_labeler()
    if labeler is not None:
        return labeler
    
    console.print("[info]Loading speaker embedding model...[/info]")
    try:
        embedding_model = Model.from_pretrained("pyannote/embedding")
        embedding_inference = Inference(embedding_model, window="whole")
        set_embedding_inference(embedding_inference)
        
        speaker_state_path = get_speaker_state_path()
        if speaker_state_path.exists():
            try:
                with open(speaker_state_path, 'r') as f:
                    state = json.load(f)
                labeler = SegmentSpeakerLabeler.from_dict(
                    state,
                    embedding_model=embedding_inference,
                )
                set_speaker_labeler(labeler)
                console.print(
                    f"[success]Restored speaker state: "
                    f"{labeler.speaker_count} speaker(s), "
                    f"{labeler.total_segments_processed} segments processed[/success]"
                )
                return labeler
            except Exception as e:
                console.print(f"[warning]Could not restore speaker state: {e}[/warning]")
        
        labeler = SegmentSpeakerLabeler(
            embedding_model=embedding_inference,
            debug=True,
        )
        set_speaker_labeler(labeler)
        console.print("[success]Speaker labeler initialized[/success]")
    except Exception as e:
        console.print(f"[error]Failed to initialize speaker labeler: {e}[/error]")
        raise
    return labeler


def should_reset_context(header: dict) -> bool:
    """Determine if we should reset the context buffer based on time gap or silence."""
    return True


def should_label_speaker(text: str, min_chars: int = 2) -> bool:
    """
    Determine if speaker labeling should be performed based on text content.
    """
    clean_text = text.strip()
    meaningful_chars = sum(
        1 for c in clean_text
        if c.isalnum()
        or '\u3040' <= c <= '\u309f'
        or '\u30a0' <= c <= '\u30ff'
        or '\u4e00' <= c <= '\u9fff'
        or '\u3400' <= c <= '\u4dbf'
        or '\uac00' <= c <= '\ud7af'
        or '\u0600' <= c <= '\u06ff'
        or '\u0900' <= c <= '\u097f'
        or '\u0400' <= c <= '\u04ff'
        or '\u0370' <= c <= '\u03ff'
        or '\u0e00' <= c <= '\u0e7f'
    )
    return meaningful_chars >= min_chars


def get_speaker_diarization() -> Dict:
    """Get current speaker diarization summary with speaker list support."""
    from core.state import get_current_speaker
    
    labeler = get_speaker_labeler()
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
        "current_speaker": get_current_speaker(),
        "total_segments_processed": labeler.total_segments_processed,
        "known_speakers": labeler.known_speakers,
        "speaker_count": labeler.speaker_count,
        "speakers_info": dict(sorted_speakers),
    }


def label_speakers_for_segment(
    waveform: np.ndarray,
    sample_rate: int,
    timestamp: Optional[float] = None,
    return_multiple: bool = True,
) -> tuple:
    """Label speakers for an audio segment using the progressive labeler."""
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
    
    current_speaker = get_current_speaker()
    last_change_time = get_last_speaker_change_time()
    
    context = {
        "previous_speaker": current_speaker,
        "time_since_last_change": (
            timestamp - last_change_time
            if last_change_time > 0
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
    
    if primary_label and primary_label != current_speaker:
        console.print(
            f"[speaker]🔊 Speaker change: {current_speaker} → {primary_label} "
            f"(confidence: {primary_confidence:.3f})[/speaker]"
        )
        set_current_speaker(primary_label)
        set_last_speaker_change_time(timestamp)
    
    if labeler.total_segments_processed % 10 == 0:
        save_speaker_state()
    
    return speaker_results, primary_label, primary_confidence, metadata


def blocking_process_audio(audio_bytes: bytes, header: dict) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work.
    
    Handles both Japanese and non-Japanese audio processing with
    speaker labeling, translation, and file persistence.
    """
    from core.state import (
        get_context_buffer,
        get_audio_language_detector,
        get_last_n_segments_dir,
        get_live_audio_buffer_dir,
        get_segment_index_path,
        get_n_segment_results,
        get_current_speaker,
    )
    
    context_buffer = get_context_buffer()
    live_audio_buffer_dir = get_live_audio_buffer_dir()
    last_n_segments_dir = get_last_n_segments_dir()
    segment_index_path = get_segment_index_path()
    n_segment_results = get_n_segment_results()
    
    uuid_ = header.get("uuid")
    if not uuid_:
        console.print("[error]Missing UUID in header[/error]")
        return {"message": "missing uuid", "success": False}
    
    sample_rate = header.get("sample_rate", 16000)
    language = header.get("language", "auto")
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Language detection
    if not language or language == "auto":
        console.print("[info]Detecting language with AudioLanguageDetector...[/info]")
        try:
            detector = get_audio_language_detector()
            audio_tensor = torch.from_numpy(audio_np).float() / 32768.0
            audio_tensor = audio_tensor.unsqueeze(0)
            detected_lang = detector.detect_from_bytes(
                audio_tensor, sample_rate=sample_rate
            )
            console.print(f"[success]Detected language: {detected_lang}[/success]")
            language = detected_lang
        except Exception as e:
            console.print(f"[error]Language detection failed: {e}. Falling back to 'ja'[/error]")
            language = "ja"
    
    console.print(f"[info]Transcribing with language: {language}[/info]")
    
    # Context management
    if should_reset_context(header):
        context_buffer.reset()
    
    new_audio_duration_sec = len(audio_np) / sample_rate
    context_duration_sec = context_buffer.get_total_duration()
    max_duration_sec = context_buffer.max_duration_sec
    
    context_audio_int16, actual_context_sec, segments_used = (
        context_buffer.get_context_audio_within_limit(new_audio_duration_sec)
    )
    
    combined_naive_sec = context_duration_sec + new_audio_duration_sec
    if combined_naive_sec > max_duration_sec:
        dropped_segments = len(context_buffer.segments) - segments_used
        console.print(
            f"[warning]⚠️  Combined audio ({combined_naive_sec:.2f}s) would exceed "
            f"max_duration_sec ({max_duration_sec:.2f}s). "
            f"Dropped {dropped_segments} oldest segment(s) to stay within limit. "
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
    
    console.print(f"[info]VAD Reason:[/info] [value]{header['vad_reason']}[/value]")
    console.print(
        f"[info]Context Duration:[/info] [time]{actual_context_sec:.2f}s[/time] used "
        f"/ [time]{context_duration_sec:.2f}s[/time] buffered "
        f"({segments_used}/{len(context_buffer.segments)} segments)"
    )
    console.print(f"[info]New Audio Duration:[/info] [time]{header['duration_sec']:.2f}s[/time]")
    console.print(
        f"[info]Full Duration:[/info] "
        f"[time]{actual_full_duration_sec:.2f}s[/time] / [time]{max_duration_sec:.2f}s[/time] max"
    )
    
    # Transcription
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
    
    is_spaceless = language in SPACELESS_LANGUAGES
    if is_spaceless:
        full_word_segments_text = "".join(s["word"] for s in full_word_segments)
    else:
        full_word_segments_text = " ".join(s["word"] for s in full_word_segments)
    
    console.print("[bold green]📝 Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{full_word_segments_text}[/bright_white]")
    console.print(f"[dim]Language: {language} | Words: {len(full_word_segments)}[/dim]")
    
    # Determine if this is Japanese or non-Japanese
    if language != "ja" and language != "jpn":
        return _process_non_japanese(
            audio_bytes=audio_bytes,
            audio_np=audio_np,
            full_audio_int16=full_audio_int16,
            header=header,
            uuid_=uuid_,
            sample_rate=sample_rate,
            language=language,
            full_word_segments=full_word_segments,
            full_word_segments_text=full_word_segments_text,
            full_phrase_segments=full_phrase_segments,
            full_metadata=full_metadata,
            context_buffer=context_buffer,
            live_audio_buffer_dir=live_audio_buffer_dir,
            last_n_segments_dir=last_n_segments_dir,
            segment_index_path=segment_index_path,
            n_segment_results=n_segment_results,
        )
    else:
        return _process_japanese(
            audio_bytes=audio_bytes,
            audio_np=audio_np,
            full_audio_int16=full_audio_int16,
            header=header,
            uuid_=uuid_,
            sample_rate=sample_rate,
            language=language,
            full_word_segments=full_word_segments,
            full_word_segments_text=full_word_segments_text,
            full_phrase_segments=full_phrase_segments,
            full_metadata=full_metadata,
            context_buffer=context_buffer,
            live_audio_buffer_dir=live_audio_buffer_dir,
            last_n_segments_dir=last_n_segments_dir,
            segment_index_path=segment_index_path,
            n_segment_results=n_segment_results,
        )


def _perform_speaker_labeling(
    audio_np: np.ndarray,
    sample_rate: int,
    header: dict,
    full_word_segments_text: str,
) -> tuple:
    """Perform speaker labeling if text content is sufficient."""
    text_has_sufficient_content = should_label_speaker(full_word_segments_text, min_chars=2)
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
    
    return text_has_sufficient_content, speaker_results, primary_label, primary_confidence, speaker_metadata


def _save_segment_files(
    segment_dir: Path,
    segment_dir_name: str,
    segment_num: int,
    header: dict,
    audio_bytes: bytes,
    audio_np: np.ndarray,
    full_audio_int16: np.ndarray,
    sample_rate: int,
    language: str,
    primary_label: Optional[str],
    primary_confidence: float,
    speaker_metadata: dict,
    speaker_results: list,
    old_ja_sents: list,
    new_ja_sents: list,
    old_en_sents: list,
    new_en_sents: list,
    ja_text: str,
    en_text: str,
) -> dict:
    """Save all segment-related files to disk."""
    # Header
    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)
    
    # Audio files
    audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)
    wavfile.write(str(segment_dir / "full_sound.wav"), sample_rate, full_audio_int16)
    
    # Sentence files
    with open(segment_dir / "ja_sents.json", "w", encoding="utf-8") as f:
        json.dump({"old_ja_sents": old_ja_sents, "new_ja_sents": new_ja_sents}, f, ensure_ascii=False, indent=2)
    
    with open(segment_dir / "en_sents.json", "w", encoding="utf-8") as f:
        json.dump({"old_en_sents": old_en_sents, "new_en_sents": new_en_sents}, f, ensure_ascii=False, indent=2)
    
    # Speaker info
    with open(segment_dir / "speaker_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "speaker_label": primary_label,
            "speaker_confidence": primary_confidence,
            "speaker_metadata": speaker_metadata,
            "speakers": speaker_results,
            "diarization": get_speaker_diarization(),
        }, f, ensure_ascii=False, indent=2)
    
    # Results markdown
    if len(speaker_results) > 1:
        speaker_lines = [f"- {r['label']} ({r['confidence']:.3f}, {r['match_type']})" for r in speaker_results[:5]]
        speaker_md = "\n".join(speaker_lines)
        md_results = (
            f"**Segment:** {segment_dir_name} (#{segment_num})\n\n"
            f"**Language:** {language}\n\n"
            f"**Speakers:**\n{speaker_md}\n\n"
            f"**Primary:** {primary_label} (confidence: {primary_confidence:.3f})\n\n"
        )
    else:
        md_results = (
            f"**Segment:** {segment_dir_name} (#{segment_num})\n\n"
            f"**Language:** {language}\n\n"
            f"**Speaker:** {primary_label} (confidence: {primary_confidence:.3f})\n\n"
        )
    
    if ja_text:
        md_results += f"JA: {ja_text}\n\n"
    if en_text:
        md_results += f"EN: {en_text}\n"
    
    with open(segment_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(md_results)
    
    # Metadata
    metadata_out = {
        "uuid": header.get("uuid"),
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
    
    return metadata_out


def _save_full_audio_files(
    full_audio_dir: Path,
    full_audio_int16: np.ndarray,
    sample_rate: int,
    context_buffer,
    full_trans_result: dict,
    full_metadata: dict,
    full_word_segments: list,
    full_word_segments_text: str,
    full_phrase_segments: list,
    full_ja_sents: Optional[list] = None,
) -> None:
    """Save full audio analysis files."""
    if full_audio_int16.size > 0:
        wavfile.write(str(full_audio_dir / "full_sound.wav"), sample_rate, full_audio_int16)
    
    # Summary
    context_summary = {
        "total_duration_sec": round(context_buffer.get_total_duration(), 3),
        "num_chunks": len(context_buffer.segments),
        "max_duration_sec": context_buffer.max_duration_sec,
        "sample_rate": context_buffer.sample_rate,
        "last_updated": datetime.now().isoformat(),
        "context_includes_current_segment": True,
        "current_speaker": get_current_speaker(),
        "speaker_count": get_speaker_labeler().speaker_count if get_speaker_labeler() else 0,
    }
    with open(full_audio_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(context_summary, f, ensure_ascii=False, indent=2)
    
    # Metadata lists
    full_audio_metadata = context_buffer.get_list_metadata()
    with open(full_audio_dir / "full_audio_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_audio_metadata, f, ensure_ascii=False, indent=2)
    
    with open(full_audio_dir / "full_transcription.json", "w", encoding="utf-8") as f:
        json.dump(full_trans_result, f, ensure_ascii=False, indent=2)
    
    with open(full_audio_dir / "full_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, ensure_ascii=False, indent=2)
    
    # Word segments
    with open(full_audio_dir / "full_word_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": "word",
            "count": len(full_word_segments),
            "text": full_word_segments_text,
            "segments": full_word_segments
        }, f, ensure_ascii=False, indent=2)
    
    # Phrase segments
    with open(full_audio_dir / "full_phrase_segments.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": "phrase",
            "count": len(full_phrase_segments),
            "phrases": [p["phrase"] for p in full_phrase_segments],
            "segments": full_phrase_segments
        }, f, ensure_ascii=False, indent=2)
    
    # Japanese sentences (if provided)
    if full_ja_sents is not None:
        with open(full_audio_dir / "full_ja_sents.json", "w", encoding="utf-8") as f:
            json.dump(full_ja_sents, f, ensure_ascii=False, indent=2)


def _process_non_japanese(
    audio_bytes: bytes,
    audio_np: np.ndarray,
    full_audio_int16: np.ndarray,
    header: dict,
    uuid_: str,
    sample_rate: int,
    language: str,
    full_word_segments: list,
    full_word_segments_text: str,
    full_phrase_segments: list,
    full_metadata: dict,
    context_buffer,
    live_audio_buffer_dir: Path,
    last_n_segments_dir: Path,
    segment_index_path: Path,
    n_segment_results: int,
) -> dict:
    """Process non-Japanese audio segment."""
    en_text = full_word_segments_text.strip()
    console.print("[bold green]📝 Non-Japanese Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{en_text}[/bright_white]")
    console.print(f"[dim]Language: {language} | Words: {len(full_word_segments)}[/dim]")
    
    # Speaker labeling
    text_has_sufficient_content, speaker_results, primary_label, primary_confidence, speaker_metadata = (
        _perform_speaker_labeling(audio_np, sample_rate, header, full_word_segments_text)
    )
    
    # Segment directory
    segment_num = get_next_segment_number()
    segment_dir = prepare_segment_directory(
        segment_num,
        segments_dir=last_n_segments_dir,
        segment_index_path=segment_index_path,
        n_results=n_segment_results,
    )
    segment_dir_name = f"segment_{segment_num:03d}"
    console.print(
        f"[info]Segment directory:[/info] [uuid]{segment_dir_name}[/uuid] "
        f"(#{segment_num}, keeping last {n_segment_results})"
    )
    
    # Save segment files
    _save_segment_files(
        segment_dir=segment_dir,
        segment_dir_name=segment_dir_name,
        segment_num=segment_num,
        header=header,
        audio_bytes=audio_bytes,
        audio_np=audio_np,
        full_audio_int16=full_audio_int16,
        sample_rate=sample_rate,
        language=language,
        primary_label=primary_label,
        primary_confidence=primary_confidence,
        speaker_metadata=speaker_metadata,
        speaker_results=speaker_results,
        old_ja_sents=[],
        new_ja_sents=[],
        old_en_sents=[],
        new_en_sents=[en_text] if en_text else [],
        ja_text="",
        en_text=en_text,
    )
    
    # Save full audio files
    _save_full_audio_files(
        full_audio_dir=live_audio_buffer_dir,
        full_audio_int16=full_audio_int16,
        sample_rate=sample_rate,
        context_buffer=context_buffer,
        full_trans_result={},
        full_metadata=full_metadata,
        full_word_segments=full_word_segments,
        full_word_segments_text=full_word_segments_text,
        full_phrase_segments=full_phrase_segments,
    )
    
    # Add to context buffer
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
    
    if get_speaker_labeler() and get_speaker_labeler().total_segments_processed % 5 == 0:
        save_speaker_state()
    
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
            "current_speaker": get_current_speaker(),
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


def _process_japanese(
    audio_bytes: bytes,
    audio_np: np.ndarray,
    full_audio_int16: np.ndarray,
    header: dict,
    uuid_: str,
    sample_rate: int,
    language: str,
    full_word_segments: list,
    full_word_segments_text: str,
    full_phrase_segments: list,
    full_metadata: dict,
    context_buffer,
    live_audio_buffer_dir: Path,
    last_n_segments_dir: Path,
    segment_index_path: Path,
    n_segment_results: int,
) -> dict:
    """Process Japanese audio segment with translation."""
    full_ja_text = full_word_segments_text
    full_ja_sents = split_sentences_ja(full_ja_text)
    
    console.print("[bold green]📝 Japanese Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")
    console.print(f"[dim]Sentences: {len(full_ja_sents)}[/dim]")
    
    # Speaker labeling
    text_has_sufficient_content, speaker_results, primary_label, primary_confidence, speaker_metadata = (
        _perform_speaker_labeling(audio_np, sample_rate, header, full_word_segments_text)
    )
    
    # Process Japanese text and translate
    prev_full_ja_text = None
    prev_full_en_text = None
    new_ja_text = full_ja_text
    new_ja_start_index = None
    new_ja_similarity = None
    history = None
    old_ja_sents = []
    old_en_sents = []
    new_ja_sents = full_ja_sents
    new_en_sents = []
    ja_text = full_ja_text
    en_text = ""
    full_en_text = ""
    
    new_audio_duration_sec = len(audio_np) / sample_rate
    
    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_ja_text = last_meta.get("full_ja_text", "")
        prev_full_en_text = last_meta.get("full_en_text", "")
        
        # Extract new text
        new_ja_text_res = extract_new_ja_text(prev_full_ja_text, full_ja_text)
        new_ja_text = new_ja_text_res["new_text"]
        new_ja_start_index = new_ja_text_res["start_index"]
        new_ja_similarity = new_ja_text_res["similarity"]
        
        # Fuzzy match
        MATCH_SCORE_CUTOFF = 75
        match_result = fuzzy_shortest_best_match_contains(
            query=new_ja_text,
            texts=full_ja_text,
            score_cutoff=MATCH_SCORE_CUTOFF,
            max_extra_chars=30,
        )
        
        if match_result["score"] >= MATCH_SCORE_CUTOFF and match_result["start"] != -1:
            console.print("[success bold]✅ Accepted[/success bold]")
            new_text = new_ja_text
        else:
            console.print("[error]❌ Below threshold[/error]")
            console.print(f"[warning]Fuzzy match too weak (score={match_result['score']:.1f}).[/warning]")
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
        old_en_sents = split_sentences_ja(prev_full_en_text)
        new_ja_sents = split_sentences_ja(new_text)
        ja_text = "".join(new_ja_sents).strip()
        
        if ja_text:
            # Get history for translation context
            hist_result = context_buffer.get_context_history_by_duration(
                max_duration_sec=context_buffer.max_duration_sec,
                reserved_duration_sec=new_audio_duration_sec,
            )
            history = hist_result["history"]
            
            # Translate
            trans_en = translate_japanese_to_english(
                text=ja_text,
                history=history,
            )
            en_text = trans_en["text"].strip()
            console.print("[bold cyan]🌐 English Translation:[/bold cyan]")
            console.print(f"[bright_white]{en_text}[/bright_white]")
        
        # Build full EN text
        if prev_full_en_text:
            if new_ja_text_res["start_index"] == 0:
                full_en_text = en_text.strip()
            else:
                full_en_text = (prev_full_en_text + "\n" + en_text).strip() if en_text else prev_full_en_text
        else:
            full_en_text = en_text
    else:
        # First segment
        ja_text = full_ja_text
        curr_clean = ja_text.rstrip('.。！？、…・「」『』').rstrip()
        if curr_clean:
            full_trans_en = translate_japanese_to_english(text=ja_text)
            new_ja_sents = full_ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
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
    
    new_en_sents = split_sentences_ja(full_en_text)
    
    # Prefix matching
    prefix_result = fuzzy_match_prefix_texts({
        "prev_ja": prev_full_ja_text,
        "prev_en": prev_full_en_text,
        "full_ja": full_ja_text,
        "full_en": full_en_text,
    })
    ja_text = prefix_result["new_ja"]
    en_text = prefix_result["new_en"]
    
    # Segment directory
    segment_num = get_next_segment_number()
    segment_dir = prepare_segment_directory(
        segment_num,
        segments_dir=last_n_segments_dir,
        segment_index_path=segment_index_path,
        n_results=n_segment_results,
    )
    segment_dir_name = f"segment_{segment_num:03d}"
    console.print(
        f"[info]Segment directory:[/info] [uuid]{segment_dir_name}[/uuid] "
        f"(#{segment_num}, keeping last {n_segment_results})"
    )
    
    # Save segment files
    _save_segment_files(
        segment_dir=segment_dir,
        segment_dir_name=segment_dir_name,
        segment_num=segment_num,
        header=header,
        audio_bytes=audio_bytes,
        audio_np=audio_np,
        full_audio_int16=full_audio_int16,
        sample_rate=sample_rate,
        language=language,
        primary_label=primary_label,
        primary_confidence=primary_confidence,
        speaker_metadata=speaker_metadata,
        speaker_results=speaker_results,
        old_ja_sents=old_ja_sents,
        new_ja_sents=new_ja_sents,
        old_en_sents=old_en_sents,
        new_en_sents=new_en_sents,
        ja_text=ja_text,
        en_text=en_text,
    )
    
    # Save full audio files
    _save_full_audio_files(
        full_audio_dir=live_audio_buffer_dir,
        full_audio_int16=full_audio_int16,
        sample_rate=sample_rate,
        context_buffer=context_buffer,
        full_trans_result={},
        full_metadata=full_metadata,
        full_word_segments=full_word_segments,
        full_word_segments_text=full_word_segments_text,
        full_phrase_segments=full_phrase_segments,
        full_ja_sents=full_ja_sents,
    )
    
    # Add to context buffer
    context_buffer.add_audio_segment(audio_np, {
        "uuid": header["uuid"],
        "forced": header["forced"],
        "vad_reason": header["vad_reason"],
        "start_sec": header["start_sec"],
        "end_sec": header["end_sec"],
        "duration_sec": header["duration_sec"],
        "started_at": header["started_at"],
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
    
    if get_speaker_labeler() and get_speaker_labeler().total_segments_processed % 5 == 0:
        save_speaker_state()
    
    console.print("[bold green]✅ Japanese Response Summary:[/bold green]")
    console.print(f"  UUID: [uuid]{uuid_[-6:]}[/uuid]")
    console.print(f"  Language: [value]{language}[/value]")
    console.print(f"  JA text: [number]{len(ja_text)}[/number] chars")
    console.print(f"  EN text: [number]{len(en_text)}[/number] chars")
    console.print(f"  Speaker: [speaker]{primary_label}[/speaker]")
    
    return {
        "uuid": uuid_,
        "new_duration": header['duration_sec'],
        "context_uuid": context_buffer.get_context_uuid() or uuid_,
        "context_duration": context_buffer.get_total_duration(),
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
            "current_speaker": get_current_speaker(),
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
