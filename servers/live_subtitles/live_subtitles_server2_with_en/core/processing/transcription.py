"""
Transcription pipeline: main entry point, Japanese and non-Japanese processing.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from rich.console import Console

from core.state import (
    get_audio_language_detector,
    get_context_buffer,
    get_current_speaker,
    get_last_n_segments_dir,
    get_live_audio_buffer_dir,
    get_n_segment_results,
    get_segment_index_path,
    get_speaker_diarization,
    get_speaker_labeler,
    save_speaker_state,
)
from core.processing.audio_tagging import perform_audio_tagging
from core.processing.speaker_labeling import (
    perform_speaker_labeling,
    save_segment_audio_for_playback,
    save_segmentation_tests_async,
)
from services.audio_config import SAMPLE_RATE
from services.audio_utils import get_audio_duration
from services.diff_utils import extract_new_ja_text
from services.live_subtitles_server_utils import (
    get_next_segment_number,
    prepare_segment_directory,
)
from services.save_utils import (
    save_full_audio_files,
    save_segment_files,
    save_tagging_to_segment,
)
from services.sentence_matcher_ja import (
    fuzzy_match_prefix_texts,
    fuzzy_shortest_best_match_contains,
)
from services.sentence_utils import split_sentences_ja
from services.transcribe_funasr import transcribe_audio
from services.translate_jp_en_llm_prefixed import translate_japanese_to_english

console = Console()
SPACELESS_LANGUAGES = {"ja", "jpn", "zh", "chi", "zho", "ko", "kor", "th", "tha"}


def should_reset_context(header: dict) -> bool:
    """Determine if we should reset the context buffer based on time gap or silence."""
    return True


def blocking_process_audio(audio_bytes: bytes, header: dict) -> dict:
    """
    Runs in thread pool — contains the blocking CPU/GPU heavy work.
    """
    from core.state import get_current_speaker
    context_buffer = get_context_buffer()
    live_audio_buffer_dir = get_live_audio_buffer_dir()
    last_n_segments_dir = get_last_n_segments_dir()
    segment_index_path = get_segment_index_path()
    n_segment_results = get_n_segment_results()
    uuid_ = header.get("uuid")
    segment_id = header.get("segment_id")
    segment_number = header.get("segment_number")
    if not segment_id:
        console.print("[error]Missing segment ID in header[/error]")
        return {"message": "missing segment_id", "success": False}
    sample_rate = header.get("sample_rate", SAMPLE_RATE)
    language = header.get("language", "auto")
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    console.print("[info]🎵 Performing early audio tagging...[/info]")
    try:
        tagging_events = perform_audio_tagging(
            audio_np=audio_np,
            sample_rate=sample_rate,
            segment_dir=None,
        )
        speech_detected = tagging_events.get("speech_detected", False)
        console.print(
            f"[info]Speech detected: {'✅' if speech_detected else '❌'} "
            f"(probability: {tagging_events.get('max_speech_probability', 0.0):.3f})[/info]"
        )
    except Exception as e:
        console.print(f"[error]Early audio tagging failed: {e}[/error]")
        console.print("[warning]Assuming speech detected to continue processing[/warning]")
        tagging_events = {
            "speech_detected": True,
            "max_speech_probability": 0.0,
            "error": str(e),
            "processing_mode": "failed",
            "top_predictions": [],
        }
        speech_detected = True

    if not speech_detected:
        console.print("[warning]⚠️ No speech detected, skipping further processing[/warning]")
        return {
            "uuid": uuid_,
            "segment_id": segment_id,
            "segment_number": segment_number,
            "new_duration": header.get("duration_sec", 0),
            "context_uuid": uuid_,
            "context_duration": 0,
            "success": False,
            "ja_text": "",
            "en_text": "",
            "language": language,
            "event": "silence",
            "emo": "neutral",
            "transcribed_duration_sec": 0,
            "transcribed_duration_pctg": 0,
            "coverage_label": "no_speech",
            "speaker_label": "SPEAKER_UNKNOWN",
            "speaker_confidence": 0.0,
            "speaker_match_type": "skipped",
            "speakers": [],
            "diarization": {
                "current_speaker": get_current_speaker(),
                "known_speakers": [],
                "speaker_count": 0,
                "speakers_info": {},
                "total_segments_processed": 0,
                "note": "No speech detected - processing skipped",
            },
            "speaker_labeling_performed": False,
            "old_ja_sents": [],
            "new_ja_sents": [],
            "old_en_sents": [],
            "new_en_sents": [],
            "phrase_segments": [],
            "new_ja_similarity": None,
            "new_ja_start_index": None,
            "segment_number": 0,
            "segment_dir": "no_speech",
            "tagging_events": tagging_events,
            "message": "No speech detected, skipping processing",
        }

    if not language or language == "auto":
        console.print("[info]Detecting language with AudioLanguageDetector...[/info]")
        try:
            detector = get_audio_language_detector()
            audio_tensor = torch.from_numpy(audio_np).float() / 32768.0
            audio_tensor = audio_tensor.unsqueeze(0)
            detected_lang = detector.detect_from_bytes(audio_tensor, sample_rate=sample_rate)
            console.print(f"[success]Detected language: {detected_lang}[/success]")
            language = detected_lang
        except Exception as e:
            console.print(f"[error]Language detection failed: {e}. Falling back to 'ja'[/error]")
            language = "ja"

    console.print(f"[info]Transcribing with language: {language}[/info]")
    if should_reset_context(header):
        context_buffer.reset()

    new_audio_duration_sec = get_audio_duration(audio_np, sr=sample_rate)
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
    actual_full_duration_sec = get_audio_duration(full_audio_int16, sr=sample_rate)
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
            tagging_events=tagging_events,
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
            tagging_events=tagging_events,
        )


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
    tagging_events: dict,
) -> dict:
    """Process non-Japanese audio segment."""
    en_text = full_word_segments_text.strip()
    console.print("[bold green]📝 Non-Japanese Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{en_text}[/bright_white]")
    console.print(f"[dim]Language: {language} | Words: {len(full_word_segments)}[/dim]")
    segment_id = header.get("segment_id")
    segment_num = header.get("segment_number")
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
    (
        text_has_sufficient_content,
        speaker_results,
        primary_label,
        primary_confidence,
        speaker_metadata,
    ) = perform_speaker_labeling(
        audio_np,
        sample_rate,
        header,
        full_word_segments_text,
        segment_id=segment_id,
        segment_dir=segment_dir,
    )
    if segment_id:
        save_segment_audio_for_playback(
            audio_np=audio_np,
            segment_id=segment_id,
            sample_rate=sample_rate,
            metadata={
                "segment_number": segment_num,
                "speaker_label": primary_label,
                "timestamp": header.get("start_sec", time.time()),
                "language": language,
                "text": en_text[:100] if en_text else "",
            }
        )
    if segment_dir and tagging_events:
        try:
            save_tagging_to_segment(
                segment_dir,
                tagging_events,
                tagging_events.get("speech_detected", False),
                tagging_events.get("max_speech_probability", 0.0),
            )
        except Exception as e:
            console.print(f"[warning]Could not save tagging to segment: {e}[/warning]")
    save_segment_files(
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
        tagging_events=tagging_events,
    )
    save_segmentation_tests_async(
        segment_dir=segment_dir,
        audio_np=full_audio_int16,
        sample_rate=sample_rate,
    )
    save_full_audio_files(
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
    context_buffer.add_audio_segment(
        audio_np,
        {
            "uuid": header["uuid"],
            "segment_id": segment_id,
            "segment_number": segment_num,
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
            "tagging_events": tagging_events,
        },
    )
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
        "segment_id": segment_id,
        "segment_number": segment_num,
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
        "diarization": get_speaker_diarization()
        if text_has_sufficient_content
        else {
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
        "tagging_events": tagging_events,
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
    tagging_events: dict,
) -> dict:
    """Process Japanese audio segment with translation."""
    full_ja_text = full_word_segments_text
    full_ja_sents = split_sentences_ja(full_ja_text)
    console.print("[bold green]📝 Japanese Transcribed Text:[/bold green]")
    console.print(f"[bright_white]{full_ja_text}[/bright_white]")
    console.print(f"[dim]Sentences: {len(full_ja_sents)}[/dim]")
    segment_id = header.get("segment_id")
    segment_num = header.get("segment_number")
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
    (
        text_has_sufficient_content,
        speaker_results,
        primary_label,
        primary_confidence,
        speaker_metadata,
    ) = perform_speaker_labeling(
        audio_np,
        sample_rate,
        header,
        full_word_segments_text,
        segment_id=segment_id,
        segment_dir=segment_dir,
    )
    if segment_dir and tagging_events:
        try:
            save_tagging_to_segment(
                segment_dir,
                tagging_events,
                tagging_events.get("speech_detected", False),
                tagging_events.get("max_speech_probability", 0.0),
            )
        except Exception as e:
            console.print(f"[warning]Could not save tagging to segment: {e}[/warning]")

    prev_full_ja_text = None
    prev_full_en_text = None
    new_ja_text = full_ja_text
    new_ja_start_index = None
    new_ja_similarity = None
    old_ja_sents = []
    old_en_sents = []
    new_ja_sents = full_ja_sents
    new_en_sents = []
    ja_text = full_ja_text
    en_text = ""
    full_en_text = ""
    new_audio_duration_sec = get_audio_duration(audio_np, sr=sample_rate)

    if context_buffer.segments:
        _, last_meta = context_buffer.get_last_segment()
        prev_full_ja_text = last_meta.get("full_ja_text", "")
        prev_full_en_text = last_meta.get("full_en_text", "")
        new_ja_text_res = extract_new_ja_text(prev_full_ja_text, full_ja_text)
        new_ja_text = new_ja_text_res["new_text"]
        new_ja_start_index = new_ja_text_res["start_index"]
        new_ja_similarity = new_ja_text_res["similarity"]
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
        new_clean = new_text.rstrip(".。！？、…・「」『』").rstrip()
        if not new_clean:
            if segment_id:
                save_segment_audio_for_playback(
                    audio_np=audio_np,
                    segment_id=segment_id,
                    sample_rate=sample_rate,
                    metadata={
                        "segment_number": segment_num,
                        "speaker_label": primary_label,
                        "timestamp": header.get("start_sec", time.time()),
                        "language": language,
                        "text_ja": "",
                        "text_en": "",
                        "note": "Same text as previous segment",
                    }
                )
            return {
                "uuid": uuid_,
                "segment_id": segment_id,
                "segment_number": segment_num,
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
            hist_result = context_buffer.get_context_history_by_duration(
                max_duration_sec=context_buffer.max_duration_sec,
                reserved_duration_sec=new_audio_duration_sec,
            )
            history = hist_result["history"]
            trans_en = translate_japanese_to_english(
                text=ja_text,
                history=history,
            )
            en_text = trans_en["text"].strip()
            console.print("[bold cyan]🌐 English Translation:[/bold cyan]")
            console.print(f"[bright_white]{en_text}[/bright_white]")
        if prev_full_en_text:
            if new_ja_text_res["start_index"] == 0:
                full_en_text = en_text.strip()
            else:
                full_en_text = (
                    (prev_full_en_text + "\n" + en_text).strip()
                    if en_text
                    else prev_full_en_text
                )
        else:
            full_en_text = en_text
    else:
        ja_text = full_ja_text
        curr_clean = ja_text.rstrip(".。！？、…・「」『』").rstrip()
        if curr_clean:
            full_trans_en = translate_japanese_to_english(text=ja_text)
            new_ja_sents = full_ja_sents
            full_en_text = full_trans_en["text"].strip()
            en_text = full_en_text
            console.print("[bold cyan]🌐 English Translation (First Segment):[/bold cyan]")
            console.print(f"[bright_white]{en_text}[/bright_white]")
        else:
            if segment_id:
                save_segment_audio_for_playback(
                    audio_np=audio_np,
                    segment_id=segment_id,
                    sample_rate=sample_rate,
                    metadata={
                        "segment_number": segment_num,
                        "speaker_label": primary_label,
                        "timestamp": header.get("start_sec", time.time()),
                        "language": language,
                        "text_ja": "",
                        "text_en": "",
                        "note": "Empty transcription after cleaning",
                    }
                )
            return {
                "uuid": uuid_,
                "segment_id": segment_id,
                "segment_number": segment_num,
                "ja_text": "",
                "en_text": "",
                "speaker_label": primary_label,
                "speaker_confidence": primary_confidence,
                "success": False,
                "message": "Empty transcription after cleaning",
            }

    new_en_sents = split_sentences_ja(full_en_text)
    prefix_result = fuzzy_match_prefix_texts(
        {
            "prev_ja": prev_full_ja_text,
            "prev_en": prev_full_en_text,
            "full_ja": full_ja_text,
            "full_en": full_en_text,
        }
    )
    ja_text = prefix_result["new_ja"]
    en_text = prefix_result["new_en"]

    if segment_id:
        save_segment_audio_for_playback(
            audio_np=audio_np,
            segment_id=segment_id,
            sample_rate=sample_rate,
            metadata={
                "segment_number": segment_num,
                "speaker_label": primary_label,
                "timestamp": header.get("start_sec", time.time()),
                "language": language,
                "text_ja": ja_text[:100] if ja_text else "",
                "text_en": en_text[:100] if en_text else "",
            }
        )

    save_segment_files(
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
        tagging_events=tagging_events,
    )
    save_segmentation_tests_async(
        segment_dir=segment_dir,
        audio_np=full_audio_int16,
        sample_rate=sample_rate,
    )
    save_full_audio_files(
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
    context_buffer.add_audio_segment(
        audio_np,
        {
            "uuid": header["uuid"],
            "segment_id": segment_id,
            "segment_number": segment_num,
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
            "tagging_events": tagging_events,
        },
    )
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
        "segment_id": segment_id,
        "segment_number": segment_num,
        "new_duration": header["duration_sec"],
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
        "diarization": get_speaker_diarization()
        if text_has_sufficient_content
        else {
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
        "tagging_events": tagging_events,
    }
