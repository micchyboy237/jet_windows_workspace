import os
import fnmatch
import argparse
import subprocess
import json
import tiktoken
from rich.console import Console
from tqdm import tqdm
from _utils_copy_for_prompt import (
    find_files,
    format_file_structure,
    clean_newlines,
    clean_content,
    remove_parent_paths,
    copy_to_clipboard,
)
from headroom import compress

logger = Console()

exclude_files = [
    "**/.git/",
    "**/.gitignore",
    "**/.DS_Store",
    "**/*.pyc",
    "**/_copy*.py",
    "**/__pycache__/",
    "**/.pytest_cache/",
    "**/node_modules/",
    "**/*lock.json",
    "**/*.lock",
    "**/public/",
    "**/mocks/",
    "**/.venv/",
    "**/dream/",
    "**/jupyter/",
    "**/*.png",
    "**/*.svg",
    # "**/_*",
    # "**/.cache/",
    "**/_git_stats.json",
    "**/stats_results/",
    # "**/generated/",
    # "**/.*",

    # Custom
    # "**/*.sh"
    # "**/__init__.py",
    # "*.md",
]
include_files = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Examples\.vscode\launch.json",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\main.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\websocket.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger_types.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\speaker_reference.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\segment_types.py",
    r"",
]

structure_include = [
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers",
]
structure_exclude = []

include_content = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\tagger",
]
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

COMPRESSION_MODEL = "gpt-4o"
TOKEN_BUDGET = 8000

DEFAULT_QUERY_MESSAGE = r"""
Where can we call extract_high_confidence_speech_segments or tag_audio_segments to improve intra and inter speakers?

# C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py

def label_speakers_for_segment(
    waveform: np.ndarray,
    sample_rate: int,
    timestamp: Optional[float] = None,
    return_multiple: bool = True,
    segment_id: Optional[str] = None,
) -> tuple:
    \"\"\"Label speakers for an audio segment using the progressive labeler.\"\"\"
    if waveform.size == 0:
        empty_result = [
            {
                "label": "SPEAKER_UNKNOWN",
                "confidence": 0.0,
                "match_type": "empty_waveform",
                "is_primary": True,
                "is_new_speaker": False,
            }
        ]
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
            timestamp - last_change_time if last_change_time > 0 else float("inf")
        ),
        "segment_duration": len(waveform) / sample_rate,
    }

    if return_multiple:
        # label_segments now returns List[SegmentGroup]
        segment_groups = labeler.label_segments(
            waveform=waveform_tensor,
            sample_rate=sample_rate,
            timestamp=timestamp,
            context=context,
            segment_id=segment_id,
        )
        # Extract matches from the latest segment group
        latest_group = segment_groups[-1] if segment_groups else None
        speaker_results = latest_group["matches"] if latest_group else []
        
        primary = (
            speaker_results[0]
            if speaker_results
            else {
                "label": "SPEAKER_UNKNOWN",
                "confidence": 0.0,
                "match_type": "unknown",
                "is_primary": True,
                "is_new_speaker": False,
            }
        )
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
            segment_id=segment_id,
        )
        primary_label = label
        primary_confidence = confidence
        speaker_results = [
            {
                "label": label,
                "confidence": confidence,
                "match_type": metadata.get("match_type", "unknown"),
                "is_primary": True,
                "is_new_speaker": metadata.get("is_new_speaker", False),
            }
        ]
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
    \"\"\"
    Runs in thread pool — contains the blocking CPU/GPU heavy work.
    Handles both Japanese and non-Japanese audio processing with
    speaker labeling, translation, and file persistence.
    \"\"\"
    from core.state import (
        get_current_speaker,
    )

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

    # ===== STEP 1: Audio Tagging (moved here before transcription) =====
    console.print("[info]🎵 Performing early audio tagging...[/info]")
    try:
        tagging_events = perform_audio_tagging(
            audio_np=audio_np,
            sample_rate=sample_rate,
            segment_dir=None,  # No segment dir yet, we'll save later
        )
        speech_detected = tagging_events.get("speech_detected", False)
        console.print(
            f"[info]Speech detected: {'✅' if speech_detected else '❌'} "
            f"(probability: {tagging_events.get('max_speech_probability', 0.0):.3f})[/info]"
        )
    except Exception as e:
        console.print(f"[error]Early audio tagging failed: {e}[/error]")
        console.print(
            "[warning]Assuming speech detected to continue processing[/warning]"
        )
        tagging_events = {
            "speech_detected": True,  # Default to True on failure to avoid skipping
            "max_speech_probability": 0.0,
            "error": str(e),
            "processing_mode": "failed",
            "top_predictions": [],
        }
        speech_detected = True

    # ===== STEP 2: Check if speech was detected =====
    if not speech_detected:
        console.print(
            "[warning]⚠️ No speech detected, skipping further processing[/warning]"
        )
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

    # ===== STEP 3: Language Detection =====
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
            console.print(
                f"[error]Language detection failed: {e}. Falling back to 'ja'[/error]"
            )
            language = "ja"

    console.print(f"[info]Transcribing with language: {language}[/info]")

    # ===== STEP 4: Context Buffer Management =====
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
    console.print(
        f"[info]New Audio Duration:[/info] [time]{header['duration_sec']:.2f}s[/time]"
    )
    console.print(
        f"[info]Full Duration:[/info] "
        f"[time]{actual_full_duration_sec:.2f}s[/time] / [time]{max_duration_sec:.2f}s[/time] max"
    )

    # ===== STEP 5: Transcription =====
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

    # ===== STEP 6: Route to language-specific processor =====
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
            tagging_events=tagging_events,  # Pass pre-computed tagging events
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
            tagging_events=tagging_events,  # Pass pre-computed tagging events
        )


def _perform_speaker_labeling(
    audio_np: np.ndarray,
    sample_rate: int,
    header: dict,
    full_word_segments_text: str,
    segment_id: Optional[str] = None,
) -> tuple:
    \"\"\"Perform speaker labeling if text content is sufficient.\"\"\"
    text_has_sufficient_content = should_label_speaker(
        full_word_segments_text, min_chars=2
    )
    speaker_results = []
    primary_label = None
    primary_confidence = 0.0
    speaker_metadata = {"match_type": "skipped_no_text"}

    if text_has_sufficient_content:
        segment_timestamp = header.get("start_sec", time.time())
        segment_duration = header.get(
            "duration_sec", 
            get_audio_duration(audio_np, sr=sample_rate)
        )
        use_multiple = segment_duration >= 3.0
        speaker_results, primary_label, primary_confidence, speaker_metadata = (
            label_speakers_for_segment(
                waveform=audio_np,
                sample_rate=sample_rate,
                timestamp=segment_timestamp,
                return_multiple=use_multiple,
                segment_id=segment_id,
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

    return (
        text_has_sufficient_content,
        speaker_results,
        primary_label,
        primary_confidence,
        speaker_metadata,
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
    \"\"\"Process non-Japanese audio segment.\"\"\"
    from services.audio_utils import get_audio_duration
    
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
    ) = _perform_speaker_labeling(
        audio_np,
        sample_rate,
        header,
        full_word_segments_text,
        segment_id=segment_id,
    )

    # Save audio permanently for segment detail playback
    # ✅ All variables (en_text, primary_label, segment_num) are now defined
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

    # Save tagging events to segment directory
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

    if (
        get_speaker_labeler()
        and get_speaker_labeler().total_segments_processed % 5 == 0
    ):
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
    \"\"\"Process Japanese audio segment with translation.\"\"\"
    from services.audio_utils import get_audio_duration
    
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
    ) = _perform_speaker_labeling(
        audio_np,
        sample_rate,
        header,
        full_word_segments_text,
        segment_id=segment_id,
    )

    # Save tagging events to segment directory
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
    history = None
    old_ja_sents = []
    old_en_sents = []
    new_ja_sents = full_ja_sents
    new_en_sents = []
    ja_text = full_ja_text
    en_text = ""
    full_en_text = ""

    # ✅ Use utility function instead of manual calculation
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
            console.print(
                f"[warning]Fuzzy match too weak (score={match_result['score']:.1f}).[/warning]"
            )
            new_text = full_ja_text.strip()

        new_clean = new_text.rstrip(".。！？、…・「」『』").rstrip()
        if not new_clean:
            # ✅ Save audio even for empty returns (early exit with audio saved)
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
            console.print(
                "[bold cyan]🌐 English Translation (First Segment):[/bold cyan]"
            )
            console.print(f"[bright_white]{en_text}[/bright_white]")
        else:
            # ✅ Save audio even for empty returns (early exit with audio saved)
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

    # ✅ Save audio permanently for segment detail playback
    # All variables (ja_text, en_text, primary_label) are now properly defined
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

    if (
        get_speaker_labeler()
        and get_speaker_labeler().total_segments_processed % 5 == 0
    ):
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


def perform_audio_tagging(
    audio_np: np.ndarray,
    sample_rate: int,
    segment_dir: Optional[Path] = None,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap_duration: float = DEFAULT_CHUNK_OVERLAP,
    speech_prob_threshold: float = DEFAULT_SPEECH_PROB_THRESHOLD,
    min_speech_duration: float = 2.0,  # NEW parameter
) -> Dict[str, Any]:
    \"\"\"
    Perform audio tagging on an audio segment and save results.
    
    UPDATED: Now returns speech_duration for minimum duration threshold checks.
    
    Args:
        audio_np: Audio samples as int16 numpy array
        sample_rate: Sample rate in Hz
        segment_dir: Optional directory to save results
        chunk_duration: Duration of each chunk for long audio
        overlap_duration: Overlap between chunks
        min_speech_duration: Minimum speech duration in seconds 
                            (used for speech_detected logic)
    
    Returns:
        Dictionary with speech_detected, speech_duration, 
        max_speech_probability, and detailed predictions
    \"\"\"
    console.print("[info]🎵 Starting audio tagging...[/info]")
    console.print(
        f"[info]Audio shape: {audio_np.shape}, Sample rate: {sample_rate}[/info]"
    )
    audio_duration = get_audio_duration(audio_np, sr=sample_rate)
    console.print(f"[info]Audio duration: {audio_duration:.2f}s[/info]")

    try:
        # Get tagger singleton
        tagger = get_audio_tagger()

        # Convert to float32 for the tagger
        audio_float = audio_np.astype(np.float32) / 32768.0

        console.print(
            f"[info]Using chunked processing "
            f"(audio {audio_duration:.2f}s > {chunk_duration * 2:.1f}s)[/info]"
        )
        chunked_summary = tagger.tag_audio_chunks(
            audio=audio_float,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
        )
        
        # ── UPDATED: Include speech_duration in results ──────────────
        speech_duration = chunked_summary.get("speech_duration", 0.0)
        avg_speech_prob = chunked_summary.get("avg_speech_probability", 0.0)
        max_speech_prob = chunked_summary.get("max_speech_probability", 0.0)
        
        # Speech detected only if BOTH:
        # 1. Probability threshold met (from chunked summary)
        # 2. Speech duration >= minimum (NEW)
        speech_detected_by_prob = chunked_summary.get("speech_detected", False)
        speech_detected_by_duration = speech_duration >= min_speech_duration
        
        has_speech = speech_detected_by_prob and speech_detected_by_duration
        
        console.print(
            f"[info]Average speech probability: {avg_speech_prob:.3f}[/info]"
        )
        console.print(
            f"[info]Max speech probability: {max_speech_prob:.3f} "
            f"(detected: {'✅' if speech_detected_by_prob else '❌'})[/info]"
        )
        console.print(
            f"[info]Speech duration: {speech_duration:.2f}s "
            f"(≥{min_speech_duration:.1f}s: {'✅' if speech_detected_by_duration else '❌'})[/info]"
        )
        console.print(
            f"[info]Final speech decision: {'✅' if has_speech else '❌'}[/info]"
        )

        tagging_results = {
            "speech_detected": has_speech,
            "speech_duration": round(speech_duration, 4),  # NEW FIELD
            "avg_speech_probability": round(avg_speech_prob, 4),
            "max_speech_probability": round(max_speech_prob, 4),
            "speech_prob_threshold": speech_prob_threshold,
            "min_speech_duration_threshold": min_speech_duration,  # For reference
            "speech_detected_by_prob": speech_detected_by_prob,
            "speech_detected_by_duration": speech_detected_by_duration,
            "overall_top_predictions": chunked_summary["overall_top_predictions"],
            "total_chunks": chunked_summary["total_chunks"],
            "chunk_duration": chunked_summary["chunk_duration"],
            "processing_mode": "chunked",
            "chunks": [
                {
                    "chunk_index": chunk["chunk_index"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "speech_detected": chunk.get("speech_detected", False),
                    "speech_probability": chunk.get("speech_probability", 0.0),
                    "predictions": chunk["predictions"][:3],
                }
                for chunk in chunked_summary["chunks"]
            ],
            "processing_time": chunked_summary["total_processing_time"],
            "real_time_factor": chunked_summary["real_time_factor"],
        }

        # ── Display results ──────────────────────────────────────────────
        console.print("[bold green]🎵 Audio Tagging Results:[/bold green]")
        console.print(
            f"  Speech detected (prob): {'✅' if speech_detected_by_prob else '❌'} "
            f"(prob: {max_speech_prob:.3f})"
        )
        console.print(
            f"  Speech duration: {speech_duration:.2f}s "
            f"(≥{min_speech_duration:.1f}s: {'✅' if speech_detected_by_duration else '❌'})"
        )
        console.print(
            f"  Final decision: {'✅ SPEECH' if has_speech else '❌ NO SPEECH / TOO BRIEF'}"
        )
        
        top_preds = tagging_results.get(
            "overall_top_predictions"
        ) or tagging_results.get("top_predictions", [])
        for pred in top_preds[:3]:
            console.print(f"  - {pred['name']}: {pred['prob']:.3f}")

        if segment_dir:
            save_tagging_to_segment(
                segment_dir, tagging_results, has_speech, max_speech_prob
            )

        return tagging_results

    except Exception as e:
        console.print(f"[error]Audio tagging failed: {e}[/error]")
        console.print("[warning]Continuing without audio tagging results[/warning]")
        return {
            "speech_detected": False,
            "speech_duration": 0.0,  # NEW FIELD
            "avg_speech_probability": 0.0,
            "max_speech_probability": 0.0,
            "speech_prob_threshold": speech_prob_threshold,
            "min_speech_duration_threshold": min_speech_duration,
            "error": str(e),
            "processing_mode": "failed",
            "top_predictions": [],
        }


def save_segment_audio_for_playback(
    audio_np: np.ndarray,
    segment_id: str,
    sample_rate: int = SAMPLE_RATE,
    metadata: Optional[Dict] = None,
) -> Optional[Path]:
    \"\"\"
    Save segment audio as WAV file permanently for playback in segment detail page.
    Organizes by segment_id so it can always be found.
    
    Args:
        audio_np: Audio samples as int16 numpy array
        segment_id: Unique segment identifier (UUID)
        sample_rate: Sample rate in Hz (defaults to SAMPLE_RATE from services.audio_config)
        metadata: Optional metadata to store alongside audio
        
    Returns:
        Path to saved audio file, or None if failed
    \"\"\"
    import wave
    import json
    from services.config import SEGMENT_AUDIO_DIR, SEGMENT_AUDIO_INDEX
    
    if audio_np.size == 0:
        console.print(f"[warning]Cannot save empty audio for segment {segment_id}[/]")
        return None
    
    try:
        # ✅ Ensure directory exists (handles race conditions and deleted directories)
        SEGMENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create audio file path using segment_id
        audio_path = SEGMENT_AUDIO_DIR / f"{segment_id}.wav"
        
        # Convert to int16 if needed
        if audio_np.dtype != np.int16:
            if audio_np.dtype == np.float64 or audio_np.dtype == np.float32:
                # Convert from float to int16
                audio_int16 = (np.clip(audio_np.astype(np.float64), -1.0, 1.0) * 32767).astype(np.int16)
            else:
                audio_int16 = audio_np.astype(np.int16)
        else:
            audio_int16 = audio_np
        
        # Write WAV file
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        # Use utility for duration
        duration_sec = get_audio_duration(audio_int16, sr=sample_rate)
        console.print(
            f"[success]Saved segment audio: {segment_id}.wav "
            f"({len(audio_int16)} samples, {duration_sec:.2f}s) → {audio_path}[/]"
        )
        
        # Update audio index
        audio_index = {}
        if SEGMENT_AUDIO_INDEX.exists():
            try:
                with open(SEGMENT_AUDIO_INDEX, 'r') as f:
                    audio_index = json.load(f)
            except Exception as e:
                console.print(f"[warning]Could not read audio index: {e}, starting fresh[/]")
        
        audio_index[segment_id] = {
            "file": f"{segment_id}.wav",
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sample_rate,
            "samples": len(audio_int16),
            "saved_at": time.time(),
            "metadata": metadata or {},
        }
        
        # Keep only last 500 entries to prevent index from growing too large
        if len(audio_index) > 500:
            # Sort by saved_at and keep newest 500
            sorted_items = sorted(
                audio_index.items(), 
                key=lambda x: x[1].get('saved_at', 0), 
                reverse=True
            )[:500]
            old_count = len(audio_index) - len(sorted_items)
            audio_index = dict(sorted_items)
            
            # Clean up old files not in index
            console.print(f"[dim]Cleaning up {old_count} old audio files...[/]")
            for audio_file in SEGMENT_AUDIO_DIR.glob("*.wav"):
                if audio_file.stem not in audio_index:
                    try:
                        audio_file.unlink()
                        console.print(f"[dim]Cleaned up old audio: {audio_file.name}[/]")
                    except Exception as e:
                        console.print(f"[warning]Could not delete old audio {audio_file.name}: {e}[/]")
        
        # Write index atomically using temp file
        import tempfile
        index_dir = SEGMENT_AUDIO_INDEX.parent
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            dir=index_dir,
            delete=False
        ) as tmp:
            json.dump(audio_index, tmp, indent=2)
            tmp_path = Path(tmp.name)
        
        # Atomic rename
        tmp_path.replace(SEGMENT_AUDIO_INDEX)
        
        return audio_path
        
    except Exception as e:
        console.print(f"[error]Failed to save segment audio for {segment_id}: {e}[/]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        return None




# C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger.py

class AudioTagger:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = AUDIO_TAGGING_MODEL,
        labels_path: Optional[Union[str, Path]] = CLASS_LABELS_INDICES_CSV,
        top_k: int = DEFAULT_TOP_K,
        num_threads: int = DEFAULT_NUM_THREADS,
        provider: str = DEFAULT_PROVIDER,
        debug: bool = False,
        speech_prob_threshold: Optional[float] = None,
        speech_top_n: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        chunk_overlap: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
    ) -> None:
        \"\"\"
        Initialize the AudioTagger with model configuration.
        Args:
            model_path: Path to ONNX model file
            labels_path: Path to class labels CSV
            top_k: Number of top predictions to return
            num_threads: Number of CPU threads
            provider: Computation provider ("cpu", "cuda", etc.)
            debug: Enable debug logging for Sherpa-ONNX
            speech_prob_threshold: Minimum speech probability (default: 0.5)
            speech_top_n: Check the top N predictions for speech classes (default: 3)
            chunk_duration: Default chunk duration in seconds (default: 1.0s)
            chunk_overlap: Default overlap between chunks in seconds (default: 0.5s)
            min_chunk_duration: Minimum valid chunk duration (default: 0.5s)
        \"\"\"
        self.model_path: Path = (
            Path(model_path) if model_path else DEFAULT_MODEL_PATH
        )
        self.labels_path: Path = (
            Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        )
        self.top_k: int = top_k
        self.num_threads: int = num_threads
        self.provider: str = provider
        self.debug: bool = debug
        
        # Set speech detection parameters with validation
        self.speech_prob_threshold: float = (
            speech_prob_threshold
            if speech_prob_threshold is not None
            else DEFAULT_SPEECH_PROB_THRESHOLD
        )
        self.speech_top_n: int = (
            speech_top_n if speech_top_n is not None else DEFAULT_SPEECH_TOP_N
        )
        
        # Validate speech threshold - prevent overly low values that cause false positives
        if self.speech_prob_threshold < DEFAULT_MIN_SPEECH_PROB_THRESHOLD:
            console.print(
                f"[yellow]⚠ Speech probability threshold {self.speech_prob_threshold} "
                f"is below minimum valid value {DEFAULT_MIN_SPEECH_PROB_THRESHOLD}. "
                f"Using {DEFAULT_SPEECH_PROB_THRESHOLD} to prevent false positives.[/yellow]"
            )
            self.speech_prob_threshold = DEFAULT_SPEECH_PROB_THRESHOLD
        
        self.chunk_duration: float = (
            chunk_duration
            if chunk_duration is not None
            else DEFAULT_CHUNK_DURATION
        )
        self.chunk_overlap: float = (
            chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        )
        self.min_chunk_duration: float = (
            min_chunk_duration
            if min_chunk_duration is not None
            else DEFAULT_MIN_CHUNK_DURATION
        )
        
        self._validate_chunking_config()
        self._tagger: Optional[sherpa_onnx.AudioTagging] = None
        self._labels_map: Optional[Dict[int, str]] = None
        
        console.print(
            Panel.fit(
                f"[bold green]AudioTagger Initialized[/bold green]\n"
                f"Model: {linkify(str(self.model_path))}\n"
                f"Labels: {linkify(str(self.labels_path))}\n"
                f"Speech Threshold: {self.speech_prob_threshold}\n"
                f"Speech Top N: {self.speech_top_n}\n"
                f"Chunk Duration: {self.chunk_duration}s\n"
                f"Chunk Overlap: {self.chunk_overlap}s\n"
                f"Min Chunk Duration: {self.min_chunk_duration}s",
                title="AudioTagger Configuration",
                border_style="blue",
            )
        )

    def tag_audio_chunks(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
    ) -> AudioChunksTaggingSummary:
        \"\"\"
        Process long audio by splitting into overlapping chunks and tagging each.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: 16000)
            chunk_duration: Duration of each chunk in seconds.
            overlap_duration: Overlap between chunks in seconds.
            min_chunk_duration: Minimum duration for the last chunk.

        Returns:
            AudioChunksTaggingSummary with per-chunk results, overall aggregation,
            speech_duration, and avg_speech_probability
        \"\"\"
        _chunk_dur = chunk_duration if chunk_duration is not None else self.chunk_duration
        _overlap = overlap_duration if overlap_duration is not None else self.chunk_overlap
        _min_chunk = (
            min_chunk_duration if min_chunk_duration is not None else self.min_chunk_duration
        )

        if _chunk_dur < _min_chunk:
            console.print(
                f"[yellow]⚠ Chunk duration {_chunk_dur}s < min {_min_chunk}s, "
                f"using min value[/yellow]"
            )
            _chunk_dur = _min_chunk

        if _overlap >= _chunk_dur:
            console.print(
                f"[yellow]⚠ Overlap {_overlap}s >= chunk duration {_chunk_dur}s, "
                f"using half chunk duration[/yellow]"
            )
            _overlap = _chunk_dur / 2.0

        overall_start = time.time()
        try:
            waveform, actual_sr = load_audio(
                audio, sr=sample_rate or SAMPLE_RATE, mono=True
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to load audio: {e}[/red]")
            raise

        total_samples = len(waveform)
        total_duration = total_samples / actual_sr
        console.print(
            f"[dim]📊 Audio loaded: {total_duration:.2f}s, "
            f"{actual_sr}Hz, {total_samples} samples[/dim]"
        )

        if isinstance(audio, (str, Path)):
            audio_path_str = str(audio)
        elif isinstance(audio, bytes):
            audio_path_str = f"bytes_input_{len(audio)}bytes"
        else:
            audio_path_str = f"array_input_{waveform.shape}"

        chunk_samples = int(_chunk_dur * actual_sr)
        hop_samples = int((_chunk_dur - _overlap) * actual_sr)
        if hop_samples < 1:
            hop_samples = 1

        console.print(
            f"[dim]🔧 Chunk config: {_chunk_dur}s chunks, "
            f"{_overlap}s overlap, hop={hop_samples} samples[/dim]"
        )

        chunk_positions = self._calculate_chunk_positions(
            total_samples=total_samples,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            min_chunk_duration=_min_chunk,
            sample_rate=actual_sr,
        )
        console.print(f"[dim]📏 Calculated {len(chunk_positions)} chunk positions[/dim]")

        if not chunk_positions:
            elapsed = time.time() - overall_start
            console.print("[yellow]⚠ No valid chunk positions found[/yellow]")
            return AudioChunksTaggingSummary(
                audio_path=audio_path_str,
                total_duration=total_duration,
                sample_rate=actual_sr,
                chunk_duration=_chunk_dur,
                overlap_duration=_overlap,
                total_chunks=0,
                chunks=[],
                overall_top_predictions=[],
                total_processing_time=elapsed,
                real_time_factor=elapsed / total_duration if total_duration > 0 else 0.0,
                speech_duration=0.0,
                speech_detected=False,
                max_speech_probability=0.0,
                avg_speech_probability=0.0,
            )

        chunks: List[ChunkTaggingResult] = []
        all_predictions: Dict[str, List[float]] = {}
        any_speech_detected = False
        global_max_speech_prob = 0.0
        speech_probabilities: List[float] = []

        for idx, (start_sample, end_sample) in enumerate(chunk_positions):
            chunk_start_time = time.time()
            start_sec = start_sample / actual_sr
            end_sec = end_sample / actual_sr

            console.print(
                f"[dim]🔍 Processing chunk {idx + 1}/{len(chunk_positions)}: "
                f"{start_sec:.2f}s - {end_sec:.2f}s[/dim]"
            )

            chunk_waveform = waveform[start_sample:end_sample].copy()

            try:
                chunk_predictions = self._tag_waveform(chunk_waveform, actual_sr)
                console.print(
                    f"[dim]   ✅ Tagged successfully: "
                    f"{len(chunk_predictions)} predictions[/dim]"
                )
            except Exception as e:
                console.print(f"[red]   ❌ Tagging failed: {e}[/red]")
                chunk_predictions = []

            speech_detected, chunk_speech_prob = self._chunk_has_speech(chunk_predictions)

            if speech_detected:
                any_speech_detected = True
                speech_probabilities.append(chunk_speech_prob)
                console.print(
                    f"[green]   🎤 Speech detected! "
                    f"speech_prob={chunk_speech_prob:.4f}[/green]"
                )
            else:
                console.print(
                    f"[dim]   🔇 No speech detected "
                    f"(speech_prob={chunk_speech_prob:.4f})[/dim]"
                )

            if chunk_speech_prob > global_max_speech_prob:
                global_max_speech_prob = chunk_speech_prob

            chunk_elapsed = time.time() - chunk_start_time

            for pred in chunk_predictions:
                name = pred["name"]
                if name not in all_predictions:
                    all_predictions[name] = []
                all_predictions[name].append(pred["prob"])

            chunk_result = ChunkTaggingResult(
                chunk_index=idx,
                start_time=round(start_sec, 3),
                end_time=round(end_sec, 3),
                duration=round(end_sec - start_sec, 3),
                predictions=chunk_predictions,
                processing_time=round(chunk_elapsed, 4),
                speech_detected=speech_detected,
                speech_probability=round(chunk_speech_prob, 4),
            )
            chunks.append(chunk_result)

        speech_duration = self._calculate_speech_duration(chunks, _overlap)

        if speech_probabilities:
            avg_speech_prob = float(np.mean(speech_probabilities))
            console.print(
                f"[dim]📊 Avg speech probability: {avg_speech_prob:.4f} "
                f"(from {len(speech_probabilities)} speech chunks)[/dim]"
            )
        else:
            avg_speech_prob = 0.0
            console.print("[dim]📊 No speech chunks for avg calculation[/dim]")

        overall_top = self._aggregate_chunk_predictions(all_predictions, self.top_k)
        total_elapsed = time.time() - overall_start
        rtf = total_elapsed / total_duration if total_duration > 0 else 0.0

        console.print(
            f"[dim]⏱ Total processing: {total_elapsed:.2f}s, "
            f"RTF: {rtf:.3f}x[/dim]"
        )

        summary = AudioChunksTaggingSummary(
            audio_path=audio_path_str,
            total_duration=round(total_duration, 3),
            sample_rate=actual_sr,
            chunk_duration=_chunk_dur,
            overlap_duration=_overlap,
            total_chunks=len(chunks),
            chunks=chunks,
            overall_top_predictions=overall_top,
            total_processing_time=round(total_elapsed, 4),
            real_time_factor=round(rtf, 4),
            speech_duration=round(speech_duration, 3),
            speech_detected=any_speech_detected,
            max_speech_probability=round(global_max_speech_prob, 4),
            avg_speech_probability=round(avg_speech_prob, 4),
        )
        return summary

    def tag_audio_segments(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        speech_threshold: Optional[float] = None,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_DURATION_SEC,
        resolution_ms: float = DEFAULT_RESOLUTION_MS,
        include_non_speech: bool = False,
    ) -> AudioSegmentsResult:
        \"\"\"
        Tag audio by splitting into chunks, detecting speech, and identifying
        continuous speech/non-speech segments.
        This combines tag_audio_chunks() with timeline-based segment detection
        into a single call that returns structured segment data without writing
        to disk (use save_speech_segments() for persistence).
        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor).
            sample_rate: Sample rate for raw audio data (default: SAMPLE_RATE).
            chunk_duration: Duration of each analysis chunk in seconds.
            overlap_duration: Overlap between consecutive chunks.
            min_chunk_duration: Minimum duration for the last chunk.
            speech_threshold: Speech probability threshold (default: self.speech_prob_threshold).
            min_silence_duration_sec: Continuous non-speech gap to close a segment (default: 1.0s).
            min_speech_duration_sec: Minimum duration for a valid speech segment (default: 1.0s).
            resolution_ms: Timeline resolution in ms (default: HOP_STEP_MS).
            include_non_speech: If True, also detect non-speech segments (default: False).
        Returns:
            AudioSegmentsResult with chunks, speech_segments, non_speech_segments,
            and aggregate statistics.
        Example:
            >>> tagger = AudioTagger()
            >>> result = tagger.tag_audio_segments("recording.wav", min_silence_duration_sec=1.0)
            >>> for seg in result["speech_segments"]:
            ...     print(f"Speech: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
        Debug logs trace:
            - Chunk tagging progress (from tag_audio_chunks)
            - Timeline building statistics
            - Speech/non-speech transition detection
            - Segment count and duration summary
        \"\"\"
        _speech_threshold = speech_threshold if speech_threshold is not None else self.speech_prob_threshold
        if _speech_threshold <= 0.0 or _speech_threshold > 1.0:
            console.print(
                f"[yellow]⚠ Invalid speech threshold {_speech_threshold}, using {DEFAULT_SPEECH_PROB_THRESHOLD}[/yellow]"
            )
            _speech_threshold = DEFAULT_SPEECH_PROB_THRESHOLD
        overall_start = time.time()
        console.print(
            Panel.fit(
                f"[bold cyan]tag_audio_segments[/bold cyan]\n"
                f"speech_threshold={_speech_threshold:.2f} | "
                f"min_silence={min_silence_duration_sec}s | "
                f"min_speech={min_speech_duration_sec}s | "
                f"resolution={resolution_ms}ms | "
                f"include_non_speech={include_non_speech}",
                title="Segment-Based Audio Tagging",
                border_style="cyan",
            )
        )
        # Step 1: Run chunk-level tagging
        chunk_summary = self.tag_audio_chunks(
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
        )
        chunks = chunk_summary.get("chunks", [])
        actual_sr = chunk_summary.get("sample_rate", SAMPLE_RATE)
        total_duration = chunk_summary.get("total_duration", 0.0)
        audio_path_str = chunk_summary.get("audio_path", "unknown")
        if not chunks:
            elapsed = time.time() - overall_start
            console.print("[yellow]⚠ No chunks produced, returning empty result[/yellow]")
            return AudioSegmentsResult(
                audio_path=audio_path_str,
                total_duration=total_duration,
                sample_rate=actual_sr,
                chunk_duration=chunk_summary.get("chunk_duration", self.chunk_duration),
                overlap_duration=chunk_summary.get("overlap_duration", self.chunk_overlap),
                total_chunks=0,
                speech_threshold=_speech_threshold,
                min_silence_duration_sec=min_silence_duration_sec,
                min_speech_duration_sec=min_speech_duration_sec,
                resolution_ms=resolution_ms,
                chunks=[],
                speech_segments=[],
                non_speech_segments=[],
                total_speech_duration=0.0,
                total_non_speech_duration=0.0,
                overall_top_predictions=[],
                total_processing_time=round(elapsed, 4),
                real_time_factor=round(elapsed / total_duration, 4) if total_duration > 0 else 0.0,
            )
        # Step 2: Build probability timeline
        times, probs = self._build_prob_timeline(chunks, resolution_ms=resolution_ms)
        if len(times) == 0:
            elapsed = time.time() - overall_start
            console.print("[yellow]⚠ Empty probability timeline[/yellow]")
            return AudioSegmentsResult(
                audio_path=audio_path_str,
                total_duration=total_duration,
                sample_rate=actual_sr,
                chunk_duration=chunk_summary.get("chunk_duration", self.chunk_duration),
                overlap_duration=chunk_summary.get("overlap_duration", self.chunk_overlap),
                total_chunks=len(chunks),
                speech_threshold=_speech_threshold,
                min_silence_duration_sec=min_silence_duration_sec,
                min_speech_duration_sec=min_speech_duration_sec,
                resolution_ms=resolution_ms,
                chunks=chunks,
                speech_segments=[],
                non_speech_segments=[],
                total_speech_duration=0.0,
                total_non_speech_duration=0.0,
                overall_top_predictions=chunk_summary.get("overall_top_predictions", []),
                total_processing_time=round(elapsed, 4),
                real_time_factor=round(elapsed / total_duration, 4) if total_duration > 0 else 0.0,
            )
        console.print(f"[dim]🎚 Using speech threshold: {_speech_threshold}[/dim]")
        # Step 3: Detect speech segments from timeline
        step = resolution_ms / 1000.0
        min_silence_cells = max(1, int(np.ceil(min_silence_duration_sec / step)))
        min_speech_cells = max(1, int(np.ceil(min_speech_duration_sec / step)))
        is_speech = probs >= _speech_threshold
        speech_cell_count = np.sum(is_speech)
        total_cells = len(is_speech)
        console.print(
            f"[dim]📊 Timeline: {speech_cell_count}/{total_cells} cells above threshold "
            f"({speech_cell_count/total_cells*100:.1f}%)[/dim]"
        )
        raw_segments: List[Tuple[float, float]] = []
        in_speech = False
        seg_start_idx = 0
        silence_run = 0
        speech_cells_in_current = 0
        for i, sp in enumerate(is_speech):
            if not in_speech:
                if sp:
                    in_speech = True
                    seg_start_idx = i
                    silence_run = 0
                    speech_cells_in_current = 1
                    console.print(
                        f"[dim]🎤 Speech start at cell {i} (time={times[i]:.3f}s)[/dim]"
                    )
            else:
                if sp:
                    silence_run = 0
                    speech_cells_in_current += 1
                else:
                    silence_run += 1
                    if silence_run >= min_silence_cells:
                        seg_end_idx = i - silence_run + 1
                        seg_start_time = times[seg_start_idx]
                        seg_end_time = times[seg_end_idx - 1]
                        raw_segments.append((seg_start_time, seg_end_time))
                        console.print(
                            f"[dim]🔇 Speech end at cell {i} (time={times[i]:.3f}s) | "
                            f"segment: {seg_start_time:.3f}s-{seg_end_time:.3f}s "
                            f"(silence={silence_run*step:.3f}s)[/dim]"
                        )
                        in_speech = False
                        silence_run = 0
                        speech_cells_in_current = 0
        if in_speech:
            seg_start_time = times[seg_start_idx]
            seg_end_time = times[-1]
            raw_segments.append((seg_start_time, seg_end_time))
            console.print(
                f"[dim]🎤 Trailing speech segment: {seg_start_time:.3f}s-{seg_end_time:.3f}s[/dim]"
            )
        # Step 4: Filter by minimum speech duration
        speech_segments: List[Tuple[float, float]] = []
        for s, e in raw_segments:
            duration = e - s
            if duration >= min_speech_duration_sec:
                speech_segments.append((s, e))
            else:
                console.print(
                    f"[dim]⏭ Discarding short segment: {s:.3f}s-{e:.3f}s "
                    f"(dur={duration:.3f}s < min_speech={min_speech_duration_sec}s)[/dim]"
                )
        console.print(f"[bold green]✅ {len(speech_segments)} speech segment(s) detected[/bold green]")
        # Step 5: Detect non-speech segments (if requested)
        non_speech_segments: List[Tuple[float, float]] = []
        if include_non_speech:
            all_segments_sorted = sorted(speech_segments, key=lambda x: x[0])
            prev_end = 0.0
            total_end = times[-1] if len(times) > 0 else max(c["end_time"] for c in chunks)
            for seg_start, seg_end in all_segments_sorted:
                if seg_start > prev_end:
                    gap_duration = seg_start - prev_end
                    if gap_duration >= min_silence_duration_sec:
                        non_speech_segments.append((prev_end, seg_start))
                prev_end = max(prev_end, seg_end)
            if prev_end < total_end:
                gap_duration = total_end - prev_end
                if gap_duration >= min_silence_duration_sec:
                    non_speech_segments.append((prev_end, total_end))
            console.print(
                f"[dim]🔇 {len(non_speech_segments)} non-speech segment(s) detected[/dim]"
            )
        # Step 6: Build structured segment results
        speech_segment_results: List[SpeechSegmentResult] = []
        for seg_num, (seg_start, seg_end) in enumerate(speech_segments):
            result = self._build_segment_result(
                seg_num=seg_num,
                seg_start=seg_start,
                seg_end=seg_end,
                is_speech=True,
                times=times,
                probs=probs,
                chunks=chunks,
                speech_threshold=_speech_threshold,
            )
            speech_segment_results.append(result)
        non_speech_segment_results: List[SpeechSegmentResult] = []
        if include_non_speech:
            for seg_num, (seg_start, seg_end) in enumerate(non_speech_segments):
                result = self._build_segment_result(
                    seg_num=seg_num,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    is_speech=False,
                    times=times,
                    probs=probs,
                    chunks=chunks,
                    speech_threshold=_speech_threshold,
                )
                non_speech_segment_results.append(result)
        total_speech_duration = sum(e - s for s, e in speech_segments)
        total_non_speech_duration = sum(e - s for s, e in non_speech_segments)
        total_elapsed = time.time() - overall_start
        rtf = total_elapsed / total_duration if total_duration > 0 else 0.0
        console.print(
            f"[dim]⏱ Segment detection complete: {total_elapsed:.2f}s, RTF: {rtf:.3f}x[/dim]"
        )
        final_result: AudioSegmentsResult = {
            "audio_path": audio_path_str,
            "total_duration": round(total_duration, 3),
            "sample_rate": actual_sr,
            "chunk_duration": chunk_summary.get("chunk_duration", self.chunk_duration),
            "overlap_duration": chunk_summary.get("overlap_duration", self.chunk_overlap),
            "total_chunks": len(chunks),
            "speech_threshold": _speech_threshold,
            "min_silence_duration_sec": min_silence_duration_sec,
            "min_speech_duration_sec": min_speech_duration_sec,
            "resolution_ms": resolution_ms,
            "chunks": chunks,
            "speech_segments": speech_segment_results,
            "non_speech_segments": non_speech_segment_results,
            "total_speech_duration": round(total_speech_duration, 3),
            "total_non_speech_duration": round(total_non_speech_duration, 3),
            "overall_top_predictions": chunk_summary.get("overall_top_predictions", []),
            "total_processing_time": round(total_elapsed, 4),
            "real_time_factor": round(rtf, 4),
        }
        return final_result

    def extract_high_confidence_speech_segments(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        min_duration: float = 2.0,
        require_confidence: Optional[List[str]] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None,
        min_chunk_duration: Optional[float] = None,
        speech_threshold: Optional[float] = None,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_DURATION_SEC,
    ) -> Tuple[List[SpeechSegmentResult], List[np.ndarray]]:
        \"\"\"
        Extract high speech segments and their audio from the input.
        A segment qualifies if duration > min_duration and segment_type == "speech".
        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (default: SAMPLE_RATE)
            min_duration: Minimum segment duration in seconds to include (default: 2.0)
            require_confidence: (deprecated) No longer used. Kept for backward compatibility.
            chunk_duration: Duration of each analysis chunk in seconds
            overlap_duration: Overlap between consecutive chunks
            min_chunk_duration: Minimum duration for the last chunk
            speech_threshold: Speech probability threshold
            min_silence_duration_sec: Continuous non-speech gap to close a segment
            min_speech_duration_sec: Minimum duration for a valid speech segment
        Returns:
            Tuple of:
                - List[SpeechSegmentResult]: Filtered speech segments
                - List[np.ndarray]: Corresponding audio arrays for each segment
        Example:
            >>> tagger = AudioTagger()
            >>> segments, audios = tagger.extract_high_confidence_speech_segments(
            ...     "recording.wav", min_duration=2.0
            ... )
            >>> for seg, aud in zip(segments, audios):
            ...     print(f"{seg['start_time']:.1f}s-{seg['end_time']:.1f}s: {len(aud)} samples")
        \"\"\"
        import soundfile as sf
        overall_start = time.time()
        console.print(
            Panel.fit(
                f"[bold cyan]extract_high_confidence_speech_segments[/bold cyan]\n"
                f"min_duration={min_duration}s | "
                f"filter: duration > {min_duration}s AND segment_type == 'speech'",
                title="High Speech Segments Extraction",
                border_style="cyan",
            )
        )
        console.print("[dim]🔍 Running tag_audio_segments...[/dim]")
        segments_result = self.tag_audio_segments(
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
            speech_threshold=speech_threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            include_non_speech=False,
        )
        speech_segments = segments_result.get("speech_segments", [])
        console.print(f"[dim]📊 Found {len(speech_segments)} total speech segments[/dim]")
        high_speech_segments: List[SpeechSegmentResult] = []
        for segment in speech_segments:
            duration = segment.get("duration", 0.0)
            segment_type = segment.get("segment_type", "")
            is_high_speech = duration > min_duration and segment_type == "speech"
            if is_high_speech:
                high_speech_segments.append(segment)
                console.print(
                    f"[green]   ✅ Segment {segment['segment_index']}: "
                    f"{segment['start_time']:.2f}s-{segment['end_time']:.2f}s "
                    f"(dur={duration:.2f}s, type={segment_type})[/green]"
                )
            else:
                reasons = []
                if duration <= min_duration:
                    reasons.append(f"duration {duration:.2f}s <= {min_duration}s")
                if segment_type != "speech":
                    reasons.append(f"segment_type is '{segment_type}'")
                console.print(
                    f"[dim]   ⏭ Skipped segment {segment['segment_index']}: "
                    f"{', '.join(reasons) if reasons else 'unknown reason'}[/dim]"
                )
        console.print(
            f"[bold green]✅ Filtered {len(high_speech_segments)} "
            f"high speech segments (duration > {min_duration}s, type=speech)[/bold green]"
        )
        high_speech_audios: List[np.ndarray] = []
        if high_speech_segments:
            try:
                audio_data, actual_sr = load_audio(
                    audio, sr=sample_rate or SAMPLE_RATE, mono=True
                )
                console.print(
                    f"[dim]📂 Loaded audio for extraction: "
                    f"{len(audio_data)/actual_sr:.2f}s @ {actual_sr}Hz[/dim]"
                )
            except Exception as e:
                console.print(f"[red]❌ Failed to load audio for extraction: {e}[/red]")
                audio_data = np.array([], dtype=np.float32)
                actual_sr = sample_rate or SAMPLE_RATE
            for segment in high_speech_segments:
                start_sample = int(segment["start_time"] * actual_sr)
                end_sample = int(segment["end_time"] * actual_sr)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                if end_sample > start_sample:
                    segment_audio = audio_data[start_sample:end_sample].copy()
                    high_speech_audios.append(segment_audio)
                    seg_dur = len(segment_audio) / actual_sr
                    console.print(
                        f"[dim]   ✂ Extracted {seg_dur:.2f}s audio "
                        f"({len(segment_audio)} samples)[/dim]"
                    )
                else:
                    console.print(
                        f"[yellow]⚠ Empty audio range for segment "
                        f"{segment['segment_index']}: "
                        f"{start_sample}-{end_sample} samples[/yellow]"
                    )
                    high_speech_audios.append(np.array([], dtype=np.float32))
        total_elapsed = time.time() - overall_start
        total_extracted_duration = sum(
            len(a) / actual_sr for a in high_speech_audios if len(a) > 0
        )
        console.print(
            f"[dim]⏱ Extraction complete: {total_elapsed:.2f}s | "
            f"Total extracted: {total_extracted_duration:.2f}s[/dim]"
        )
        return high_speech_segments, high_speech_audios

""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:
- Browse when beneficial or requested.
- Keep explanations simple and clear.

When coding:
- Provide step-by-step analysis and explain the flow.
- Use visuals, diagrams, or tables when helpful.
- For additions, show full code for new files, classes, methods, or functions.
- For changes, show full code for updated functions or methods; otherwise, show only the changed lines with surrounding context.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Always add logs so we can trace and know if all features work correctly.
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
""".strip()

# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z
# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- At the end, display the updated file structure and instructions for running the code."
# "\n- Provide complete working code for each file (should match file structure)"
# )
# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)


def get_language_from_extension(filename: str) -> str:
    """
    Simple file extension → markdown code fence language mapping
    Returns 'text' as safe fallback
    """
    ext = os.path.splitext(filename.lower())[1]
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".md": "markdown",
        ".mdx": "mdx",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".sh": "bash",
        ".bash": "bash",
        ".sql": "sql",
        ".prisma": "prisma",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".php": "php",
        ".rb": "ruby",
    }
    return mapping.get(ext, "text")


def main():
    global exclude_files, include_files, include_content, exclude_content
    print("Running _copy_for_prompt.py")
    # Parse command-line options
    parser = argparse.ArgumentParser(
        description="Generate clipboard content from specified files."
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=file_dir,
        help="Base directory to search files in (default: current directory)",
    )
    parser.add_argument(
        "-if",
        "--include-files",
        nargs="*",
        default=include_files,
        help="Patterns of files to include (default: schema.prisma, episode)",
    )
    parser.add_argument(
        "-ef",
        "--exclude-files",
        nargs="*",
        default=exclude_files,
        help="Directories or files to exclude (default: node_modules)",
    )
    parser.add_argument(
        "-ic",
        "--include-content",
        nargs="*",
        default=include_content,
        help="Patterns of file content to include",
    )
    parser.add_argument(
        "-ec",
        "--exclude-content",
        nargs="*",
        default=exclude_content,
        help="Patterns of file content to exclude",
    )
    parser.add_argument(
        "-cs",
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Make content pattern matching case-sensitive",
    )
    parser.add_argument(
        "-sf",
        "--shorten-funcs",
        action="store_true",
        default=SHORTEN_FUNCTS,
        help="Shorten function and class definitions",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=DEFAULT_QUERY_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-i",
        "--instructions",
        default=DEFAULT_INSTRUCTIONS_MESSAGE,
        help="Instructions to include in the clipboard content",
    )
    parser.add_argument(
        "-fo",
        "--filenames-only",
        action="store_true",
        help="Only copy the relative filenames, not their contents",
    )
    parser.add_argument(
        "-nl",
        "--no-length",
        action="store_true",
        default=INCLUDE_FILE_STRUCTURE,
        help="Do not show file character length",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=False,
        help="Enable compression of the clipboard content before copying (default: False)",
    )
    args = parser.parse_args()
    base_dir = args.base_dir
    include = args.include_files
    exclude = args.exclude_files
    include_content = args.include_content
    exclude_content = args.exclude_content
    case_sensitive = args.case_sensitive
    shorten_funcs = args.shorten_funcs
    query_message = args.message
    system_message = args.system
    instructions_message = args.instructions
    filenames_only = args.filenames_only
    show_file_length = not args.no_length
    compress_enabled = args.compress
    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(
        base_dir, include, exclude, include_content, exclude_content, case_sensitive
    )
    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"Compress enabled: {compress_enabled}")
    print(
        f"\nFound files ({len(context_files)}):\n{json.dumps(context_files, indent=2)}"
    )
    print("\n")
    # Initialize the clipboard content
    clipboard_content = ""
    if not context_files:
        print("No context files found matching the given patterns.")
    else:
        # Append relative filenames to the clipboard content
        for file in tqdm(
            context_files, desc=f"Processing {len(context_files)} files..."
        ):
            rel_path = os.path.relpath(path=file, start=file_dir)
            cleaned_rel_path = remove_parent_paths(rel_path)
            prefix = f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n"
            if filenames_only:
                clipboard_content += f"{prefix}"
            else:
                file_path = os.path.relpath(os.path.join(base_dir, file))
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                            content = clean_content(content, file, shorten_funcs)
                            # ── NEW: Add fenced code block ───────────────────────────────
                            lang = get_language_from_extension(file)
                            fenced_content = f"```{lang}\n{content.rstrip()}\n```"
                            clipboard_content += f"{prefix}{fenced_content}\n\n"
                    except Exception:
                        # Continue to the next file
                        continue
                else:
                    clipboard_content += f"{prefix}\n"
        clipboard_content = clean_newlines(clipboard_content).strip()
    # Generate and format the file structure
    structure_include_files = structure_include
    if include:
        structure_include_files += include
    structure_exclude_files = structure_exclude
    if exclude:
        structure_exclude_files += exclude
    files_structure = format_file_structure(
        base_dir,
        include_files=structure_include_files,
        exclude_files=structure_exclude_files,
        include_content=include_content,
        exclude_content=exclude_content,
        case_sensitive=case_sensitive,
        shorten_funcs=shorten_funcs,
        show_file_length=show_file_length,
    )
    # Prepend system and query to the clipboard content then append instructions
    clipboard_content_parts = []
    if system_message:
        clipboard_content_parts.append(f"<system>\n{system_message}\n</system>")
    # Query should come before instructions
    clipboard_content_parts.append(f"<query>\n{query_message}\n</query>")
    if instructions_message:
        clipboard_content_parts.append(f"<instructions>\n{instructions_message}\n</instructions>")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}\n")
    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}\n"
        )
    clipboard_content = "\n\n".join(clipboard_content_parts)
    # Compress to reduce tokens (optional)
    if compress_enabled:
        messages = [{"role": "user", "content": clipboard_content}]
        result = compress(
            messages,
            model=COMPRESSION_MODEL,  # headroom uses this for strategy selection only
            token_budget=TOKEN_BUDGET,  # enforce fit within llama-server context
            ccr_enabled=True,  # reversible compression (default)
            compress_user_messages=True,
            target_ratio=0.5,  # keep 50% — safe for mixed prose + code
            protect_recent=0,  # only 1 message, nothing to protect
            protect_analysis_context=False,  # do not protect code from compression
            # kompress_model="disabled",
        )
        # Log compression stats using logger.log for each result.*
        logger.log("Tokens before:", f"{result.tokens_before:,}")
        logger.log("Tokens after:", f"{result.tokens_after:,}")
        logger.log(
            "Tokens saved:",
            f"{result.tokens_saved:,} ({result.compression_ratio:.1%})",
        )
        logger.log(
            "Transforms applied:",
            str(result.transforms_applied),
        )
    else:
        logger.log("Compression skipped (use -c or --compress to enable)")
    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)
    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))
    logger.log("Tokens Count (gpt-4o):", count_tokens(clipboard_content))
    # Newline
    print("\n")


def count_tokens(
    text: str,
    model: str = "gpt-4o",  # Best default
    encoding_name: str | None = None,
) -> int:
    """
    Count the number of tokens in a string using tiktoken.
    Args:
        text: The input string to tokenize.
        model: OpenAI model name to determine the encoding
               (default: "gpt-4o" — recommended).
        encoding_name: Optional direct encoding name
                       (e.g., "o200k_base", "cl100k_base").
                       Takes precedence over model.
    Returns:
        Number of tokens.
    """
    if encoding_name:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


if __name__ == "__main__":
    main()
