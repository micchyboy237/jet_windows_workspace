# save_utils.py

import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import scipy.io.wavfile as wavfile
from core.state import (
    get_current_speaker,
    get_last_n_segments_dir,
    get_speaker_labeler,
    get_speaker_diarization,
)
from rich.console import Console

console = Console()


def _atomic_write_json(file_path: Path, data: Any, indent: int = 2) -> None:
    """
    Atomically write JSON data to a file using a temp file + rename pattern.
    
    This is safe for:
    - All platforms (Windows, Linux, macOS)
    - Multi-threaded environments
    - Multi-process environments
    
    How it works:
    1. Write to a temporary file in the same directory
    2. Flush and sync to disk
    3. Atomically rename temp file to target file
    
    If anything fails during writing, the original file remains untouched.
    
    Args:
        file_path: Target file path
        data: Data to serialize as JSON
        indent: JSON indentation level
    
    Debug logs trace:
        - Temp file creation
        - Write success/failure
        - Rename success/failure
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in the same directory (ensures same filesystem for atomic rename)
    temp_fd = None
    temp_path = None
    try:
        # Create a temporary file in the same directory as the target
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(file_path.parent),
            prefix=f".{file_path.name}.",
            suffix=".tmp"
        )
        
        console.print(f"[dim]Writing to temp file: {Path(temp_path).name}[/dim]")
        
        # Write JSON to the temp file
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        console.print(f"[dim]Temp file written successfully[/dim]")
        
        # Atomic rename (this is the key - it's atomic on all modern OSes)
        os.replace(temp_path, str(file_path))
        
        console.print(f"[dim]Atomic rename successful: {file_path.name}[/dim]")
        
    except Exception as e:
        console.print(f"[error]Atomic write failed for {file_path.name}: {e}[/error]")
        # Clean up temp file if it exists and we failed
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                console.print(f"[dim]Cleaned up temp file: {Path(temp_path).name}[/dim]")
            except Exception as cleanup_err:
                console.print(f"[warning]Could not clean up temp file: {cleanup_err}[/warning]")
        raise


def _read_json_safe(file_path: Path, default: Any = None) -> Any:
    """
    Safely read a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is corrupted
    
    Returns:
        Parsed JSON data or default value
    
    Debug logs trace:
        - File existence check
        - Read attempt
        - Parse attempt
        - Any errors encountered
    """
    if default is None:
        default = []
    
    if not file_path.exists():
        console.print(f"[dim]File does not exist: {file_path.name}[/dim]")
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        console.print(f"[dim]Successfully read {file_path.name}[/dim]")
        return data
    except json.JSONDecodeError as e:
        console.print(f"[warning]Corrupted JSON in {file_path.name}: {e}. Using default.[/warning]")
        return default
    except Exception as e:
        console.print(f"[error]Error reading {file_path.name}: {e}. Using default.[/error]")
        return default


def _append_to_global_json(
    file_path: Path,
    new_entry: Dict[str, Any],
    max_entries: int = 200,
    key_field: str = "segment_dir",
) -> None:
    """
    Append a new entry to a global JSON array file using atomic writes.
    
    This function is safe for concurrent access because:
    1. It reads the entire file into memory
    2. Modifies the data in memory
    3. Writes using atomic temp file + rename pattern
    
    The atomic write ensures that:
    - Readers always see a complete, valid JSON file
    - A crash during writing won't corrupt the file
    - Concurrent readers won't see partial writes
    
    Args:
        file_path: Path to the global JSON file
        new_entry: New entry to append
        max_entries: Maximum number of entries to keep
        key_field: Field to use for deduplication
    
    Debug logs trace:
        - File existence check
        - Read existing entries count
        - Deduplication check
        - Write operation
        - Entry count after write
    """
    console.print(f"[dim]Updating global file: {file_path.name}[/dim]")
    
    try:
        # Read existing entries (safely)
        existing_entries = _read_json_safe(file_path, default=[])
        
        if not isinstance(existing_entries, list):
            console.print(
                f"[warning]{file_path.name} is not a list, resetting[/warning]"
            )
            existing_entries = []
        
        console.print(
            f"[dim]Read {len(existing_entries)} existing entries from {file_path.name}[/dim]"
        )
        
        # Check for duplicate entries (same key_field value)
        key_value = new_entry.get(key_field)
        if key_value:
            # Remove any existing entry with the same key
            original_count = len(existing_entries)
            existing_entries = [
                entry for entry in existing_entries 
                if entry.get(key_field) != key_value
            ]
            removed = original_count - len(existing_entries)
            if removed > 0:
                console.print(
                    f"[dim]Removed {removed} duplicate(s) for {key_field}={key_value}[/dim]"
                )
        
        # Add timestamp if not present
        if 'updated_at' not in new_entry:
            new_entry['updated_at'] = datetime.now().isoformat()
        
        # Append new entry
        existing_entries.append(new_entry)
        
        # Trim to max_entries (keep most recent)
        if len(existing_entries) > max_entries:
            trimmed = len(existing_entries) - max_entries
            existing_entries = existing_entries[-max_entries:]
            console.print(
                f"[dim]Trimmed {trimmed} old entries from {file_path.name}[/dim]"
            )
        
        # Write atomically (temp file + rename)
        _atomic_write_json(file_path, existing_entries)
        
        console.print(
            f"[success]Updated {file_path.name}: {len(existing_entries)} entries[/success]"
        )
        
    except Exception as e:
        console.print(f"[error]Failed to update {file_path.name}: {e}[/error]")
        console.print(f"[error]Traceback: {traceback.format_exc()}[/error]")


def update_global_speakers_file(
    speaker_data: Dict[str, Any], 
    segment_dir_name: str, 
    segment_num: int
) -> None:
    """
    Update the global all_speakers.json file with speaker information from a segment.
    
    Args:
        speaker_data: Speaker information from the segment
        segment_dir_name: Segment directory name
        segment_num: Segment number
    
    Debug logs trace:
        - Entry creation
        - File update status
    """
    last_n_segments_dir = get_last_n_segments_dir()
    global_speakers_path = last_n_segments_dir / "all_speakers.json"
    
    console.print(
        f"[dim]Updating global speakers file with segment {segment_dir_name}[/dim]"
    )
    
    # Create the entry matching the structure of speaker_info.json
    global_entry = {
        "segment_dir": segment_dir_name,
        "segment_number": segment_num,
        "speaker_label": speaker_data.get("speaker_label"),
        "speaker_confidence": speaker_data.get("speaker_confidence"),
        "speaker_metadata": speaker_data.get("speaker_metadata", {}),
        "speakers": speaker_data.get("speakers", []),
        "diarization": speaker_data.get("diarization", {}),
        "timestamp": datetime.now().isoformat(),
    }
    
    console.print(
        f"[dim]Speaker entry: {global_entry.get('speaker_label')} "
        f"(confidence: {global_entry.get('speaker_confidence', 0):.3f})[/dim]"
    )
    
    _append_to_global_json(
        file_path=global_speakers_path,
        new_entry=global_entry,
        max_entries=200,
        key_field="segment_dir",
    )


def update_global_tag_events_file(
    tagging_data: Dict[str, Any], 
    segment_dir_name: str, 
    segment_num: int
) -> None:
    """
    Update the global all_tag_events.json file with audio tagging information.
    
    Args:
        tagging_data: Audio tagging results from the segment
        segment_dir_name: Segment directory name
        segment_num: Segment number
    
    Debug logs trace:
        - Entry creation
        - Speech detection status
        - Top predictions
        - File update status
    """
    last_n_segments_dir = get_last_n_segments_dir()
    global_tag_path = last_n_segments_dir / "all_tag_events.json"
    
    console.print(
        f"[dim]Updating global tag events file with segment {segment_dir_name}[/dim]"
    )
    
    # Extract key tagging information
    has_speech = tagging_data.get("speech_detected", False)
    speech_prob = tagging_data.get("max_speech_probability", 0.0)
    processing_mode = tagging_data.get("processing_mode", "unknown")
    
    # Get top predictions (handle both chunked and single modes)
    top_preds = (
        tagging_data.get("overall_top_predictions") or 
        tagging_data.get("top_predictions", [])
    )
    
    # Create simplified entry
    global_entry = {
        "segment_dir": segment_dir_name,
        "segment_number": segment_num,
        "speech_detected": has_speech,
        "speech_probability": round(speech_prob, 4),
        "processing_mode": processing_mode,
        "top_predictions": top_preds[:3] if top_preds else [],
        "total_chunks": tagging_data.get("total_chunks", 0),
        "timestamp": datetime.now().isoformat(),
    }
    
    console.print(
        f"[dim]Tag entry: speech={has_speech}, prob={speech_prob:.3f}, "
        f"mode={processing_mode}[/dim]"
    )
    
    if top_preds:
        top_names = ", ".join(
            [f"{p.get('name', '?')}({p.get('prob', 0):.3f})" for p in top_preds[:3]]
        )
        console.print(f"[dim]Top predictions: {top_names}[/dim]")
    
    _append_to_global_json(
        file_path=global_tag_path,
        new_entry=global_entry,
        max_entries=200,
        key_field="segment_dir",
    )


def save_tagging_to_segment(
    segment_dir: Path,
    tagging_results: Dict[str, Any],
    has_speech: bool,
    speech_prob: float,
) -> None:
    """
    Save audio tagging results to segment directory.
    
    Args:
        segment_dir: Directory to save results
        tagging_results: Tagging results dictionary
        has_speech: Whether speech was detected
        speech_prob: Maximum speech probability
    
    Debug logs trace:
        - Directory creation
        - File writing
        - Success/failure status
    """
    try:
        tagging_dir = segment_dir / "tagging"
        tagging_dir.mkdir(exist_ok=True)
        
        tagging_file = tagging_dir / "audio_tags.json"
        with open(tagging_file, "w", encoding="utf-8") as f:
            json.dump(tagging_results, f, ensure_ascii=False, indent=2)
        
        summary_file = tagging_dir / "tagging_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Audio Tagging Summary\n")
            f.write("====================\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Speech Detected: {'Yes' if has_speech else 'No'}\n")
            f.write(f"Speech Probability: {speech_prob:.3f}\n")
            f.write(
                f"Processing Mode: {tagging_results.get('processing_mode', 'unknown')}\n"
            )
            top_preds = (
                tagging_results.get("overall_top_predictions") or 
                tagging_results.get("top_predictions", [])
            )
            if top_preds:
                f.write("\nTop Predictions:\n")
                for pred in top_preds[:5]:
                    f.write(f"  - {pred['name']}: {pred['prob']:.3f}\n")
        
        console.print(
            f"[success]Audio tagging results saved to: {tagging_dir}[/success]"
        )
    except Exception as e:
        console.print(f"[warning]Failed to save tagging results: {e}[/warning]")


def save_segment_files(
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
    tagging_events: Optional[Dict[str, Any]] = None,
) -> dict:
    """Save all segment-related files to disk."""
    with open(segment_dir / "header.json", "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)
    audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(segment_dir / "sound.wav"), sample_rate, audio_np_int16)
    wavfile.write(str(segment_dir / "full_sound.wav"), sample_rate, full_audio_int16)
    with open(segment_dir / "ja_sents.json", "w", encoding="utf-8") as f:
        json.dump({"old_ja_sents": old_ja_sents, "new_ja_sents": new_ja_sents}, f, ensure_ascii=False, indent=2)
    with open(segment_dir / "en_sents.json", "w", encoding="utf-8") as f:
        json.dump({"old_en_sents": old_en_sents, "new_en_sents": new_en_sents}, f, ensure_ascii=False, indent=2)
    
    # Create speaker info data
    speaker_info_data = {
        "speaker_label": primary_label,
        "speaker_confidence": primary_confidence,
        "speaker_metadata": speaker_metadata,
        "speakers": speaker_results,
        "diarization": get_speaker_diarization(),
    }
    with open(segment_dir / "speaker_info.json", "w", encoding="utf-8") as f:
        json.dump(speaker_info_data, f, ensure_ascii=False, indent=2)
    
    # NEW: Update global speakers file
    update_global_speakers_file(
        speaker_data=speaker_info_data,
        segment_dir_name=segment_dir_name,
        segment_num=segment_num,
    )
    
    # Save tagging results if available
    if tagging_events:
        save_tagging_to_segment(
            segment_dir=segment_dir,
            tagging_results=tagging_events,
            has_speech=tagging_events.get("speech_detected", False),
            speech_prob=tagging_events.get("max_speech_probability", 0.0),
        )
        
        # NEW: Update global tag events file
        update_global_tag_events_file(
            tagging_data=tagging_events,
            segment_dir_name=segment_dir_name,
            segment_num=segment_num,
        )
    
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


def save_full_audio_files(
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
        wavfile.write(
            str(full_audio_dir / "full_sound.wav"), sample_rate, full_audio_int16
        )
    context_summary = {
        "total_duration_sec": round(context_buffer.get_total_duration(), 3),
        "num_chunks": len(context_buffer.segments),
        "max_duration_sec": context_buffer.max_duration_sec,
        "sample_rate": context_buffer.sample_rate,
        "last_updated": datetime.now().isoformat(),
        "context_includes_current_segment": True,
        "current_speaker": get_current_speaker(),
        "speaker_count": get_speaker_labeler().speaker_count
        if get_speaker_labeler()
        else 0,
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
        json.dump(
            {
                "level": "word",
                "count": len(full_word_segments),
                "text": full_word_segments_text,
                "segments": full_word_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(full_audio_dir / "full_phrase_segments.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "level": "phrase",
                "count": len(full_phrase_segments),
                "phrases": [p["phrase"] for p in full_phrase_segments],
                "segments": full_phrase_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if full_ja_sents is not None:
        with open(full_audio_dir / "full_ja_sents.json", "w", encoding="utf-8") as f:
            json.dump(full_ja_sents, f, ensure_ascii=False, indent=2)
