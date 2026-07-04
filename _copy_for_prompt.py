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
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_config.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\state.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger_types.py",
    r"",
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
Update the logic of _perform_speaker_labeling to remove full audio labeling part.
This is so we only label those that have high speech confidence.

# C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger.py
class AudioTagger:
    def extract_high_confidence_speech_segments(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        min_duration: float = 1.5,
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


def _perform_speaker_labeling(
    audio_np: np.ndarray,
    sample_rate: int,
    header: dict,
    full_word_segments_text: str,
    segment_id: Optional[str] = None,
) -> tuple:
    \"\"\"Perform speaker labeling if text content is sufficient.
    
    When multiple high-confidence speech segments are extracted, each segment
    is labeled individually to capture potential speaker changes within the
    audio chunk. Results are then aggregated with the highest-confidence
    speaker becoming the primary label.
    
    Returns:
        tuple: (text_has_sufficient_content, speaker_results, primary_label,
                primary_confidence, speaker_metadata)
    \"\"\"
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
        use_multiple = segment_duration >= 2.0  # Changed from 3.0 to 2.0 as discussed
        
        audio_for_labeler = audio_np
        extraction_info = {
            "attempted": False,
            "successful": False,
            "segments_found": 0,
            "used_segment_duration": segment_duration,
            "original_duration": segment_duration,
            "individual_segment_results": [],  # NEW: track per-segment results
        }
        
        all_segment_speaker_results = []  # NEW: collect results from all segments

        if segment_duration >= 2.0:
            extraction_info["attempted"] = True
            try:
                tagger = get_audio_tagger()
                if tagger is not None:
                    audio_float = audio_np.astype(np.float32) / 32768.0
                    console.print(
                        f"[info]🎯 Attempting high-confidence speech extraction "
                        f"(audio: {segment_duration:.2f}s)...[/info]"
                    )
                    high_conf_segments, high_conf_audios = (
                        tagger.extract_high_confidence_speech_segments(
                            audio=audio_float,
                            sample_rate=sample_rate,
                        )
                    )
                    extraction_info["segments_found"] = len(high_conf_audios)
                    
                    if high_conf_audios and len(high_conf_audios) > 0:
                        # ──────────────────────────────────────────────
                        # NEW: Label EACH high-confidence segment separately
                        # ──────────────────────────────────────────────
                        console.print(
                            f"[success]🎯 Extracted {len(high_conf_audios)} high-confidence "
                            f"speech segment(s) — labeling each individually:[/success]"
                        )
                        
                        for i, (seg, aud) in enumerate(zip(high_conf_segments, high_conf_audios)):
                            seg_dur = len(aud) / sample_rate if len(aud) > 0 else 0
                            seg_start = seg.get('start_time', 0)
                            seg_end = seg.get('end_time', 0)
                            seg_prob = seg.get('avg_speech_probability', 0)
                            console.print(
                                f"[dim]  [{i}] {seg_start:.2f}s-{seg_end:.2f}s "
                                f"({seg_dur:.2f}s, prob={seg_prob:.3f}) → labeling...[/dim]"
                            )
                            seg_audio_int16 = (
                                np.clip(aud, -1.0, 1.0) * 32767.0
                            ).astype(np.int16)
                            sub_segment_id = f"{segment_id}_sub{i}" if segment_id else None
                            
                            # Save sub-segment audio
                            if sub_segment_id:
                                save_segment_audio_for_playback(
                                    audio_np=seg_audio_int16,
                                    segment_id=sub_segment_id,
                                    sample_rate=sample_rate,
                                    metadata={
                                        "parent_segment_id": segment_id,
                                        "sub_segment_index": i,
                                        "start_time": seg_start,
                                        "end_time": seg_end,
                                        "duration": seg_dur,
                                        "avg_speech_probability": seg_prob,
                                        "timestamp": segment_timestamp + seg_start,
                                    }
                                )
                            
                            seg_results, seg_primary, seg_conf, seg_meta = (
                                label_speakers_for_segment(
                                    waveform=seg_audio_int16,
                                    sample_rate=sample_rate,
                                    timestamp=segment_timestamp + seg_start,
                                    return_multiple=False,
                                    segment_id=sub_segment_id,
                                )
                            )
                            
                            # Store per-segment info
                            extraction_info["individual_segment_results"].append({
                                "index": i,
                                "start_time": seg_start,
                                "end_time": seg_end,
                                "duration": seg_dur,
                                "avg_speech_probability": seg_prob,
                                "primary_label": seg_primary,
                                "primary_confidence": seg_conf,
                                "match_type": seg_meta.get("match_type", "unknown"),
                                "speaker_results": seg_results,
                            })
                            
                            # Collect all speaker results across segments
                            all_segment_speaker_results.append({
                                "segment_index": i,
                                "start_time": seg_start,
                                "end_time": seg_end,
                                "label": seg_primary,
                                "confidence": seg_conf,
                                "match_type": seg_meta.get("match_type", "unknown"),
                                "is_primary": False,  # Will be set after aggregation
                            })
                            
                            console.print(
                                f"[dim]     → Speaker: {seg_primary} "
                                f"(confidence: {seg_conf:.3f}, type: {seg_meta.get('match_type', 'unknown')})[/dim]"
                            )
                        
                        # ──────────────────────────────────────────────
                        # Aggregate results across all segments
                        # ──────────────────────────────────────────────
                        # Count which speaker appeared most / with highest confidence
                        speaker_aggregates = {}
                        for result in all_segment_speaker_results:
                            label = result["label"]
                            if label not in speaker_aggregates:
                                speaker_aggregates[label] = {
                                    "label": label,
                                    "confidences": [],
                                    "total_duration": 0.0,
                                    "appearances": 0,
                                    "match_types": [],
                                }
                            agg = speaker_aggregates[label]
                            agg["confidences"].append(result["confidence"])
                            agg["appearances"] += 1
                            agg["match_types"].append(result["match_type"])
                            # Find corresponding segment info for duration
                            for seg_info in extraction_info["individual_segment_results"]:
                                if seg_info["index"] == result["segment_index"]:
                                    agg["total_duration"] += seg_info["duration"]
                                    break
                        
                        # Build final speaker_results list (ranked by confidence/duration)
                        ranked_speakers = sorted(
                            speaker_aggregates.values(),
                            key=lambda x: (
                                x["appearances"],  # More appearances = more confident
                                sum(x["confidences"]) / len(x["confidences"]),  # Avg confidence
                                x["total_duration"],  # More speech = more reliable
                            ),
                            reverse=True,
                        )
                        
                        if ranked_speakers:
                            primary = ranked_speakers[0]
                            primary_label = primary["label"]
                            primary_confidence = (
                                sum(primary["confidences"]) / len(primary["confidences"])
                                if primary["confidences"] else 0.0
                            )
                            
                            # Build final speaker_results
                            speaker_results = []
                            for rank, spk in enumerate(ranked_speakers):
                                is_primary = (rank == 0)
                                # Find the best match_type for this speaker
                                match_types = spk["match_types"]
                                best_match_type = max(
                                    match_types,
                                    key=lambda mt: {
                                        "strong_match": 5,
                                        "early_match": 4,
                                        "context_match": 4,
                                        "possible_match": 3,
                                        "weak_match": 2,
                                        "new_speaker": 1,
                                        "first_speaker": 1,
                                        "unknown": 0,
                                    }.get(mt, 0),
                                ) if match_types else "unknown"
                                
                                speaker_results.append({
                                    "label": spk["label"],
                                    "confidence": round(
                                        sum(spk["confidences"]) / len(spk["confidences"]), 4
                                    ) if spk["confidences"] else 0.0,
                                    "match_type": best_match_type,
                                    "is_primary": is_primary,
                                    "is_new_speaker": False,  # Will be updated below
                                    "segment_count": spk["appearances"],
                                    "total_speech_duration": round(spk["total_duration"], 3),
                                })
                            
                            # Mark new speakers
                            for result in speaker_results:
                                if result["match_type"] in ("new_speaker", "first_speaker"):
                                    result["is_new_speaker"] = True
                            
                            speaker_metadata = {
                                "match_type": speaker_results[0]["match_type"] if speaker_results else "unknown",
                                "speaker_list": speaker_results,
                                "total_speakers": len(speaker_results),
                                "speech_extraction": extraction_info,
                                "aggregation_method": "multi_segment_voting",
                            }
                            
                            extraction_info["successful"] = True
                            extraction_info["used_segment_duration"] = sum(
                                s["duration"] for s in extraction_info["individual_segment_results"]
                            )
                            
                            console.print(
                                f"[success]🎯 Aggregated {len(high_conf_audios)} segments → "
                                f"{len(speaker_results)} unique speaker(s)[/success]"
                            )
                            for spk in speaker_results:
                                console.print(
                                    f"[dim]   {spk['label']}: avg_conf={spk['confidence']:.3f}, "
                                    f"appearances={spk['segment_count']}, "
                                    f"duration={spk['total_speech_duration']:.2f}s, "
                                    f"type={spk['match_type']}{' ★ PRIMARY' if spk['is_primary'] else ''}[/dim]"
                                )
                        else:
                            # Fallback: no speakers found in any segment
                            console.print(
                                f"[warning]⚠️ No speakers identified in any sub-segment, "
                                f"using full audio[/warning]"
                            )
                            # Fall through to full-audio labeling below
                            all_segment_speaker_results = []
                    else:
                        console.print(
                            f"[dim]🔇 No high-confidence speech segments found "
                            f"(tagger found {extraction_info['segments_found']} segments, "
                            f"but none met the min_duration=1.5s threshold), "
                            f"using full audio for labeling[/dim]"
                        )
                else:
                    console.print(
                        "[dim]🔇 Audio tagger not available (get_audio_tagger() returned None), "
                        "skipping speech extraction[/dim]"
                    )
            except Exception as e:
                console.print(
                    f"[warning]⚠️ extract_high_confidence_speech_segments failed: {e}, "
                    f"using full audio for labeling[/warning]"
                )
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(
                f"[dim]🔇 Segment too short for speech extraction "
                f"({segment_duration:.2f}s < 2.0s), using full audio[/dim]"
            )

        # ─────────────────────────────────────────────────────────────
        # If no sub-segment results, fall back to full audio labeling
        # ─────────────────────────────────────────────────────────────
        if not all_segment_speaker_results:
            speaker_results, primary_label, primary_confidence, speaker_metadata = (
                label_speakers_for_segment(
                    waveform=audio_for_labeler,
                    sample_rate=sample_rate,
                    timestamp=segment_timestamp,
                    return_multiple=use_multiple,
                    segment_id=segment_id,
                )
            )
            speaker_metadata["speech_extraction"] = extraction_info

        # ─────────────────────────────────────────────────────────────
        # Console output
        # ─────────────────────────────────────────────────────────────
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

""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:
- Browse when beneficial or requested.
- Keep explanations simple and clear.

When coding:
- Provide step-by-step analysis and explain the flow.
- Use visuals, diagrams, or tables when helpful.
- For new files, classes, methods, or functions: show the full code.
- For updates to existing files: show only the changed sections with context. Never output the full file unless it's small.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Always add logs for traceability and verification.
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
    # Disable special-token checks entirely — the input is arbitrary file
    # content/prompt text, not something where special tokens should be
    # interpreted as control tokens. This prevents ValueError crashes when
    # source files happen to contain strings like "<|endoftext|>".
    return len(encoding.encode(text, disallowed_special=()))


if __name__ == "__main__":
    main()
