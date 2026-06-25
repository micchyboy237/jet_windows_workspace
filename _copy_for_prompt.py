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
Replace high-confidence speech extraction logic to use get_valid_speech_waves from speech_waves instead of tagger.extract_high_confidence_speech_segments


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
                            
                            # Convert this segment to int16 for the labeler
                            seg_audio_int16 = (
                                np.clip(aud, -1.0, 1.0) * 32767.0
                            ).astype(np.int16)
                            
                            # Generate a unique segment_id for this sub-segment
                            sub_segment_id = f"{segment_id}_sub{i}" if segment_id else None
                            
                            # Label THIS individual segment
                            seg_results, seg_primary, seg_conf, seg_meta = (
                                label_speakers_for_segment(
                                    waveform=seg_audio_int16,
                                    sample_rate=sample_rate,
                                    timestamp=segment_timestamp + seg_start,
                                    return_multiple=False,  # Individual segments are short
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



# C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speech_waves.py

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from rich.console import Console

try:
    from services._types import AudioInput, SpeechWave
    from services.audio_config import HOP_SIZE, SAMPLE_RATE
    from services.energy import compute_rms_per_frame
    from services.audio_utils import load_audio
    from services.norm_speech_loudness import normalize_audio_for_vad
    from services.vad_firered import extract_speech_timestamps
    from services.dtype_conversion import convert_audio_dtype
except ImportError:
    from _types import AudioInput, SpeechWave
    from audio_config import HOP_SIZE, SAMPLE_RATE
    from energy import compute_rms_per_frame
    from audio_utils import load_audio
    from norm_speech_loudness import normalize_audio_for_vad
    from vad_firered import extract_speech_timestamps
    from dtype_conversion import convert_audio_dtype

DEFAULT_THRESHOLD = 0.3

DEFAULT_MIN_PROMINENCE = 0.05
DEFAULT_MIN_EXCURSION = 0.04
DEFAULT_MIN_PEAK_PROB = 0.55
DEFAULT_MIN_FRAMES = 3
DEFAULT_MIN_DURATION_SEC = 1.0
DEFAULT_BASELINE_THRESHOLD = 0.1

DEFAULT_MIN_SPEECH_DURATION_MS = 1000
DEFAULT_MIN_SILENCE_DURATION_MS = 100

WaveState = Literal["below", "above"]

console = Console()

@dataclasses.dataclass
class WaveShapeConfig:
    \"\"\"
    Tunable thresholds that decide whether a probability wave has a real
    mountain shape rather than being a flat plateau or a tiny ripple.

    Attributes:
        min_prominence: How much the peak must rise above the average of the
            two surrounding valley endpoints.
        min_excursion: The minimum difference between the highest and lowest
            probability inside the wave window.
        min_peak_prob: Absolute floor — the peak frame must reach at least
            this probability (guards against waves that never really fire).
        min_frames: Waves shorter than this many frames are discarded.
        min_duration_sec: Minimum wall-clock duration in seconds. Waves
            shorter than this are rejected even if they pass frame and shape
            checks. Derived independently of min_frames so both constraints
            must be satisfied.
        baseline_threshold: Probability threshold used to determine when a
            wave has truly fallen back to baseline/silence level. Used to
            detect wave boundaries and preroll adjustments.
    \"\"\"

    min_prominence: float = DEFAULT_MIN_PROMINENCE
    min_excursion: float = DEFAULT_MIN_EXCURSION
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB
    min_frames: int = DEFAULT_MIN_FRAMES
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD


def is_prominent_wave(
    wave_probs: List[float],
    entry_prob: float,
    exit_prob: float,
    cfg: WaveShapeConfig,
) -> tuple[bool, dict]:
    \"\"\"
    Decide whether a slice of probabilities forms a genuine mountain shape.

    The algorithm:
      1. Baseline = average of entry_prob and exit_prob (the "ground level").
      2. Peak     = maximum probability inside the slice.
      3. Prominence = peak - baseline.
      4. Excursion  = max - min inside the slice (vertical range).

    Returns:
        (passed: bool, diagnostics: dict)
    \"\"\"
    if not wave_probs:
        return False, {}

    peak_prob = max(wave_probs)
    min_prob = min(wave_probs)
    baseline = (entry_prob + exit_prob) / 2.0
    prominence = peak_prob - baseline
    excursion = peak_prob - min_prob
    n_frames = len(wave_probs)

    passed = (
        prominence >= cfg.min_prominence
        and excursion >= cfg.min_excursion
        and peak_prob >= cfg.min_peak_prob
        and n_frames >= cfg.min_frames
    )

    diagnostics = {
        "baseline": round(baseline, 6),
        "peak_prob": round(peak_prob, 6),
        "prominence": round(prominence, 6),
        "excursion": round(excursion, 6),
        "n_frames": n_frames,
        "shape_passed": passed,
    }
    return passed, diagnostics


def check_speech_waves(
    speech_probs: List[float],
    threshold: float = DEFAULT_THRESHOLD,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> List[SpeechWave]:
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()

    if not speech_probs:
        return []

    waves: List[SpeechWave] = []
    current_wave: SpeechWave | None = None
    state: WaveState = "below"
    rise_frame_idx: int | None = None

    if speech_probs:
        if speech_probs[0] < shape_cfg.baseline_threshold:
            current_wave = SpeechWave(
                has_risen=False,
                has_multi_passed=False,
                has_fallen=False,
                is_valid=False,
                start_sec=0.0,
                end_sec=0.0,
                details={
                    "frame_start": 0,
                    "frame_end": 0,
                    "frame_len": 0,
                    "duration_sec": 0.0,
                    "min_prob": speech_probs[0],
                    "max_prob": speech_probs[0],
                    "avg_prob": speech_probs[0],
                    "std_prob": 0.0,
                    "composite_score": 0.0,
                },
            )
            state = "below"

        elif speech_probs[0] >= threshold:
            state = "above"

    for i, prob in enumerate(speech_probs):
        frame_time_sec = i * HOP_SIZE / sampling_rate

        if state == "below":
            if prob >= threshold:
                rise_frame_idx = i

                # ── Preroll: walk back from rise_frame_idx until we find a
                #    frame strictly below baseline_threshold (or hit index 0).
                preroll_start = rise_frame_idx
                while (
                    preroll_start > 0
                    and speech_probs[preroll_start - 1] >= shape_cfg.baseline_threshold
                ):
                    preroll_start -= 1
                preroll_start_sec = preroll_start * HOP_SIZE / sampling_rate

                current_wave = SpeechWave(
                    has_risen=current_wave["has_risen"] if current_wave else True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=preroll_start_sec,
                    end_sec=preroll_start_sec,
                    details={
                        "frame_start": preroll_start,
                        "frame_end": preroll_start,
                        "frame_len": 0,
                        "duration_sec": 0.0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "avg_prob": prob,
                        "std_prob": 0.0,
                        "composite_score": 0.0,
                    },
                )

                state = "above"
        else:
            if prob >= threshold:
                if current_wave is not None:
                    current_wave["has_multi_passed"] = True
            else:
                if current_wave is not None:
                    if prob <= shape_cfg.baseline_threshold:
                        current_wave["has_fallen"] = True

                    # frame_start uses the preroll-adjusted value stored in details
                    frame_start = current_wave["details"]["frame_start"]
                    frame_end = i
                    wave_probs = speech_probs[frame_start:frame_end]
                    frame_len = frame_end - frame_start

                    # entry_prob: the frame immediately before the preroll start
                    entry_prob = (
                        speech_probs[frame_start - 1] if frame_start > 0 else 0.0
                    )
                    exit_prob = prob

                    shape_ok, shape_diag = is_prominent_wave(
                        wave_probs, entry_prob, exit_prob, shape_cfg
                    )

                    duration_sec = frame_time_sec - current_wave["start_sec"]
                    duration_ok = duration_sec >= shape_cfg.min_duration_sec

                    current_wave["is_valid"] = (
                        current_wave["has_risen"]
                        and current_wave["has_multi_passed"]
                        and current_wave["has_fallen"]
                        and shape_ok
                        and duration_ok
                    )
                    current_wave["end_sec"] = frame_time_sec
                    current_wave["details"] = {
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "frame_len": frame_len,
                        "duration_sec": duration_sec,
                        "min_prob": min(wave_probs) if wave_probs else 0.0,
                        "max_prob": max(wave_probs) if wave_probs else 0.0,
                        "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                        "std_prob": statistics.stdev(wave_probs)
                        if frame_len > 1
                        else 0.0,
                        "duration_ok": duration_ok,
                        **shape_diag,
                        "composite_score": 0.0,
                    }
                    current_wave["details"]["composite_score"] = (
                        compute_composite_score(current_wave)
                    )

                # FIX: Only append if current_wave is not None
                if prob < shape_cfg.baseline_threshold:
                    if current_wave is not None:
                        waves.append(current_wave)
                    current_wave = None
                    rise_frame_idx = None
                    state = "below"

    # FIX: Handle a wave that never fell back below the threshold
    # Guard ensures we only append if current_wave exists
    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False
        current_wave["end_sec"] = len(speech_probs) * HOP_SIZE / sampling_rate

        if rise_frame_idx is not None:
            # frame_start is already preroll-adjusted in details
            frame_start = current_wave["details"]["frame_start"]
            frame_end = len(speech_probs)
            wave_probs = speech_probs[frame_start:frame_end]
            frame_len = frame_end - frame_start
            duration_sec = current_wave["end_sec"] - current_wave["start_sec"]
            entry_prob = speech_probs[frame_start - 1] if frame_start > 0 else 0.0
            exit_prob = threshold
            shape_ok, shape_diag = is_prominent_wave(
                wave_probs, entry_prob, exit_prob, shape_cfg
            )
            current_wave["details"] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_len": frame_len,
                "duration_sec": duration_sec,
                "min_prob": min(wave_probs) if wave_probs else 0.0,
                "max_prob": max(wave_probs) if wave_probs else 0.0,
                "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
                "duration_ok": False,
                **shape_diag,
                "composite_score": 0.0,
            }
            current_wave["details"]["composite_score"] = compute_composite_score(
                current_wave
            )

        waves.append(current_wave)

    return waves


def get_speech_waves(
    audio: AudioInput,
    speech_probs: List[float],
    threshold: float = DEFAULT_THRESHOLD,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
    with_audio: bool = False,
) -> List[SpeechWave] | List[Tuple[SpeechWave, np.ndarray]]:
    \"\"\"
    Identify complete speech waves (rise → sustained high → fall) from FireRedVAD probabilities.

    Follows the same pipeline as _main_speech_waves.main():
      1. Runs shape analysis on pre-computed VAD scores
      2. Filters to valid waves
      3. Optionally loads audio and extracts segments

    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        speech_probs: Speech probability scores from VAD
        threshold: VAD probability threshold
        sampling_rate: Audio sample rate in Hz
        shape_cfg: Configuration for wave shape validation (defaults to WaveShapeConfig())
        with_audio: If True, returns list of tuples (SpeechWave, np.ndarray) with 
                   the audio data for each wave extracted from the loaded audio

    Returns:
        If with_audio=False: List[SpeechWave] containing valid speech waves
        If with_audio=True: List[Tuple[SpeechWave, np.ndarray]] containing valid 
                           speech waves paired with their audio segments
    \"\"\"
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"get_speech_waves called with with_audio={with_audio}, threshold={threshold}"
    )

    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()
        logger.debug(f"Using default WaveShapeConfig: {shape_cfg}")

    # Step 1: Shape analysis on existing scores (same as _main_speech_waves)
    all_waves = check_speech_waves(
        speech_probs=speech_probs,
        threshold=threshold,
        sampling_rate=sampling_rate,
        shape_cfg=shape_cfg,
    )
    logger.info(f"Total waves detected: {len(all_waves)}")

    # Step 2: Filter to valid waves only
    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave.get("is_valid", False):
            valid_waves.append(wave)

    logger.info(f"Valid waves (without audio): {len(valid_waves)}")

    # Step 3: Return early if audio extraction not requested
    if not with_audio:
        return valid_waves

    # Step 4: Load audio only when needed for extraction
    loaded_audio, loaded_sr = load_audio(audio, sr=sampling_rate, mono=True)
    logger.debug(f"Audio loaded for extraction: shape={loaded_audio.shape}, sr={loaded_sr}")

    # Step 5: Extract audio segments
    valid_waves_with_audio: List[Tuple[SpeechWave, np.ndarray]] = []
    for wave in valid_waves:
        frame_start = wave["details"]["frame_start"]
        frame_end = wave["details"]["frame_end"]

        start_sample = frame_start * HOP_SIZE
        end_sample = (frame_end + 1) * HOP_SIZE
        start_sample = max(0, start_sample)
        end_sample = min(len(loaded_audio), end_sample)

        if end_sample > start_sample:
            wave_audio = loaded_audio[start_sample:end_sample].copy()
            valid_waves_with_audio.append((wave, wave_audio))
            logger.debug(
                f"Wave audio extracted: frames [{frame_start}:{frame_end}], "
                f"samples [{start_sample}:{end_sample}], "
                f"duration={wave['details']['duration_sec']:.3f}s"
            )

    logger.info(f"Valid waves (with audio): {len(valid_waves_with_audio)}")
    return valid_waves_with_audio


def get_valid_speech_waves(
    audio: AudioInput,
    sampling_rate: int = SAMPLE_RATE,
    vad_threshold: float = DEFAULT_THRESHOLD,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_excursion: float = DEFAULT_MIN_EXCURSION,
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    with_audio: bool = False,
) -> List[SpeechWave] | List[Tuple[SpeechWave, np.ndarray]]:
    \"\"\"
    Identify valid speech waves from audio using VAD and shape analysis.

    This function follows the same pipeline as _main_speech_waves.main():
      1. Loads audio (accepts file path, bytes, numpy array, or torch tensor)
      2. Runs VAD (extract_speech_timestamps) to get probability scores
      3. Identifies speech waves via shape analysis (check_speech_waves)
      4. Filters to only valid (is_valid=True) waves
      5. Optionally extracts audio segments for each wave

    All parameters default to the module-level DEFAULT_* constants,
    matching the all-defaults usage in _main_speech_waves.

    Args:
        audio: Audio input — file path (str/Path), bytes, numpy array, or torch tensor.
               Accepts the same types as load_audio() (AudioInput union).
        sampling_rate: Audio sampling rate in Hz (used when audio is not a file)
        vad_threshold: VAD probability threshold (above = speech)
        min_prominence: Minimum peak prominence above baseline
        min_excursion: Minimum peak-to-valley excursion
        min_peak_prob: Minimum peak probability
        min_frames: Minimum frames per wave
        min_duration_sec: Minimum wave duration in seconds
        baseline_threshold: Probability threshold for silence/baseline
        min_speech_duration_ms: Minimum speech segment for VAD
        min_silence_duration_ms: Minimum silence gap for VAD
        with_audio: If True, returns list of tuples (SpeechWave, np.ndarray)
                   with the audio data for each wave extracted from the input audio

    Returns:
        If with_audio=False: List[SpeechWave] containing valid speech waves.
        If with_audio=True: List[Tuple[SpeechWave, np.ndarray]] containing valid 
                           speech waves paired with their audio segments.
        Returns empty list if no valid speech found or VAD fails.

    Example:
        >>> # With file path
        >>> waves = get_valid_speech_waves("recording.wav")
        >>> 
        >>> # With audio extraction
        >>> waves_with_audio = get_valid_speech_waves("recording.wav", with_audio=True)
        >>> for wave, audio_chunk in waves_with_audio:
        ...     print(f"Duration: {wave['details']['duration_sec']:.2f}s")
    \"\"\"
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"get_valid_speech_waves called with with_audio={with_audio}, "
        f"vad_threshold={vad_threshold}, min_duration={min_duration_sec}s"
    )

    # Build WaveShapeConfig from parameters (all default to module constants)
    shape_cfg = WaveShapeConfig(
        min_prominence=min_prominence,
        min_excursion=min_excursion,
        min_peak_prob=min_peak_prob,
        min_frames=min_frames,
        min_duration_sec=min_duration_sec,
        baseline_threshold=baseline_threshold,
    )
    logger.debug(f"WaveShapeConfig: {shape_cfg}")

    # Step 1: Load audio (handles file path, bytes, numpy array, or torch tensor)
    #          Uses the same load_audio as _main_speech_waves.main()
    audio_np, sr = load_audio(audio, sr=sampling_rate, mono=True)
    logger.debug(f"Audio loaded: shape={audio_np.shape}, sr={sr}, dtype={audio_np.dtype}")

    # Step 2: Run VAD — same call as _main_speech_waves.main()
    try:
        _, scores = extract_speech_timestamps(
            audio=audio_np,
            include_non_speech=False,
            threshold=vad_threshold,
            min_speech_duration_sec=min_speech_duration_ms / 1000.0,
            min_silence_duration_sec=min_silence_duration_ms / 1000.0,
            with_scores=True,
        )
    except Exception as e:
        logger.error(f"VAD extraction failed: {e}")
        console.print(f"[error]VAD extraction failed: {e}[/error]")
        return []

    if not scores:
        logger.warning("No speech scores returned from VAD")
        return []

    logger.info(f"VAD produced {len(scores)} probability scores")

    # Step 3: Shape analysis — same call chain as _main_speech_waves.main()
    all_waves = check_speech_waves(
        speech_probs=scores,
        threshold=vad_threshold,
        sampling_rate=sr,
        shape_cfg=shape_cfg,
    )
    logger.info(f"Total waves detected by shape analysis: {len(all_waves)}")

    # Step 4: Filter to valid waves only (is_valid=True)
    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave is None:
            continue
        if not isinstance(wave, dict):
            continue
        if not wave.get("is_valid", False):
            continue
        valid_waves.append(wave)

    logger.info(f"Valid waves after filtering: {len(valid_waves)}")

    # Step 5: Return early if audio extraction not requested
    if not with_audio:
        return valid_waves

    # Step 6: Extract audio segments for each valid wave
    valid_waves_with_audio: List[Tuple[SpeechWave, np.ndarray]] = []
    for wave in valid_waves:
        frame_start = wave["details"]["frame_start"]
        frame_end = wave["details"]["frame_end"]

        # Convert frame indices to sample indices using HOP_SIZE (160 samples = 10ms)
        start_sample = frame_start * HOP_SIZE
        end_sample = (frame_end + 1) * HOP_SIZE

        # Clamp to valid range within the loaded audio
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_np), end_sample)

        if end_sample > start_sample:
            wave_audio = audio_np[start_sample:end_sample].copy()
            valid_waves_with_audio.append((wave, wave_audio))
            logger.debug(
                f"Wave audio extracted: frames [{frame_start}:{frame_end}], "
                f"samples [{start_sample}:{end_sample}], "
                f"duration={wave['details']['duration_sec']:.3f}s"
            )
        else:
            logger.warning(
                f"Skipping wave with invalid sample range: "
                f"frames [{frame_start}:{frame_end}] → "
                f"samples [{start_sample}:{end_sample}] "
                f"(audio_np length={len(audio_np)})"
            )

    logger.info(f"Valid waves (with audio): {len(valid_waves_with_audio)}")
    return valid_waves_with_audio


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
