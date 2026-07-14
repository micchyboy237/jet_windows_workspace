"""
Speaker labeling logic: label extraction, aggregation, and segment audio saving.
"""
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from rich.console import Console
from core.state import (
    get_audio_tagger,
    get_current_speaker,
    get_embedding_model,  # NEW: access shared embedding model
    get_last_speaker_change_time,
    get_speaker_diarization,
    get_speaker_labeler,
    save_speaker_state,
    set_current_speaker,
    set_last_speaker_change_time,
)
from services.audio_info import display_audio_info
from services.audio_utils import get_audio_duration
from services.norm_speech_loudness import normalize_audio_for_vad
from services.quant import quantize_audio
from services.overlap_aware_diarization import split_speaker_segments  # NEW import

console = Console()

def should_label_speaker(text: str, min_chars: int = 2) -> bool:
    """
    Determine if speaker labeling should be performed based on text content.
    """
    clean_text = text.strip()
    meaningful_chars = sum(
        1
        for c in clean_text
        if c.isalnum()
        or "\u3040" <= c <= "\u309f"
        or "\u30a0" <= c <= "\u30ff"
        or "\u4e00" <= c <= "\u9fff"
        or "\u3400" <= c <= "\u4dbf"
        or "\uac00" <= c <= "\ud7af"
        or "\u0600" <= c <= "\u06ff"
        or "\u0900" <= c <= "\u097f"
        or "\u0400" <= c <= "\u04ff"
        or "\u0370" <= c <= "\u03ff"
        or "\u0e00" <= c <= "\u0e7f"
    )
    return meaningful_chars >= min_chars


def label_speakers_for_segment(
    waveform: np.ndarray,
    sample_rate: int,
    timestamp: Optional[float] = None,
    return_multiple: bool = True,
    segment_id: Optional[str] = None,
) -> tuple:
    """Label speakers for an audio segment using the progressive labeler."""
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
    
    labeler = get_speaker_labeler()
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
        segment_groups = labeler.label_segments(
            waveform=waveform_tensor,
            sample_rate=sample_rate,
            timestamp=timestamp,
            context=context,
            segment_id=segment_id,
        )
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


def perform_speaker_labeling(
    audio_np: np.ndarray,
    sample_rate: int,
    header: dict,
    full_word_segments_text: str,
    segment_id: Optional[str] = None,
    min_label_duration: float = 2.0,
    max_label_duration: float = 7.0,
    speech_threshold: float = 0.1,
    use_diarization: bool = True,  # NEW: toggle diarization vs old method
) -> tuple:
    """Perform speaker labeling using split_speaker_segments for healthier
    intra/inter speaker segmentation.
    
    The diarization approach (split_speaker_segments) produces properly 
    separated speaker turns with overlap awareness, resulting in cleaner 
    speaker assignments compared to the old speech-probability-based extraction.
    
    Falls back to the old extract_high_confidence_speech_segments method
    if diarization fails or use_diarization=False.
    """
    text_has_sufficient_content = should_label_speaker(
        full_word_segments_text, min_chars=2
    )
    
    speaker_results = []
    primary_label = None
    primary_confidence = 0.0
    speaker_metadata = {"match_type": "skipped_no_text"}
    
    if not text_has_sufficient_content:
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
    
    segment_timestamp = header.get("start_sec", time.time())
    segment_duration = header.get(
        "duration_sec",
        get_audio_duration(audio_np, sr=sample_rate)
    )
    
    extraction_info = {
        "attempted": False,
        "successful": False,
        "segments_found": 0,
        "used_segment_duration": segment_duration,
        "original_duration": segment_duration,
        "min_label_duration": min_label_duration,
        "max_label_duration": max_label_duration,
        "individual_segment_results": [],
        "method": "diarization" if use_diarization else "speech_extraction",
    }
    
    if segment_duration < min_label_duration:
        console.print(
            f"[dim]🔇 Segment too short for speaker labeling "
            f"({segment_duration:.2f}s < {min_label_duration}s), "
            f"skipping speaker labeling[/dim]"
        )
        speaker_metadata = {
            "match_type": "skipped_too_short",
            "speaker_list": [],
            "total_speakers": 0,
            "speech_extraction": extraction_info,
            "aggregation_method": "none",
        }
        return (
            text_has_sufficient_content,
            speaker_results,
            primary_label,
            primary_confidence,
            speaker_metadata,
        )
    
    extraction_info["attempted"] = True
    
    # ========================================================================
    # NEW: Primary path — use split_speaker_segments (overlap-aware diarization)
    # ========================================================================
    if use_diarization:
        try:
            embedding_model = get_embedding_model()
            if embedding_model is None:
                console.print(
                    "[warning]⚠️ Embedding model not available in state, "
                    "falling back to speech extraction method[/warning]"
                )
            else:
                console.print(
                    f"[info]🎯 Using overlap-aware diarization "
                    f"({segment_duration:.2f}s audio)...[/info]"
                )
                
                # Normalize and prepare audio
                audio_np_norm, _ = normalize_audio_for_vad(audio_np, sample_rate)
                
                # Run split_speaker_segments with the shared embedding model
                diarization_result, speaker_segments = split_speaker_segments(
                    audio_path=audio_np_norm,
                    strategy="resegment",  # balanced approach
                    condition="noisy",
                    embedding_model=embedding_model,  # Pass shared model instance
                    min_turn_dur=0.5,  # minimum turn duration
                )
                
                extraction_info["segments_found"] = len(speaker_segments)
                
                if not speaker_segments:
                    console.print(
                        f"[dim]🔇 Diarization found no speaker segments, "
                        f"falling back to speech extraction[/dim]"
                    )
                else:
                    console.print(
                        f"[success]🎯 Diarization found {len(speaker_segments)} "
                        f"speaker segment(s) with {diarization_result.n_speakers} "
                        f"speaker(s):[/success]"
                    )
                    
                    all_segment_speaker_results = []
                    
                    for i, (seg_info, seg_audio) in enumerate(speaker_segments):
                        seg_dur = seg_info["duration"]
                        seg_start = seg_info["start"]
                        seg_end = seg_info["end"]
                        seg_speaker = seg_info["speaker"]
                        seg_score = seg_info["score"]
                        seg_label = seg_info["label"]
                        
                        # Skip segments that are too short or too long
                        if seg_dur < 0.3:
                            console.print(
                                f"[dim]  [{i}] Skipping too-short segment "
                                f"({seg_dur:.2f}s < 0.3s)[/dim]"
                            )
                            continue
                        
                        # Truncate if too long for labeling
                        original_dur = seg_dur
                        if seg_dur > max_label_duration:
                            max_samples = int(max_label_duration * sample_rate)
                            seg_audio = seg_audio[:max_samples]
                            seg_dur = max_label_duration
                            console.print(
                                f"[dim]  [{i}] {seg_start:.2f}s-{seg_end:.2f}s "
                                f"(orig={original_dur:.2f}s, speaker={seg_speaker}) → "
                                f"truncated to {max_label_duration}s[/dim]"
                            )
                        else:
                            console.print(
                                f"[dim]  [{i}] {seg_start:.2f}s-{seg_end:.2f}s "
                                f"({seg_dur:.2f}s, speaker={seg_speaker}, "
                                f"score={seg_score:.3f}, label={seg_label})[/dim]"
                            )
                        
                        # Normalize and quantize segment audio
                        seg_audio_norm, _ = normalize_audio_for_vad(
                            seg_audio, sample_rate
                        )
                        seg_audio_int16, _ = quantize_audio(
                            seg_audio_norm, target_dtype="int16", 
                            sr=sample_rate, verbose=False,
                        )
                        
                        sub_segment_id = f"{segment_id}_sub{i}" if segment_id else None
                        
                        # Save segment audio for playback
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
                                    "original_duration": original_dur,
                                    "diarization_speaker": seg_speaker,
                                    "diarization_score": seg_score,
                                    "diarization_label": seg_label,
                                    "timestamp": segment_timestamp + seg_start,
                                    "truncated": seg_dur != original_dur,
                                    "max_label_duration": max_label_duration,
                                    "method": "diarization",
                                }
                            )
                        
                        # Label this specific segment
                        seg_results, seg_primary, seg_conf, seg_meta = (
                            label_speakers_for_segment(
                                waveform=seg_audio_int16,
                                sample_rate=sample_rate,
                                timestamp=segment_timestamp + seg_start,
                                return_multiple=False,
                                segment_id=sub_segment_id,
                            )
                        )
                        
                        extraction_info["individual_segment_results"].append({
                            "index": i,
                            "start_time": seg_start,
                            "end_time": seg_end,
                            "duration": seg_dur,
                            "original_duration": original_dur,
                            "diarization_speaker": seg_speaker,
                            "diarization_score": seg_score,
                            "diarization_label": seg_label,
                            "primary_label": seg_primary,
                            "primary_confidence": seg_conf,
                            "match_type": seg_meta.get("match_type", "unknown"),
                            "speaker_results": seg_results,
                        })
                        
                        all_segment_speaker_results.append({
                            "segment_index": i,
                            "start_time": seg_start,
                            "end_time": seg_end,
                            "label": seg_primary,
                            "confidence": seg_conf,
                            "match_type": seg_meta.get("match_type", "unknown"),
                            "diarization_speaker": seg_speaker,
                            "diarization_score": seg_score,
                            "is_primary": False,
                        })
                        
                        console.print(
                            f"[dim]     → Speaker: {seg_primary} "
                            f"(confidence: {seg_conf:.3f}, "
                            f"diarization: {seg_speaker}, "
                            f"type: {seg_meta.get('match_type', 'unknown')})[/dim]"
                        )
                    
                    # Aggregate results
                    speaker_results, primary_label, primary_confidence, speaker_metadata = \
                        _aggregate_speaker_results(
                            all_segment_speaker_results, extraction_info
                        )
                    
                    if speaker_results:
                        if len(speaker_results) > 1:
                            speakers_str = ", ".join(
                                f"{r['label']}({r['confidence']:.2f})" 
                                for r in speaker_results[:3]
                            )
                            console.print(
                                f"[speaker]Speakers: [{speakers_str}] "
                                f"(primary: {primary_label}, "
                                f"type: {speaker_metadata.get('match_type', 'unknown')})[/speaker]"
                            )
                        else:
                            console.print(
                                f"[speaker]Speaker: {primary_label} "
                                f"(confidence: {primary_confidence:.3f}, "
                                f"type: {speaker_metadata.get('match_type', 'unknown')})[/speaker]"
                            )
                    
                    return (
                        text_has_sufficient_content,
                        speaker_results,
                        primary_label,
                        primary_confidence,
                        speaker_metadata,
                    )
        
        except Exception as e:
            console.print(
                f"[warning]⚠️ Diarization failed: {e}, "
                f"falling back to speech extraction method[/warning]"
            )
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    # ========================================================================
    # FALLBACK: Original speech extraction method
    # ========================================================================
    try:
        tagger = get_audio_tagger()
        if tagger is None:
            console.print(
                "[dim]🔇 Audio tagger not available, "
                "skipping speaker labeling[/dim]"
            )
            speaker_metadata = {
                "match_type": "skipped_tagger_unavailable",
                "speaker_list": [],
                "total_speakers": 0,
                "speech_extraction": extraction_info,
                "aggregation_method": "none",
            }
            return (
                text_has_sufficient_content,
                speaker_results,
                primary_label,
                primary_confidence,
                speaker_metadata,
            )
        
        audio_np, _ = normalize_audio_for_vad(audio_np, sample_rate)
        audio_np, _ = quantize_audio(
            audio_np, target_dtype="int16", sr=sample_rate, verbose=False,
        )
        
        console.print(
            f"[info]🎯 Falling back to high-confidence speech extraction "
            f"(audio: {segment_duration:.2f}s, "
            f"min_label={min_label_duration}s, "
            f"max_label={max_label_duration}s)...[/info]"
        )
        
        display_audio_info(audio_np)
        
        high_conf_segments, high_conf_audios = (
            tagger.extract_high_confidence_speech_segments(
                audio=audio_np,
                sample_rate=sample_rate,
                speech_threshold=speech_threshold,
            )
        )
        
        extraction_info["segments_found"] = len(high_conf_audios)
        extraction_info["method"] = "speech_extraction"
        
        if not high_conf_audios:
            console.print(
                f"[dim]🔇 No high-confidence speech segments found, "
                f"skipping speaker labeling[/dim]"
            )
            speaker_metadata = {
                "match_type": "skipped_no_high_confidence_speech",
                "speaker_list": [],
                "total_speakers": 0,
                "speech_extraction": extraction_info,
                "aggregation_method": "none",
            }
            return (
                text_has_sufficient_content,
                speaker_results,
                primary_label,
                primary_confidence,
                speaker_metadata,
            )
        
        console.print(
            f"[success]🎯 Extracted {len(high_conf_audios)} high-confidence "
            f"speech segment(s) — labeling each individually "
            f"(max {max_label_duration}s per segment):[/success]"
        )
        
        all_segment_speaker_results = []
        
        for i, (seg, aud) in enumerate(zip(high_conf_segments, high_conf_audios)):
            seg_dur = len(aud) / sample_rate if len(aud) > 0 else 0
            seg_start = seg.get('start_time', 0)
            seg_end = seg.get('end_time', 0)
            seg_prob = seg.get('avg_speech_probability', 0)
            original_dur = seg_dur
            
            if seg_dur > max_label_duration:
                max_samples = int(max_label_duration * sample_rate)
                aud = aud[:max_samples]
                seg_dur = max_label_duration
                console.print(
                    f"[dim]  [{i}] {seg_start:.2f}s-{seg_end:.2f}s "
                    f"(orig={original_dur:.2f}s, prob={seg_prob:.3f}) → "
                    f"truncated to {max_label_duration}s for labeling...[/dim]"
                )
            else:
                console.print(
                    f"[dim]  [{i}] {seg_start:.2f}s-{seg_end:.2f}s "
                    f"({seg_dur:.2f}s, prob={seg_prob:.3f}) → labeling...[/dim]"
                )
            
            aud, _ = normalize_audio_for_vad(aud, sample_rate)
            seg_audio_int16, _ = quantize_audio(
                aud, target_dtype="int16", sr=sample_rate, verbose=False,
            )
            
            sub_segment_id = f"{segment_id}_sub{i}" if segment_id else None
            
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
                        "original_duration": original_dur,
                        "avg_speech_probability": seg_prob,
                        "timestamp": segment_timestamp + seg_start,
                        "truncated": seg_dur != original_dur,
                        "max_label_duration": max_label_duration,
                        "method": "speech_extraction",
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
            
            extraction_info["individual_segment_results"].append({
                "index": i,
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": seg_dur,
                "original_duration": original_dur,
                "avg_speech_probability": seg_prob,
                "primary_label": seg_primary,
                "primary_confidence": seg_conf,
                "match_type": seg_meta.get("match_type", "unknown"),
                "speaker_results": seg_results,
            })
            
            all_segment_speaker_results.append({
                "segment_index": i,
                "start_time": seg_start,
                "end_time": seg_end,
                "label": seg_primary,
                "confidence": seg_conf,
                "match_type": seg_meta.get("match_type", "unknown"),
                "is_primary": False,
            })
            
            console.print(
                f"[dim]     → Speaker: {seg_primary} "
                f"(confidence: {seg_conf:.3f}, type: {seg_meta.get('match_type', 'unknown')})[/dim]"
            )
        
        speaker_results, primary_label, primary_confidence, speaker_metadata = \
            _aggregate_speaker_results(all_segment_speaker_results, extraction_info)
        
        if speaker_results:
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
                f"[dim]🔇 Speaker labeling skipped: {speaker_metadata.get('match_type', 'unknown')}[/dim]"
            )
    
    except Exception as e:
        console.print(
            f"[warning]⚠️ Speech extraction failed: {e}, "
            f"skipping speaker labeling[/warning]"
        )
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        speaker_metadata = {
            "match_type": "skipped_extraction_error",
            "speaker_list": [],
            "total_speakers": 0,
            "speech_extraction": extraction_info,
            "aggregation_method": "none",
        }
    
    return (
        text_has_sufficient_content,
        speaker_results,
        primary_label,
        primary_confidence,
        speaker_metadata,
    )


def _aggregate_speaker_results(
    all_segment_speaker_results: list,
    extraction_info: dict,
) -> tuple:
    """Aggregate per-sub-segment speaker results into ranked speaker list."""
    speaker_results = []
    primary_label = None
    primary_confidence = 0.0

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
        for seg_info in extraction_info["individual_segment_results"]:
            if seg_info["index"] == result["segment_index"]:
                agg["total_duration"] += seg_info["duration"]
                break

    ranked_speakers = sorted(
        speaker_aggregates.values(),
        key=lambda x: (
            x["appearances"],
            sum(x["confidences"]) / len(x["confidences"]),
            x["total_duration"],
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
        for rank, spk in enumerate(ranked_speakers):
            is_primary = (rank == 0)
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
                "is_new_speaker": False,
                "segment_count": spk["appearances"],
                "total_speech_duration": round(spk["total_duration"], 3),
            })
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
        f"[success]🎯 Aggregated {len(all_segment_speaker_results)} segments → "
        f"{len(speaker_results)} unique speaker(s)[/success]"
    )
    for spk in speaker_results:
        console.print(
            f"[dim]   {spk['label']}: avg_conf={spk['confidence']:.3f}, "
            f"appearances={spk['segment_count']}, "
            f"duration={spk['total_speech_duration']:.2f}s, "
            f"type={spk['match_type']}{' ★ PRIMARY' if spk['is_primary'] else ''}[/dim]"
        )

    return speaker_results, primary_label, primary_confidence, speaker_metadata


def save_segment_audio_for_playback(
    audio_np: np.ndarray,
    segment_id: str,
    sample_rate: int = 16000,
    metadata: Optional[Dict] = None,
) -> Optional[Path]:
    """
    Save segment audio as WAV file permanently for playback in segment detail page.
    """
    import wave
    import json
    from services.config import SEGMENT_AUDIO_DIR, SEGMENT_AUDIO_INDEX
    if audio_np.size == 0:
        console.print(f"[warning]Cannot save empty audio for segment {segment_id}[/]")
        return None
    try:
        SEGMENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = SEGMENT_AUDIO_DIR / f"{segment_id}.wav"
        if audio_np.dtype != np.int16:
            audio_np, _ = normalize_audio_for_vad(audio_np, sample_rate)
            audio_int16, _ = quantize_audio(
                audio_np, target_dtype="int16", sr=sample_rate, verbose=False,
            )
        else:
            audio_int16 = audio_np
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        duration_sec = get_audio_duration(audio_int16, sr=sample_rate)
        console.print(
            f"[success]Saved segment audio: {segment_id}.wav "
            f"({len(audio_int16)} samples, {duration_sec:.2f}s) → {audio_path}[/]"
        )
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
        if len(audio_index) > 500:
            sorted_items = sorted(
                audio_index.items(),
                key=lambda x: x[1].get('saved_at', 0),
                reverse=True
            )[:500]
            old_count = len(audio_index) - len(sorted_items)
            audio_index = dict(sorted_items)
            console.print(f"[dim]Cleaning up {old_count} old audio files...[/]")
            for audio_file in SEGMENT_AUDIO_DIR.glob("*.wav"):
                if audio_file.stem not in audio_index:
                    try:
                        audio_file.unlink()
                        console.print(f"[dim]Cleaned up old audio: {audio_file.name}[/]")
                    except Exception as e:
                        console.print(f"[warning]Could not delete old audio {audio_file.name}: {e}[/]")
        import tempfile
        index_dir = SEGMENT_AUDIO_INDEX.parent
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', dir=index_dir, delete=False
        ) as tmp:
            json.dump(audio_index, tmp, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(SEGMENT_AUDIO_INDEX)
        return audio_path
    except Exception as e:
        console.print(f"[error]Failed to save segment audio for {segment_id}: {e}[/]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        return None
