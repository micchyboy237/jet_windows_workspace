"""
Audio tagging: speech detection, chunked processing, and result aggregation.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from rich.console import Console

from core.state import get_audio_tagger
from services.audio_tagger import (
    DEFAULT_CHUNK_DURATION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SPEECH_PROB_THRESHOLD,
)
from services.audio_utils import get_audio_duration
from services.norm_speech_loudness import normalize_audio_for_vad
from services.quant import quantize_audio
from services.save_utils import save_tagging_to_segment

console = Console()


def perform_audio_tagging(
    audio_np: np.ndarray,
    sample_rate: int,
    segment_dir: Optional[Path] = None,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap_duration: float = DEFAULT_CHUNK_OVERLAP,
    speech_prob_threshold: float = DEFAULT_SPEECH_PROB_THRESHOLD,
    min_speech_duration: float = 0.8,
) -> Dict[str, Any]:
    """
    Perform audio tagging on an audio segment and save results.
    """
    console.print("[info]🎵 Starting audio tagging...[/info]")
    console.print(
        f"[info]Audio shape: {audio_np.shape}, Sample rate: {sample_rate}[/info]"
    )
    audio_duration = get_audio_duration(audio_np, sr=sample_rate)
    console.print(f"[info]Audio duration: {audio_duration:.2f}s[/info]")
    try:
        tagger = get_audio_tagger()
        audio_np, _ = normalize_audio_for_vad(audio_np, sample_rate)
        audio_np, _ = quantize_audio(
            audio_np, target_dtype="int16", sr=sample_rate, verbose=False,
        )
        console.print(
            f"[info]Using chunked processing "
            f"(audio {audio_duration:.2f}s > {chunk_duration * 2:.1f}s)[/info]"
        )
        chunked_summary = tagger.tag_audio_chunks(
            audio=audio_np,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
        )
        speech_duration = chunked_summary.get("speech_duration", 0.0)
        avg_speech_prob = chunked_summary.get("avg_speech_probability", 0.0)
        max_speech_prob = chunked_summary.get("max_speech_probability", 0.0)
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
            "speech_duration": round(speech_duration, 4),
            "avg_speech_probability": round(avg_speech_prob, 4),
            "max_speech_probability": round(max_speech_prob, 4),
            "speech_prob_threshold": speech_prob_threshold,
            "min_speech_duration_threshold": min_speech_duration,
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
            "speech_duration": 0.0,
            "avg_speech_probability": 0.0,
            "max_speech_probability": 0.0,
            "speech_prob_threshold": speech_prob_threshold,
            "min_speech_duration_threshold": min_speech_duration,
            "error": str(e),
            "processing_mode": "failed",
            "top_predictions": [],
        }
