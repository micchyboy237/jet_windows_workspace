# servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_audio_tagger.py

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_rich_traceback
from custom_logging import linkify
from audio_tagger import (
    AUDIO_TAGGING_MODEL,
    CLASS_LABELS_INDICES_CSV,
    AudioChunksTaggingSummary,
    AudioTagger,
    ChunkTaggingResult,
    DEFAULT_MIN_SPEECH_PROB_THRESHOLD,
    DEFAULT_SPEECH_PROB_THRESHOLD,
    DEFAULT_SPEECH_TOP_N,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MIN_SILENCE_DURATION_SEC,
    DEFAULT_MIN_SPEECH_DURATION_SEC,
    DEFAULT_RESOLUTION_MS,
    calculate_confidence_tier,
    GENDER_CLASS_INDICES,
    DEFAULT_GENDER_CONFIDENCE_THRESHOLD,
)
from serialization_utils import serialize
from audio_config import FRAME_SHIFT_MS, SAMPLE_RATE
install_rich_traceback(show_locals=True)
console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_args(
    default_audio: Optional[str] = None,
    default_output_dir: Optional[str | Path] = None,
):
    DEFAULT_AUDIO = default_audio or r"C:\Users\druiv\.cache\files\audio\sub_audio\start_5s_recording_1_speaker.wav"
    parser = argparse.ArgumentParser(
        description="Audio tagging with Sherpa-ONNX models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        default=DEFAULT_AUDIO,
        help="Path to input audio file",
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default=str(AUDIO_TAGGING_MODEL),
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "-l", "--labels-path",
        type=str,
        default=str(CLASS_LABELS_INDICES_CSV),
        help="Path to class labels CSV file",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top predictions to return",
    )
    parser.add_argument(
        "-j", "--num-threads",
        type=int,
        default=1,
        help="Number of CPU threads (jobs)",
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="Computation provider",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode for Sherpa-ONNX",
    )
    parser.add_argument(
        "-mt", "--min-speech-threshold",
        type=float,
        default=DEFAULT_MIN_SPEECH_PROB_THRESHOLD,
        help="Minimum valid probability for speech detection",
    )
    parser.add_argument(
        "-t", "--speech-threshold",
        type=float,
        default=DEFAULT_SPEECH_PROB_THRESHOLD,
        help="Minimum probability for speech detection",
    )
    parser.add_argument(
        "-n", "--speech-top-n",
        type=int,
        default=DEFAULT_SPEECH_TOP_N,
        help="Check top N predictions for speech",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=str(default_output_dir or OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "-s", "--check-speech",
        action="store_true",
        help="Check if audio contains speech",
    )
    parser.add_argument(
        "-C", "--chunk",
        action="store_true",
        default=True,
        help="Use tag_audio_chunks instead of tag_audio",
    )
    parser.add_argument(
        "-D", "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION,
        help="Duration of each chunk in seconds",
    )
    parser.add_argument(
        "-O", "--chunk-overlap",
        type=float,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap between chunks in seconds",
    )
    parser.add_argument(
        "--min-silence", type=float, default=DEFAULT_MIN_SILENCE_DURATION_SEC,
        help="Min silence duration (s) to split speech segments"
    )
    parser.add_argument(
        "--min-speech", type=float, default=DEFAULT_MIN_SPEECH_DURATION_SEC,
        help="Min speech duration (s) to keep a segment"
    )
    parser.add_argument(
        "--resolution-ms", type=float, default=float(FRAME_SHIFT_MS),
        help=f"Timeline resolution in ms for segment detection (default: {FRAME_SHIFT_MS}ms = 1 frame hop)"
    )
    args = parser.parse_args()
    return args


def _create_play_button(audio_path: Path) -> str:
    """
    Create a clickable play button for an audio file.
    
    Args:
        audio_path: Path to the audio file (e.g., sound.wav)
        
    Returns:
        Rich-formatted string with play button link
    """
    if audio_path.exists():
        # Convert to absolute path with forward slashes for file:// URL
        file_url = f"file:///{str(audio_path.absolute()).replace(os.sep, '/')}"
        return f"[bold cyan][link={file_url}]▶ Play[/link][/bold cyan]"
    return "[dim]—[/dim]"


def _create_clickable_segment_name(segment_dir: Path) -> str:
    """
    Create a clickable segment name that opens the directory.
    
    Args:
        segment_dir: Path to the segment directory
        
    Returns:
        Rich-formatted string with clickable directory link
    """
    dir_url = f"file:///{str(segment_dir.absolute()).replace(os.sep, '/')}"
    return f"[link={dir_url}]{segment_dir.name}[/link]"


def _format_predictions_with_emphasis(predictions, threshold=0.3, max_display=3):
    """
    Format multiple predictions with visual emphasis based on probability magnitude.
    
    Args:
        predictions: List of prediction dicts with 'name' and 'prob' keys
        threshold: Minimum probability to display
        max_display: Maximum number of predictions to show
        
    Returns:
        Rich Text object with color-coded predictions
        
    Probability magnitude emphasis:
        - High (≥0.7): Bold green
        - Medium (0.4-0.7): Yellow
        - Low (0.3-0.4): Dim white
        - Below threshold: Not shown
    """
    text = Text()
    qualified = [p for p in predictions if p.get("prob", 0) >= threshold]
    qualified.sort(key=lambda x: x.get("prob", 0), reverse=True)
    
    if not qualified:
        return Text("—", style="dim")
    
    for i, pred in enumerate(qualified[:max_display]):
        prob = pred["prob"]
        name = pred["name"]
        display_name = name[:35] + "…" if len(name) > 35 else name
        
        if prob >= 0.7:
            style = "bold green"
            emoji = "🔴"
        elif prob >= 0.4:
            style = "yellow"
            emoji = "🟡"
        else:
            style = "dim white"
            emoji = "⚪"
        
        if i > 0:
            text.append("\n")
        
        prob_bar = _get_probability_bar(prob)
        text.append(f"{emoji} ", style="")
        text.append(f"{display_name} ", style=style)
        text.append(f"{prob:.1%} ", style=style)
        text.append(f"[{prob_bar}]", style="dim")
    
    return text


def _get_probability_bar(probability, width=10):
    """
    Create a visual bar indicating probability magnitude.
    
    Args:
        probability: Float between 0 and 1
        width: Total width of the bar in characters
        
    Returns:
        String with filled and empty blocks representing probability
    """
    filled = int(probability * width)
    empty = width - filled
    
    if probability >= 0.7:
        bar_char = "█"
    elif probability >= 0.4:
        bar_char = "▓"
    else:
        bar_char = "▒"
    
    return f"{bar_char * filled}{'░' * empty}"


def display_per_chunk_analysis(
    summary: AudioChunksTaggingSummary,
    probability_threshold: float = 0.3,
    max_predictions_display: int = 3,
    show_summary_table: bool = True,
    show_chunk_table: bool = True,
    title: str = "Per-Chunk Analysis",
) -> None:
    """
    Display detailed per-chunk analysis from a chunked tagging summary.
    Updated to include gender distribution stats and full probability precision.
    """
    console.print(f"[dim]📊 display_per_chunk_analysis: {len(summary.get('chunks', []))} chunks[/dim]")

    if show_summary_table:
        chunk_summary_table = Table(
            title="Chunk Analysis Summary",
            border_style="blue",
            show_header=True,
            header_style="bold cyan",
        )
        chunk_summary_table.add_column("Metric", style="cyan")
        chunk_summary_table.add_column("Value", style="yellow")

        # --- Gender Statistics Calculation ---
        chunks = summary.get("chunks", [])
        male_count = sum(1 for c in chunks if c.get("dominant_gender") == "male")
        female_count = sum(1 for c in chunks if c.get("dominant_gender") == "female")
        unknown_count = sum(1 for c in chunks if c.get("dominant_gender") != "male" and c.get("dominant_gender") != "female")
        total_chunks = len(chunks) if chunks else 1 # Avoid div by zero

        male_pct = (male_count / total_chunks) * 100
        female_pct = (female_count / total_chunks) * 100
        unknown_pct = (unknown_count / total_chunks) * 100
        
        gender_stats_str = (
            f"[blue]{male_count} Male ({male_pct:.1f}%)[/blue] | "
            f"[magenta]{female_count} Female ({female_pct:.1f}%)[/magenta] | "
            f"[dim]{unknown_count} Unknown ({unknown_pct:.1f}%)[/dim]"
        )
        # --------------------------------------

        chunk_summary_table.add_row("Total Duration", f"{summary['total_duration']:.2f}s")
        chunk_summary_table.add_row("Total Chunks", str(summary["total_chunks"]))
        chunk_summary_table.add_row("Chunk Duration", f"{summary['chunk_duration']:.2f}s")
        chunk_summary_table.add_row("Overlap", f"{summary['overlap_duration']:.2f}s")
        chunk_summary_table.add_row(
            "Speech Detected",
            "✅ Yes" if summary["speech_detected"] else "❌ No",
        )
        chunk_summary_table.add_row(
            "Speech Duration",
            f"{summary['speech_duration']:.2f}s"
            f" ({summary['speech_duration'] / summary['total_duration'] * 100:.1f}% of total)"
            if summary["total_duration"] > 0
            else "0.00s",
        )
        chunk_summary_table.add_row(
            "Max Speech Probability", f"{summary['max_speech_probability']}" # Full precision
        )
        chunk_summary_table.add_row(
            "Avg Speech Probability",
            f"{summary['avg_speech_probability']}" # Full precision
            if summary["avg_speech_probability"] > 0
            else "N/A (no speech chunks)",
        )
        
        # Add Gender Row
        chunk_summary_table.add_row("Gender Distribution", gender_stats_str)

        chunk_summary_table.add_row(
            "Processing Time", f"{summary['total_processing_time']:.3f}s"
        )
        chunk_summary_table.add_row(
            "Real-Time Factor", f"{summary['real_time_factor']:.3f}x"
        )
        console.print(chunk_summary_table)
        console.print(f"[dim]✅ Chunk summary table displayed[/dim]")

    if show_chunk_table:
        console.print("\n[bold]Overall Top Predictions:[/bold]")
        if summary.get("overall_top_predictions"):
            predictions_table = Table(
                title="Overall Top Predictions (Aggregated)",
                border_style="green",
                show_header=True,
                header_style="bold cyan",
            )
            predictions_table.add_column("Rank", style="cyan", justify="right")
            predictions_table.add_column("Name", style="green")
            predictions_table.add_column("Mean Probability", style="magenta", justify="right")
            for i, pred in enumerate(summary["overall_top_predictions"], 1):
                prob_color = "green" if pred["prob"] >= 0.5 else "yellow"
                # Use full precision here as well
                predictions_table.add_row(
                    str(i),
                    pred["name"],
                    f"[{prob_color}]{pred['prob']}[/{prob_color}]",
                )
            console.print(predictions_table)

        chunk_table = Table(
            title=title,
            border_style="blue",
            show_header=True,
            header_style="bold cyan",
        )
        chunk_table.add_column("Chunk", justify="right", style="cyan")
        chunk_table.add_column("Time Range", style="yellow")
        chunk_table.add_column("Duration", justify="right")
        chunk_table.add_column("Speech", justify="center", style="green")
        chunk_table.add_column("Gender", justify="center", style="magenta")
        chunk_table.add_column("Top Predictions", style="green", min_width=40)
        chunk_table.add_column("Proc Time", justify="right")

        for chunk in summary["chunks"]:
            predictions = chunk.get("predictions", [])
            predictions_display = _format_predictions_with_emphasis(
                predictions,
                threshold=probability_threshold,
                max_display=max_predictions_display,
            )

            speech_indicator = (
                f"✅ {chunk['speech_probability']}" # Full precision
                if chunk.get("speech_detected", False)
                else "❌ --"
            )

            # --- New Gender Logic ---
            dominant_gender = chunk.get("dominant_gender", "unknown")
            male_prob = chunk.get("male_prob", 0.0)
            female_prob = chunk.get("female_prob", 0.0)
            
            if dominant_gender == "male":
                gender_icon = "👨"
                gender_color = "blue"
            elif dominant_gender == "female":
                gender_icon = "👩"
                gender_color = "magenta"
            else:
                gender_icon = "❓"
                gender_color = "dim"
                
            # Show full precision for gender probs
            gender_display = f"[{gender_color}]{gender_icon} M:{male_prob} F:{female_prob}[/{gender_color}]"
            # ------------------------

            chunk_table.add_row(
                str(chunk["chunk_index"]),
                f"{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s",
                f"{chunk['duration']:.2f}s",
                speech_indicator,
                gender_display,
                predictions_display,
                f"{chunk['processing_time'] * 1000:.1f}ms",
            )

        console.print(chunk_table)
        console.print(f"[dim]✅ Per-chunk table displayed with {len(summary['chunks'])} rows[/dim]")


def display_per_segment_analysis(
    segment_dirs: List[Path],
    show_speech_segments: bool = True,
    show_non_speech_segments: bool = True,
    show_details: bool = True,
) -> None:
    """
    Display analysis of speech and non-speech segments saved to disk in table format.
    
    Confidence Levels:
        ✨ High   - Strong, consistent speech signal (avg_prob ≥ 0.6 AND density ≥ 70%)
        ⚠ Medium - Probable speech with some uncertainty
        — Low    - Weak or inconsistent signal
    
    This function reads the segment.json files from saved segment directories
    and displays them in rich-formatted tables with timing, speech statistics,
    insights, and clickable play buttons for audio files.
    
    Args:
        segment_dirs: List of Path objects pointing to segment directories
        show_speech_segments: Whether to display speech segment tables
        show_non_speech_segments: Whether to display non-speech segment tables
        show_details: Whether to show detailed statistics for each segment
    
    Example:
        >>> segment_dirs = save_speech_segments(summary, output_dir)
        >>> display_per_segment_analysis(segment_dirs)
    
    Debug logs trace:
        - Number of segment directories found
        - Classification of speech vs non-speech segments
        - Table generation for each segment type
        - Audio file availability for play buttons
        - Speech duration calculations with threshold context
        - Confidence tier distribution statistics
    """
    console.print(f"[dim]📂 display_per_segment_analysis: {len(segment_dirs)} segment dirs[/dim]")

    speech_dirs = [d for d in segment_dirs if "non_speech" not in d.name]
    non_speech_dirs = [d for d in segment_dirs if "non_speech" in d.name]

    console.print(
        f"[dim]📊 Found {len(speech_dirs)} speech, {len(non_speech_dirs)} non-speech segments[/dim]"
    )

    if show_speech_segments and speech_dirs:
        speech_table = Table(
            title="🗣️ Speech Segments Analysis",
            border_style="green",
            show_header=True,
            header_style="bold cyan",
        )
        speech_table.add_column("Segment", style="cyan", justify="right")
        speech_table.add_column("Time Range", style="yellow")
        speech_table.add_column("Duration", justify="right")
        if show_details:
            speech_table.add_column("Avg Speech Prob", justify="right", style="green")
            speech_table.add_column("Gender", justify="center", style="magenta") # New Column
            speech_table.add_column("Top Prediction", style="magenta")
            speech_table.add_column("Confidence", justify="center", style="bold")
            speech_table.add_column("", justify="center", style="bold cyan")

        segments_with_audio = 0
        total_speech_duration = 0.0
        threshold_used = None
        
        # Gender stats
        male_count = 0
        female_count = 0
        unknown_count = 0

        for seg_dir in speech_dirs:
            segment_json = seg_dir / "segment.json"
            if segment_json.exists():
                try:
                    with open(segment_json, "r", encoding="utf-8") as f:
                        seg_info = json.load(f)
                    
                    duration = seg_info.get("duration", 0)
                    speech_stats = seg_info.get("speech_stats", {})
                    insights = seg_info.get("insights", {})
                    parameters = seg_info.get("parameters", {})
                    
                    if threshold_used is None:
                        threshold_used = parameters.get("speech_threshold", 0.5)
                    
                    speech_density = speech_stats.get("speech_density", 0)
                    segment_speech_duration = duration * speech_density
                    total_speech_duration += segment_speech_duration
                    
                    confidence_tier = insights.get("confidence_tier", "low")
                    confidence_label = insights.get("confidence_label", "❓")
                    
                    if confidence_tier == "high":
                        conf_style = "bold green"
                    elif confidence_tier == "medium":
                        conf_style = "yellow"
                    else:
                        conf_style = "dim"

                    # --- New Gender Logic ---
                    dominant_gender = insights.get("dominant_gender", "unknown")
                    male_prob = insights.get("male_probability", 0.0)
                    female_prob = insights.get("female_probability", 0.0)
                    gender_conf = insights.get("gender_confidence", "low")
                    
                    if dominant_gender == "male":
                        gender_icon = "👨"
                        gender_color = "blue"
                        male_count += 1
                    elif dominant_gender == "female":
                        gender_icon = "👩"
                        gender_color = "magenta"
                        female_count += 1
                    else:
                        gender_icon = "❓"
                        gender_color = "dim"
                        unknown_count += 1
                        
                    gender_display = f"[{gender_color}]{gender_icon} {dominant_gender.capitalize()} ({gender_conf})[/{gender_color}]"
                    # ------------------------

                    segment_name = _create_clickable_segment_name(seg_dir)
                    row = [
                        segment_name,
                        f"{seg_info.get('start_time', 0):.3f}s – {seg_info.get('end_time', 0):.3f}s",
                        f"{duration:.3f}s",
                    ]
                    if show_details:
                        row.extend([
                            f"{speech_stats.get('avg_speech_probability', 0):.3f}",
                            gender_display, # Added to row
                            f"{insights.get('top_prediction', 'Unknown')} "
                            f"({insights.get('top_prediction_prob', 0):.3f})",
                            f"[{conf_style}]{confidence_label}[/{conf_style}]",
                        ])
                    
                    wav_path = seg_dir / "sound.wav"
                    play_button = _create_play_button(wav_path)
                    if "▶ Play" in play_button:
                        segments_with_audio += 1
                    row.append(play_button)
                    
                    speech_table.add_row(*row)

                except Exception as e:
                    console.print(
                        f"[yellow]⚠️ Failed to read segment.json for {seg_dir}: {e}[/yellow]"
                    )
        
        console.print(speech_table)
        
        if speech_dirs:
            console.print(
                f"[dim]📊 Speech segments summary: "
                f"{len(speech_dirs)} segments, "
                f"total speech duration: {total_speech_duration:.2f}s[/dim]"
            )
            console.print(
                f"[dim]👥 Gender distribution: "
                f"[blue]{male_count} Male[/blue] | "
                f"[magenta]{female_count} Female[/magenta] | "
                f"[dim]{unknown_count} Unknown[/dim][/dim]"
            )
        
        console.print(f"[dim]✅ Speech segments table displayed[/dim]")
        
        if segments_with_audio > 0:
            console.print(
                f"[dim]🎵 {segments_with_audio}/{len(speech_dirs)} speech segments have playable audio[/dim]"
            )
        else:
            console.print(
                "[yellow]⚠ No audio files found for speech segments. Play buttons disabled.[/yellow]"
            )
    
    if show_non_speech_segments and non_speech_dirs:
        non_speech_table = Table(
            title="🔇 Non-Speech Segments Analysis",
            border_style="yellow",
            show_header=True,
            header_style="bold cyan",
        )
        non_speech_table.add_column("Segment", style="yellow", justify="right")
        non_speech_table.add_column("Time Range", style="yellow")
        non_speech_table.add_column("Duration", justify="right")
        if show_details:
            non_speech_table.add_column("Avg Speech Prob", justify="right", style="dim")
            non_speech_table.add_column("Speech Density", justify="right", style="dim")
            non_speech_table.add_column("Speech Leakage", justify="right", style="yellow")
        non_speech_table.add_column("", justify="center", style="bold cyan")
        
        segments_with_audio = 0
        total_non_speech_duration = 0.0
        total_speech_leakage = 0.0
        threshold_used = None
        
        for seg_dir in non_speech_dirs:
            segment_json = seg_dir / "segment.json"
            if segment_json.exists():
                try:
                    with open(segment_json, "r", encoding="utf-8") as f:
                        seg_info = json.load(f)
                    
                    duration = seg_info.get("duration", 0)
                    speech_stats = seg_info.get("speech_stats", {})
                    parameters = seg_info.get("parameters", {})
                    
                    if threshold_used is None:
                        threshold_used = parameters.get("speech_threshold", 0.5)
                    
                    speech_density = speech_stats.get("speech_density", 0)
                    segment_speech_leakage = duration * speech_density
                    total_non_speech_duration += duration
                    total_speech_leakage += segment_speech_leakage
                    
                    segment_name = _create_clickable_segment_name(seg_dir)
                    
                    row = [
                        segment_name,
                        f"{seg_info.get('start_time', 0):.3f}s – {seg_info.get('end_time', 0):.3f}s",
                        f"{duration:.3f}s",
                    ]
                    
                    if show_details:
                        if segment_speech_leakage > 0.5:
                            leakage_style = "yellow"
                            leakage_emoji = "⚠"
                        elif segment_speech_leakage > 0.1:
                            leakage_style = "dim yellow"
                            leakage_emoji = "🔇"
                        else:
                            leakage_style = "dim"
                            leakage_emoji = "✅"
                        
                        row.extend([
                            f"{speech_stats.get('avg_speech_probability', 0):.3f}",
                            f"{speech_density:.1%}",
                            f"{leakage_emoji} {segment_speech_leakage:.3f}s",
                        ])
                    
                    wav_path = seg_dir / "sound.wav"
                    play_button = _create_play_button(wav_path)
                    if "▶ Play" in play_button:
                        segments_with_audio += 1
                    row.append(play_button)
                    
                    non_speech_table.add_row(*row)
                    
                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Failed to read segment.json for {seg_dir}: {e}[/yellow]"
                    )
                    segment_name = _create_clickable_segment_name(seg_dir)
                    wav_path = seg_dir / "sound.wav"
                    non_speech_table.add_row(
                        segment_name,
                        "N/A",
                        "N/A",
                        "—",
                        "—",
                        "N/A",
                        _create_play_button(wav_path),
                    )
        
        console.print(non_speech_table)
        
        if non_speech_dirs:
            leakage_percentage = (
                total_speech_leakage / total_non_speech_duration * 100
            ) if total_non_speech_duration > 0 else 0
            threshold_info = f" (threshold: {threshold_used:.2f})" if threshold_used else ""
            console.print(
                f"[dim]📊 Non-speech segments summary: "
                f"{len(non_speech_dirs)} segments, "
                f"total duration: {total_non_speech_duration:.2f}s, "
                f"speech leakage: {total_speech_leakage:.2f}s "
                f"({leakage_percentage:.1f}%){threshold_info}[/dim]"
            )
            if leakage_percentage > 10:
                console.print(
                    f"[yellow]⚠ High speech leakage ({leakage_percentage:.1f}%) in non-speech segments. "
                    f"Consider adjusting speech threshold or min_silence parameters.[/yellow]"
                )
        
        console.print(f"[dim]✅ Non-speech segments table displayed[/dim]")
        
        if segments_with_audio > 0:
            console.print(
                f"[dim]🎵 {segments_with_audio}/{len(non_speech_dirs)} non-speech segments have playable audio[/dim]"
            )
        else:
            console.print(
                "[yellow]⚠ No audio files found for non-speech segments. Play buttons disabled.[/yellow]"
            )


def _build_prob_timeline(
    chunks: List[ChunkTaggingResult],
    resolution_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a continuous speech-probability timeline from overlapping chunks.
    
    For each (time, speech_prob) observation contributed by a chunk, the
    probability is spread uniformly over the chunk's interval.  Where chunks
    overlap the per-cell values are accumulated via a weighted sum and then
    divided by the total weight (coverage count), giving a coverage-weighted
    average — the correct formula for non-uniform overlap grids.

    Formula for each timeline cell t covered by chunk i:
        prob_timeline[t] += chunk_speech_prob[i]   # accumulate
        weight[t]        += 1                       # count coverage
    
    Final:
        prob_timeline[t] /= weight[t]              # weighted mean

    Args:
        chunks: Chunk results with start_time, end_time, speech_probability.
        resolution_ms: Timeline resolution in milliseconds (default 10 ms).
        
    Returns:
        (times_sec, probs) arrays of equal length:
            times_sec — centre of each cell in seconds
            probs     — consolidated speech probability at that time
    """
    if not chunks:
        return np.array([]), np.array([])
    
    total_end = max(c["end_time"] for c in chunks)
    step = resolution_ms / 1000.0
    n_cells = max(1, int(np.ceil(total_end / step)))
    
    prob_acc = np.zeros(n_cells, dtype=np.float64)
    weight   = np.zeros(n_cells, dtype=np.float64)
    
    for chunk in chunks:
        t0  = chunk["start_time"]
        t1  = chunk["end_time"]
        sp  = chunk.get("speech_probability", 0.0)
        i0 = int(t0 / step)
        i1 = min(int(np.ceil(t1 / step)), n_cells)
        prob_acc[i0:i1] += sp
        weight[i0:i1]   += 1.0
    
    covered = weight > 0
    probs = np.where(covered, prob_acc / np.where(covered, weight, 1.0), 0.0)
    times = (np.arange(n_cells) + 0.5) * step
    
    console.print(
        f"[dim]🕑 Built prob timeline: {n_cells} cells @ {resolution_ms}ms "
        f"resolution, total_end={total_end:.3f}s[/dim]"
    )
    
    return times.astype(np.float32), probs.astype(np.float32)


def save_speech_segments(
    summary: AudioChunksTaggingSummary,
    output_dir: Path,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_DURATION_SEC,
    resolution_ms: float = DEFAULT_RESOLUTION_MS,
    include_non_speech: bool = False,
    waveform: Optional[np.ndarray] = None,
    sample_rate: Optional[int] = None,
    speech_threshold: float = 0.5,
) -> List[Path]:
    """
    Merge consecutive speech/non-speech chunks and persist each speech segment.
    Only meaningful for chunks mode (requires AudioChunksTaggingSummary).
    Algorithm
    ---------
    1. Build a coverage-weighted speech-probability timeline from all chunks,
       consolidating overlapping regions with the proper weighted-mean formula.
    2. Walk the timeline to identify speech ON/OFF transitions:
       - A run of non-speech cells lasting >= min_silence_duration_sec ends the
         current speech segment.
       - A completed speech segment shorter than min_speech_duration_sec is
         discarded.
    3. For each surviving segment write under:
         output_dir/segments/segment_001/
             segment.json  — timing, speech stats, overall insights
             chunks.json   — array of chunks whose windows overlap this segment
             probs.json    — array of {time, prob} objects at timeline resolution
             plots.png     — speech prob chart + predictions heatmap
             sound.wav     — extracted audio for this segment
    4. If include_non_speech=True, also saves non-speech segments as:
         output_dir/segments/non_speech_segment_001/
             (same file structure as speech segments)
    Args:
        summary: Result of AudioTagger.tag_audio_chunks().
        output_dir: Root directory; segments go under output_dir/segments/.
        min_silence_duration_sec: Continuous non-speech gap (seconds) needed to
            close the current speech segment. Default 1.0 s.
        min_speech_duration_sec: Discard segments shorter than this. Default 1.0 s.
        resolution_ms: Resolution of the internal probability timeline in ms.
        include_non_speech: If True, also save non-speech segments. Default False.
        waveform: Optional numpy array of the original audio waveform.
            If not provided, will try to load from summary['audio_path'].
        sample_rate: Sample rate of the provided waveform. Required if waveform is provided.
        speech_threshold: Speech probability threshold (0.0-1.0). Default 0.5.
    Returns:
        List of Paths to each created segment directory.
    Debug logs trace:
        - Threshold validation
        - Timeline building with cell statistics
        - Speech segment detection with transition logging
        - Non-speech segment identification (with deduplication)
        - Segment saving details
    """
    import soundfile as sf
    if speech_threshold <= 0.0 or speech_threshold > 1.0:
        console.print(
            f"[yellow]⚠ Invalid speech threshold {speech_threshold}, using 0.5[/yellow]"
        )
        speech_threshold = 0.5
    chunks = summary.get("chunks", [])
    if not chunks:
        console.print("[yellow]⚠ save_speech_segments: no chunks in summary, skipping[/yellow]")
        return []
    actual_sr = sample_rate or summary.get("sample_rate", 16000)
    if waveform is None:
        audio_path = summary.get("audio_path", "")
        if (audio_path and
            Path(audio_path).exists() and
            not audio_path.startswith("bytes_input_") and
            not audio_path.startswith("array_input_")):
            try:
                from audio_utils import load_audio
                waveform, actual_sr = load_audio(audio_path, sr=actual_sr, mono=True)
                console.print(
                    f"[dim]📊 Loaded original audio from file: "
                    f"{len(waveform)/actual_sr:.2f}s, {actual_sr}Hz[/dim]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Failed to load original audio file: {e}. "
                    f"Segment audio extraction disabled.[/yellow]"
                )
                waveform = None
        else:
            console.print(
                "[yellow]⚠ Original audio file not found or not a file path. "
                "Segment audio extraction disabled. "
                "Pass waveform parameter directly to enable.[/yellow]"
            )
            waveform = None
    else:
        console.print(
            f"[dim]📊 Using provided waveform: "
            f"{len(waveform)/actual_sr:.2f}s, {actual_sr}Hz[/dim]"
        )
    console.print(
        Panel.fit(
            f"[bold cyan]save_speech_segments[/bold cyan]\n"
            f"chunks={len(chunks)} | "
            f"speech_threshold={speech_threshold:.2f} | "
            f"min_silence={min_silence_duration_sec}s | "
            f"min_speech={min_speech_duration_sec}s | "
            f"resolution={resolution_ms}ms | "
            f"include_non_speech={include_non_speech}",
            title="Save Speech Segments",
            border_style="cyan",
        )
    )
    times, probs = _build_prob_timeline(chunks, resolution_ms=resolution_ms)
    if len(times) == 0:
        console.print("[yellow]⚠ Empty probability timeline[/yellow]")
        return []
    console.print(f"[dim]🎚 Using speech threshold: {speech_threshold}[/dim]")
    step = resolution_ms / 1000.0
    min_silence_cells = max(1, int(np.ceil(min_silence_duration_sec / step)))
    min_speech_cells = max(1, int(np.ceil(min_speech_duration_sec / step)))
    console.print(
        f"[dim]🔧 min_silence_cells={min_silence_cells} "
        f"(={min_silence_duration_sec}s) | "
        f"min_speech_cells={min_speech_cells} "
        f"(={min_speech_duration_sec}s)[/dim]"
    )
    is_speech = probs >= speech_threshold
    speech_cell_count = np.sum(is_speech)
    total_cells = len(is_speech)
    total_timeline_duration = times[-1] - times[0] if len(times) > 0 else 0
    console.print(
        f"[dim]📊 Timeline: {speech_cell_count}/{total_cells} cells above threshold "
        f"({speech_cell_count/total_cells*100:.1f}%) | "
        f"span={total_timeline_duration:.3f}s[/dim]"
    )

    # ===== DEBUG: Timeline Diagnostics =====
    console.print(f"\n[bold yellow]🔍 DEBUG: Timeline Diagnostics[/bold yellow]")
    
    # Show key chunk boundaries
    chunk_starts = sorted(set(c["start_time"] for c in chunks))
    chunk_ends = sorted(set(c["end_time"] for c in chunks))
    console.print(f"[yellow]Chunk starts: {[f'{t:.3f}s' for t in chunk_starts]}[/yellow]")
    console.print(f"[yellow]Chunk ends:   {[f'{t:.3f}s' for t in chunk_ends]}[/yellow]")
    
    # Show timeline in critical regions (around chunk 11 and segment boundaries)
    critical_regions = [
        (0.0, 3.5, "Full first 3.5s"),
        (2.7, 3.3, "Chunk 11 boundary region (2.75-3.25)"),
        (1.4, 1.6, "Segment 001/non_speech boundary (~1.495s)"),
    ]
    
    for region_start, region_end, label in critical_regions:
        mask = (times >= region_start) & (times <= region_end)
        region_times = times[mask]
        region_probs = probs[mask]
        
        if len(region_times) == 0:
            continue
            
        console.print(f"\n[yellow]📊 {label}:[/yellow]")
        console.print(f"[dim]   {'Time':>8s}  {'Prob':>8s}  {'Above?':>6s}  Visualization[/dim]")
        
        for t, p in zip(region_times, region_probs):
            above = "✅" if p >= speech_threshold else "❌"
            bar_len = min(int(p * 50), 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            console.print(f"   {t:>8.3f}  {p:>8.4f}  {above:>6s}  {bar}")
        
        # Show which chunks cover this region
        console.print(f"[dim]   Chunks covering this region:[/dim]")
        for c in chunks:
            if c["end_time"] > region_start and c["start_time"] < region_end:
                sp = c.get("speech_probability", 0.0)
                detected = c.get("speech_detected", False)
                console.print(
                    f"[dim]     Chunk {c['chunk_index']:>2d}: "
                    f"{c['start_time']:.3f}-{c['end_time']:.3f}s "
                    f"speech_prob={sp:.4f} detected={detected}[/dim]"
                )
    
    # Show speech segment detection parameters
    console.print(f"\n[yellow]🔧 Segment detection parameters:[/yellow]")
    console.print(f"   speech_threshold={speech_threshold}")
    console.print(f"   min_silence_cells={min_silence_cells} (={min_silence_duration_sec}s)")
    console.print(f"   min_speech_cells={min_speech_cells} (={min_speech_duration_sec}s)")
    console.print(f"   step={step*1000:.1f}ms per cell")
    
    # Count silence runs in the speech/non-speech timeline
    console.print(f"\n[yellow]📊 Silence run analysis (gaps > 0.5s):[/yellow]")
    silence_start = None
    for i in range(len(is_speech)):
        if not is_speech[i] and silence_start is None:
            silence_start = i
        elif is_speech[i] and silence_start is not None:
            silence_dur = (i - silence_start) * step
            if silence_dur > 0.5:
                console.print(
                    f"   Silence: {times[silence_start]:.3f}s - {times[i]:.3f}s "
                    f"(dur={silence_dur:.3f}s, cells={i-silence_start})"
                )
            silence_start = None
    if silence_start is not None:
        silence_dur = (len(is_speech) - silence_start) * step
        if silence_dur > 0.5:
            console.print(
                f"   Trailing silence: {times[silence_start]:.3f}s - "
                f"{times[-1]:.3f}s (dur={silence_dur:.3f}s)"
            )
    
    console.print(f"[bold yellow]🔍 End Timeline Diagnostics[/bold yellow]\n")
    # ===== END DEBUG =====

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
                    f"[bold green]🎤 Speech START at cell {i} "
                    f"(time={times[i]:.3f}s, prob={probs[i]:.4f})[/bold green]"
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
                    seg_duration = seg_end_time - seg_start_time
                    console.print(
                        f"[bold red]🔇 Speech END at cell {i} "
                        f"(time={times[i]:.3f}s, prob={probs[i]:.4f}) | "
                        f"segment: {seg_start_time:.3f}s-{seg_end_time:.3f}s "
                        f"(dur={seg_duration:.3f}s, "
                        f"cells={speech_cells_in_current}) | "
                        f"silence_run={silence_run} cells "
                        f"(={silence_run*step:.3f}s, "
                        f"need {min_silence_cells}={min_silence_duration_sec}s)[/bold red]"
                    )
                    raw_segments.append((seg_start_time, seg_end_time))
                    in_speech = False
                    silence_run = 0
                    speech_cells_in_current = 0
    if in_speech:
        seg_start_time = times[seg_start_idx]
        seg_end_time = times[-1]
        seg_duration = seg_end_time - seg_start_time
        console.print(
            f"[dim]🎤 Trailing speech segment: {seg_start_time:.3f}s-{seg_end_time:.3f}s "
            f"(dur={seg_duration:.3f}s) | "
            f"speech_cells={speech_cells_in_current} "
            f"({speech_cells_in_current*step:.3f}s)[/dim]"
        )
        raw_segments.append((seg_start_time, seg_end_time))
    console.print(
        f"[dim]🔍 Raw segments before duration filter: {len(raw_segments)}[/dim]"
    )
    for i, (s, e) in enumerate(raw_segments):
        console.print(f"[dim]   Raw seg {i+1}: {s:.3f}s-{e:.3f}s (dur={e-s:.3f}s)[/dim]")
    segments: List[Tuple[float, float]] = []
    for s, e in raw_segments:
        duration = e - s
        if duration >= min_speech_duration_sec:
            segments.append((s, e))
            console.print(
                f"[green]✅ Keeping segment: {s:.3f}s-{e:.3f}s "
                f"(dur={duration:.3f}s >= min_speech={min_speech_duration_sec}s)[/green]"
            )
        else:
            console.print(
                f"[dim]⏭ Discarding segment: {s:.3f}s-{e:.3f}s "
                f"(dur={duration:.3f}s < min_speech={min_speech_duration_sec}s)[/dim]"
            )
    n_discarded = len(raw_segments) - len(segments)
    if n_discarded:
        console.print(
            f"[yellow]⏭ Discarded {n_discarded} segment(s) shorter than "
            f"{min_speech_duration_sec}s[/yellow]"
        )
    console.print(f"[bold green]✅ {len(segments)} speech segment(s) to save[/bold green]")
    timeline_speech_duration = sum(e - s for s, e in segments)
    console.print(
        f"[dim]📊 Timeline-based speech duration: {timeline_speech_duration:.3f}s "
        f"({timeline_speech_duration/total_timeline_duration*100:.1f}% of audio)[/dim]"
        if total_timeline_duration > 0
        else f"[dim]📊 Timeline-based speech duration: {timeline_speech_duration:.3f}s[/dim]"
    )
    non_speech_segments: List[Tuple[float, float]] = []
    if include_non_speech:
        all_segments_sorted = sorted(segments, key=lambda x: x[0])
        prev_end = 0.0
        total_end = times[-1] if len(times) > 0 else max(c["end_time"] for c in chunks)
        console.print(f"[dim]🔍 Finding non-speech gaps between {len(all_segments_sorted)} speech segments[/dim]")
        for seg_idx, (seg_start, seg_end) in enumerate(all_segments_sorted):
            if seg_start > prev_end:
                gap_duration = seg_start - prev_end
                if gap_duration >= min_silence_duration_sec:
                    non_speech_segments.append((prev_end, seg_start))
                    console.print(
                        f"[dim]🔇 Non-speech gap {len(non_speech_segments)}: "
                        f"{prev_end:.3f}s - {seg_start:.3f}s "
                        f"(dur={gap_duration:.3f}s)[/dim]"
                    )
                else:
                    console.print(
                        f"[dim]⏭ Gap too small to be non-speech segment: "
                        f"{prev_end:.3f}s - {seg_start:.3f}s "
                        f"(dur={gap_duration:.3f}s < min_silence={min_silence_duration_sec}s)[/dim]"
                    )
            prev_end = max(prev_end, seg_end)
        if prev_end < total_end:
            gap_duration = total_end - prev_end
            if gap_duration >= min_silence_duration_sec:
                non_speech_segments.append((prev_end, total_end))
                console.print(
                    f"[dim]🔇 Non-speech gap {len(non_speech_segments)} (trailing): "
                    f"{prev_end:.3f}s - {total_end:.3f}s "
                    f"(dur={gap_duration:.3f}s)[/dim]"
                )
            else:
                console.print(
                    f"[dim]⏭ Trailing gap too small: "
                    f"{prev_end:.3f}s - {total_end:.3f}s "
                    f"(dur={gap_duration:.3f}s < min_silence={min_silence_duration_sec}s)[/dim]"
                )
        console.print(
            f"[dim]🔇 Found {len(non_speech_segments)} non-speech segment(s) "
            f"(total gap duration: {sum(e-s for s,e in non_speech_segments):.3f}s)[/dim]"
        )
    if not segments and not non_speech_segments:
        console.print("[yellow]⚠ No segments survive the minimum-duration filter[/yellow]")
        return []
    segments_root = output_dir / "segments"
    segments_root.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    def _save_segment(
        seg_num: int,
        seg_start: float,
        seg_end: float,
        is_speech: bool,
        prefix: str = "segment",
    ) -> Path:
        """Save a single segment (speech or non-speech) to disk."""
        seg_duration = seg_end - seg_start
        seg_dir = segments_root / f"{prefix}_{seg_num:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_type = "speech" if is_speech else "non-speech"
        console.print(
            f"\n[bold cyan]📁 {segment_type.title()} Segment {seg_num}: "
            f"{seg_start:.3f}s – {seg_end:.3f}s "
            f"(dur={seg_duration:.3f}s)[/bold cyan]"
        )
        
        # 1. Extract Audio
        if waveform is not None:
            try:
                start_sample = int(seg_start * actual_sr)
                end_sample = int(seg_end * actual_sr)
                start_sample = max(0, start_sample)
                end_sample = min(len(waveform), end_sample)
                if end_sample > start_sample:
                    segment_audio = waveform[start_sample:end_sample].copy()
                    wav_path = seg_dir / "sound.wav"
                    if segment_audio.dtype != np.float32:
                        segment_audio = segment_audio.astype(np.float32)
                    sf.write(
                        str(wav_path),
                        segment_audio,
                        samplerate=actual_sr,
                        subtype='PCM_16',
                    )
                    wav_size = wav_path.stat().st_size
                    wav_duration = len(segment_audio) / actual_sr
                    console.print(
                        f"[dim]   ✅ sound.wav saved ({wav_size:,} bytes, "
                        f"{wav_duration:.3f}s, {len(segment_audio)} samples)[/dim]"
                    )
                else:
                    console.print("[yellow]   ⚠ No valid audio samples for this segment[/yellow]")
            except Exception as e:
                console.print(f"[red]   ❌ Failed to save sound.wav: {e}[/red]")
        else:
            console.print("[dim]   ℹ No audio waveform available for extraction[/dim]")

        # 2. Calculate Stats & Prepare Info
        mask = (times >= seg_start) & (times <= seg_end)
        seg_times = times[mask]
        seg_probs = probs[mask]
        avg_prob = float(np.mean(seg_probs)) if len(seg_probs) else 0.0
        max_prob = float(np.max(seg_probs)) if len(seg_probs) else 0.0
        min_prob = float(np.min(seg_probs)) if len(seg_probs) else 0.0
        speech_density = float(np.mean(seg_probs >= speech_threshold)) if len(seg_probs) else 0.0
        seg_chunks = [
            c for c in chunks
            if c["start_time"] < seg_end and c["end_time"] > seg_start
        ]
        
        console.print(
            f"[dim]   Timeline cells: {len(seg_times)} | "
            f"Overlapping chunks: {len(seg_chunks)}[/dim]"
        )
        
        pred_acc: Dict[str, List[float]] = {}
        for c in seg_chunks:
            for p in c.get("predictions", []):
                pred_acc.setdefault(p["name"], []).append(p["prob"])
        top_preds = sorted(
            [
                {"name": name, "mean_prob": round(float(np.mean(ps)), 4), "count": len(ps)}
                for name, ps in pred_acc.items()
            ],
            key=lambda x: x["mean_prob"],
            reverse=True,
        )[:10]
        
        speech_chunk_count = sum(
            1 for c in seg_chunks
            if c.get("speech_probability", 0.0) >= speech_threshold
        )

        confidence_tier, confidence_label, is_high_confidence, is_medium_confidence = \
            calculate_confidence_tier(
                avg_prob=avg_prob,
                speech_density=speech_density,
                duration=seg_duration,
                speech_chunk_ratio=round(speech_chunk_count / len(seg_chunks), 4)
                if seg_chunks else 0.0,
            )
        
        duration_note = ""
        if seg_duration < 0.3 and confidence_tier in ("high", "medium"):
            duration_note = "⚠ Very short - verify manually"
        elif seg_duration < 0.5 and confidence_tier == "medium":
            duration_note = "Short duration may be noise"
        elif seg_duration < 1.0 and confidence_tier == "low":
            duration_note = "Too short for reliable classification"
        elif seg_duration > 10.0:
            duration_note = "Long segment - check for mixed content"
        elif seg_duration > 30.0 and confidence_tier == "high":
            duration_note = "Unusually long - may contain multiple speakers"
        
        if duration_note:
            console.print(f"[dim]   📝 Duration note: {duration_note}[/dim]")

        # Initialize segment_info WITHOUT gender yet
        segment_info = {
            "segment_index": seg_num - 1,
            "segment_type": segment_type,
            "start_time": round(float(seg_start), 3),
            "end_time": round(float(seg_end), 3),
            "duration": round(float(seg_duration), 3),
            "audio_file": "sound.wav" if waveform is not None else None,
            "speech_stats": {
                "avg_speech_probability": round(avg_prob, 4),
                "max_speech_probability": round(max_prob, 4),
                "min_speech_probability": round(min_prob, 4),
                "speech_density": round(speech_density, 4),
                "speech_chunk_count": speech_chunk_count,
                "total_chunk_count": len(seg_chunks),
                "speech_chunk_ratio": round(speech_chunk_count / len(seg_chunks), 4)
                if seg_chunks else 0.0,
                "threshold_used": speech_threshold,
            },
            "insights": {
                "is_high_confidence": is_high_confidence,
                "is_medium_confidence": is_medium_confidence,
                "confidence_tier": confidence_tier,
                "confidence_label": confidence_label,
                "confidence_duration_note": duration_note,
                "is_dense_speech": speech_density >= 0.8,
                "top_prediction": top_preds[0]["name"] if top_preds else "Unknown",
                "top_prediction_prob": top_preds[0]["mean_prob"] if top_preds else 0.0,
                "top_predictions": top_preds[:5],
            },
            "source_audio": {
                "audio_path": summary.get("audio_path", ""),
                "sample_rate": summary.get("sample_rate", 0),
                "total_audio_duration": summary.get("total_duration", 0.0),
                "chunk_duration": summary.get("chunk_duration", 0.0),
                "overlap_duration": summary.get("overlap_duration", 0.0),
            },
            "parameters": {
                "min_silence_duration_sec": min_silence_duration_sec,
                "min_speech_duration_sec": min_speech_duration_sec,
                "speech_threshold": speech_threshold,
                "resolution_ms": resolution_ms,
                "include_non_speech": include_non_speech,
            },
        }

        # 3. Save Chunks & Probs JSONs (These don't depend on gender)
        chunks_data = []
        for c in seg_chunks:
            overlap_start = max(c["start_time"], seg_start)
            overlap_end = min(c["end_time"], seg_end)
            overlap_dur = max(0.0, overlap_end - overlap_start)
            chunk_speech_prob = c.get("speech_probability", 0.0)
            chunk_speech_detected = chunk_speech_prob >= speech_threshold
            chunks_data.append({
                "chunk_index": c["chunk_index"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "duration": c["duration"],
                "speech_detected": chunk_speech_detected,
                "speech_probability": chunk_speech_prob,
                "overlap_with_segment": {
                    "start": round(overlap_start, 3),
                    "end": round(overlap_end, 3),
                    "duration": round(overlap_dur, 3),
                    "coverage_ratio": round(overlap_dur / c["duration"], 4)
                    if c["duration"] > 0 else 0.0,
                },
                "predictions": [
                    {
                        "name": p["name"],
                        "class_index": p.get("class_index", -1),
                        "prob": round(p["prob"], 4),
                    }
                    for p in c.get("predictions", [])[:10]
                ],
            })
        chunks_json_path = seg_dir / "chunks.json"
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(serialize(chunks_data), f, indent=2, ensure_ascii=False)
        console.print(
            f"[dim]   ✅ chunks.json ({chunks_json_path.stat().st_size:,} bytes) | "
            f"speech_chunks={speech_chunk_count}/{len(seg_chunks)}[/dim]"
        )

        probs_data = [
            {"time": round(float(t), 4), "speech_prob": round(float(p), 4)}
            for t, p in zip(seg_times, seg_probs)
        ]
        probs_json_path = seg_dir / "probs.json"
        with open(probs_json_path, "w", encoding="utf-8") as f:
            json.dump(serialize(probs_data), f, indent=2, ensure_ascii=False)
        console.print(f"[dim]   ✅ probs.json ({probs_json_path.stat().st_size:,} bytes)[/dim]")

        # 4. Generate Plots AND Inject Gender Data into segment_info
        _save_segment_plots(
            seg_dir=seg_dir,
            seg_times=seg_times,
            seg_probs=seg_probs,
            seg_chunks=seg_chunks,
            seg_start=seg_start,
            seg_end=seg_end,
            speech_threshold=speech_threshold,
            segment_info=segment_info, # This dict is modified in-place by _save_segment_plots
        )

        # 5. NOW Save segment.json (which now contains the gender data injected by plots function)
        seg_json_path = seg_dir / "segment.json"
        with open(seg_json_path, "w", encoding="utf-8") as f:
            json.dump(serialize(segment_info), f, indent=2, ensure_ascii=False)
        console.print(f"[dim]   ✅ segment.json ({seg_json_path.stat().st_size:,} bytes)[/dim]")

        console.print(
            f"[green]   📁 Saved: {linkify(str(seg_dir))}[/green]"
        )
        return seg_dir

    for seg_num, (seg_start, seg_end) in enumerate(segments, start=1):
        seg_dir = _save_segment(
            seg_num=seg_num,
            seg_start=seg_start,
            seg_end=seg_end,
            is_speech=True,
            prefix="segment",
        )
        created.append(seg_dir)
    if include_non_speech:
        for seg_num, (seg_start, seg_end) in enumerate(non_speech_segments, start=1):
            seg_dir = _save_segment(
                seg_num=seg_num,
                seg_start=seg_start,
                seg_end=seg_end,
                is_speech=False,
                prefix="non_speech_segment",
            )
            created.append(seg_dir)
    total_speech_duration = sum(e - s for s, e in segments)
    total_non_speech_duration = sum(e - s for s, e in non_speech_segments) if include_non_speech else 0
    summary_parts = [
        f"[bold green]✅ {len(segments)} speech segment(s) saved "
        f"({total_speech_duration:.3f}s total)[/bold green]"
    ]
    if include_non_speech and non_speech_segments:
        summary_parts.append(
            f"[bold yellow]🔇 {len(non_speech_segments)} non-speech segment(s) saved "
            f"({total_non_speech_duration:.3f}s total)[/bold yellow]"
        )
    summary_parts.append(f"Root: {linkify(str(segments_root))}")
    summary_parts.append(f"Threshold: {speech_threshold:.2f} | "
                        f"Resolution: {resolution_ms}ms | "
                        f"Min silence: {min_silence_duration_sec}s | "
                        f"Min speech: {min_speech_duration_sec}s")
    console.print(
        Panel.fit(
            "\n".join(summary_parts),
            border_style="green",
        )
    )
    return created


def _save_segment_plots(
    seg_dir: Path,
    seg_times: np.ndarray,
    seg_probs: np.ndarray,
    seg_chunks: List[ChunkTaggingResult],
    seg_start: float,
    seg_end: float,
    speech_threshold: float,
    segment_info: dict,
) -> None:
    """
    Save multi-panel PNGs for one speech segment, including a new gender analysis chart.
    Updated to inject gender data into segment_info for JSON persistence.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        console.print("[yellow]   ⚠️ matplotlib not available – skipping plots[/yellow]")
        return

    # --- 1. Gender Aggregation & Injection ---
    male_probs = [c.get('male_prob', 0.0) for c in seg_chunks]
    female_probs = [c.get('female_prob', 0.0) for c in seg_chunks]
    
    avg_male_prob = float(np.mean(male_probs)) if male_probs else 0.0
    avg_female_prob = float(np.mean(female_probs)) if female_probs else 0.0
    
    # Determine segment dominant gender using the low threshold
    if avg_male_prob > avg_female_prob and avg_male_prob > DEFAULT_GENDER_CONFIDENCE_THRESHOLD:
        segment_dominant_gender = "male"
    elif avg_female_prob > avg_male_prob and avg_female_prob > DEFAULT_GENDER_CONFIDENCE_THRESHOLD:
        segment_dominant_gender = "female"
    else:
        segment_dominant_gender = "unknown"
        
    # Simple confidence metric for gender
    gender_diff = abs(avg_male_prob - avg_female_prob)
    if gender_diff > 0.3:
        gender_confidence = "high"
    elif gender_diff > 0.1:
        gender_confidence = "medium"
    else:
        gender_confidence = "low"

    # INJECT into segment_info so it gets saved to segment.json
    segment_info["insights"]["dominant_gender"] = segment_dominant_gender
    segment_info["insights"]["male_probability"] = round(avg_male_prob, 6)
    segment_info["insights"]["female_probability"] = round(avg_female_prob, 6)
    segment_info["insights"]["gender_confidence"] = gender_confidence
    
    console.print(
        f"[dim]   👥 Gender Calculated: {segment_dominant_gender.capitalize()} "
        f"(M:{avg_male_prob:.4f}, F:{avg_female_prob:.4f})[/dim]"
    )
    # -----------------------------------------

    # --- 2. Standard Plots (Speech Timeline, Heatmap, Bar Chart) ---
    fig = plt.figure(figsize=(14, 10))
    segment_type = segment_info.get("segment_type", "speech")
    
    # Add Gender to Title
    gender_str = f" | Gender: {segment_dominant_gender.capitalize()}" if segment_dominant_gender != "unknown" else ""
    
    fig.suptitle(
        f"{segment_type.title()} Segment {segment_info['segment_index'] + 1}  "
        f"[{seg_start:.3f}s – {seg_end:.3f}s]  "
        f"dur={segment_info['duration']:.3f}s{gender_str}",
        fontsize=13,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)
    
    # Panel 1: Speech Probability Timeline
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(seg_times, seg_probs, color="#1f77b4", linewidth=1.2, label="speech prob")
    ax1.fill_between(seg_times, seg_probs, alpha=0.25, color="#1f77b4")
    ax1.axhline(speech_threshold, color="red", linestyle="--", linewidth=0.9,
                label=f"threshold ({speech_threshold:.2f})")
    ax1.set_xlim(seg_start, seg_end)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Speech Probability")
    ax1.set_title("Speech Probability Timeline")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Predictions Heatmap
    ax2 = fig.add_subplot(gs[1])
    if seg_chunks:
        pred_names_set: Dict[str, float] = {}
        for c in seg_chunks:
            for p in c.get("predictions", []):
                n = p["name"]
                pred_names_set[n] = max(pred_names_set.get(n, 0.0), p["prob"])
        top_labels = sorted(pred_names_set, key=pred_names_set.get, reverse=True)[:8]
        n_labels = len(top_labels)
        n_chunks = len(seg_chunks)
        heatmap_data = np.zeros((n_labels, n_chunks), dtype=np.float32)
        chunk_labels = []
        for ci, c in enumerate(seg_chunks):
            chunk_labels.append(f"C{c['chunk_index']}\n{c['start_time']:.1f}s")
            pred_dict = {p["name"]: p["prob"] for p in c.get("predictions", [])}
            for li, label in enumerate(top_labels):
                heatmap_data[li, ci] = pred_dict.get(label, 0.0)
        im = ax2.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                        vmin=0.0, vmax=1.0, interpolation="nearest")
        ax2.set_xticks(range(n_chunks))
        ax2.set_xticklabels(chunk_labels, fontsize=6)
        ax2.set_yticks(range(n_labels))
        short_labels = [lb[:30] + "…" if len(lb) > 30 else lb for lb in top_labels]
        ax2.set_yticklabels(short_labels, fontsize=7)
        ax2.set_title("Predictions Heatmap")
        fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02, label="Probability")
    else:
        ax2.text(0.5, 0.5, "No overlapping chunks", ha="center", va="center", transform=ax2.transAxes)

    # Panel 3: Per-Chunk Speech Bar Chart
    ax3 = fig.add_subplot(gs[2])
    if seg_chunks:
        chunk_speech = [c.get("speech_probability", 0.0) for c in seg_chunks]
        chunk_centers = [(c["start_time"] + c["end_time"]) / 2 for c in seg_chunks]
        bar_colors = ["#2ca02c" if sp >= speech_threshold else "#d62728" for sp in chunk_speech]
        bar_width = min((seg_end - seg_start) / max(len(seg_chunks), 1) * 0.7, 0.7)
        ax3.bar(chunk_centers, chunk_speech, width=bar_width, color=bar_colors, alpha=0.85)
        ax3.axhline(speech_threshold, color="red", linestyle="--", linewidth=0.9)
        ax3.set_xlim(seg_start, seg_end)
        ax3.set_ylim(0, 1.05)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Speech Probability")
        ax3.set_title("Per-Chunk Speech Probability")
        ax3.grid(axis="y", alpha=0.3)

    plot_path = seg_dir / "plots.png"
    fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[dim]   ✅ plots.png saved[/dim]")

    # --- 3. NEW: Dedicated Gender Analysis Chart ---
    try:
        fig_g, ax_g = plt.subplots(figsize=(12, 4))
        
        # Prepare data for gender timeline
        # We map each chunk's gender probs to its time range
        chunk_starts = [c["start_time"] for c in seg_chunks]
        chunk_ends = [c["end_time"] for c in seg_chunks]
        m_probs = [c.get("male_prob", 0.0) for c in seg_chunks]
        f_probs = [c.get("female_prob", 0.0) for c in seg_chunks]
        
        # Plot Male and Female probabilities as step lines or filled areas
        # Using a simple line plot connecting chunk centers for clarity
        centers = [(s + e) / 2 for s, e in zip(chunk_starts, chunk_ends)]
        
        ax_g.plot(centers, m_probs, 'b-o', label='Male Prob', linewidth=2, markersize=4)
        ax_g.plot(centers, f_probs, 'm-o', label='Female Prob', linewidth=2, markersize=4)
        ax_g.fill_between(centers, m_probs, alpha=0.1, color='blue')
        ax_g.fill_between(centers, f_probs, alpha=0.1, color='magenta')
        
        ax_g.set_title(f"Gender Probability Analysis ({segment_dominant_gender.capitalize()})")
        ax_g.set_xlabel("Time (s)")
        ax_g.set_ylabel("Probability")
        ax_g.set_xlim(seg_start, seg_end)
        ax_g.legend(loc="upper right")
        ax_g.grid(True, alpha=0.3)
        
        # Add threshold line for gender confidence
        ax_g.axhline(DEFAULT_GENDER_CONFIDENCE_THRESHOLD, color='gray', linestyle=':', label=f'Confidence Threshold ({DEFAULT_GENDER_CONFIDENCE_THRESHOLD})')
        
        gender_plot_path = seg_dir / "gender_analysis.png"
        fig_g.savefig(str(gender_plot_path), dpi=120, bbox_inches="tight")
        plt.close(fig_g)
        console.print(f"[dim]   ✅ gender_analysis.png saved[/dim]")
        
    except Exception as e:
        console.print(f"[yellow]   ⚠️ Failed to save gender chart: {e}[/yellow]")


def save_overall_gender_analysis(
    summary: AudioChunksTaggingSummary,
    output_dir: Path,
    audio_name: str,
) -> Path:
    """
    Calculate and save overall gender statistics for the entire audio file.
    
    Args:
        summary: The AudioChunksTaggingSummary containing chunk data with gender info.
        output_dir: Directory to save the analysis file.
        audio_name: Base name of the audio file for naming the output.
        
    Returns:
        Path to the saved JSON file.
    """
    chunks = summary.get("chunks", [])
    if not chunks:
        return Path()

    total_chunks = len(chunks)
    male_count = 0
    female_count = 0
    unknown_count = 0
    
    total_male_prob = 0.0
    total_female_prob = 0.0
    
    for chunk in chunks:
        gender = chunk.get("dominant_gender", "unknown")
        m_prob = chunk.get("male_prob", 0.0)
        f_prob = chunk.get("female_prob", 0.0)
        
        if gender == "male":
            male_count += 1
        elif gender == "female":
            female_count += 1
        else:
            unknown_count += 1
            
        total_male_prob += m_prob
        total_female_prob += f_prob
        
    avg_male_prob = total_male_prob / total_chunks if total_chunks > 0 else 0.0
    avg_female_prob = total_female_prob / total_chunks if total_chunks > 0 else 0.0
    
    # Determine overall dominant gender based on average probabilities
    if avg_male_prob > avg_female_prob and avg_male_prob > DEFAULT_GENDER_CONFIDENCE_THRESHOLD:
        overall_dominant_gender = "male"
    elif avg_female_prob > avg_male_prob and avg_female_prob > DEFAULT_GENDER_CONFIDENCE_THRESHOLD:
        overall_dominant_gender = "female"
    else:
        overall_dominant_gender = "unknown"
        
    gender_diff = abs(avg_male_prob - avg_female_prob)
    if gender_diff > 0.3:
        overall_confidence = "high"
    elif gender_diff > 0.1:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"

    analysis_data = {
        "audio_file": summary.get("audio_path", "unknown"),
        "total_duration": summary.get("total_duration", 0.0),
        "total_chunks_analyzed": total_chunks,
        "chunk_distribution": {
            "male": {"count": male_count, "percentage": round((male_count / total_chunks) * 100, 2)},
            "female": {"count": female_count, "percentage": round((female_count / total_chunks) * 100, 2)},
            "unknown": {"count": unknown_count, "percentage": round((unknown_count / total_chunks) * 100, 2)},
        },
        "probability_averages": {
            "avg_male_probability": round(avg_male_prob, 6),
            "avg_female_probability": round(avg_female_prob, 6),
        },
        "overall_result": {
            "dominant_gender": overall_dominant_gender,
            "confidence": overall_confidence,
            "gender_difference_score": round(gender_diff, 6)
        }
    }
    
    output_path = output_dir / "gender_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
    console.print(f"[green]Overall gender analysis saved to: {linkify(str(output_path))}[/green]")
    return output_path


def summary_chunk_duration_or_default(chunks: List[ChunkTaggingResult]) -> float:
    """Return a representative chunk duration from the chunk list, default 1.0."""
    if not chunks:
        return 1.0
    durations = [c.get("duration", 0.0) for c in chunks if c.get("duration", 0.0) > 0]
    return float(np.median(durations)) if durations else 1.0


def main():
    args = get_args()
    audio_path = args.audio_path
    
    tagger = AudioTagger(
        model_path=args.model_path,
        labels_path=args.labels_path,
        top_k=args.top_k,
        num_threads=args.num_threads,
        provider=args.provider,
        debug=args.debug,
        speech_prob_threshold=args.speech_threshold,
        speech_top_n=args.speech_top_n,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
    )
    
    console.print(
        Panel.fit(
            "[bold cyan]Audio Tagging Analysis[/bold cyan]",
            border_style="cyan",
        )
    )
    
    try:
        console.print(f"\n[bold]Analyzing audio: {linkify(audio_path)}[/bold]\n")
        audio_name = Path(audio_path).stem
        if args.chunk:
            # Use the new tag_audio_segments for combined chunk + segment detection
            segments_result = tagger.tag_audio_segments(
                audio_path,
                chunk_duration=args.chunk_duration,
                overlap_duration=args.chunk_overlap,
                speech_threshold=args.speech_threshold,
                min_silence_duration_sec=args.min_silence,
                min_speech_duration_sec=args.min_speech,
                resolution_ms=args.resolution_ms,
                include_non_speech=True,
            )
            
            # Build a compatible summary dict for display functions and save_speech_segments
            summary: AudioChunksTaggingSummary = {
                "audio_path": segments_result["audio_path"],
                "total_duration": segments_result["total_duration"],
                "sample_rate": segments_result["sample_rate"],
                "chunk_duration": segments_result["chunk_duration"],
                "overlap_duration": segments_result["overlap_duration"],
                "total_chunks": segments_result["total_chunks"],
                "chunks": segments_result["chunks"],
                "overall_top_predictions": segments_result["overall_top_predictions"],
                "total_processing_time": segments_result["total_processing_time"],
                "real_time_factor": segments_result["real_time_factor"],
                "speech_duration": segments_result["total_speech_duration"],
                "speech_detected": len(segments_result["speech_segments"]) > 0,
                "max_speech_probability": max(
                    (c.get("speech_probability", 0.0) for c in segments_result["chunks"]),
                    default=0.0,
                ),
                "avg_speech_probability": float(np.mean(
                    [c.get("speech_probability", 0.0) for c in segments_result["chunks"]
                     if c.get("speech_detected", False)]
                )) if any(c.get("speech_detected", False) for c in segments_result["chunks"]) else 0.0,
            }
            
            console.print(
                f"[dim]Showing predictions with probability ≥ {args.speech_threshold:.0%}[/dim]"
            )
            console.print(
                f"[dim]Speech threshold: {args.speech_threshold:.0%} | "
                f"Speech duration: {summary['speech_duration']:.2f}s | "
                f"Avg speech prob: {summary['avg_speech_probability']:.4f}[/dim]"
            )
            
            # Chunk visualizations
            try:
                from audio_tagger_chunk_plots import save_chunk_plots
                plot_paths = save_chunk_plots(
                    summary=summary,
                    output_dir=Path(args.output_dir),
                    top_n_display=min(args.top_k, 10),
                    probability_threshold=args.speech_threshold,
                )
                console.print(
                    Panel(
                        "\n".join(
                            f"[cyan]{i + 1}. {linkify(str(p))}[/cyan]"
                            for i, p in enumerate(plot_paths)
                        ),
                        title="📊 Chunk Visualization Plots",
                        border_style="blue",
                    )
                )
            except ImportError:
                console.print(
                    "[yellow]⚠ Plot module not available — skipping chunk visualizations[/yellow]"
                )
            except Exception as e:
                console.print(f"[red]⚠ Chunk plot generation failed: {e}[/red]")
            
            # Save chunk summary JSON
            summary_output = Path(args.output_dir) / "chunks_summary.json"
            serializable = {
                **summary,
                "chunks": [{**chunk} for chunk in summary["chunks"]],
                "overall_top_predictions": summary["overall_top_predictions"],
            }
            with open(summary_output, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            console.print(
                f"[green]Chunked results saved to: {linkify(str(summary_output))}[/green]"
            )
            
            # Save speech segments to disk using the existing function
            console.print("\n[bold]Saving speech segments…[/bold]")
            segment_dirs = save_speech_segments(
                summary=summary,
                output_dir=Path(args.output_dir),
                min_silence_duration_sec=args.min_silence,
                min_speech_duration_sec=args.min_speech,
                resolution_ms=args.resolution_ms,
                include_non_speech=True,
                speech_threshold=args.speech_threshold,
            )

            # --- Save Overall Gender Analysis ---
            console.print("\n[bold cyan]💾 Saving Overall Gender Analysis[/bold cyan]")
            save_overall_gender_analysis(
                summary=summary,
                output_dir=Path(args.output_dir),
                audio_name=audio_name,
            )
            
            # Display per-chunk analysis
            console.print("\n[bold cyan]📊 Per-Chunk Analysis[/bold cyan]")
            display_per_chunk_analysis(
                summary=summary,
                probability_threshold=args.speech_threshold,
                max_predictions_display=min(args.top_k, 10),
                show_summary_table=True,
                show_chunk_table=True,
            )
            
            # Display per-segment analysis
            if segment_dirs:
                console.print("\n[bold cyan]📊 Per-Segment Analysis[/bold cyan]")
                display_per_segment_analysis(
                    segment_dirs=segment_dirs,
                    show_speech_segments=True,
                    show_non_speech_segments=True,
                    show_details=True,
                )

        else:
            # Single-pass audio tagging (no chunking)
            results = tagger.tag_audio(audio_path)
            tagger.display_results(results)
            
            # Save results
            json_output = Path(args.output_dir) / f"{audio_name}_tags.json"
            tagger.save_results(results, json_output, format="json")
            
            txt_output = Path(args.output_dir) / f"{audio_name}_tags.txt"
            tagger.save_results(results, txt_output, format="txt")
            
            # Check speech if requested
            if args.check_speech:
                console.print("\n[bold]Speech Detection Analysis[/bold]")
                is_speech = tagger.contains_speech(audio_path)
                speech_prob = tagger.get_speech_probability(audio_path)
                
                speech_table = Table(
                    title="Speech Detection Results", border_style="green"
                )
                speech_table.add_column("Metric", style="cyan")
                speech_table.add_column("Value", style="yellow")
                speech_table.add_row(
                    "Speech Detected", "✅ Yes" if is_speech else "❌ No"
                )
                speech_table.add_row("Max Speech Probability", f"{speech_prob:.4f}")
                speech_table.add_row("Threshold", str(args.speech_threshold))
                console.print(speech_table)
                
                speech_result = {
                    "audio_path": audio_path,
                    "speech_detected": is_speech,
                    "max_speech_probability": speech_prob,
                    "threshold": args.speech_threshold,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                speech_output = (
                    Path(args.output_dir) / f"{audio_name}_speech_detection.json"
                )
                with open(speech_output, "w", encoding="utf-8") as f:
                    json.dump(speech_result, f, indent=2, ensure_ascii=False)
                console.print(
                    f"[green]Speech detection saved to: {linkify(str(speech_output))}[/green]"
                )
            
            # Generate comprehensive summary
            console.print("\n[bold]Generating Comprehensive Summary[/bold]")
            summary = tagger.get_tagging_summary(audio_path, audio_path=audio_path)
            
            summary_table = Table(title="Audio Tagging Summary", border_style="magenta")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="yellow")
            summary_table.add_row("Audio File", summary["audio_path"])
            summary_table.add_row("Duration", f"{summary['duration_seconds']:.2f}s")
            summary_table.add_row("Sample Rate", f"{summary['sample_rate']} Hz")
            summary_table.add_row("Results Count", str(summary["num_results"]))
            summary_table.add_row(
                "Speech Detected",
                "✅ Yes" if summary["speech_detected"] else "❌ No",
            )
            summary_table.add_row(
                "Max Speech Prob", f"{summary['max_speech_probability']:.4f}"
            )
            summary_table.add_row(
                "Speech Duration",
                f"{summary['speech_duration']:.2f}s"
                f" ({summary['speech_duration'] / summary['duration_seconds'] * 100:.1f}%)"
                if summary["duration_seconds"] > 0
                else "0.00s",
            )
            summary_table.add_row(
                "Processing Time", f"{summary['processing_time_seconds']:.3f}s"
            )
            summary_table.add_row(
                "Real-Time Factor", f"{summary['real_time_factor']:.3f}"
            )
            console.print(summary_table)
            
            summary_output = Path(args.output_dir) / f"{audio_name}_summary.json"
            with open(summary_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            console.print(
                f"[green]Summary saved to: {linkify(str(summary_output))}[/green]"
            )
        
        console.print(
            Panel.fit(
                f"[bold green]✅ Analysis Complete[/bold green]\n"
                f"Results saved in: {linkify(str(args.output_dir))}",
                border_style="green",
            )
        )
        
    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error Processing Audio[/bold red]\n{str(e)}",
                border_style="red",
            )
        )
        raise


if __name__ == "__main__":
    main()
