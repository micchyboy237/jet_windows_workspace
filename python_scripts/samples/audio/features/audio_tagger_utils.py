"""
audio_tagger_utils.py
=====================
Standalone utility functions for sherpa-onnx audio tagging.
Extracted from audio_tagger_base.py to keep the base module focused on
the BaseAudioTagger abstract class and its core dataclasses.

Saves 5 output files:
  1. results.json        — Aggregated top-K predictions
  2. metadata.json       — Processing metadata and performance metrics
  3. chunk_results.json  — Per-chunk raw probabilities
  4. chunk_timeline.png  — Event probability timeline plot
  5. results_bar.png     — Aggregated results bar chart

Aligned with FireRed VAD:
    - Uses FRAME_SHIFT_SAMPLE (160 samples) as the fundamental unit
    - Windows are multiples of FireRed frames (100 frames = 1s window)
    - Supports per-segment tagging with absolute UTC timestamps
"""
from __future__ import annotations
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from audio_tagger_core import (
    HOP_LENGTH,
    SAMPLE_RATE,
    TaggingEvent,
    TaggingResult,
    console,
    log,
)
from fireredvad.core.constants import FRAME_SHIFT_SAMPLE
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


def read_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Read any soundfile-supported file.
    Returns (mono float32 samples, original sample rate).
    """
    p = Path(audio_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\nPlease check the path and try again."
        )
    data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    samples = np.ascontiguousarray(data[:, 0])
    log.debug(
        f"Read [cyan]{len(samples):,}[/cyan] samples at [cyan]{sr} Hz[/cyan] "
        f"from [cyan]{audio_path}[/cyan]"
    )
    return samples, sr


def resample_if_needed(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Linear-interpolation resample; no-op when rates already match."""
    if orig_sr == target_sr:
        return samples
    log.info(f"Resampling [cyan]{orig_sr} Hz → {target_sr} Hz[/cyan]")
    target_len = int(len(samples) / orig_sr * target_sr)
    idx = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


def find_model_file(model_path: Path, model_int8_path: Path) -> str:
    """
    Return the path to whichever model file exists, preferring int8.
    Raises FileNotFoundError with a clear message when neither exists.
    """
    if model_int8_path.is_file():
        log.info("Using int8 quantised model")
        return str(model_int8_path)
    if model_path.is_file():
        log.info("Using standard (fp32) model")
        return str(model_path)
    raise FileNotFoundError(
        "No model file found. Checked:\n"
        f"  • {model_int8_path}\n"
        f"  • {model_path}\n"
        "Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
    )


def _validate_firered_alignment(expected_frames: int, hop_length: int) -> None:
    """
    Validate that tagger parameters align with FireRed VAD frame boundaries.
    FireRed operates on 10ms frames (160 samples at 16kHz).
    The tagger's window and hop must be multiples of this frame size
    to ensure perfect alignment when processing speech segments.
    Args:
        expected_frames: Number of frames per tagger window
        hop_length: Hop length in samples (should equal FRAME_SHIFT_SAMPLE)
    Raises:
        ValueError: If alignment check fails
    """
    window_samples = expected_frames * hop_length
    hop_samples = window_samples // 2
    if window_samples % FRAME_SHIFT_SAMPLE != 0:
        raise ValueError(
            f"Tagger window ({window_samples} samples) is not aligned to FireRed "
            f"frame size ({FRAME_SHIFT_SAMPLE} samples). "
            f"Window must be a multiple of {FRAME_SHIFT_SAMPLE}."
        )
    if hop_samples % FRAME_SHIFT_SAMPLE != 0:
        raise ValueError(
            f"Tagger hop ({hop_samples} samples) is not aligned to FireRed "
            f"frame size ({FRAME_SHIFT_SAMPLE} samples). "
            f"Hop must be a multiple of {FRAME_SHIFT_SAMPLE}."
        )
    log.debug(
        f"✓ FireRed alignment verified: window={window_samples}samples "
        f"({window_samples // FRAME_SHIFT_SAMPLE} frames), "
        f"hop={hop_samples}samples ({hop_samples // FRAME_SHIFT_SAMPLE} frames)"
    )


def process_audio_chunks(
    audio_tagger,
    samples: np.ndarray,
    sample_rate: int,
    expected_frames: int = 100,
    hop_length: int = HOP_LENGTH,
    segment_start_utc: Optional[object] = None,
) -> List[dict]:
    """
    Slide a 50%-overlapping window over the audio and collect raw events.
    Window  = expected_frames × hop_length samples  (1.0 s at 16 kHz with 100 frames)
    Step    = window // 2  (50 % overlap)
    Padding — short clips are zero-padded to at least one full window.
    Now aligned with FireRed VAD:
        - Validates that window/hop are multiples of FRAME_SHIFT_SAMPLE
        - Supports optional UTC timestamps for speech segments
    Args:
        audio_tagger: Configured sherpa-onnx AudioTagging instance
        samples: Audio samples (mono, float32)
        sample_rate: Sample rate in Hz (must be 16000 for FireRed alignment)
        expected_frames: Frames per window (default 100 = 1s)
        hop_length: Hop length in samples (default FRAME_SHIFT_SAMPLE = 160)
        segment_start_utc: Optional UTC timestamp for segment start
    Returns:
        List of event dicts with timing information
    """
    _validate_firered_alignment(expected_frames, hop_length)
    window_samples = expected_frames * hop_length
    hop_samples = window_samples // 2
    total_samples = len(samples)
    log.info(
        f"Audio: [cyan]{total_samples:,}[/cyan] samples "
        f"([cyan]{total_samples / sample_rate:.2f}s[/cyan]) | "
        f"Window: [cyan]{window_samples / sample_rate:.2f}s[/cyan] "
        f"({expected_frames} frames × {hop_length} samples)"
    )
    if total_samples < window_samples:
        log.info(f"Padding short audio ({total_samples} → {window_samples} samples)")
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:total_samples] = samples
        samples = padded
        total_samples = window_samples
    num_chunks = max(1, (total_samples - window_samples) // hop_samples + 1)
    all_events: List[dict] = []
    log.info(f"Processing [cyan]{num_chunks}[/cyan] overlapping chunk(s)…")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Tagging chunks…", total=num_chunks)
        for i in range(num_chunks):
            start = i * hop_samples
            end = min(start + window_samples, total_samples)
            chunk = np.zeros(window_samples, dtype=np.float32)
            available = min(window_samples, total_samples - start)
            chunk[:available] = samples[start : start + available]
            stream = audio_tagger.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
            result = audio_tagger.compute(stream)
            chunk_start_sec = start / sample_rate
            chunk_end_sec = end / sample_rate
            time_utc_start = None
            time_utc_end = None
            if segment_start_utc is not None:
                time_utc_start = segment_start_utc + timedelta(seconds=chunk_start_sec)
                time_utc_end = segment_start_utc + timedelta(seconds=chunk_end_sec)
            log.debug(
                f"Chunk {i}: [{chunk_start_sec:.2f}s – {chunk_end_sec:.2f}s] "
                f"events={len(result)}"
            )
            for event in result:
                all_events.append(
                    {
                        "name": getattr(event, "name", None),
                        "index": getattr(event, "index", None),
                        "prob": getattr(event, "prob", None),
                        "chunk_start": chunk_start_sec,
                        "chunk_end": chunk_end_sec,
                        "chunk_index": i,
                        "time_utc_start": time_utc_start,
                        "time_utc_end": time_utc_end,
                    }
                )
            progress.update(task, advance=1)
    return all_events


def aggregate_chunk_results(
    chunk_events: List[dict],
    top_k: int = 5,
) -> List[TaggingEvent]:
    """
    Average probabilities for the same event across all chunks.
    Returns top-k TaggingEvent objects sorted by mean probability (desc).
    """
    if not chunk_events:
        return []
    groups: dict[tuple, dict] = {}
    for ev in chunk_events:
        key = (ev["name"], ev["index"])
        if key not in groups:
            groups[key] = {
                "name": ev["name"],
                "class_index": ev["index"],
                "probs": [],
                "occurrences": 0,
                "chunk_start": ev["chunk_start"],
                "chunk_end": ev["chunk_end"],
                "time_utc_start": ev.get("time_utc_start"),
                "time_utc_end": ev.get("time_utc_end"),
            }
        groups[key]["probs"].append(ev["prob"])
        groups[key]["occurrences"] += 1
        groups[key]["chunk_start"] = min(groups[key]["chunk_start"], ev["chunk_start"])
        groups[key]["chunk_end"] = max(groups[key]["chunk_end"], ev["chunk_end"])
    aggregated = [
        TaggingEvent(
            name=g["name"],
            class_index=g["class_index"],
            prob=float(np.mean(g["probs"])),
            max_prob=float(np.max(g["probs"])),
            occurrences=g["occurrences"],
            chunk_start=g["chunk_start"],
            chunk_end=g["chunk_end"],
            time_utc_start=g["time_utc_start"],
            time_utc_end=g["time_utc_end"],
        )
        for g in groups.values()
    ]
    aggregated.sort(key=lambda e: e.prob, reverse=True)
    return aggregated[:top_k]


def print_results_table(
    events: List[TaggingEvent], chunk_count: int, backend_name: str
) -> None:
    """Display aggregated results in a Rich table."""
    tbl = Table(
        title=f"🎵 Audio Tagging Results — {backend_name} (aggregated)",
        header_style="bold magenta",
    )
    tbl.add_column("Rank", style="cyan", width=6)
    tbl.add_column("Label", style="green")
    tbl.add_column("Index", style="yellow", width=8)
    tbl.add_column("Avg prob", style="bold white", justify="right")
    tbl.add_column("Max prob", style="dim white", justify="right")
    tbl.add_column("Occurrences", style="blue", justify="right")
    tbl.add_column("Bar", style="magenta")
    for i, ev in enumerate(events):
        bar_len = int(ev.prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        label = ev.name or "N/A"
        if ev.time_utc_start:
            label += f"\n[dim]{ev.time_utc_start.strftime('%H:%M:%S.%f')[:-3]}[/dim]"
        tbl.add_row(
            str(i + 1),
            label,
            str(ev.class_index) if ev.class_index is not None else "N/A",
            f"{ev.prob * 100:.2f}%",
            f"{ev.max_prob * 100:.2f}%",
            f"{ev.occurrences}/{chunk_count}",
            bar,
        )
    console.print(tbl)


def print_perf_table(result: TaggingResult) -> None:
    """Display performance metrics in a Rich table."""
    rtf = result.real_time_factor
    rtf_style = "green" if rtf < 1.0 else "yellow"
    rtf_icon = "✓" if rtf < 1.0 else "⚠"
    perf = Table(title="⚡ Performance metrics")
    perf.add_column("Metric", style="cyan")
    perf.add_column("Value", style="green")
    perf.add_column("Status", style=rtf_style)
    perf.add_row("Processing time", f"{result.elapsed_time:.3f}s", "")
    perf.add_row("Audio duration", f"{result.duration:.3f}s", "")
    perf.add_row(
        "Real-time factor",
        f"{rtf:.3f}x",
        f"{rtf_icon} {'Real-time' if rtf < 1.0 else 'Slower than real-time'}",
    )
    perf.add_row(
        "Processing speed", f"{result.duration / result.elapsed_time:.1f}x", ""
    )
    perf.add_row("Chunks processed", str(result.chunk_count), "")
    if result.is_speech_segment:
        perf.add_row("Type", "[cyan]Speech Segment[/cyan]", "")
        if result.segment_start_utc:
            perf.add_row(
                "UTC Start",
                result.segment_start_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "",
            )
    console.print(perf)


def save_results(
    result: TaggingResult,
    output_dir: Path,
    chunk_events: Optional[List[dict]] = None,
) -> None:
    """
    Persist all 5 output files and print summary tables.

    Always saves:
      1. results.json       — Aggregated top-K predictions
      2. metadata.json      — Processing metadata and performance metrics

    When chunk_events is provided, also saves:
      3. chunk_results.json — Per-chunk raw probabilities
      4. chunk_timeline.png — Event probability timeline plot
      5. results_bar.png    — Aggregated results bar chart

    Args:
        result: The aggregated TaggingResult
        output_dir: Directory to save all output files
        chunk_events: Optional raw chunk events from process_audio_chunks.
                      When provided, enables per-chunk JSON and plots.
    """
    print_results_table(result.events, result.chunk_count, result.backend_name)
    print_perf_table(result)

    # Resolve output_dir to absolute path for terminal links
    output_dir = output_dir.resolve()

    # --- 1. Save results.json ---
    events_as_dicts = [
        {
            "rank": i + 1,
            "name": ev.name,
            "class_index": ev.class_index,
            "prob": ev.prob,
            "max_prob": ev.max_prob,
            "occurrences": ev.occurrences,
            "chunk_start": ev.chunk_start,
            "chunk_end": ev.chunk_end,
            "time_utc_start": ev.time_utc_start.isoformat()
            if ev.time_utc_start
            else None,
            "time_utc_end": ev.time_utc_end.isoformat() if ev.time_utc_end else None,
        }
        for i, ev in enumerate(result.events)
    ]

    results_json = output_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(events_as_dicts, f, indent=2, ensure_ascii=False)
    log.debug(f"Saved: {results_json}")

    # --- 2. Save metadata.json ---
    metadata = {
        "audio_file": result.audio_path,
        "sample_rate": result.sample_rate,
        "audio_duration_seconds": round(result.duration, 3),
        "processing_time_seconds": round(result.elapsed_time, 3),
        "real_time_factor": round(result.real_time_factor, 3),
        "backend": result.backend_name,
        "model_variant": result.model_variant,
        "top_k": result.top_k,
        "chunks_processed": result.chunk_count,
        "is_speech_segment": result.is_speech_segment,
        "segment_start_utc": result.segment_start_utc.isoformat()
        if result.segment_start_utc
        else None,
        "segment_end_utc": result.segment_end_utc.isoformat()
        if result.segment_end_utc
        else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregation_method": "average_probability_with_max",
    }

    metadata_json = output_dir / "metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.debug(f"Saved: {metadata_json}")

    # Helper to format terminal source link: file:///absolute/path
    def _link(path: Path, label: str) -> str:
        """Create a Rich-compatible terminal link with file:// URI."""
        uri = path.as_uri()  # file:///C:/Users/... or file:///home/...
        return f"[link={uri}]{label}[/link]"

    # Collect paths for summary panel with terminal links
    saved_files = [
        f"[cyan]1. Results:[/cyan]  {_link(results_json, results_json.name)}",
        f"[cyan]2. Metadata:[/cyan] {_link(metadata_json, metadata_json.name)}",
    ]

    # --- 3, 4, 5. Save chunk_results.json and plots (if chunk_events provided) ---
    if chunk_events:
        log.debug(
            f"save_results: received {len(chunk_events)} raw chunk events — "
            f"generating per-chunk JSON and plots"
        )

        # 3. Save chunk_results.json
        try:
            chunk_path = _save_chunk_results(chunk_events, output_dir, result.top_k)
            saved_files.append(
                f"[cyan]3. Chunks:[/cyan]   {_link(chunk_path, chunk_path.name)}"
            )
        except Exception as e:
            log.error(f"Failed to save chunk_results.json: {e}", exc_info=True)
            saved_files.append("[red]3. Chunks:[/red]   FAILED — see log")

        # 4 & 5. Generate and save plots
        try:
            timeline_path, bar_path = _generate_and_save_plots(
                chunk_events=chunk_events,
                aggregated_events=result.events,
                output_dir=output_dir,
                backend_name=result.backend_name,
                audio_path=result.audio_path,
            )
            saved_files.extend([
                f"[cyan]4. Timeline:[/cyan] {_link(timeline_path, timeline_path.name)}",
                f"[cyan]5. Bar chart:[/cyan] {_link(bar_path, bar_path.name)}",
            ])
        except Exception as e:
            log.error(f"Failed to generate plots: {e}", exc_info=True)
            saved_files.append("[red]4-5. Plots:[/red]  FAILED — see log")
    else:
        log.debug(
            "save_results: no chunk_events provided — "
            "skipping chunk_results.json and plots"
        )
        saved_files.append("[dim]3-5. (no chunk data — skipping plots)[/dim]")

    console.print(
        Panel(
            "\n".join(saved_files),
            title="💾 Saved files",
            border_style="green",
        )
    )


def _save_chunk_results(
    chunk_events: List[dict],
    output_dir: Path,
    top_k: int = 5,
) -> Path:
    """
    Save per-chunk results to chunk_results.json.

    Groups events by chunk_index and saves the raw probabilities for each chunk,
    allowing detailed analysis of how predictions change over time.

    Args:
        chunk_events: List of raw event dicts from process_audio_chunks
        output_dir: Directory to save the JSON file
        top_k: Number of top events to include per chunk

    Returns:
        Path to the saved file
    """
    # Group events by chunk index
    chunks_by_index: Dict[int, List[dict]] = {}
    for event in chunk_events:
        chunk_idx = event["chunk_index"]
        if chunk_idx not in chunks_by_index:
            chunks_by_index[chunk_idx] = []
        chunks_by_index[chunk_idx].append(event)

    log.debug(
        f"_save_chunk_results: {len(chunk_events)} events across "
        f"{len(chunks_by_index)} chunks"
    )

    chunk_results = []
    for chunk_idx in sorted(chunks_by_index.keys()):
        events = chunks_by_index[chunk_idx]
        chunk_data = {
            "chunk_index": chunk_idx,
            "chunk_start": events[0]["chunk_start"],
            "chunk_end": events[0]["chunk_end"],
            "time_utc_start": events[0]["time_utc_start"].isoformat()
            if events[0].get("time_utc_start")
            else None,
            "time_utc_end": events[0]["time_utc_end"].isoformat()
            if events[0].get("time_utc_end")
            else None,
            "total_events": len(events),
            "top_events": sorted(
                [
                    {
                        "name": e["name"],
                        "index": e["index"],
                        "prob": e["prob"],
                    }
                    for e in events
                ],
                key=lambda x: x["prob"],
                reverse=True,
            )[:top_k],
            "all_events": sorted(
                [
                    {
                        "name": e["name"],
                        "index": e["index"],
                        "prob": e["prob"],
                    }
                    for e in events
                ],
                key=lambda x: x["prob"],
                reverse=True,
            ),
        }
        chunk_results.append(chunk_data)

    chunk_results_path = output_dir / "chunk_results.json"
    with open(chunk_results_path, "w", encoding="utf-8") as f:
        json.dump(chunk_results, f, indent=2, ensure_ascii=False)

    log.info(
        f"Saved [cyan]{len(chunk_results)}[/cyan] chunk results to "
        f"[cyan]{chunk_results_path}[/cyan]"
    )
    return chunk_results_path


def _generate_and_save_plots(
    chunk_events: List[dict],
    aggregated_events: List[TaggingEvent],
    output_dir: Path,
    backend_name: str,
    audio_path: str,
) -> Tuple[Path, Path]:
    """
    Generate and save visualization plots for audio tagging results.

    Creates two plots:
    1. chunk_timeline.png — How top event probabilities change over time/chunks
    2. results_bar.png    — Bar chart of final aggregated results

    Args:
        chunk_events: Raw per-chunk events from process_audio_chunks
        aggregated_events: Aggregated TaggingEvent objects
        output_dir: Directory to save plot files
        backend_name: Name of the backend (CED, Zipformer)
        audio_path: Path to the audio file (for plot titles)

    Returns:
        Tuple of (chunk_timeline_path, results_bar_path)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file saving
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    log.debug(
        f"_generate_and_save_plots: {len(chunk_events)} chunk events, "
        f"{len(aggregated_events)} aggregated events"
    )

    # Style configuration
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(aggregated_events), 1)))

    # =====================================================================
    # Plot 1: Chunk Timeline — Top events probability over time
    # =====================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Get unique top event names (up to 5 for readability)
    top_event_names = [e.name for e in aggregated_events[:5] if e.name]

    # Group events by chunk
    chunks_by_index: Dict[int, List[dict]] = {}
    for event in chunk_events:
        chunk_idx = event["chunk_index"]
        if chunk_idx not in chunks_by_index:
            chunks_by_index[chunk_idx] = []
        chunks_by_index[chunk_idx].append(event)

    chunk_indices = sorted(chunks_by_index.keys())
    chunk_midpoints = [
        (chunks_by_index[i][0]["chunk_start"] + chunks_by_index[i][0]["chunk_end"]) / 2
        for i in chunk_indices
    ]

    log.debug(
        f"Timeline plot: {len(chunk_indices)} chunks, "
        f"{len(top_event_names)} event traces"
    )

    # Plot timeline for each top event
    for idx, event_name in enumerate(top_event_names):
        probs = []
        for chunk_idx in chunk_indices:
            chunk_events_list = chunks_by_index[chunk_idx]
            # Find this event in the chunk, default to 0.0 if absent
            found_prob = 0.0
            for e in chunk_events_list:
                if e["name"] == event_name:
                    found_prob = e["prob"]
                    break
            probs.append(found_prob)

        ax1.plot(
            chunk_midpoints,
            probs,
            marker='o',
            linewidth=2,
            markersize=4,
            color=colors[idx],
            label=event_name[:50],  # Truncate long names for legend
            alpha=0.8,
        )

    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'Event Probabilities Over Time — {backend_name}\n'
        f'{Path(audio_path).name}',
        fontsize=14,
        fontweight='bold',
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()

    # Save chunk timeline
    chunk_timeline_path = output_dir / "chunk_timeline.png"
    fig1.savefig(
        chunk_timeline_path, dpi=150, bbox_inches='tight', facecolor='white'
    )
    plt.close(fig1)
    log.debug(f"Saved timeline plot: {chunk_timeline_path}")

    # =====================================================================
    # Plot 2: Aggregated Results Bar Chart
    # =====================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    event_names_display = [
        (e.name[:60] + '...') if e.name and len(e.name) > 60
        else (e.name or 'Unknown')
        for e in aggregated_events
    ]
    avg_probs = [e.prob for e in aggregated_events]
    max_probs = [e.max_prob for e in aggregated_events]

    y_pos = range(len(aggregated_events))

    # Create horizontal bars
    ax2.barh(
        y_pos, avg_probs, height=0.6, color=colors, alpha=0.8,
        label='Average Probability'
    )
    ax2.barh(
        y_pos, max_probs, height=0.3, color='lightgray', alpha=0.5,
        label='Max Probability'
    )

    # Add probability labels on bars
    for i, (avg, max_p) in enumerate(zip(avg_probs, max_probs)):
        ax2.text(
            avg + 0.02, i,
            f'{avg*100:.1f}% (max: {max_p*100:.1f}%)',
            va='center',
            fontsize=9,
            fontweight='bold',
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(event_names_display, fontsize=10)
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_title(
        f'Aggregated Audio Tagging Results — {backend_name}\n'
        f'{Path(audio_path).name}',
        fontsize=14,
        fontweight='bold',
    )
    ax2.set_xlim(0, 1.1)
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.2, axis='x')
    ax2.invert_yaxis()  # Highest probability on top

    plt.tight_layout()

    # Save bar chart
    results_bar_path = output_dir / "results_bar.png"
    fig2.savefig(
        results_bar_path, dpi=150, bbox_inches='tight', facecolor='white'
    )
    plt.close(fig2)
    log.debug(f"Saved bar chart: {results_bar_path}")

    log.info(
        f"Saved plots:\n"
        f"  [cyan]Chunk timeline:[/cyan] {chunk_timeline_path}\n"
        f"  [cyan]Results bar:[/cyan]    {results_bar_path}"
    )

    return chunk_timeline_path, results_bar_path
