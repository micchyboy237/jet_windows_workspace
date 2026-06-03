"""
audio_tagger_chunk_plots.py
===========================
Visualization utilities for chunked audio tagging results.
Produces 4 plot files:
  1. chunk_heatmap.png    — Heatmap showing top-K event probabilities across chunks
  2. events_timeline.png  — Line plot tracking event probabilities over time
  3. results_bar.png      — Horizontal bar chart of aggregated results
  4. chunks_summary.png   — Per-chunk mini bar chart with annotations

EMPHASIS ON PROBABILITY MAGNITUDE:
  - Marker sizes proportional to probability in timeline
  - Cell opacity and font weight vary with probability in heatmap
  - Bar color intensity scales with probability
  - Highlight threshold crossing with visual markers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib
from audio_tagger import AudioChunksTaggingSummary

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

COLORS = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
    "#009E73",
    "#56B4E9",
    "#E69F00",
    "#F0E442",
    "#0072B2",
]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "audio_heatmap",
    ["#FFFFFF", "#648FFF", "#1A1A6E"],
    N=256,
)

# Threshold for highlighting significant events
DEFAULT_PROBABILITY_THRESHOLD = 0.3
HIGH_PROBABILITY_THRESHOLD = 0.7
MEDIUM_PROBABILITY_THRESHOLD = 0.4


def save_chunk_plots(
    summary: AudioChunksTaggingSummary,
    output_dir: Path,
    top_n_display: int = 5,
    figsize_heatmap: Tuple[int, int] = (14, 8),
    figsize_timeline: Tuple[int, int] = (14, 7),
    figsize_bar: Tuple[int, int] = (10, 6),
    figsize_summary: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> List[Path]:
    """
    Generate all 4 visualization plots for chunked audio tagging results.

    Args:
        summary: AudioChunksTaggingSummary from tag_audio_chunks()
        output_dir: Directory to save plot files
        top_n_display: How many top events to show in heatmap/timeline
        figsize_heatmap: Figure size for heatmap (width, height)
        figsize_timeline: Figure size for timeline (width, height)
        figsize_bar: Figure size for bar chart (width, height)
        figsize_summary: Figure size for chunk summary (width, height)
        dpi: Dots per inch for output images
        probability_threshold: Minimum probability to emphasize in plots

    Returns:
        List of Paths to saved plot files

    Debug logs trace:
        - Plot generation start/completion
        - Number of data points per plot
        - Any rendering errors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = summary.get("chunks", [])
    overall_top = summary.get("overall_top_predictions", [])
    audio_path = summary.get("audio_path", "unknown")
    total_duration = summary.get("total_duration", 0)
    rtf = summary.get("real_time_factor", 0)

    logger.debug(
        f"save_chunk_plots: {len(chunks)} chunks, "
        f"{len(overall_top)} overall predictions, "
        f"top_n_display={top_n_display}, "
        f"probability_threshold={probability_threshold}"
    )

    saved_paths: List[Path] = []

    try:
        heatmap_path = _plot_chunk_heatmap(
            chunks=chunks,
            overall_top=overall_top,
            output_dir=output_dir,
            top_n_display=top_n_display,
            audio_path=audio_path,
            total_duration=total_duration,
            rtf=rtf,
            figsize=figsize_heatmap,
            dpi=dpi,
            probability_threshold=probability_threshold,
        )
        saved_paths.append(heatmap_path)
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}", exc_info=True)
        saved_paths.append(output_dir / "chunk_heatmap.png")

    try:
        timeline_path = _plot_events_timeline(
            chunks=chunks,
            overall_top=overall_top,
            output_dir=output_dir,
            top_n_display=top_n_display,
            audio_path=audio_path,
            total_duration=total_duration,
            rtf=rtf,
            figsize=figsize_timeline,
            dpi=dpi,
            probability_threshold=probability_threshold,
        )
        saved_paths.append(timeline_path)
    except Exception as e:
        logger.error(f"Failed to generate timeline: {e}", exc_info=True)
        saved_paths.append(output_dir / "events_timeline.png")

    try:
        bar_path = _plot_results_bar(
            overall_top=overall_top,
            output_dir=output_dir,
            audio_path=audio_path,
            total_duration=total_duration,
            rtf=rtf,
            figsize=figsize_bar,
            dpi=dpi,
            probability_threshold=probability_threshold,
        )
        saved_paths.append(bar_path)
    except Exception as e:
        logger.error(f"Failed to generate bar chart: {e}", exc_info=True)
        saved_paths.append(output_dir / "results_bar.png")

    try:
        summary_path = _plot_chunk_summary(
            chunks=chunks,
            output_dir=output_dir,
            top_n_display=min(top_n_display, 3),
            audio_path=audio_path,
            total_duration=total_duration,
            figsize=figsize_summary,
            dpi=dpi,
            probability_threshold=probability_threshold,
        )
        saved_paths.append(summary_path)
    except Exception as e:
        logger.error(f"Failed to generate chunk summary: {e}", exc_info=True)
        saved_paths.append(output_dir / "chunks_summary.png")

    return saved_paths


def _get_marker_size_and_alpha(probability, threshold):
    """
    Calculate marker size and alpha based on probability magnitude.

    Args:
        probability: Float between 0 and 1
        threshold: Minimum probability threshold

    Returns:
        Tuple of (marker_size, alpha, linewidth) for visual emphasis
    """
    if probability >= HIGH_PROBABILITY_THRESHOLD:
        # High probability: large markers, full opacity, thick lines
        return (120, 1.0, 2.5)
    elif probability >= MEDIUM_PROBABILITY_THRESHOLD:
        # Medium probability: moderate markers, good opacity
        return (80, 0.85, 2.0)
    elif probability >= threshold:
        # Low but above threshold: smaller markers, reduced opacity
        return (50, 0.7, 1.5)
    else:
        # Below threshold: minimal visibility
        return (20, 0.3, 1.0)


def _plot_chunk_heatmap(
    chunks: List[dict],
    overall_top: List[dict],
    output_dir: Path,
    top_n_display: int,
    audio_path: str,
    total_duration: float,
    rtf: float,
    figsize: Tuple[int, int],
    dpi: int,
    probability_threshold: float,
) -> Path:
    """
    Generate a heatmap showing top-K event probabilities across all chunks
    with EMPHASIS on probability magnitude:
    - Cell text size and boldness increase with probability
    - Highlight cells above threshold with border
    - Add glow effect for high-probability cells
    """
    if not chunks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunks to display",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Chunk Probability Heatmap (No Data)")
        path = output_dir / "chunk_heatmap.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    top_names = _get_top_event_names(overall_top, chunks, top_n_display)
    if not top_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No predictions available",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Chunk Probability Heatmap (No Predictions)")
        path = output_dir / "chunk_heatmap.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    n_chunks = len(chunks)
    n_events = len(top_names)
    data = np.zeros((n_events, n_chunks))

    for col_idx, chunk in enumerate(chunks):
        pred_map = {p["name"]: p["prob"] for p in chunk.get("predictions", [])}
        for row_idx, name in enumerate(top_names):
            data[row_idx, col_idx] = pred_map.get(name, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    aspect = "auto" if n_chunks > 20 else n_chunks / max(n_events, 1)

    im = ax.imshow(
        data,
        cmap=HEATMAP_CMAP,
        aspect=aspect,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Add cells with probability magnitude emphasis
    for row_idx in range(n_events):
        for col_idx in range(n_chunks):
            prob = data[row_idx, col_idx]
            if prob > 0:
                # Text styling based on probability magnitude
                fontsize = 7 if n_chunks > 15 else 9
                if prob >= HIGH_PROBABILITY_THRESHOLD:
                    fontsize += 2
                    fontweight = "bold"
                    text_color = "white"
                    # Add a subtle border for high-probability cells
                    ax.add_patch(
                        plt.Rectangle(
                            (col_idx - 0.5, row_idx - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="yellow",
                            linewidth=2,
                            alpha=0.8,
                        )
                    )
                elif prob >= MEDIUM_PROBABILITY_THRESHOLD:
                    fontsize += 1
                    fontweight = "bold"
                    text_color = "white"
                elif prob >= probability_threshold:
                    fontweight = "normal"
                    text_color = "white"
                else:
                    fontweight = "normal"
                    text_color = "black"

                ax.text(
                    col_idx,
                    row_idx,
                    f"{prob:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                    fontweight=fontweight,
                )
            else:
                ax.text(
                    col_idx,
                    row_idx,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )

    ax.set_yticks(range(n_events))
    ax.set_yticklabels(
        [name[:40] + "…" if len(name) > 40 else name for name in top_names],
        fontsize=9,
    )

    if n_chunks <= 30:
        chunk_labels = [f"C{c['chunk_index']}\n{c['start_time']:.1f}s" for c in chunks]
        ax.set_xticks(range(n_chunks))
        ax.set_xticklabels(chunk_labels, fontsize=7, rotation=45, ha="right")
    else:
        step = max(1, n_chunks // 15)
        ax.set_xticks(range(0, n_chunks, step))
        ax.set_xticklabels(
            [f"{chunks[i]['start_time']:.1f}s" for i in range(0, n_chunks, step)],
            fontsize=7,
            rotation=45,
            ha="right",
        )

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Probability", fontsize=10, fontweight="bold")

    # Add threshold lines on colorbar
    cbar.ax.axhline(
        y=probability_threshold, color="orange", linestyle="--", linewidth=1, alpha=0.7
    )
    cbar.ax.axhline(
        y=HIGH_PROBABILITY_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    ax.set_title(
        f"Event Probability Heatmap Across Chunks\n"
        f"{Path(audio_path).name}\n"
        f"Duration: {total_duration:.1f}s | Chunks: {n_chunks} | RTF: {rtf:.3f}\n"
        f"Threshold: {probability_threshold:.0%} | Yellow border = High prob (≥{HIGH_PROBABILITY_THRESHOLD:.0%})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Chunk (start time)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Event Label", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "chunk_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_events_timeline(
    chunks: List[dict],
    overall_top: List[dict],
    output_dir: Path,
    top_n_display: int,
    audio_path: str,
    total_duration: float,
    rtf: float,
    figsize: Tuple[int, int],
    dpi: int,
    probability_threshold: float,
) -> Path:
    """
    Generate a line plot tracking top event probabilities over time
    with EMPHASIS on probability magnitude:
    - Marker sizes proportional to probability
    - High-probability points get larger markers with glow effect
    - Threshold crossing highlighted with vertical indicators
    """
    if not chunks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunks to display",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Event Probability Timeline (No Data)")
        path = output_dir / "events_timeline.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    top_names = _get_top_event_names(overall_top, chunks, top_n_display)
    if not top_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No predictions available",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Event Probability Timeline (No Predictions)")
        path = output_dir / "events_timeline.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=figsize)
    chunk_times = [(c["start_time"] + c["end_time"]) / 2 for c in chunks]

    for idx, event_name in enumerate(top_names):
        color = COLORS[idx % len(COLORS)]
        probs = []
        for chunk in chunks:
            pred_map = {p["name"]: p["prob"] for p in chunk.get("predictions", [])}
            probs.append(pred_map.get(event_name, np.nan))

        valid_indices = [i for i, p in enumerate(probs) if not np.isnan(p)]
        valid_times = [chunk_times[i] for i in valid_indices]
        valid_probs = [probs[i] for i in valid_indices]

        if len(valid_probs) >= 2:
            # Draw line with varying alpha based on probability
            ax.plot(
                valid_times,
                valid_probs,
                color=color,
                linewidth=2,
                alpha=0.85,
                label=event_name[:50],
                zorder=2,
            )

            # Add markers with size proportional to probability
            for t, p in zip(valid_times, valid_probs):
                if p >= probability_threshold:
                    marker_size, marker_alpha, line_width = _get_marker_size_and_alpha(
                        p, probability_threshold
                    )

                    # Main marker
                    ax.scatter(
                        [t],
                        [p],
                        s=marker_size,
                        color=color,
                        alpha=marker_alpha,
                        edgecolors="white",
                        linewidth=1,
                        zorder=3,
                    )

                    # Add glow effect for high-probability points
                    if p >= HIGH_PROBABILITY_THRESHOLD:
                        ax.scatter(
                            [t],
                            [p],
                            s=marker_size * 2,
                            color=color,
                            alpha=0.2,
                            edgecolors="none",
                            zorder=2,
                        )
                else:
                    # Small markers for below-threshold points
                    ax.scatter(
                        [t],
                        [p],
                        s=30,
                        color=color,
                        alpha=0.4,
                        zorder=2,
                    )

            # Add shaded area for above-threshold regions
            above_threshold = [p >= probability_threshold for p in valid_probs]
            if any(above_threshold):
                ax.fill_between(
                    valid_times,
                    [probability_threshold] * len(valid_times),
                    valid_probs,
                    where=above_threshold,
                    alpha=0.15,
                    color=color,
                    interpolate=True,
                )

        elif len(valid_probs) == 1:
            p = valid_probs[0]
            marker_size, marker_alpha, _ = _get_marker_size_and_alpha(
                p, probability_threshold
            )
            ax.scatter(
                valid_times,
                valid_probs,
                s=marker_size,
                color=color,
                alpha=marker_alpha,
                marker="D",
                label=event_name[:50],
                edgecolors="white",
                linewidth=1,
                zorder=5,
            )

    # Add threshold lines with labels
    ax.axhline(
        y=probability_threshold,
        color="orange",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"Threshold ({probability_threshold:.0%})",
    )
    ax.axhline(
        y=HIGH_PROBABILITY_THRESHOLD,
        color="red",
        linestyle=":",
        alpha=0.4,
        linewidth=1,
        label=f"High prob ({HIGH_PROBABILITY_THRESHOLD:.0%})",
    )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, total_duration + 0.5 if total_duration > 0 else 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Top {top_n_display} Event Probabilities Over Time\n"
        f"{Path(audio_path).name}\n"
        f"Duration: {total_duration:.1f}s | Chunks: {len(chunks)} | RTF: {rtf:.3f}\n"
        f"Marker size ∝ probability | Shaded = above threshold",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    path = output_dir / "events_timeline.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_results_bar(
    overall_top: List[dict],
    output_dir: Path,
    audio_path: str,
    total_duration: float,
    rtf: float,
    figsize: Tuple[int, int],
    dpi: int,
    probability_threshold: float,
) -> Path:
    """
    Generate a horizontal bar chart of overall aggregated results
    with EMPHASIS on probability magnitude:
    - Bar color intensity scales with probability
    - High-probability bars get gradient effect
    - Add threshold indicator
    """
    if not overall_top:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No aggregated predictions",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Aggregated Results (No Data)")
        path = output_dir / "results_bar.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=figsize)

    names = [
        p["name"][:60] + "…"
        if len(p.get("name", "")) > 60
        else p.get("name", "Unknown")
        for p in overall_top
    ]
    probs = [p.get("prob", 0) for p in overall_top]

    names = names[::-1]
    probs = probs[::-1]
    y_pos = range(len(names))

    # Create bars with emphasis on probability magnitude
    bars = []
    for i, (name, prob) in enumerate(zip(names, probs)):
        # Color intensity based on probability
        if prob >= HIGH_PROBABILITY_THRESHOLD:
            # High probability: use gradient from blue to dark blue
            color = _interpolate_color("#648FFF", "#1A1A6E", prob)
            alpha = 1.0
            edge_color = "#FFD700"  # Gold border for high prob
            edge_width = 2
            hatch = ""
        elif prob >= MEDIUM_PROBABILITY_THRESHOLD:
            color = _interpolate_color("#648FFF", "#DC267F", prob)
            alpha = 0.9
            edge_color = "white"
            edge_width = 1
            hatch = ""
        elif prob >= probability_threshold:
            color = "#648FFF"
            alpha = 0.8
            edge_color = "white"
            edge_width = 1
            hatch = ""
        else:
            color = "#AAAAAA"
            alpha = 0.6
            edge_color = "gray"
            edge_width = 1
            hatch = "//"

        bar = ax.barh(
            i,
            prob,
            height=0.6,
            color=color,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=edge_width,
            hatch=hatch,
        )
        bars.append(bar)

    # Add probability labels with emphasis
    for i, (prob, bar_group) in enumerate(zip(probs, bars)):
        label_x = prob + 0.02
        if prob >= HIGH_PROBABILITY_THRESHOLD:
            label_size = 13
            label_weight = "bold"
            label_color = "#1A1A6E"
            # Add star for high probability
            label_text = f"★ {prob:.1%}"
        elif prob >= MEDIUM_PROBABILITY_THRESHOLD:
            label_size = 12
            label_weight = "bold"
            label_color = "#333333"
            label_text = f"{prob:.1%}"
        elif prob >= probability_threshold:
            label_size = 11
            label_weight = "normal"
            label_color = "#555555"
            label_text = f"{prob:.1%}"
        else:
            label_size = 10
            label_weight = "normal"
            label_color = "#888888"
            label_text = f"{prob:.1%}"

        ax.text(
            label_x,
            bar_group[0].get_y() + bar_group[0].get_height() / 2,
            label_text,
            va="center",
            fontsize=label_size,
            fontweight=label_weight,
            color=label_color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Add threshold line
    ax.axvline(
        x=probability_threshold,
        color="orange",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Threshold ({probability_threshold:.0%})",
    )

    ax.set_xlabel("Mean Probability Across Chunks", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Aggregated Audio Tagging Results\n"
        f"{Path(audio_path).name}\n"
        f"Duration: {total_duration:.1f}s | RTF: {rtf:.3f}\n"
        f"★ = High confidence (≥{HIGH_PROBABILITY_THRESHOLD:.0%}) | Hatch = Below threshold",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.2, axis="x")
    ax.legend(fontsize=8, loc="lower right")

    # Add rank numbers with emphasis
    for i in range(len(names)):
        if probs[i] >= HIGH_PROBABILITY_THRESHOLD:
            rank_color = "red"
            rank_weight = "bold"
        else:
            rank_color = "gray"
            rank_weight = "normal"

        ax.text(
            -0.03,
            i,
            f"#{len(names) - i}",
            va="center",
            ha="right",
            fontsize=8,
            color=rank_color,
            fontweight=rank_weight,
        )

    plt.tight_layout()
    path = output_dir / "results_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_chunk_summary(
    chunks: List[dict],
    output_dir: Path,
    top_n_display: int,
    audio_path: str,
    total_duration: float,
    figsize: Tuple[int, int],
    dpi: int,
    probability_threshold: float,
) -> Path:
    """
    Generate a multi-panel summary showing per-chunk mini bar charts
    with EMPHASIS on probability magnitude:
    - Border color indicates max probability in chunk
    - Bar opacity varies with probability
    """
    if not chunks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunks to display",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Per-Chunk Summary (No Data)")
        path = output_dir / "chunks_summary.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    MAX_DISPLAY_CHUNKS = 24
    display_chunks = chunks[:MAX_DISPLAY_CHUNKS]
    n_chunks = len(display_chunks)

    if n_chunks == 1:
        n_rows, n_cols = 1, 1
    elif n_chunks <= 4:
        n_rows, n_cols = 2, 2
    elif n_chunks <= 9:
        n_rows, n_cols = 3, 3
    elif n_chunks <= 16:
        n_rows, n_cols = 4, 4
    else:
        n_rows, n_cols = 4, 6

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )
    fig.suptitle(
        f"Per-Chunk Top-{top_n_display} Predictions\n"
        f"{Path(audio_path).name} ({total_duration:.1f}s, {len(chunks)} total chunks)\n"
        f"Border color = confidence level | Bar opacity ∝ probability",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx in range(n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if idx < n_chunks:
            chunk = display_chunks[idx]
            predictions = chunk.get("predictions", [])[:top_n_display]

            # Determine max probability for border color
            max_prob = max([p["prob"] for p in predictions]) if predictions else 0

            if predictions:
                names = [p["name"][:25] for p in predictions][::-1]
                probs = [p["prob"] for p in predictions][::-1]
                y_pos = range(len(names))

                # Bar colors and opacity based on probability
                for j, (name, prob) in enumerate(zip(names, probs)):
                    if prob >= HIGH_PROBABILITY_THRESHOLD:
                        color = _interpolate_color("#648FFF", "#1A1A6E", prob)
                        alpha = 1.0
                    elif prob >= MEDIUM_PROBABILITY_THRESHOLD:
                        color = _interpolate_color("#648FFF", "#DC267F", prob)
                        alpha = 0.85
                    elif prob >= probability_threshold:
                        color = "#648FFF"
                        alpha = 0.7
                    else:
                        color = "#AAAAAA"
                        alpha = 0.5

                    ax.barh(
                        j,
                        prob,
                        height=0.7,
                        color=color,
                        alpha=alpha,
                        edgecolor="white",
                        linewidth=0.5,
                    )

                if probs:
                    # Add label for top prediction with emphasis
                    top_prob = probs[-1]
                    if top_prob >= HIGH_PROBABILITY_THRESHOLD:
                        label_style = "bold"
                        label_color = "#1A1A6E"
                        label_prefix = "★ "
                    else:
                        label_style = "normal"
                        label_color = "#333333"
                        label_prefix = ""

                    ax.text(
                        top_prob + 0.05,
                        len(probs) - 1,
                        f"{label_prefix}{top_prob:.0%}",
                        va="center",
                        fontsize=8,
                        fontweight=label_style,
                        color=label_color,
                    )

                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=6)
                ax.set_xlim(0, 1.1)
                ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
                ax.tick_params(axis="x", labelsize=6)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No predictions",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                    transform=ax.transAxes,
                )

            # Add border color based on max probability
            if max_prob >= HIGH_PROBABILITY_THRESHOLD:
                border_color = "gold"
                border_width = 2
            elif max_prob >= MEDIUM_PROBABILITY_THRESHOLD:
                border_color = "#648FFF"
                border_width = 1.5
            elif max_prob >= probability_threshold:
                border_color = "gray"
                border_width = 1
            else:
                border_color = "lightgray"
                border_width = 0.5

            for spine in ax.spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(border_width)

            time_label = (
                f"C{chunk['chunk_index']}: "
                f"{chunk['start_time']:.1f}-{chunk['end_time']:.1f}s"
            )
            ax.set_title(time_label, fontsize=7, fontweight="bold", pad=3)
        else:
            ax.axis("off")

    for idx in range(n_chunks, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "chunks_summary.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _get_top_event_names(
    overall_top: List[dict],
    chunks: List[dict],
    top_n: int,
) -> List[str]:
    """
    Extract top-N event names, prioritizing overall_top, falling back to
    collecting unique names from chunks.
    """
    if overall_top:
        return [p["name"] for p in overall_top[:top_n] if p.get("name")]

    seen = set()
    names = []
    for chunk in chunks:
        for pred in chunk.get("predictions", []):
            name = pred.get("name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
                if len(names) >= top_n:
                    return names
    return names


def _interpolate_color(hex_start: str, hex_end: str, factor: float) -> str:
    """
    Linear interpolation between two hex colors.
    factor=0 → start color, factor=1 → end color.
    """
    factor = max(0.0, min(1.0, factor))

    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"

    r1, g1, b1 = _hex_to_rgb(hex_start)
    r2, g2, b2 = _hex_to_rgb(hex_end)

    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)

    return _rgb_to_hex(r, g, b)
