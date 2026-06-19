import argparse
import json
import shutil
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_rich_traceback
from custom_logging import linkify
from audio_tagger import (
    AUDIO_TAGGING_MODEL,
    CLASS_LABELS_INDICES_CSV,
    AudioTagger,
)
from config import SAMPLE_RATE
from main._main_audio_tagger import get_args, _format_predictions_with_emphasis, _get_probability_bar

install_rich_traceback(show_locals=True)
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves\waves\segment_001_wave_002\sound.wav"
args = get_args(audio_path)

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

console.print(f"\n[bold]Analyzing audio: {linkify(audio_path)}[/bold]\n")

# ── FIX: Pass output_dir when --save-speech-chunks is set ──
output_dir_param = Path(args.output_dir)

# ── NEW: Save filtered speech audio ──
import soundfile as sf

filtered_speech_only_audio = tagger.extract_speech_only(audio_path, prob_threshold=args.speech_threshold)
filtered_audio_path = output_dir_param / "filtered_speech_audio.wav"
sf.write(str(filtered_audio_path), filtered_speech_only_audio, SAMPLE_RATE)
console.print(f"[green]💾 Filtered speech audio saved to: {linkify(str(filtered_audio_path))}[/green]")
# ───────────────────────────────────────

summary = tagger.tag_audio_chunks(
    filtered_speech_only_audio,
    chunk_duration=args.chunk_duration,
    overlap_duration=args.chunk_overlap,
    output_dir=output_dir_param,  # ← KEY FIX
)
# ───────────────────────────────────────────────────────────

chunk_summary_table = Table(
    title="Chunk Analysis Summary",
    border_style="blue",
    show_header=True,
    header_style="bold cyan"
)
chunk_summary_table.add_column("Metric", style="cyan")
chunk_summary_table.add_column("Value", style="yellow")
chunk_summary_table.add_row(
    "Total Duration", f"{summary['total_duration']:.2f}s"
)
chunk_summary_table.add_row(
    "Total Chunks", str(summary['total_chunks'])
)
chunk_summary_table.add_row(
    "Chunk Duration", f"{summary['chunk_duration']:.2f}s"
)
chunk_summary_table.add_row(
    "Overlap", f"{summary['overlap_duration']:.2f}s"
)
chunk_summary_table.add_row(
    "Speech Detected",
    "✅ Yes" if summary['speech_detected'] else "❌ No"
)
chunk_summary_table.add_row(
    "Speech Duration",
    f"{summary['speech_duration']:.2f}s"
    f" ({summary['speech_duration']/summary['total_duration']*100:.1f}% of total)"
    if summary['total_duration'] > 0
    else "0.00s"
)
chunk_summary_table.add_row(
    "Max Speech Probability",
    f"{summary['max_speech_probability']:.4f}"
)
# ── NEW: Display avg speech probability ──
chunk_summary_table.add_row(
    "Avg Speech Probability",
    f"{summary['avg_speech_probability']:.4f}"
    if summary['avg_speech_probability'] > 0
    else "N/A (no speech chunks)"
)
# ─────────────────────────────────────────
chunk_summary_table.add_row(
    "Processing Time",
    f"{summary['total_processing_time']:.3f}s"
)
chunk_summary_table.add_row(
    "Real-Time Factor",
    f"{summary['real_time_factor']:.3f}x"
)
console.print(chunk_summary_table)

console.print("\n[bold]Overall Top Predictions:[/bold]")
tagger.display_results(summary["overall_top_predictions"])

chunk_table = Table(
    title="Per-Chunk Analysis",
    border_style="blue",
    show_header=True,
    header_style="bold cyan"
)
chunk_table.add_column("Chunk", justify="right", style="cyan")
chunk_table.add_column("Time Range", style="yellow")
chunk_table.add_column("Duration", justify="right")
chunk_table.add_column("Speech", justify="center", style="green")
chunk_table.add_column("Top Predictions", style="green", min_width=40)
chunk_table.add_column("Proc Time", justify="right")

for chunk in summary["chunks"]:
    predictions = chunk.get("predictions", [])
    predictions_display = _format_predictions_with_emphasis(
        predictions,
        threshold=args.display_threshold,
        max_display=3,
    )
    # ── FIX: Use speech_probability from chunk ──
    speech_indicator = (
        f"✅ {chunk['speech_probability']:.0%}"
        if chunk.get('speech_detected', False)
        else "❌ —"
    )
    # ────────────────────────────────────────────
    chunk_table.add_row(
        str(chunk["chunk_index"]),
        f"{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s",
        f"{chunk['duration']:.2f}s",
        speech_indicator,
        predictions_display,
        f"{chunk['processing_time'] * 1000:.1f}ms",
    )
console.print(chunk_table)
console.print(
    f"[dim]Showing predictions with probability ≥ {args.display_threshold:.0%}[/dim]"
)
console.print(
    f"[dim]Speech threshold: {args.speech_threshold:.0%} | "
    f"Speech duration: {summary['speech_duration']:.2f}s | "
    f"Avg speech prob: {summary['avg_speech_probability']:.4f}[/dim]"
)

# Generate plots if available
try:
    from audio_tagger_chunk_plots import (
        save_chunk_plots,
    )
    plot_paths = save_chunk_plots(
        summary=summary,
        output_dir=Path(args.output_dir),
        top_n_display=min(args.top_k, 10),
        probability_threshold=args.display_threshold,
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
        "[yellow]⚠ Plot module not available — skipping visualizations[/yellow]"
    )
except Exception as e:
    console.print(f"[red]⚠ Plot generation failed: {e}[/red]")

# Save chunk summary JSON
summary_output = Path(args.output_dir) / f"chunks_summary.json"
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

# ── NEW: Confirm speech chunks saved ──
speech_chunks_dir = Path(args.output_dir) / "speech_chunks"
if speech_chunks_dir.exists():
    num_chunks = len(list(speech_chunks_dir.iterdir()))
    console.print(
        f"[green]💾 {num_chunks} speech chunks saved to: "
        f"{linkify(str(speech_chunks_dir))}[/green]"
    )
# ───────────────────────────────────────

