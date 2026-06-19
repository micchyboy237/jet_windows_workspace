import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional
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

install_rich_traceback(show_locals=True)
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_args(default_audio: Optional[str] = None):
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
        help="Path to input audio file"
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
        default=5,
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
        "-t", "--speech-threshold",
        type=float,
        default=AudioTagger.DEFAULT_SPEECH_PROB_THRESHOLD,
        help="Minimum probability for speech detection",
    )
    parser.add_argument(
        "-n", "--speech-top-n",
        type=int,
        default=AudioTagger.DEFAULT_SPEECH_TOP_N,
        help="Check top N predictions for speech",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "-c", "--check-speech",
        action="store_true",
        help="Check if audio contains speech",
    )
    parser.add_argument(
        "-S", "--save-summary",
        action="store_true",
        help="Save comprehensive summary",
    )
    parser.add_argument(
        "-C", "--chunk",
        action="store_true",
        help="Use tag_audio_chunks instead of tag_audio",
    )
    parser.add_argument(
        "-D", "--chunk-duration",
        type=float,
        default=AudioTagger.DEFAULT_CHUNK_DURATION,
        help="Duration of each chunk in seconds",
    )
    parser.add_argument(
        "-O", "--chunk-overlap",
        type=float,
        default=AudioTagger.DEFAULT_CHUNK_OVERLAP,
        help="Overlap between chunks in seconds",
    )
    parser.add_argument(
        "-T", "--display-threshold",
        type=float,
        default=0.3,
        help="Minimum probability to display predictions in chunk table",
    )
    
    args = parser.parse_args()
    return args


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
            # ── FIX: Pass output_dir when --save-speech-chunks is set ──
            output_dir_param = Path(args.output_dir)
            
            summary = tagger.tag_audio_chunks(
                audio_path,
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
            summary_output = Path(args.output_dir) / f"{audio_name}_chunks_summary.json"
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
        
        else:
            results = tagger.tag_audio(audio_path)
            tagger.display_results(results)
            
            json_output = Path(args.output_dir) / f"{audio_name}_tags.json"
            tagger.save_results(results, json_output, format="json")
            
            txt_output = Path(args.output_dir) / f"{audio_name}_tags.txt"
            tagger.save_results(results, txt_output, format="txt")
            
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
            
            if args.save_summary:
                console.print("\n[bold]Generating Comprehensive Summary[/bold]")
                summary = tagger.get_tagging_summary(audio_path, audio_path=audio_path)
                
                summary_table = Table(
                    title="Audio Tagging Summary", border_style="magenta"
                )
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
                    f" ({summary['speech_duration']/summary['duration_seconds']*100:.1f}%)"
                    if summary['duration_seconds'] > 0
                    else "0.00s"
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
