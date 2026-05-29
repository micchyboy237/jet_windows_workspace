# audio_tagger_zipformer.py

"""
This script shows how to use audio tagging Python APIs to tag a file.
Uses sherpa-onnx Zipformer audio tagging models.

Available models:
- Standard: sherpa-onnx-zipformer-audio-tagging-2024-04-09 (288 MB)
- Small: sherpa-onnx-zipformer-small-audio-tagging-2024-04-15 (106 MB)

Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
"""

import json
import time
import argparse
import shutil
from pathlib import Path
from typing import List, Any, Optional

import numpy as np
import sherpa_onnx
import soundfile as sf
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import logging

# Setup rich console
console = Console()

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
log = logging.getLogger("rich")

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()

# Model definitions
MODELS = {
    "standard": {
        "name": "sherpa-onnx-zipformer-audio-tagging-2024-04-09",
        "size": "288 MB",
        "description": "Standard (larger, more accurate)",
        "expected_frames": 80,  # 0.8 seconds at 10ms hop (for 16kHz audio)
    },
    "small": {
        "name": "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15",
        "size": "106 MB",
        "description": "Small (faster, less accurate)",
        "expected_frames": 80,  # 0.8 seconds at 10ms hop (for 16kHz audio)
    }
}

# Audio processing constants
SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10ms at 16kHz


def get_model_paths(use_small: bool = False):
    """Get model and label paths based on model selection."""
    model_key = "small" if use_small else "standard"
    model_info = MODELS[model_key]
    model_name = model_info["name"]
    
    return {
        "model": BASE_DIR / model_name / "model.onnx",
        "model_int8": BASE_DIR / model_name / "model.int8.onnx",
        "labels": BASE_DIR / model_name / "class_labels_indices.csv",
        "test_wavs_dir": BASE_DIR / model_name / "test_wavs",
        "tokens": BASE_DIR / model_name / "tokens.txt",
        "model_info": model_info
    }


def find_model_file(paths: dict) -> str:
    """Find the actual model file, trying int8 first then regular."""
    if paths["model_int8"].is_file():
        log.info("Using int8 quantized model")
        return str(paths["model_int8"])
    elif paths["model"].is_file():
        log.info("Using standard model")
        return str(paths["model"])
    else:
        raise FileNotFoundError(
            f"No model file found. Checked:\n"
            f"  • {paths['model_int8']}\n"
            f"  • {paths['model']}"
        )


def read_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Read audio file and return samples and sample rate."""
    audio_path_obj = Path(audio_path)
    
    if not audio_path_obj.is_file():
        raise ValueError(
            f"Audio file not found: {audio_path}\n"
            "Please check the path and try again."
        )

    # Read audio file
    data, sample_rate = sf.read(
        audio_path,
        always_2d=True,
        dtype="float32",
    )
    
    # Use only the first channel
    data = data[:, 0]
    samples = np.ascontiguousarray(data)
    
    return samples, sample_rate


def resample_if_needed(samples: np.ndarray, orig_sample_rate: int, target_sample_rate: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate if needed."""
    if orig_sample_rate == target_sample_rate:
        return samples
    
    log.info(f"Resampling from {orig_sample_rate}Hz to {target_sample_rate}Hz")
    
    # Simple linear resampling
    duration = len(samples) / orig_sample_rate
    target_length = int(duration * target_sample_rate)
    indices = np.linspace(0, len(samples) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    
    return resampled.astype(np.float32)


def process_audio_chunks(
    audio_tagger: sherpa_onnx.AudioTagging,
    samples: np.ndarray,
    sample_rate: int,
    expected_frames: int = 80,
    hop_length: int = 160
) -> List[Any]:
    """
    Process audio in chunks that match the model's expected input size.
    
    Args:
        audio_tagger: The initialized audio tagger
        samples: Audio samples array
        sample_rate: Sample rate of the audio
        expected_frames: Number of frames the model expects (default: 80 for 0.8s)
        hop_length: Hop length in samples (default: 160 for 10ms at 16kHz)
    
    Returns:
        Combined results from all chunks
    """
    expected_samples = expected_frames * hop_length
    total_samples = len(samples)
    
    log.info(f"Audio length: {total_samples} samples ({total_samples/sample_rate:.2f}s)")
    log.info(f"Model expects: {expected_samples} samples ({expected_frames} frames, {expected_samples/sample_rate:.2f}s)")
    
    # If audio is shorter than expected, pad it
    if total_samples < expected_samples:
        log.info(f"Padding short audio from {total_samples} to {expected_samples} samples")
        padded = np.zeros(expected_samples, dtype=np.float32)
        padded[:total_samples] = samples
        samples = padded
        total_samples = expected_samples
    
    # Process in overlapping chunks
    chunk_size = expected_samples
    hop_size = chunk_size // 2  # 50% overlap for smoother results
    
    all_events = []
    num_chunks = max(1, (total_samples - chunk_size) // hop_size + 1)
    
    log.info(f"Processing {num_chunks} overlapping chunks...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing chunks...", total=num_chunks)
        
        for i in range(num_chunks):
            start = i * hop_size
            end = min(start + chunk_size, total_samples)
            
            # Extract chunk (pad if last chunk is partial)
            chunk = np.zeros(chunk_size, dtype=np.float32)
            chunk_len = end - start
            chunk[:chunk_len] = samples[start:end]
            
            # Process chunk
            stream = audio_tagger.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
            result = audio_tagger.compute(stream)
            
            # Collect events with position info
            for event in result:
                all_events.append({
                    "name": getattr(event, "name", None),
                    "index": getattr(event, "index", None),
                    "prob": getattr(event, "prob", None),
                    "chunk_start": start / sample_rate,
                    "chunk_end": end / sample_rate,
                    "chunk_index": i,
                })
            
            progress.update(task, advance=1)
    
    return all_events


def aggregate_chunk_results(chunk_events: List[dict], top_k: int = 5) -> List[dict]:
    """
    Aggregate results from multiple chunks by averaging probabilities
    for the same event types.
    """
    if not chunk_events:
        return []
    
    # Group by event name/index
    event_groups = {}
    for event in chunk_events:
        key = (event["name"], event["index"])
        if key not in event_groups:
            event_groups[key] = {
                "name": event["name"],
                "index": event["index"],
                "probs": [],
                "occurrences": 0,
            }
        event_groups[key]["probs"].append(event["prob"])
        event_groups[key]["occurrences"] += 1
    
    # Calculate average probability and sort
    aggregated = []
    for key, data in event_groups.items():
        aggregated.append({
            "name": data["name"],
            "index": data["index"],
            "prob": np.mean(data["probs"]),
            "max_prob": np.max(data["probs"]),
            "occurrences": data["occurrences"],
        })
    
    # Sort by probability and return top_k
    aggregated.sort(key=lambda x: x["prob"], reverse=True)
    return aggregated[:top_k]


def create_audio_tagger(use_small: bool = False, top_k: int = 5):
    """Create and configure the audio tagger."""
    paths = get_model_paths(use_small)
    model_file = find_model_file(paths)
    label_file = paths["labels"]

    if not Path(label_file).is_file():
        raise ValueError(
            f"Labels file not found: {label_file}\n"
            "Please download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                model=str(model_file),          # ← correct: zipformer config
            ),
            num_threads=1,
            debug=True,
            provider="cpu",
        ),
        labels=str(label_file),
        top_k=top_k,
    )

    if not config.validate():
        raise ValueError(f"Invalid configuration: {config}")

    model_key = "small" if use_small else "standard"
    config_table = Table(title="🎯 Audio Tagger Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model Variant", paths["model_info"]["description"])
    config_table.add_row("Model Size", paths["model_info"]["size"])
    config_table.add_row("Model Path", str(model_file))
    config_table.add_row("Labels Path", str(label_file))
    config_table.add_row("Top K", str(top_k))
    config_table.add_row("Provider", "cpu")
    config_table.add_row("Debug", "True")
    config_table.add_row("Threads", "1")
    console.print(config_table)

    return sherpa_onnx.AudioTagging(config)


def save_results(
    results: List[dict],
    output_dir: Path,
    audio_path: str,
    sample_rate: int,
    duration: float,
    elapsed_time: float,
    use_small: bool,
    top_k: int,
    chunk_count: int,
):
    """Save results to JSON and create a summary display."""
    
    # Save results to JSON
    results_json = output_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Display results table
    table = Table(
        title="🎵 Audio Tagging Results (Aggregated)",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Label", style="green")
    table.add_column("Index", style="yellow", width=8)
    table.add_column("Avg Prob", style="bold white", justify="right")
    table.add_column("Max Prob", style="dim white", justify="right")
    table.add_column("Occurrences", style="blue", justify="right")
    table.add_column("Distribution", style="magenta")
    
    for i, item in enumerate(results):
        avg_prob_percent = f"{item['prob']*100:.2f}%"
        max_prob_percent = f"{item['max_prob']*100:.2f}%"
        
        # Create probability bar
        bar_length = int(item['prob'] * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        table.add_row(
            str(i + 1),
            item['name'] or "N/A",
            str(item['index']) if item['index'] is not None else "N/A",
            avg_prob_percent,
            max_prob_percent,
            f"{item['occurrences']}/{chunk_count}",
            bar
        )
    
    console.print(table)
    
    # Save metadata
    metadata = {
        "audio_file": str(audio_path),
        "sample_rate": sample_rate,
        "audio_duration_seconds": round(duration, 3),
        "processing_time_seconds": round(elapsed_time, 3),
        "real_time_factor": round(elapsed_time / duration, 3) if duration > 0 else 0,
        "model_type": "small" if use_small else "standard",
        "model_info": MODELS["small" if use_small else "standard"],
        "top_k": top_k,
        "chunks_processed": chunk_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregation_method": "average_probability_with_max",
    }
    
    metadata_json = output_dir / "metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Display metadata summary
    meta_table = Table(title="📊 Processing Metadata", show_header=True)
    meta_table.add_column("Metric", style="cyan")
    meta_table.add_column("Value", style="green")
    meta_table.add_row("Audio Duration", f"{duration:.3f}s")
    meta_table.add_row("Processing Time", f"{elapsed_time:.3f}s")
    meta_table.add_row("Real-Time Factor", f"{elapsed_time/duration:.3f}x" if duration > 0 else "N/A")
    meta_table.add_row("Sample Rate", f"{sample_rate} Hz")
    meta_table.add_row("Model Type", metadata["model_type"])
    meta_table.add_row("Chunks Processed", str(chunk_count))
    
    console.print(meta_table)
    
    # Show saved file paths with clickable links
    saved_files = Panel(
        f"[cyan]Results:[/cyan] [link=file://{results_json.absolute()}]{results_json}[/link]\n"
        f"[cyan]Metadata:[/cyan] [link=file://{metadata_json.absolute()}]{metadata_json}[/link]",
        title="💾 Saved Files",
        border_style="green"
    )
    console.print(saved_files)


def main():
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="[bold cyan]Audio Tagging with Zipformer[/bold cyan] - Tag audio files using sherpa-onnx",
        epilog=(
            "Examples:\n"
            "  %(prog)s audio.wav\n"
            "  %(prog)s audio.wav --small -k 10\n"
            "  %(prog)s audio.wav --small -o ./my_results\n"
            "\n"
            "[bold]Available Models:[/bold]\n"
            f"  • Standard: {MODELS['standard']['name']} ({MODELS['standard']['size']})\n"
            f"  • Small:   {MODELS['small']['name']} ({MODELS['small']['size']})\n"
            "\n"
            "Download from: [link=https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models]"
            "GitHub Releases[/link]"
        )
    )
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        help="Path to input wave file (optional if using default test file)"
    )
    parser.add_argument(
        "-s",
        "--small",
        action="store_true",
        help="Use the small model (106 MB) instead of the standard model (288 MB). "
             "Small model is faster but less accurate."
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help=f"Output directory (default: '{OUTPUT_DIR}')"
    )
    args = parser.parse_args()

    # Set default audio path if not provided
    if args.audio_path is None:
        paths = get_model_paths(args.small)
        default_audio = paths["test_wavs_dir"] / "6.wav"
        if default_audio.is_file():
            args.audio_path = str(default_audio)
            log.info(f"No audio path provided, using default test file")
        else:
            console.print(
                "[red]No audio file provided and no default test file found.[/red]\n"
                f"Expected at: {default_audio}\n"
                "Please download test files from the GitHub releases page."
            )
            exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        log.info(f"Cleaning output directory: [cyan]{output_dir}[/cyan]")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Welcome banner
    model_type = "Small" if args.small else "Standard"
    model_info = MODELS["small" if args.small else "standard"]
    
    console.print(Panel.fit(
        f"[bold yellow]🎵 Audio Tagging Tool[/bold yellow]\n"
        f"[dim]Model: {model_type} ({model_info['size']}) | Top-K: {args.top_k}[/dim]\n"
        f"[dim]Chunk size: {model_info['expected_frames']} frames (0.8s) with 50% overlap[/dim]",
        border_style="blue"
    ))

    # Create audio tagger with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Initializing {model_type.lower()} audio tagger...", 
            total=None
        )
        audio_tagger = create_audio_tagger(use_small=args.small, top_k=args.top_k)
        progress.update(task, completed=True, description="[green]✓ Audio tagger initialized")

    # Read audio file
    log.info(f"Reading audio file: [link=file://{Path(args.audio_path).absolute()}]{args.audio_path}[/link]")
    samples, orig_sample_rate = read_audio(args.audio_path)
    
    # Resample if needed
    if orig_sample_rate != SAMPLE_RATE:
        samples = resample_if_needed(samples, orig_sample_rate, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    else:
        sample_rate = orig_sample_rate
    
    audio_duration = len(samples) / sample_rate
    log.info(
        f"Audio loaded: [cyan]{len(samples):,}[/cyan] samples, "
        f"[cyan]{sample_rate}Hz[/cyan], [cyan]{audio_duration:.2f}s[/cyan]"
    )

    # Process audio in chunks
    log.info("Processing audio in chunks...")
    start_time = time.time()

    expected_frames = MODELS["small" if args.small else "standard"]["expected_frames"]
    
    chunk_events = process_audio_chunks(
        audio_tagger=audio_tagger,
        samples=samples,
        sample_rate=sample_rate,
        expected_frames=expected_frames,
    )
    
    # Aggregate results
    aggregated_results = aggregate_chunk_results(chunk_events, args.top_k)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    real_time_factor = elapsed_seconds / audio_duration if audio_duration > 0 else 0

    # Performance summary
    perf_icon = "✓" if real_time_factor < 1.0 else "⚠"
    perf_style = "green" if real_time_factor < 1.0 else "yellow"
    
    perf_table = Table(title="⚡ Performance Metrics", show_header=True)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")
    perf_table.add_column("Status", style=perf_style)
    perf_table.add_row(
        "Processing Time",
        f"{elapsed_seconds:.3f}s",
        ""
    )
    perf_table.add_row(
        "Audio Duration",
        f"{audio_duration:.3f}s",
        ""
    )
    perf_table.add_row(
        "Real-Time Factor",
        f"{real_time_factor:.3f}x",
        f"{perf_icon} {'Real-time' if real_time_factor < 1.0 else 'Slower than real-time'}"
    )
    perf_table.add_row(
        "Processing Speed",
        f"{audio_duration/elapsed_seconds:.1f}x",
        ""
    )
    perf_table.add_row(
        "Chunks Processed",
        str(len(set(e["chunk_index"] for e in chunk_events))),
        ""
    )
    
    console.print(perf_table)

    # Save and display results
    save_results(
        results=aggregated_results,
        output_dir=output_dir,
        audio_path=args.audio_path,
        sample_rate=sample_rate,
        duration=audio_duration,
        elapsed_time=elapsed_seconds,
        use_small=args.small,
        top_k=args.top_k,
        chunk_count=len(set(e["chunk_index"] for e in chunk_events)),
    )

    console.print("[bold green]✓ Processing complete![/bold green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Processing interrupted by user[/yellow]")
        exit(130)
    except Exception as e:
        log.exception("An error occurred during audio tagging")
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        exit(1)
