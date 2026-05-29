# audio_tagger_ced.py

"""
This script shows how to use audio tagging Python APIs to tag a file.

Please read the code to download the required model files and test wave file.
"""

import json
import time
import argparse
import shutil
from pathlib import Path
from typing import Literal, List, Dict, Any

import numpy as np
import sherpa_onnx
import soundfile as sf
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
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
DEFAULT_MODEL_SIZE: Literal["mini", "small", "base"] = "small"

def get_model_paths(model_size: str):
    """Get model and label paths based on model size."""
    model_name = f"sherpa-onnx-ced-{model_size}-audio-tagging-2024-04-19"
    return {
        "model": BASE_DIR / model_name / "model.int8.onnx",
        "labels": BASE_DIR / model_name / "class_labels_indices.csv",
        "test_wavs_dir": BASE_DIR / model_name / "test_wavs"
    }


def read_test_wave(test_wave: str):
    """Read test wave file and return samples and sample rate."""
    test_wave_path = Path(test_wave)
    
    if not test_wave_path.is_file():
        raise ValueError(
            f"Please download {test_wave} from "
            "[link=https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models]"
            "GitHub: audio-tagging-models[/link]"
        )

    # See https://python-soundfile.readthedocs.io/en/0.11.0/#soundfile.read
    data, sample_rate = sf.read(
        test_wave,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)

    return samples, sample_rate


def create_audio_tagger(model_size: str = "small", top_k: int = 5):
    """Create and configure the audio tagger."""
    paths = get_model_paths(model_size)
    model_file = paths["model"]
    label_file = paths["labels"]

    # Check model files exist
    missing_files = []
    if not Path(model_file).is_file():
        missing_files.append(("Model", model_file))
    if not Path(label_file).is_file():
        missing_files.append(("Labels", label_file))

    if missing_files:
        error_msg = "Missing required files:\n"
        for file_type, file_path in missing_files:
            error_msg += f"  • {file_type}: {file_path}\n"
        error_msg += (
            "\nPlease download from [link=https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models]"
            "GitHub: audio-tagging-models[/link]"
        )
        raise ValueError(error_msg)

    # NOTE: str() is required — sherpa_onnx is a C++ pybind11 extension that
    # does strict type checking and will not accept pathlib.Path objects.
    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            ced=str(model_file),
            num_threads=1,
            debug=True,
            provider="cpu",
        ),
        labels=str(label_file),
        top_k=top_k,
    )

    if not config.validate():
        raise ValueError(f"Please check the config: {config}")

    # Display config in a nice panel
    config_table = Table(title="Audio Tagger Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model Size", model_size)
    config_table.add_row("Model Path", str(model_file))
    config_table.add_row("Labels Path", str(label_file))
    config_table.add_row("Top K", str(top_k))
    config_table.add_row("Provider", "cpu")
    config_table.add_row("Debug", "True")
    config_table.add_row("Threads", "1")
    
    console.print(config_table)

    return sherpa_onnx.AudioTagging(config)


def save_results(
    result: List[Any],
    output_dir: Path,
    audio_path: str,
    sample_rate: int,
    duration: float,
    elapsed_time: float,
    model_size: str,
    top_k: int
):
    """Save results to JSON and create a summary display."""
    
    # Create structured result
    def to_dict(event, index):
        return {
            "index": index,
            "name": getattr(event, "name", None),
            "class_index": getattr(event, "index", None),
            "prob": getattr(event, "prob", None),
        }

    structured_result = []
    for i, e in enumerate(result):
        structured_result.append(to_dict(e, i))

    # Save to JSON
    results_json = output_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(structured_result, f, indent=2, ensure_ascii=False)
    
    # Display results table
    table = Table(
        title=f"🎵 Audio Tagging Results",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Label", style="green")
    table.add_column("Index", style="yellow", width=8)
    table.add_column("Probability", style="bold white", justify="right")
    
    for item in structured_result:
        prob_percent = f"{item['prob']*100:.2f}%" if item['prob'] else "N/A"
        table.add_row(
            str(item['index']),
            item['name'] or "N/A",
            str(item['class_index']) if item['class_index'] is not None else "N/A",
            prob_percent
        )
    
    console.print(table)
    
    # Save metadata
    metadata = {
        "audio_file": str(audio_path),
        "sample_rate": sample_rate,
        "audio_duration_seconds": round(duration, 3),
        "processing_time_seconds": round(elapsed_time, 3),
        "real_time_factor": round(elapsed_time / duration, 3) if duration > 0 else 0,
        "model_size": model_size,
        "top_k": top_k,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_json = output_dir / "metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Display metadata summary
    meta_table = Table(title="📊 Processing Metadata", show_header=True)
    meta_table.add_column("Metric", style="cyan")
    meta_table.add_column("Value", style="green")
    for key, value in metadata.items():
        meta_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(meta_table)
    
    # Show saved file paths
    saved_files = Panel(
        f"[cyan]Results JSON:[/cyan] [link=file://{results_json.absolute()}]{results_json}[/link]\n"
        f"[cyan]Metadata JSON:[/cyan] [link=file://{metadata_json.absolute()}]{metadata_json}[/link]",
        title="💾 Saved Files",
        border_style="green"
    )
    console.print(saved_files)


def main():
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="[bold cyan]CED Audio Tagging[/bold cyan] - Tag audio files using CED model",
        epilog=(
            "Examples:\n"
            "  %(prog)s audio.wav\n"
            "  %(prog)s audio.wav -s base -k 10\n"
            "  %(prog)s audio.wav -s mini -o ./my_results\n"
            "\n"
            "Model files: [link=https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models]"
            "GitHub Releases[/link]"
        )
    )
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        help="Path to input wave file (optional if using default)"
    )
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        choices=["mini", "small", "base"],
        default=DEFAULT_MODEL_SIZE,
        help=f"Model size to use (default: {DEFAULT_MODEL_SIZE})"
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
        paths = get_model_paths(args.model_size)
        args.audio_path = str(paths["test_wavs_dir"] / "6.wav")
        log.info(f"No audio path provided, using default: [cyan]{args.audio_path}[/cyan]")

    # Setup output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        log.info(f"Cleaning output directory: [cyan]{output_dir}[/cyan]")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Welcome banner
    console.print(Panel.fit(
        "[bold yellow]🎵 CED Audio Tagging Tool[/bold yellow]\n"
        f"[dim]Model: {args.model_size} | Top-K: {args.top_k}[/dim]",
        border_style="blue"
    ))

    # Create audio tagger
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Initializing audio tagger...", total=None)
        audio_tagger = create_audio_tagger(model_size=args.model_size, top_k=args.top_k)
        progress.update(task, completed=True, description="[green]✓ Audio tagger initialized")

    # Read audio file
    log.info(f"Reading audio file: [link=file://{Path(args.audio_path).absolute()}]{args.audio_path}[/link]")
    samples, sample_rate = read_test_wave(args.audio_path)
    
    audio_duration = len(samples) / sample_rate
    log.info(f"Audio loaded: [cyan]{len(samples)}[/cyan] samples, [cyan]{sample_rate}Hz[/cyan], [cyan]{audio_duration:.2f}s[/cyan]")

    # Process audio
    log.info("Processing audio...")
    start_time = time.time()

    stream = audio_tagger.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    result = audio_tagger.compute(stream)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    real_time_factor = elapsed_seconds / audio_duration if audio_duration > 0 else 0

    # Performance summary
    perf_table = Table(title="⚡ Performance Metrics", show_header=True)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")
    perf_table.add_column("Details", style="dim")
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
        "✓ Real-time" if real_time_factor < 1.0 else "⚠ Slower than real-time"
    )
    
    console.print(perf_table)

    # Save and display results
    save_results(
        result=result,
        output_dir=output_dir,
        audio_path=args.audio_path,
        sample_rate=sample_rate,
        duration=audio_duration,
        elapsed_time=elapsed_seconds,
        model_size=args.model_size,
        top_k=args.top_k
    )

    console.print("[bold green]✓ Processing complete![/bold green]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("An error occurred during audio tagging")
        console.print(f"[bold red]Error:[/bold red] {e}")
        exit(1)
