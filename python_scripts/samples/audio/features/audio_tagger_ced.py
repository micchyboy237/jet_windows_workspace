"""
Audio tagging using sherpa-onnx CED (Conformer-based Event Detector) models.

Available CED models:
- mini:  sherpa-onnx-ced-mini-audio-tagging-2024-04-19  (~30 MB)
- small: sherpa-onnx-ced-small-audio-tagging-2024-04-19 (~60 MB)
- base:  sherpa-onnx-ced-base-audio-tagging-2024-04-19  (~120 MB)

Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models

Key difference from Zipformer:
  Zipformer uses:  model=AudioTaggingModelConfig(zipformer=OfflineZipformerAudioTaggingModelConfig(model=...))
  CED uses:        model=AudioTaggingModelConfig(ced=str(model_file))  ← direct string, no wrapper class
"""

import json
import time
import argparse
import shutil
from pathlib import Path
from typing import List, Literal, Any

import numpy as np
import sherpa_onnx
import soundfile as sf
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
log = logging.getLogger(__name__)

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()

# CED model registry — three size variants, all share the same release date tag
MODELS: dict[str, dict] = {
    "mini": {
        "name": "sherpa-onnx-ced-mini-audio-tagging-2024-04-19",
        "size": "~30 MB",
        "description": "Mini (fastest, lightest)",
        "expected_frames": 80,
    },
    "small": {
        "name": "sherpa-onnx-ced-small-audio-tagging-2024-04-19",
        "size": "~60 MB",
        "description": "Small (balanced speed/accuracy)",
        "expected_frames": 80,
    },
    "base": {
        "name": "sherpa-onnx-ced-base-audio-tagging-2024-04-19",
        "size": "~120 MB",
        "description": "Base (most accurate)",
        "expected_frames": 80,
    },
}

ModelSize = Literal["mini", "small", "base"]

SAMPLE_RATE = 16_000   # CED models expect 16 kHz
HOP_LENGTH  = 160      # 10 ms per frame at 16 kHz


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_model_paths(model_size: ModelSize = "base") -> dict:
    """Return a dict of resolved paths for the given CED model size."""
    info = MODELS[model_size]
    model_dir = BASE_DIR / info["name"]
    return {
        "model":      model_dir / "model.onnx",
        "model_int8": model_dir / "model.int8.onnx",
        "labels":     model_dir / "class_labels_indices.csv",
        "test_wavs_dir": model_dir / "test_wavs",
        "model_info": info,
    }


def find_model_file(paths: dict) -> str:
    """Return the path to whichever model file exists, preferring int8."""
    if paths["model_int8"].is_file():
        log.info("Using int8 quantised model")
        return str(paths["model_int8"])
    if paths["model"].is_file():
        log.info("Using standard (fp32) model")
        return str(paths["model"])
    raise FileNotFoundError(
        "No CED model file found. Checked:\n"
        f"  • {paths['model_int8']}\n"
        f"  • {paths['model']}\n"
        "Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
    )


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def read_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Read a wave file; return (mono float32 samples, sample_rate)."""
    p = Path(audio_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\n"
            "Please check the path and try again."
        )
    data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    samples = np.ascontiguousarray(data[:, 0])   # take first channel
    log.debug(f"Read {len(samples):,} samples at {sr} Hz from [cyan]{audio_path}[/cyan]")
    return samples, sr


def resample_if_needed(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Linear-interpolation resample when sample rates differ."""
    if orig_sr == target_sr:
        return samples
    log.info(f"Resampling {orig_sr} Hz → {target_sr} Hz")
    target_len = int(len(samples) / orig_sr * target_sr)
    idx = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Tagger construction — CED-specific config
# ---------------------------------------------------------------------------

def create_audio_tagger(
    model_size: ModelSize = "base",
    top_k: int = 5,
) -> sherpa_onnx.AudioTagging:
    """
    Build a CED AudioTagging instance.

    CED config difference vs Zipformer:
        Zipformer:  AudioTaggingModelConfig(zipformer=OfflineZipformerAudioTaggingModelConfig(model=path))
        CED:        AudioTaggingModelConfig(ced=path_string)   ← direct string, no wrapper
    """
    paths = get_model_paths(model_size)
    model_file  = find_model_file(paths)
    label_file  = paths["labels"]

    if not Path(label_file).is_file():
        raise FileNotFoundError(
            f"Labels file not found: {label_file}\n"
            "Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    # ── CED uses a plain string for the `ced` field ──────────────────────────
    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            ced=model_file,       # <-- CED-specific: direct path string
            num_threads=1,
            debug=True,
            provider="cpu",
        ),
        labels=str(label_file),
        top_k=top_k,
    )

    if not config.validate():
        raise ValueError(f"Invalid AudioTaggingConfig: {config}")

    info = paths["model_info"]
    cfg_table = Table(title="🎯 CED Audio Tagger Configuration", show_header=True)
    cfg_table.add_column("Parameter", style="cyan")
    cfg_table.add_column("Value",     style="green")
    cfg_table.add_row("Model Variant",  info["description"])
    cfg_table.add_row("Model Size",     info["size"])
    cfg_table.add_row("Model Path",     model_file)
    cfg_table.add_row("Labels Path",    str(label_file))
    cfg_table.add_row("Top K",          str(top_k))
    cfg_table.add_row("Provider",       "cpu")
    cfg_table.add_row("Threads",        "1")
    console.print(cfg_table)

    return sherpa_onnx.AudioTagging(config)


# ---------------------------------------------------------------------------
# Chunked inference
# ---------------------------------------------------------------------------

def process_audio_chunks(
    audio_tagger: sherpa_onnx.AudioTagging,
    samples: np.ndarray,
    sample_rate: int,
    expected_frames: int = 80,
    hop_length: int = HOP_LENGTH,
) -> List[dict]:
    """
    Slide a window over the audio and collect per-chunk tagging events.

    Window size = expected_frames * hop_length samples (0.8 s at 16 kHz).
    Step        = window_size // 2 (50 % overlap).
    Short clips are zero-padded to at least one full window.
    """
    window_samples = expected_frames * hop_length     # e.g. 80 * 160 = 12 800
    hop_samples    = window_samples // 2              # 50 % overlap
    total_samples  = len(samples)

    log.info(
        f"Audio: {total_samples:,} samples ({total_samples/sample_rate:.2f}s) | "
        f"Window: {window_samples} samples ({window_samples/sample_rate:.2f}s)"
    )

    # Pad short audio so at least one full window is available
    if total_samples < window_samples:
        log.info(f"Padding short audio ({total_samples} → {window_samples} samples)")
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:total_samples] = samples
        samples       = padded
        total_samples = window_samples

    num_chunks = max(1, (total_samples - window_samples) // hop_samples + 1)
    log.info(f"Processing {num_chunks} overlapping chunk(s)…")

    all_events: List[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Tagging chunks…", total=num_chunks)

        for i in range(num_chunks):
            start = i * hop_samples
            end   = min(start + window_samples, total_samples)

            chunk             = np.zeros(window_samples, dtype=np.float32)
            chunk[:end-start] = samples[start:end]

            stream = audio_tagger.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
            result = audio_tagger.compute(stream)

            log.debug(f"Chunk {i}: [{start/sample_rate:.2f}s – {end/sample_rate:.2f}s]  events={len(result)}")

            for event in result:
                all_events.append({
                    "name":        getattr(event, "name",  None),
                    "index":       getattr(event, "index", None),
                    "prob":        getattr(event, "prob",  None),
                    "chunk_start": start / sample_rate,
                    "chunk_end":   end   / sample_rate,
                    "chunk_index": i,
                })

            progress.update(task, advance=1)

    return all_events


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_chunk_results(
    chunk_events: List[dict],
    top_k: int = 5,
) -> List[dict]:
    """
    Merge per-chunk events by averaging probabilities across chunks.
    Returns the top-k entries sorted by mean probability (descending).
    """
    if not chunk_events:
        return []

    groups: dict[tuple, dict] = {}
    for ev in chunk_events:
        key = (ev["name"], ev["index"])
        if key not in groups:
            groups[key] = {"name": ev["name"], "index": ev["index"], "probs": [], "occurrences": 0}
        groups[key]["probs"].append(ev["prob"])
        groups[key]["occurrences"] += 1

    aggregated = [
        {
            "name":        g["name"],
            "index":       g["index"],
            "prob":        float(np.mean(g["probs"])),
            "max_prob":    float(np.max(g["probs"])),
            "occurrences": g["occurrences"],
        }
        for g in groups.values()
    ]
    aggregated.sort(key=lambda x: x["prob"], reverse=True)
    return aggregated[:top_k]


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def save_results(
    results:      List[dict],
    output_dir:   Path,
    audio_path:   str,
    sample_rate:  int,
    duration:     float,
    elapsed_time: float,
    model_size:   ModelSize,
    top_k:        int,
    chunk_count:  int,
) -> None:
    """Write results.json + metadata.json and print Rich summary tables."""

    # ── Results table ─────────────────────────────────────────────────────────
    tbl = Table(title="🎵 CED Audio Tagging Results (Aggregated)", header_style="bold magenta")
    tbl.add_column("Rank",        style="cyan",        width=6)
    tbl.add_column("Label",       style="green")
    tbl.add_column("Index",       style="yellow",      width=8)
    tbl.add_column("Avg Prob",    style="bold white",  justify="right")
    tbl.add_column("Max Prob",    style="dim white",   justify="right")
    tbl.add_column("Occurrences", style="blue",        justify="right")
    tbl.add_column("Bar",         style="magenta")

    for i, item in enumerate(results):
        bar_len = int(item["prob"] * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        tbl.add_row(
            str(i + 1),
            item["name"] or "N/A",
            str(item["index"]) if item["index"] is not None else "N/A",
            f"{item['prob']*100:.2f}%",
            f"{item['max_prob']*100:.2f}%",
            f"{item['occurrences']}/{chunk_count}",
            bar,
        )
    console.print(tbl)

    # ── Metadata table ────────────────────────────────────────────────────────
    rtf = elapsed_time / duration if duration > 0 else 0
    meta_tbl = Table(title="📊 Processing Metadata")
    meta_tbl.add_column("Metric", style="cyan")
    meta_tbl.add_column("Value",  style="green")
    meta_tbl.add_row("Audio Duration",   f"{duration:.3f}s")
    meta_tbl.add_row("Processing Time",  f"{elapsed_time:.3f}s")
    meta_tbl.add_row("Real-Time Factor", f"{rtf:.3f}x" if duration > 0 else "N/A")
    meta_tbl.add_row("Sample Rate",      f"{sample_rate} Hz")
    meta_tbl.add_row("Model Size",       model_size)
    meta_tbl.add_row("Chunks Processed", str(chunk_count))
    console.print(meta_tbl)

    # ── Persist JSON ──────────────────────────────────────────────────────────
    results_json = output_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    metadata = {
        "audio_file":              str(audio_path),
        "sample_rate":             sample_rate,
        "audio_duration_seconds":  round(duration, 3),
        "processing_time_seconds": round(elapsed_time, 3),
        "real_time_factor":        round(rtf, 3),
        "model_type":              "ced",
        "model_size":              model_size,
        "model_info":              MODELS[model_size],
        "top_k":                   top_k,
        "chunks_processed":        chunk_count,
        "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregation_method":      "average_probability_with_max",
    }
    metadata_json = output_dir / "metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    console.print(Panel(
        f"[cyan]Results:[/cyan]  {results_json}\n"
        f"[cyan]Metadata:[/cyan] {metadata_json}",
        title="💾 Saved Files",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Audio Tagging with CED - tag audio files using sherpa-onnx CED models",
        epilog=(
            "Examples:\n"
            "  %(prog)s audio.wav\n"
            "  %(prog)s audio.wav --model-size mini -k 10\n"
            "  %(prog)s audio.wav --model-size base -o ./results\n"
            "\n"
            "Available Models:\n"
            + "\n".join(
                f"  • {k:6s}: {v['name']} ({v['size']})"
                for k, v in MODELS.items()
            )
            + "\n\nDownload: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        ),
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        type=str,
        help="Path to input .wav file (omit to use the built-in test wav)",
    )
    parser.add_argument(
        "-m", "--model-size",
        choices=["mini", "small", "base"],
        default="base",
        dest="model_size",
        help="CED model size to use (default: base)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of top predictions to return (default: 5)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        dest="output_dir",
        help=f"Directory for output files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Resolve default test audio when none supplied
    if args.audio_path is None:
        paths = get_model_paths(args.model_size)
        default_wav = paths["test_wavs_dir"] / "6.wav"
        if default_wav.is_file():
            args.audio_path = str(default_wav)
            log.info(f"No audio path given — using default test file: [cyan]{default_wav}[/cyan]")
        else:
            console.print(
                "[red]No audio file provided and default test wav not found.[/red]\n"
                f"Expected: {default_wav}\n"
                "Download test files from the GitHub releases page."
            )
            raise SystemExit(1)

    # Prepare output directory (clean previous run)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        log.info(f"Cleaning output directory: [cyan]{output_dir}[/cyan]")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    info = MODELS[args.model_size]
    console.print(Panel.fit(
        f"[bold yellow]🎵 CED Audio Tagging Tool[/bold yellow]\n"
        f"[dim]Model: {args.model_size} ({info['size']}) | Top-K: {args.top_k}[/dim]\n"
        f"[dim]Window: {info['expected_frames']} frames (0.8 s) with 50 % overlap[/dim]",
        border_style="blue",
    ))

    # Build tagger
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"[cyan]Initialising CED-{args.model_size} tagger…", total=None)
        audio_tagger = create_audio_tagger(model_size=args.model_size, top_k=args.top_k)
        p.update(task, completed=True, description="[green]✓ Tagger ready")

    # Load audio
    log.info(f"Reading: [cyan]{args.audio_path}[/cyan]")
    samples, orig_sr = read_audio(args.audio_path)
    samples           = resample_if_needed(samples, orig_sr, SAMPLE_RATE)
    sample_rate       = SAMPLE_RATE
    audio_duration    = len(samples) / sample_rate
    log.info(
        f"Audio loaded: [cyan]{len(samples):,}[/cyan] samples | "
        f"[cyan]{sample_rate} Hz[/cyan] | [cyan]{audio_duration:.2f}s[/cyan]"
    )

    # Inference
    start_time   = time.time()
    chunk_events = process_audio_chunks(
        audio_tagger=audio_tagger,
        samples=samples,
        sample_rate=sample_rate,
        expected_frames=info["expected_frames"],
    )
    aggregated   = aggregate_chunk_results(chunk_events, args.top_k)
    elapsed      = time.time() - start_time
    chunk_count  = len({e["chunk_index"] for e in chunk_events})

    # Performance summary
    rtf       = elapsed / audio_duration if audio_duration > 0 else 0
    rtf_style = "green" if rtf < 1.0 else "yellow"
    rtf_icon  = "✓" if rtf < 1.0 else "⚠"

    perf = Table(title="⚡ Performance Metrics")
    perf.add_column("Metric",    style="cyan")
    perf.add_column("Value",     style="green")
    perf.add_column("Status",    style=rtf_style)
    perf.add_row("Processing Time",  f"{elapsed:.3f}s",                 "")
    perf.add_row("Audio Duration",   f"{audio_duration:.3f}s",          "")
    perf.add_row("Real-Time Factor", f"{rtf:.3f}x",
                 f"{rtf_icon} {'Real-time' if rtf < 1.0 else 'Slower than real-time'}")
    perf.add_row("Processing Speed", f"{audio_duration/elapsed:.1f}x",  "")
    perf.add_row("Chunks Processed", str(chunk_count),                  "")
    console.print(perf)

    # Save & display
    save_results(
        results=aggregated,
        output_dir=output_dir,
        audio_path=args.audio_path,
        sample_rate=sample_rate,
        duration=audio_duration,
        elapsed_time=elapsed,
        model_size=args.model_size,
        top_k=args.top_k,
        chunk_count=chunk_count,
    )
    console.print("[bold green]✓ Done![/bold green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise SystemExit(130)
    except Exception as exc:
        log.exception("Fatal error during audio tagging")
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
