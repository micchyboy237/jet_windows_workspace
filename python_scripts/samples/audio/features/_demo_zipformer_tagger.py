"""
_demo_zipformer_tagger.py
========================
Comprehensive demo showcasing all features of the Zipformer audio tagger.

Features demonstrated:
  1. Basic file tagging with default settings
  2. Custom top_k and variant selection
  3. Processing pre-loaded audio samples (process_audio API)
  4. Speech segment tagging with UTC timestamps
  5. Performance comparison across model variants
  6. Programmatic access to results
  7. Error handling and edge cases

Requirements:
  - sherpa-onnx with audio tagging support
  - Zipformer models downloaded to ~/.cache/pretrained_models/sherpa-onnx/
  - soundfile, numpy, rich

Usage:
  python demo_zipformer_tagger.py
  python demo_zipformer_tagger.py --audio path/to/audio.wav
  python demo_zipformer_tagger.py --audio path/to/audio.wav --skip-performance
"""
from __future__ import annotations
import argparse
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from audio_tagger_core import (
    SAMPLE_RATE,
    TaggingEvent,
    TaggingResult,
    console,
    log,
)
from audio_tagger_utils import (
    read_audio,
    resample_if_needed,
    save_results,
    aggregate_chunk_results,
    process_audio_chunks,
)
from audio_tagger_zipformer import (
    ZIPFORMER_MODELS,
    ZipformerAudioTagger,
)
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.layout import Layout


# ──────────────────────────────────────────────────────────────────────
# Demo Configuration
# ──────────────────────────────────────────────────────────────────────
DEMO_OUTPUT_BASE = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(DEMO_OUTPUT_BASE, ignore_errors=True)


def print_header(title: str) -> None:
    """Print a styled demo section header."""
    console.print()
    console.print(Rule(Text(title, style="bold yellow")))
    console.print()


def print_subsection(title: str) -> None:
    """Print a styled sub-section header."""
    console.print(f"\n[bold cyan]▸ {title}[/bold cyan]")


def check_model_availability(variant: str = "small") -> bool:
    """
    Check if the specified Zipformer model is downloaded.
    
    Args:
        variant: Model variant to check ('small' or 'standard')
    
    Returns:
        True if model files exist, False otherwise
    """
    from audio_tagger_core import BASE_DIR
    
    model_info = ZIPFORMER_MODELS.get(variant)
    if not model_info:
        console.print(f"[red]Unknown variant: {variant}[/red]")
        return False
    
    model_dir = BASE_DIR / model_info["name"]
    model_file = model_dir / "model.onnx"
    model_int8 = model_dir / "model.int8.onnx"
    labels_file = model_dir / "class_labels_indices.csv"
    
    has_model = model_file.is_file() or model_int8.is_file()
    has_labels = labels_file.is_file()
    
    if not has_model:
        console.print(
            f"[yellow]⚠ Model not found for '{variant}' at:[/yellow]\n"
            f"  {model_dir}\n"
            f"[dim]Download from: "
            f"https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models[/dim]"
        )
        return False
    
    if not has_labels:
        console.print(f"[yellow]⚠ Labels file missing at: {labels_file}[/yellow]")
        return False
    
    return True


# ══════════════════════════════════════════════════════════════════════
# Demo 1: Basic File Tagging
# ══════════════════════════════════════════════════════════════════════

def demo_basic_tagging(audio_path: str) -> Optional[TaggingResult]:
    """
    Demonstrate basic file tagging with default settings.
    
    This is the simplest use case: tag an audio file with the default
    model (standard Zipformer) and get the top 5 predictions.
    
    Args:
        audio_path: Path to the audio file to tag
    
    Returns:
        TaggingResult if successful, None if failed
    """
    print_header("Demo 1: Basic File Tagging (Default Settings)")
    
    console.print(
        "[dim]This demo shows the simplest workflow:[/dim]\n"
        "  1. Create a tagger with default variant ('standard')\n"
        "  2. Build the model\n"
        "  3. Tag an audio file\n"
        "  4. Results are automatically saved + displayed\n"
    )
    
    output_dir = DEMO_OUTPUT_BASE / "01_basic_tagging"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print_subsection("Creating ZipformerAudioTagger (variant='standard', top_k=5)")
        tagger = ZipformerAudioTagger(variant="standard", top_k=5)
        
        print_subsection("Building model...")
        tagger.build()
        console.print("[green]✓ Model built successfully[/green]")
        
        print_subsection(f"Tagging: [cyan]{audio_path}[/cyan]")
        result = tagger.tag_file(audio_path, output_dir)
        
        console.print(f"\n[green]✓ Tagging complete![/green]")
        console.print(f"  Top prediction: [bold]{result.events[0].name}[/bold] "
                       f"({result.events[0].prob:.1%})")
        console.print(f"  Processing time: {result.elapsed_time:.3f}s")
        console.print(f"  Real-time factor: {result.real_time_factor:.3f}x")
        
        return result
        
    except FileNotFoundError as e:
        console.print(f"[red]✗ File not found: {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Error in basic tagging demo: {e}[/red]")
        log.exception("Basic tagging demo failed")
        return None


# ══════════════════════════════════════════════════════════════════════
# Demo 2: Custom Configuration
# ══════════════════════════════════════════════════════════════════════

def demo_custom_config(audio_path: str) -> Optional[TaggingResult]:
    """
    Demonstrate custom top_k and variant selection.
    
    Shows how to:
    - Use the smaller/faster 'small' variant
    - Request more predictions (top_k=10)
    - Use process_audio_chunks() directly for single-pass efficiency
    - Programmatically aggregate and display results
    - Manually save results with full 5-file output
    
    Args:
        audio_path: Path to the audio file to tag
    
    Returns:
        TaggingResult if successful, None if failed
    """
    print_header("Demo 2: Custom Configuration (Small Model, Top-10)")
    
    console.print(
        "[dim]This demo shows advanced configuration:[/dim]\n"
        "  1. Use the smaller/faster 'small' model variant\n"
        "  2. Request top 10 predictions instead of 5\n"
        "  3. Process audio in a single pass (efficient)\n"
        "  4. Display results programmatically\n"
        "  5. Save all 5 output files from the same chunk data\n"
    )
    
    output_dir = DEMO_OUTPUT_BASE / "02_custom_config"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ── Step 1: Build the tagger ──────────────────────────────
        print_subsection("Creating ZipformerAudioTagger (variant='small', top_k=10)")
        tagger = ZipformerAudioTagger(variant="small", top_k=10)
        tagger.build()
        console.print("[green]✓ Small model built successfully[/green]")
        
        # ── Step 2: Load audio manually ───────────────────────────
        print_subsection("Loading audio manually")
        samples, orig_sr = read_audio(audio_path)
        samples = resample_if_needed(samples, orig_sr, SAMPLE_RATE)
        audio_duration = len(samples) / SAMPLE_RATE
        console.print(
            f"  Samples: [cyan]{len(samples):,}[/cyan] | "
            f"Duration: [cyan]{audio_duration:.2f}s[/cyan]"
        )
        
        # ── Step 3: Process ONCE — get raw chunk events ───────────
        # This is the key change: process_audio_chunks() gives us the
        # raw per-chunk data. We'll use it for BOTH display AND saving,
        # avoiding the double-processing waste.
        print_subsection("Processing audio (single pass)")
        start_time = time.time()
        
        chunk_events = process_audio_chunks(
            audio_tagger=tagger._tagger,
            samples=samples,
            sample_rate=SAMPLE_RATE,
            expected_frames=tagger.EXPECTED_FRAMES,
        )
        
        elapsed = time.time() - start_time
        
        # ── Step 4: Aggregate from chunk events ───────────────────
        # aggregate_chunk_results() computes mean/max probabilities
        # from the raw chunk events — same logic as process_audio()
        aggregated = aggregate_chunk_results(chunk_events, tagger.top_k)
        chunk_count = len({e["chunk_index"] for e in chunk_events})
        
        console.print(
            f"\n[green]✓ Processing complete in {elapsed:.3f}s[/green]"
        )
        console.print(
            f"  Chunks: [cyan]{chunk_count}[/cyan] | "
            f"Raw events: [cyan]{len(chunk_events)}[/cyan] | "
            f"Unique labels: [cyan]{len(aggregated)}[/cyan]"
        )
        
        # ── Step 5: Display results programmatically ──────────────
        console.print(f"\n[bold]Top 10 predictions:[/bold]")
        for i, event in enumerate(aggregated):
            bar_len = int(event.prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            console.print(
                f"  {i+1:2d}. [green]{event.name[:50]:<50}[/green] "
                f"[bold]{event.prob:>6.1%}[/bold]  {bar}"
            )
        
        # Show additional stats
        console.print(f"\n[dim]Per-event details:[/dim]")
        detail_table = Table(show_header=True, header_style="bold cyan")
        detail_table.add_column("Rank", width=5)
        detail_table.add_column("Label", width=35)
        detail_table.add_column("Avg Prob", width=10)
        detail_table.add_column("Max Prob", width=10)
        detail_table.add_column("Chunks Seen", width=12)
        
        for i, event in enumerate(aggregated[:5]):
            detail_table.add_row(
                str(i + 1),
                (event.name or "Unknown")[:35],
                f"{event.prob:.2%}",
                f"{event.max_prob:.2%}",
                f"{event.occurrences}/{chunk_count}",
            )
        
        console.print(detail_table)
        
        # ── Step 6: Build TaggingResult and save ──────────────────
        # Now we create the result from our already-computed data
        # and pass the raw chunk_events so save_results() can produce
        # all 5 output files (including plots) without re-processing
        result = TaggingResult(
            audio_path=audio_path,
            sample_rate=SAMPLE_RATE,
            duration=audio_duration,
            elapsed_time=elapsed,
            events=aggregated,
            chunk_count=chunk_count,
            backend_name=tagger.BACKEND_NAME,
            model_variant=tagger.variant,
            top_k=tagger.top_k,
            is_speech_segment=False,
        )
        
        # Save with chunk_events → produces all 5 files
        save_results(result, output_dir, chunk_events=chunk_events)
        
        return result
        
    except FileNotFoundError as e:
        console.print(f"[red]✗ File not found: {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Error in custom config demo: {e}[/red]")
        log.exception("Custom config demo failed")
        return None


# ══════════════════════════════════════════════════════════════════════
# Demo 3: Speech Segment Tagging (FireRed VAD simulation)
# ══════════════════════════════════════════════════════════════════════

def demo_speech_segment(audio_path: str) -> Optional[TaggingResult]:
    """
    Demonstrate speech segment tagging with UTC timestamps.
    
    Simulates the FireRed VAD pipeline by:
    1. Loading a full audio file
    2. Splitting it into "speech segments"
    3. Tagging each segment with absolute UTC timestamps
    4. Showing how results preserve temporal context
    
    Args:
        audio_path: Path to the audio file to tag
    
    Returns:
        Last TaggingResult if successful, None if failed
    """
    print_header("Demo 3: Speech Segment Tagging (FireRed VAD Simulation)")
    
    console.print(
        "[dim]This demo simulates real-time speech segment tagging:[/dim]\n"
        "  1. Load audio and split into simulated 'speech segments'\n"
        "  2. Tag each segment with absolute UTC timestamps\n"
        "  3. Show how time_utc_start/time_utc_end are preserved\n"
        "  4. Demonstrate the tag_speech_segment() API\n"
    )
    
    output_dir = DEMO_OUTPUT_BASE / "03_speech_segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print_subsection("Loading audio and simulating FireRed VAD segments")
        samples, orig_sr = read_audio(audio_path)
        samples = resample_if_needed(samples, orig_sr, SAMPLE_RATE)
        total_duration = len(samples) / SAMPLE_RATE
        
        console.print(
            f"  Total audio: [cyan]{total_duration:.2f}s[/cyan] "
            f"([cyan]{len(samples):,}[/cyan] samples)"
        )
        
        # Simulate speech segments: split audio into 1-2 second chunks
        # with gaps between them (like real VAD would produce)
        segment_duration = 1.5  # seconds per simulated segment
        gap_duration = 0.3      # seconds gap between segments
        segment_samples = int(segment_duration * SAMPLE_RATE)
        gap_samples = int(gap_duration * SAMPLE_RATE)
        
        # Create a simulated UTC start time (now - audio duration)
        now_utc = datetime.now(timezone.utc)
        simulation_start = now_utc - timedelta(seconds=total_duration)
        
        print_subsection("Building tagger for segment processing")
        tagger = ZipformerAudioTagger(variant="small", top_k=5)
        tagger.build()
        console.print("[green]✓ Tagger ready for segments[/green]")
        
        # Process each simulated segment
        segment_results = []
        pos = 0
        segment_id = 0
        
        console.print(f"\n[bold]Processing simulated speech segments:[/bold]\n")
        
        # Summary table for segment results
        seg_table = Table(
            title="Speech Segment Tagging Results",
            header_style="bold cyan",
        )
        seg_table.add_column("Segment", style="cyan", width=8)
        seg_table.add_column("Time Range (UTC)", style="green", width=30)
        seg_table.add_column("Duration", style="yellow", width=10)
        seg_table.add_column("Top Prediction", style="bold white", width=30)
        seg_table.add_column("Prob", style="magenta", width=8)
        seg_table.add_column("RTF", style="blue", width=8)
        
        while pos + segment_samples <= len(samples):
            segment_audio = samples[pos:pos + segment_samples]
            
            # Calculate UTC timestamps for this segment
            seg_start_utc = simulation_start + timedelta(
                seconds=pos / SAMPLE_RATE
            )
            seg_end_utc = simulation_start + timedelta(
                seconds=(pos + segment_samples) / SAMPLE_RATE
            )
            
            console.print(
                f"  Segment {segment_id}: "
                f"[dim]{seg_start_utc.strftime('%H:%M:%S.%f')[:-3]} → "
                f"{seg_end_utc.strftime('%H:%M:%S.%f')[:-3]}[/dim]"
            )
            
            # Tag the segment — this is the key API for FireRed VAD integration
            result = tagger.tag_speech_segment(
                segment_audio=segment_audio,
                segment_start_utc=seg_start_utc,
                segment_end_utc=seg_end_utc,
                segment_id=segment_id,
            )
            
            # Show results with UTC timestamps preserved
            if result.events:
                top_event = result.events[0]
                utc_str = (
                    f"{top_event.time_utc_start.strftime('%H:%M:%S')} → "
                    f"{top_event.time_utc_end.strftime('%H:%M:%S')}"
                    if top_event.time_utc_start
                    else "N/A"
                )
                seg_table.add_row(
                    f"#{segment_id}",
                    utc_str,
                    f"{segment_duration:.1f}s",
                    (top_event.name or "Unknown")[:30],
                    f"{top_event.prob:.1%}",
                    f"{result.real_time_factor:.3f}x",
                )
            
            segment_results.append(result)
            
            # Move to next segment (with gap)
            pos += segment_samples + gap_samples
            segment_id += 1
            
            # Limit to 5 segments for demo
            if segment_id >= 5:
                break
        
        console.print(seg_table)
        
        # Save the last segment result as example
        if segment_results:
            last_result = segment_results[-1]
            seg_output_dir = output_dir / f"segment_{segment_id - 1}"
            seg_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Re-process to get chunk events for full output
            chunk_events = process_audio_chunks(
                audio_tagger=tagger._tagger,
                samples=samples[
                    pos - segment_samples - gap_samples:pos - gap_samples
                ],
                sample_rate=SAMPLE_RATE,
                expected_frames=tagger.EXPECTED_FRAMES,
                segment_start_utc=last_result.segment_start_utc,
            )
            save_results(last_result, seg_output_dir, chunk_events=chunk_events)
            
            console.print(
                f"\n[green]✓ Processed {len(segment_results)} segments[/green]"
            )
            console.print(
                f"[dim]Last segment output saved to: {seg_output_dir}[/dim]"
            )
            
            return last_result
        
        return None
        
    except Exception as e:
        console.print(f"[red]✗ Error in speech segment demo: {e}[/red]")
        log.exception("Speech segment demo failed")
        return None


# ══════════════════════════════════════════════════════════════════════
# Demo 4: Performance Comparison
# ══════════════════════════════════════════════════════════════════════

def demo_performance_comparison(audio_path: str) -> None:
    """
    Compare performance across both Zipformer model variants.
    
    Measures and displays:
    - Processing time for each variant
    - Real-time factor
    - Number of chunks processed
    - Top prediction agreement
    
    Args:
        audio_path: Path to the audio file to tag
    """
    print_header("Demo 4: Performance Comparison (Small vs Standard)")
    
    console.print(
        "[dim]This demo compares both Zipformer model variants:[/dim]\n"
        "  • small    — 106 MB, faster, slightly less accurate\n"
        "  • standard — 288 MB, slower, more accurate\n"
    )
    
    # Check availability
    if not check_model_availability("small"):
        console.print("[yellow]⚠ Small model not available — skipping comparison[/yellow]")
        return
    
    if not check_model_availability("standard"):
        console.print("[yellow]⚠ Standard model not available — skipping comparison[/yellow]")
        return
    
    # Performance comparison table
    perf_table = Table(
        title="⚡ Zipformer Model Variant Comparison",
        header_style="bold yellow",
    )
    perf_table.add_column("Metric", style="cyan", width=25)
    perf_table.add_column("Small (106 MB)", style="green", width=25)
    perf_table.add_column("Standard (288 MB)", style="blue", width=25)
    perf_table.add_column("Ratio", style="magenta", width=15)
    
    results = {}
    
    for variant in ["small", "standard"]:
        console.print(f"\n[bold]Testing {variant} variant...[/bold]")
        
        try:
            tagger = ZipformerAudioTagger(variant=variant, top_k=5)
            tagger.build()
            
            samples, orig_sr = read_audio(audio_path)
            samples = resample_if_needed(samples, orig_sr, SAMPLE_RATE)
            
            # Warm-up run (first inference is often slower)
            _ = tagger.process_audio(
                samples[:16000],  # 1 second
                sample_rate=SAMPLE_RATE,
                show_progress=False,
            )
            
            # Timed run
            start_time = time.time()
            events = tagger.process_audio(
                samples=samples,
                sample_rate=SAMPLE_RATE,
                show_progress=True,
            )
            elapsed = time.time() - start_time
            
            duration = len(samples) / SAMPLE_RATE
            rtf = elapsed / duration
            
            results[variant] = {
                "events": events,
                "elapsed": elapsed,
                "duration": duration,
                "rtf": rtf,
            }
            
            console.print(
                f"  [green]✓ {variant}:[/green] "
                f"{elapsed:.3f}s for {duration:.2f}s audio "
                f"(RTF: {rtf:.3f}x)"
            )
            
        except Exception as e:
            console.print(f"[red]✗ {variant} failed: {e}[/red]")
            log.exception(f"Performance test failed for {variant}")
    
    # Build comparison table
    if len(results) == 2:
        small = results["small"]
        standard = results["standard"]
        
        perf_table.add_row(
            "Processing Time",
            f"{small['elapsed']:.3f}s",
            f"{standard['elapsed']:.3f}s",
            f"{small['elapsed']/standard['elapsed']:.2f}x",
        )
        perf_table.add_row(
            "Real-Time Factor",
            f"{small['rtf']:.3f}x",
            f"{standard['rtf']:.3f}x",
            f"{small['rtf']/standard['rtf']:.3f}x",
        )
        perf_table.add_row(
            "Audio Duration",
            f"{small['duration']:.2f}s",
            f"{standard['duration']:.2f}s",
            "—",
        )
        perf_table.add_row(
            "Top Prediction",
            (small['events'][0].name or "N/A")[:25],
            (standard['events'][0].name or "N/A")[:25],
            "—",
        )
        perf_table.add_row(
            "Top-1 Confidence",
            f"{small['events'][0].prob:.1%}",
            f"{standard['events'][0].prob:.1%}",
            f"{standard['events'][0].prob/small['events'][0].prob:.3f}x",
        )
        
        # Agreement check
        small_top3 = {e.name for e in small['events'][:3]}
        standard_top3 = {e.name for e in standard['events'][:3]}
        agreement = len(small_top3 & standard_top3)
        perf_table.add_row(
            "Top-3 Agreement",
            f"{agreement}/3 labels",
            f"{agreement}/3 labels",
            f"{agreement/3:.0%}",
        )
        
        console.print(perf_table)
        
        # Recommendation
        console.print()
        if small['rtf'] < 0.1 and agreement >= 2:
            console.print(
                Panel(
                    "[green]💡 Recommendation:[/green] Use the [bold]small[/bold] variant "
                    "for real-time applications. It's "
                    f"[bold]{standard['elapsed']/small['elapsed']:.1f}x faster[/bold] "
                    "with comparable accuracy.",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    "[yellow]💡 Recommendation:[/yellow] Use the [bold]standard[/bold] variant "
                    "when accuracy is critical and latency is less important.",
                    border_style="yellow",
                )
            )
    else:
        console.print("[yellow]⚠ Could not complete comparison — one or both models failed[/yellow]")


# ══════════════════════════════════════════════════════════════════════
# Demo 5: Error Handling & Edge Cases
# ══════════════════════════════════════════════════════════════════════

def demo_error_handling() -> None:
    """
    Demonstrate graceful error handling for common edge cases.
    
    Shows:
    - Invalid variant name
    - Missing model files
    - Calling methods before build()
    - Non-existent audio files
    """
    print_header("Demo 5: Error Handling & Edge Cases")
    
    console.print(
        "[dim]This demo shows how the API handles errors gracefully:[/dim]\n"
    )
    
    # Test 1: Invalid variant
    print_subsection("Test 1: Invalid model variant")
    try:
        tagger = ZipformerAudioTagger(variant="nonexistent", top_k=5)
        console.print("[red]✗ Should have raised ValueError[/red]")
    except ValueError as e:
        console.print(f"[green]✓ Correctly raised ValueError:[/green] {e}")
    
    # Test 2: Calling build() with missing model
    print_subsection("Test 2: Missing model files (if model not downloaded)")
    try:
        # Use a path we know doesn't exist
        from audio_tagger_core import BASE_DIR
        test_dir = BASE_DIR / "nonexistent-model-2024"
        if not test_dir.exists():
            console.print(
                "[dim]Skipping — requires actual missing model directory[/dim]"
            )
        else:
            tagger = ZipformerAudioTagger(variant="small", top_k=5)
            # This would fail if the model is missing
            console.print("[dim]Model exists — skipping error test[/dim]")
    except Exception as e:
        console.print(f"[green]✓ Correctly raised error:[/green] {type(e).__name__}")
    
    # Test 3: Calling tag_file() before build()
    print_subsection("Test 3: Calling tag_file() before build()")
    tagger = ZipformerAudioTagger(variant="small", top_k=5)
    try:
        # No .build() called
        _ = tagger.tag_file("test.wav", Path("/tmp"))
        console.print("[red]✗ Should have raised RuntimeError[/red]")
    except RuntimeError as e:
        console.print(f"[green]✓ Correctly raised RuntimeError:[/green] {e}")
    
    # Test 4: Non-existent audio file
    print_subsection("Test 4: Non-existent audio file")
    tagger2 = ZipformerAudioTagger(variant="small", top_k=5)
    try:
        tagger2.build()
        _ = tagger2.tag_file("/nonexistent/path/audio.wav", Path("/tmp"))
        console.print("[red]✗ Should have raised FileNotFoundError[/red]")
    except RuntimeError:
        console.print("[yellow]⚠ Model not available — skipping test[/yellow]")
    except FileNotFoundError as e:
        console.print(f"[green]✓ Correctly raised FileNotFoundError:[/green] {e}")
    except Exception as e:
        console.print(f"[dim]Expected FileNotFoundError, got {type(e).__name__}: {e}[/dim]")
    
    # Test 5: Valid variant names
    print_subsection("Test 5: Valid variant enumeration")
    console.print(
        f"  Valid variants: [cyan]{', '.join(ZipformerAudioTagger.VALID_VARIANTS)}[/cyan]"
    )
    console.print(
        f"  Default variant: [cyan]{ZipformerAudioTagger.DEFAULT_VARIANT}[/cyan]"
    )
    
    console.print(f"\n[green]✓ Error handling demo complete[/green]")


# ══════════════════════════════════════════════════════════════════════
# Main Demo Runner
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full Zipformer audio tagger demo suite."""
    parser = argparse.ArgumentParser(
        description="Zipformer Audio Tagger — Comprehensive Feature Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                    # Use default test audio\n"
            "  %(prog)s --audio speech.wav                 # Use custom audio file\n"
            "  %(prog)s --audio speech.wav --skip-performance  # Skip model comparison\n"
        ),
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file for tagging demos (uses built-in test wav if omitted)",
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip the performance comparison demo (demo 4)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run only a specific demo (1-5)",
    )
    args = parser.parse_args()
    
    # ── Welcome Banner ──────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            Text(
                "🎵 Zipformer Audio Tagger — Feature Demo\n"
                "Comprehensive showcase of all tagger capabilities",
                justify="center",
            ),
            border_style="bold magenta",
            padding=(1, 2),
        )
    )
    
    # Show available models
    model_table = Table(title="Available Zipformer Models", header_style="bold cyan")
    model_table.add_column("Variant", style="green")
    model_table.add_column("Size", style="yellow")
    model_table.add_column("Description", style="white")
    model_table.add_column("Status", style="blue")
    
    for variant, info in ZIPFORMER_MODELS.items():
        available = check_model_availability(variant)
        status = "[green]✓ Downloaded[/green]" if available else "[red]✗ Not found[/red]"
        model_table.add_row(
            variant,
            info["size"],
            info["description"],
            status,
        )
    
    console.print(model_table)
    
    # ── Resolve Audio Path ──────────────────────────────────────────
    audio_path = args.audio
    if audio_path is None:
        # Try to find the default test wav
        try:
            tagger = ZipformerAudioTagger(variant="small")
            default_wav = tagger.default_test_wav
            if default_wav.is_file():
                audio_path = str(default_wav)
                console.print(
                    f"\n[dim]Using default test audio: [cyan]{default_wav}[/cyan][/dim]"
                )
            else:
                console.print(
                    f"\n[yellow]⚠ Default test wav not found at: {default_wav}[/yellow]"
                )
        except Exception:
            pass
    
    if audio_path is None:
        console.print(
            "\n[red]No audio file available. Provide one with --audio PATH[/red]\n"
            "[dim]Download test files from: "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models[/dim]"
        )
        
        # Still run error handling demo (doesn't need audio)
        if args.demo is None or args.demo == 5:
            demo_error_handling()
        
        console.print()
        return
    
    # Check if audio file exists
    if not Path(audio_path).is_file():
        console.print(f"\n[red]✗ Audio file not found: {audio_path}[/red]")
        
        # Still run error handling demo
        if args.demo is None or args.demo == 5:
            demo_error_handling()
        
        console.print()
        return
    
    console.print(f"\n[dim]Audio file: [cyan]{audio_path}[/cyan][/dim]")
    console.print(
        f"[dim]Output base: [cyan]{DEMO_OUTPUT_BASE}[/cyan][/dim]"
    )
    
    # ── Run Demos ───────────────────────────────────────────────────
    demos_to_run = [args.demo] if args.demo else [1, 2, 3, 4, 5]
    
    for demo_num in demos_to_run:
        if demo_num == 1:
            demo_basic_tagging(audio_path)
        elif demo_num == 2:
            demo_custom_config(audio_path)
        elif demo_num == 3:
            demo_speech_segment(audio_path)
        elif demo_num == 4:
            if args.skip_performance:
                console.print("\n[yellow]⏭ Skipping performance comparison (--skip-performance)[/yellow]")
            else:
                demo_performance_comparison(audio_path)
        elif demo_num == 5:
            demo_error_handling()
    
    # ── Final Summary ───────────────────────────────────────────────
    console.print()
    console.print(Rule(Text("Demo Complete", style="bold green")))
    console.print(
        Panel(
            f"[green]✓ All demos finished![/green]\n\n"
            f"[dim]Output files saved to:[/dim]\n"
            f"[cyan]{DEMO_OUTPUT_BASE}[/cyan]\n\n"
            f"[dim]Each demo creates its own subdirectory:[/dim]\n"
            f"  • 01_basic_tagging/    — Simple file tagging\n"
            f"  • 02_custom_config/    — Custom top_k and variant\n"
            f"  • 03_speech_segments/  — Simulated VAD segments\n"
            f"  • (performance results shown inline)\n\n"
            f"[dim]API Reference:[/dim]\n"
            f"  • tagger.tag_file()        — Full pipeline, saves 5 files\n"
            f"  • tagger.tag_speech_segment() — Real-time VAD integration\n"
            f"  • tagger.process_audio()   — Low-level API, returns events\n"
            f"  • tagger.build()           — Initialize the model\n",
            border_style="green",
            title="📋 Summary",
        )
    )
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Demo interrupted by user[/yellow]")
        raise SystemExit(130)
    except Exception as e:
        log.exception("Demo failed with unexpected error")
        console.print(f"\n[red]✗ Unexpected error: {e}[/red]")
        raise SystemExit(1)
