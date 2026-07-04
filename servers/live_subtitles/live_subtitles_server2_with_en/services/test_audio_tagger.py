import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import soundfile as sf
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
from audio_config import SAMPLE_RATE
from serialization_utils import serialize
from dtype_conversion import convert_audio_dtype
from main._main_audio_tagger import get_args, _format_predictions_with_emphasis, _get_probability_bar

install_rich_traceback(show_locals=True)
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

audio_path = r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav"
# audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_002\sound.wav"

speech_threshold = 0.1
sample_rate = 16000

args = get_args(audio_path, OUTPUT_DIR)
audio_path = args.audio_path
tagger = AudioTagger(
    speech_prob_threshold=speech_threshold,
)

console.print(f"\n[bold]Analyzing audio: {linkify(audio_path)}[/bold]\n")

output_dir_param = Path(args.output_dir)
shutil.rmtree(output_dir_param, ignore_errors=True)
output_dir_param.mkdir(parents=True, exist_ok=True)

# Load original audio once for duration calculations
original_audio, original_sr = sf.read(audio_path)
original_duration = len(original_audio) / original_sr

audio_np = convert_audio_dtype(original_audio, "int16")

# Step 0: Run chunk-level tagging
chunk_summary = tagger.tag_audio_chunks(
    audio_np,
)

# ---------------------------------------------------------------------------
# Step 1: Full segment analysis (kept for detailed reporting)
# ---------------------------------------------------------------------------
segments_result = tagger.tag_audio_segments(
    audio_np,
    include_non_speech=False,
    speech_threshold=speech_threshold,
)

# ---------------------------------------------------------------------------
# Step 2: Extract high-confidence speech segments
# ---------------------------------------------------------------------------
high_speech_segments, high_speech_audios = tagger.extract_high_confidence_speech_segments(
    audio_np,
    sample_rate=sample_rate,
    speech_threshold=speech_threshold,
)

# ---------------------------------------------------------------------------
# Step 3: Extract speech-only audio (continuous trimmed audio)
# ---------------------------------------------------------------------------
console.print("\n[bold]Extracting speech-only audio...[/bold]")

# 3a: Standard extraction (removes all non-speech gaps)
speech_only_audio = tagger.extract_speech_only(
    audio_np,
    sample_rate=sample_rate,
    speech_threshold=speech_threshold,
    edges_only=False,
)
speech_only_duration = len(speech_only_audio) / sample_rate if len(speech_only_audio) > 0 else 0.0

# 3b: Edges-only extraction (trims only leading/trailing silence)
edges_only_audio = tagger.extract_speech_only(
    audio_np,
    sample_rate=sample_rate,
    speech_threshold=speech_threshold,
    edges_only=True,
)
edges_only_duration = len(edges_only_audio) / sample_rate if len(edges_only_audio) > 0 else 0.0

# ---------------------------------------------------------------------------
# Step 4: Save all results
# ---------------------------------------------------------------------------

# Save chunk summary
chunk_summary_output = output_dir_param / "chunk_summary.json"
with open(chunk_summary_output, "w", encoding="utf-8") as f:
    json.dump(chunk_summary, f, indent=2, ensure_ascii=False)
console.print(
    f"[green]Chunks summary saved to: {linkify(str(chunk_summary_output))}[/green]"
)

# Save complete segments result
segments_result_output = output_dir_param / "segments_result.json"
with open(segments_result_output, "w", encoding="utf-8") as f:
    json.dump(segments_result, f, indent=2, ensure_ascii=False)
console.print(
    f"[green]Segment summary saved to: {linkify(str(segments_result_output))}[/green]"
)

# Save filtered high-confidence segments metadata
high_speech_segments_output = output_dir_param / "high_speech_segments.json"
with open(high_speech_segments_output, "w", encoding="utf-8") as f:
    json.dump(high_speech_segments, f, indent=2, ensure_ascii=False)
console.print(
    f"[green]Filtered speech segments saved to: {linkify(str(high_speech_segments_output))}[/green]"
)

# ---------------------------------------------------------------------------
# Step 5: Save speech-only extracted audio and metadata
# ---------------------------------------------------------------------------
speech_only_dir = output_dir_param / "speech_only"
speech_only_dir.mkdir(parents=True, exist_ok=True)

# Save speech-only audio (all gaps removed)
if len(speech_only_audio) > 0:
    speech_only_wav = speech_only_dir / "speech_only.wav"
    sf.write(str(speech_only_wav), speech_only_audio, sample_rate)
    
    # Calculate audio statistics
    speech_only_peak = float(np.max(np.abs(speech_only_audio)))
    speech_only_rms = float(np.sqrt(np.mean(speech_only_audio.astype(np.float64) ** 2)))
    
    speech_only_meta = {
        "extraction_type": "speech_only",
        "description": "All non-speech segments removed, speech portions concatenated",
        "original_audio": {
            "path": str(audio_path),
            "duration_seconds": round(original_duration, 3),
            "sample_rate": original_sr,
            "total_samples": len(original_audio),
        },
        "extracted_audio": {
            "duration_seconds": round(speech_only_duration, 3),
            "sample_rate": sample_rate,
            "total_samples": len(speech_only_audio),
            "peak_amplitude": round(speech_only_peak, 4),
            "rms_amplitude": round(speech_only_rms, 4),
        },
        "reduction": {
            "duration_removed_seconds": round(original_duration - speech_only_duration, 3),
            "reduction_percentage": round((1 - speech_only_duration / original_duration) * 100, 1) if original_duration > 0 else 0.0,
        },
        "parameters": {
            "speech_threshold": speech_threshold,
            "edges_only": False,
        },
    }
    
    speech_only_meta_path = speech_only_dir / "speech_only_meta.json"
    with open(speech_only_meta_path, "w", encoding="utf-8") as f:
        json.dump(speech_only_meta, f, indent=2, ensure_ascii=False)
    
    console.print(
        f"[green]✓ Speech-only audio saved: {linkify(str(speech_only_wav))}[/green]"
    )
    console.print(
        f"[green]✓ Speech-only metadata saved: {linkify(str(speech_only_meta_path))}[/green]"
    )
    console.print(
        f"[dim]   Original: {original_duration:.2f}s → "
        f"Speech-only: {speech_only_duration:.2f}s "
        f"(-{speech_only_meta['reduction']['reduction_percentage']:.1f}%)[/dim]"
    )
else:
    console.print("[yellow]⚠ No speech detected for speech-only extraction[/yellow]")

# Save edges-only audio (only leading/trailing silence trimmed)
if len(edges_only_audio) > 0:
    edges_only_wav = speech_only_dir / "edges_only.wav"
    sf.write(str(edges_only_wav), edges_only_audio, sample_rate)
    
    edges_only_peak = float(np.max(np.abs(edges_only_audio)))
    edges_only_rms = float(np.sqrt(np.mean(edges_only_audio.astype(np.float64) ** 2)))
    
    edges_only_meta = {
        "extraction_type": "edges_only",
        "description": "Only leading and trailing non-speech trimmed, internal gaps preserved",
        "original_audio": {
            "path": str(audio_path),
            "duration_seconds": round(original_duration, 3),
            "sample_rate": original_sr,
            "total_samples": len(original_audio),
        },
        "extracted_audio": {
            "duration_seconds": round(edges_only_duration, 3),
            "sample_rate": sample_rate,
            "total_samples": len(edges_only_audio),
            "peak_amplitude": round(edges_only_peak, 4),
            "rms_amplitude": round(edges_only_rms, 4),
        },
        "reduction": {
            "duration_removed_seconds": round(original_duration - edges_only_duration, 3),
            "reduction_percentage": round((1 - edges_only_duration / original_duration) * 100, 1) if original_duration > 0 else 0.0,
        },
        "parameters": {
            "speech_threshold": speech_threshold,
            "edges_only": True,
        },
    }
    
    edges_only_meta_path = speech_only_dir / "edges_only_meta.json"
    with open(edges_only_meta_path, "w", encoding="utf-8") as f:
        json.dump(edges_only_meta, f, indent=2, ensure_ascii=False)
    
    console.print(
        f"[green]✓ Edges-only audio saved: {linkify(str(edges_only_wav))}[/green]"
    )
    console.print(
        f"[green]✓ Edges-only metadata saved: {linkify(str(edges_only_meta_path))}[/green]"
    )
    console.print(
        f"[dim]   Original: {original_duration:.2f}s → "
        f"Edges-only: {edges_only_duration:.2f}s "
        f"(-{edges_only_meta['reduction']['reduction_percentage']:.1f}%)[/dim]"
    )
else:
    console.print("[yellow]⚠ No speech detected for edges-only extraction[/yellow]")

# ---------------------------------------------------------------------------
# Step 6: Save individual high-confidence segment audio files
# ---------------------------------------------------------------------------
if high_speech_segments:
    console.print("\n[bold]Extracting individual segments...[/bold]")

    segments_dir = output_dir_param / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    for i, (segment, segment_audio) in enumerate(zip(high_speech_segments, high_speech_audios)):
        segment_index = segment.get("segment_index", i + 1)
        segment_dir = segments_dir / f"segment_{segment_index:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)

        # Save audio
        wav_path = segment_dir / "sound.wav"
        sf.write(str(wav_path), segment_audio, sample_rate)
        console.log(f"[green]✓ Saved audio: {wav_path}[/green]")

        # Save metadata
        json_path = segment_dir / "segment.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segment, f, indent=2, ensure_ascii=False)
        console.log(f"[green]✓ Saved metadata: {json_path}[/green]")

    console.print(
        f"\n[bold green]✓ Extracted {len(high_speech_segments)} segments to: "
        f"{linkify(str(segments_dir))}[/bold green]"
    )
else:
    console.print("\n[yellow]⚠ No high-confidence speech segments found[/yellow]")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
console.print("\n")
summary_table = Table(title="Extraction Summary", border_style="blue")
summary_table.add_column("Method", style="cyan")
summary_table.add_column("Duration", style="green", justify="right")
summary_table.add_column("Reduction", style="yellow", justify="right")
summary_table.add_column("Segments", style="magenta", justify="right")

summary_table.add_row(
    "Original",
    f"{original_duration:.2f}s",
    "—",
    "—",
)
summary_table.add_row(
    "Speech-only (gaps removed)",
    f"{speech_only_duration:.2f}s",
    f"{((1 - speech_only_duration / original_duration) * 100):.1f}%" if original_duration > 0 else "0%",
    "1 (concatenated)",
)
summary_table.add_row(
    "Edges-only (trimmed)",
    f"{edges_only_duration:.2f}s",
    f"{((1 - edges_only_duration / original_duration) * 100):.1f}%" if original_duration > 0 else "0%",
    "1 (trimmed)",
)
summary_table.add_row(
    "High-confidence segments",
    f"{sum(s['duration'] for s in high_speech_segments):.2f}s" if high_speech_segments else "0.00s",
    f"{((1 - sum(s['duration'] for s in high_speech_segments) / original_duration) * 100):.1f}%" if high_speech_segments and original_duration > 0 else "0%",
    str(len(high_speech_segments)),
)
summary_table.add_row(
    "Chunk-based (from summary)",
    f"{chunk_summary.get('speech_duration', 0):.2f}s",
    f"{((1 - chunk_summary.get('speech_duration', 0) / original_duration) * 100):.1f}%" if original_duration > 0 else "0%",
    "—",
)

console.print(summary_table)
