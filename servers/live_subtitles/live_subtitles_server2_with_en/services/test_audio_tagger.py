import argparse
import json
import shutil
import time
from pathlib import Path
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
from main._main_audio_tagger import get_args, _format_predictions_with_emphasis, _get_probability_bar

install_rich_traceback(show_locals=True)
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

audio_path = r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav"
# audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_002\sound.wav"

speech_threshold = None

args = get_args(audio_path, OUTPUT_DIR)
audio_path = args.audio_path
tagger = AudioTagger(
    speech_prob_threshold=speech_threshold,
)

console.print(f"\n[bold]Analyzing audio: {linkify(audio_path)}[/bold]\n")

output_dir_param = Path(args.output_dir)
shutil.rmtree(output_dir_param, ignore_errors=True)
output_dir_param.mkdir(parents=True, exist_ok=True)

# Step 0: Run chunk-level tagging
chunk_summary = tagger.tag_audio_chunks(
    audio_path,
)

# ---------------------------------------------------------------------------
# Step 1: Full segment analysis (kept for detailed reporting)
# ---------------------------------------------------------------------------
segments_result = tagger.tag_audio_segments(
    audio_path,
    include_non_speech=False,
)
speech_segments = segments_result["speech_segments"]

# ---------------------------------------------------------------------------
# Step 2: Extract high-confidence speech segments using the new method
# ---------------------------------------------------------------------------
high_speech_segments, high_speech_audios = tagger.extract_high_confidence_speech_segments(
    audio_path,
)

# ---------------------------------------------------------------------------
# Step 3: Save all results
# ---------------------------------------------------------------------------

# Save chunk summary
chunk_summary_output = output_dir_param / "chunk_summary.json"
with open(chunk_summary_output, "w", encoding="utf-8") as f:
    json.dump(chunk_summary, f, indent=2, ensure_ascii=False)
console.print(
    f"[green]Chunks summary saved to: {linkify(str(chunk_summary_output))}[/green]"
)

# Save all speech segments (full detail)
speech_segments_output = output_dir_param / "speech_segments.json"
with open(speech_segments_output, "w", encoding="utf-8") as f:
    json.dump(speech_segments, f, indent=2, ensure_ascii=False)
console.print(
    f"[green]Segment summary saved to: {linkify(str(speech_segments_output))}[/green]"
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
# Step 4: Save individual segment audio files
# ---------------------------------------------------------------------------
if high_speech_segments:
    console.print("\n[bold]Extracting individual segments...[/bold]")

    # Load audio once for sample rate (already extracted by the new method,
    # but we need sr for writing WAV files)
    audio_data, sr = sf.read(audio_path)

    segments_dir = output_dir_param / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    for i, (segment, segment_audio) in enumerate(zip(high_speech_segments, high_speech_audios)):
        segment_index = segment.get("segment_index", i + 1)
        segment_dir = segments_dir / f"segment_{segment_index:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)

        # Save audio
        wav_path = segment_dir / "sound.wav"
        sf.write(str(wav_path), segment_audio, sr)
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