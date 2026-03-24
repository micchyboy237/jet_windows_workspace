# segment_speaker_labeler.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Union

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — important for scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from fireredvad.vad import FireRedVad, FireRedVadConfig
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

AudioInput = Union[np.ndarray, bytes, bytearray, str, Path]

MODEL_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/VAD").expanduser().resolve()
)

console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SpeechSegment(TypedDict):
    start_sec: float
    end_sec: float
    duration_sec: float
    sample_rate: int
    segment_index: int
    output_path: str
    probs_info: dict


# Unused for now — kept for future speaker diarization step
class SegmentResult(TypedDict):
    path: str
    parent_dir: str
    speaker_label: int
    centroid_cosine_similarity: float
    nearest_neighbor_cosine_similarity: float


def frames_from_seconds(sec: float) -> int:
    """Consistent frame index calculation (10 ms frames)"""
    return int(round(sec * 100.0))  # 1 / 0.010 = 100


def generate_plot(
    probs: np.ndarray,
    segment_idx: int,
    duration_sec: float,
    output_path: Path,
    is_dummy: bool = False,
) -> None:
    num_frames = len(probs)
    if num_frames == 0:
        return

    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=140)

    label = "Speech probability (dummy)" if is_dummy else "Speech probability"
    color = "#ff7f0e" if is_dummy else "#2ca02c"

    ax.plot(probs, color=color, linewidth=1.8, label=label)
    ax.fill_between(range(num_frames), probs, color=color, alpha=0.14)

    ax.axhline(
        y=0.4,
        linestyle="--",
        color="#d62728",
        alpha=0.65,
        linewidth=1.2,
        label="threshold ≈ 0.4",
    )

    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, num_frames - 1)

    ax.set_ylabel("Speech Probability", fontsize=10.5)
    ax.set_xlabel(
        f"Frame (10 ms)  —  {num_frames} frames ≈ {duration_sec:.1f} s", fontsize=10.5
    )
    title = (
        f"Segment {segment_idx:03d} — {'Dummy ' if is_dummy else ''}Model Probabilities"
    )
    ax.set_title(title, fontsize=12, pad=12)

    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    fig.tight_layout(pad=0.9)
    plt.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def extract_speech_segments(
    audio: AudioInput,
    vad: Optional[FireRedVad] = None,
    vad_model_dir: str = MODEL_DIR,
    smooth_window_size: int = 5,
    speech_threshold: float = 0.3,
    min_speech_frame: int = 20,
    max_speech_frame: int = 800,
    min_silence_frame: int = 20,
    merge_silence_frame: int = 50,
    extend_speech_frame: int = 0,
    chunk_max_frame: int = 30000,
) -> Tuple[List[Tuple[SpeechSegment, np.ndarray]], Optional[np.ndarray]]:
    try:
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(str(audio))
            audio_path = Path(audio).resolve()
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = 16000
            audio_path = Path("memory_buffer")
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        waveform_np = waveform.squeeze(0).numpy()

        if vad is None:
            with console.status("[bold cyan]Loading FireRedVAD…", spinner="dots"):
                vad_config = FireRedVadConfig(
                    smooth_window_size=smooth_window_size,
                    speech_threshold=speech_threshold,
                    min_speech_frame=min_speech_frame,
                    max_speech_frame=max_speech_frame,
                    min_silence_frame=min_silence_frame,
                    merge_silence_frame=merge_silence_frame,
                    extend_speech_frame=extend_speech_frame,
                    chunk_max_frame=chunk_max_frame,
                )
                vad = FireRedVad.from_pretrained(
                    vad_model_dir, vad_config
                )

        console.rule("Voice Activity Detection")
        rprint(f"[dim]Input:[/dim] {audio_path}")
        rprint(f"[dim]Duration:[/dim] {len(waveform_np) / 16000:.1f} s")

        result, probs_tensor = vad.detect((waveform_np, 16000))
        timestamps = result.get("timestamps", [])

        if not timestamps:
            console.print("[yellow]No speech detected.[/yellow]")
            return [], None

        full_probs = probs_tensor.numpy() if probs_tensor is not None else None

        segments = []
        for i, (start_sec, end_sec) in enumerate(timestamps, 1):
            start_idx = int(start_sec * 16000)
            end_idx = int(end_sec * 16000)
            segment_wav = waveform_np[start_idx:end_idx]
            if segment_wav.size == 0:
                continue

            start_frame = frames_from_seconds(start_sec)
            end_frame = frames_from_seconds(end_sec)
            segment_probs = (
                full_probs[start_frame:end_frame] if full_probs is not None else None
            )

            probs_info = {}
            if segment_probs is not None and len(segment_probs) > 0:
                probs_info = {
                    "num_frames": len(segment_probs),
                    "mean": float(np.mean(segment_probs)),
                    "max": float(np.max(segment_probs)),
                    "min": float(np.min(segment_probs)),
                    "std": float(np.std(segment_probs)),
                    "median": float(np.median(segment_probs)),
                    "frame_rate_hz": 100,
                }

            meta: SpeechSegment = {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "sample_rate": 16000,
                "segment_index": i,
                "output_path": "",
                "probs_info": probs_info,
            }
            segments.append((meta, segment_wav))

        console.print(
            f"[green]→ {len(segments)} speech segment{'s' if len(segments) != 1 else ''}"
        )
        return segments, full_probs

    except Exception as e:
        logger.exception("Failed in VAD segmentation")
        console.print(f"[red]Error during VAD: {e}[/red]")
        return [], None


def save_segments(
    segments: List[Tuple[SpeechSegment, np.ndarray]],
    full_probs: Optional[np.ndarray],
    output_base_dir: Path,
) -> List[SpeechSegment]:
    output_base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_base_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    saved_metadata = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Saving segments + plots…", total=len(segments))

        for meta, audio_np in segments:
            idx = meta["segment_index"]
            seg_dir = segments_dir / f"segment_{idx:03d}"
            seg_dir.mkdir(exist_ok=True)

            # Waveform
            wav_path = seg_dir / "sound.wav"
            try:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                torchaudio.save(
                    str(wav_path),
                    audio_tensor,
                    16000,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            except Exception as e:
                logger.warning(f"Failed to save waveform {wav_path}: {e}")
                continue

            meta["output_path"] = str(wav_path.relative_to(output_base_dir))

            # Metadata
            with open(seg_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Probabilities & plot
            num_frames = meta["probs_info"].get("num_frames", 0)
            if num_frames > 0:
                start_frame = frames_from_seconds(meta["start_sec"])
                end_frame = frames_from_seconds(meta["end_sec"])

                if (
                    full_probs is not None
                    and start_frame < len(full_probs)
                    and end_frame <= len(full_probs)
                ):
                    segment_probs = full_probs[start_frame:end_frame]
                    is_dummy = False
                else:
                    logger.warning(
                        f"Segment {idx:03d}: falling back to dummy probs (frame range invalid)"
                    )
                    t = np.linspace(0, 1, num_frames)
                    base = 0.12 + 0.76 / (1 + np.exp(-14 * (t - 0.48)))
                    noise = np.random.normal(0, 0.035, num_frames)
                    segment_probs = np.clip(base + noise, 0.03, 0.99)
                    taper = np.sin(np.pi * t) ** 0.35
                    segment_probs *= 0.88 + 0.12 * taper
                    is_dummy = True

                    if full_probs is None:
                        console.print(
                            "[yellow]Note: using dummy probabilities (real probs not available)[/yellow]"
                        )

                # Save JSON
                with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "probs": segment_probs.tolist(),
                            "frame_shift_sec": 0.010,
                            "start_frame_global": start_frame,
                            "summary": meta["probs_info"],
                            "is_dummy": is_dummy,
                        },
                        f,
                        indent=2,
                    )

                # Plot
                plot_path = seg_dir / "speech_probs.png"
                generate_plot(
                    probs=segment_probs,
                    segment_idx=idx,
                    duration_sec=meta["duration_sec"],
                    output_path=plot_path,
                    is_dummy=is_dummy,
                )

            saved_metadata.append(meta)
            progress.advance(task)

    console.print(f"[bold green]✓ Saved {len(saved_metadata)} segments[/bold green]")
    console.print(
        f"Output: [link=file://{segments_dir.resolve()}]{segments_dir}[/link]"
    )
    return saved_metadata


if __name__ == "__main__":
    import argparse
    import json
    import shutil

    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

    parser = argparse.ArgumentParser(description="VAD segmentation with FireRedVAD")
    parser.add_argument(
        "audio_path", nargs="?", default=DEFAULT_AUDIO, help="input audio file"
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD", style="blue")

    results, full_probs = extract_speech_segments(args.audio_path)

    if not results:
        console.print("[red]No segments found.[/red]")
        raise SystemExit(0)

    saved_metas = save_segments(results, full_probs, OUTPUT_DIR)

    # Save to JSON using with open, and log success with full path.
    output_json_path = OUTPUT_DIR / "all_speech_segments.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_metas, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]✓ Segments metadata saved to:[/bold green] [link=file://{output_json_path.resolve()}]{output_json_path}[/link]"
    )

    console.rule("Done", style="green")
