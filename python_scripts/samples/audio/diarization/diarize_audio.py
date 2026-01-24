"""
Diarize a recording + optionally export raw segmentation scores (logits/probabilities)
Works with pyannote-audio 3.1+ (including current 4.0.3 version)
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from typing import Dict, Any, Tuple, Union
import numpy as np
import torch
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines.utils import get_devices
from pyannote.core import Annotation, SlidingWindowFeature
import os
import numpy.typing as npt
import librosa

# from jet.audio.speech.pyannote.utils import export_plotly_timeline
# from jet.audio.transcribers.base import AudioInput, load_audio

console = Console()

sample_rate = 16_000

AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> np.ndarray:
    """
    Robust audio loader for ASR pipelines with correct datatype, normalization, layout, and resampling.
    
    Handles:
      - File paths
      - In-memory WAV bytes
      - NumPy arrays (any shape/layout/dtype/sr)
      - Torch tensors
      - Automatically normalizes to [-1.0, 1.0] float32
      - Always resamples to target_sr
      - Correctly converts stereo → mono regardless of channel position
    Returns
    -------
    np.ndarray
        Shape (samples,), float32, [-1.0, 1.0], exactly `sr` Hz
    """
    # ─────── FIX 1: In-memory arrays/tensors have unknown original sr ───────
    import io
    current_sr: int | None
    if isinstance(audio, (str, os.PathLike)):
        y, current_sr = librosa.load(audio, sr=None, mono=False)
    elif isinstance(audio, bytes):
        y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
    elif isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = None
    elif isinstance(audio, torch.Tensor):
        y = audio.float().cpu().numpy()
        current_sr = None
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # ─────── FIX 2: Correct normalization (NumPy, not torch) ───────
    if np.issubdtype(y.dtype, np.integer):
        y = y / (2 ** (np.iinfo(y.dtype).bits - 1))
    elif np.abs(y).max() > 1.0 + 1e-6:
        y = y / np.abs(y).max()

    # ─────── FIX 3: Always make (channels, time) layout ───────
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    # Mono conversion
    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    # ─────── FIX 4: ALWAYS resample if current_sr is None or wrong ───────
    if current_sr != sr:
        y = librosa.resample(y, orig_sr=current_sr or sr, target_sr=sr)

    return y.squeeze()


def compute_scores_insights(scores: SlidingWindowFeature) -> Dict[str, Any]:
    """
    Analyze the raw segmentation scores and return useful statistics.
    Works with pyannote 3.1 where scores.data is 3D: (chunks, frames_per_chunk, speakers)
    """
    data = np.asarray(scores.data)  # shape: (chunks, frames_per_chunk, speakers)

    # Handle both 2D and 3D cases safely
    if data.ndim == 3:
        num_chunks, frames_per_chunk, num_speakers = data.shape
        total_frames = num_chunks * frames_per_chunk
        flat_data = data.reshape(total_frames, num_speakers)
    elif data.ndim == 2:
        total_frames, num_speakers = data.shape
        flat_data = data
    else:
        raise ValueError(f"Unexpected scores.data shape: {data.shape}")

    step = scores.sliding_window.step
    duration = total_frames * step

    # Per-speaker stats
    max_per_speaker = flat_data.max(axis=0)
    avg_per_speaker = flat_data.mean(axis=0)
    active_ratio = (flat_data > 0.5).mean(axis=0)  # how often model thinks speaker is active

    # Global stats
    max_prob_per_frame = flat_data.max(axis=1)
    confidence_mean = max_prob_per_frame.mean()
    confidence_std = max_prob_per_frame.std()
    overlap_frames = (flat_data.sum(axis=1) > 1.1).sum()  # rough overlap detection

    return {
        "total_duration_sec": round(float(duration), 3),
        "total_frames": int(total_frames),
        "frame_step_sec": float(step),
        "num_speakers": int(num_speakers),
        "confidence_mean": round(float(confidence_mean), 4),
        "confidence_std": round(float(confidence_std), 4),
        "overlap_frame_count": int(overlap_frames),
        "overlap_percentage": round(100 * overlap_frames / total_frames, 2),
        "per_speaker": {
            f"speaker_{i}": {
                "max_probability": round(float(m), 4),
                "avg_probability": round(float(a), 4),
                "active_ratio": round(float(r), 4),
            }
            for i, (m, a, r) in enumerate(zip(max_per_speaker, avg_per_speaker, active_ratio))
        },
    }

def diarize_file(
    audio_path: Path | str,
    output_dir: Path | str,
    *,
    pipeline_id: str = "pyannote/speaker-diarization-3.1",
    device: str | None = None,
    num_speakers: int | None = None,
    return_scores: bool = False,
) -> Tuple[Dict[str, Any], SlidingWindowFeature | None]:
    audio_path = Path(audio_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Loading pipeline [bold cyan]{pipeline_id}[/]")
    pipeline: Pipeline = Pipeline.from_pretrained(pipeline_id, revision="main")

    if device is None:
        device = get_devices(needs=1)[0]

    console.log(f"Using device: [bold green]{device}[/]")
    pipeline.to(torch.device(device))

    # ──────────────────────────────────────────────────────────────
    # Option 2 workaround: preload audio in-memory
    # waveform_np: 1D np.ndarray (samples,)
    # ──────────────────────────────────────────────────────────────
    waveform_np = load_audio(audio_path)
    assert waveform_np.ndim == 1, "Expected mono 1D waveform"

    audio_input = {
        "waveform": torch.from_numpy(waveform_np).unsqueeze(0),  # (1, samples)
        "sample_rate": sample_rate,
    }

    console.log(
        f"Running full diarization on [bold]{audio_path.name}[/] (in-memory)"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Diarizing...", total=None)
        diarization_result = pipeline(
            audio_input,
            num_speakers=num_speakers,
        )
        progress.update(task, completed=True)

    diarization: Annotation = (
        diarization_result.speaker_diarization
        if hasattr(diarization_result, "speaker_diarization")
        else diarization_result
    )

    console.log(
        f"Detected {len(diarization.labels())} speaker(s): "
        f"{sorted(str(s) for s in diarization.labels())}"
    )

    total_seconds = waveform_np.shape[0] / sample_rate
    turns: list[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────
    # Export speaker turns
    # ──────────────────────────────────────────────────────────────
    for idx, (segment, _, speaker) in enumerate(
        diarization.itertracks(yield_label=True)
    ):
        start_sec = round(segment.start, 3)
        end_sec = round(segment.end, 3)
        duration_sec = round(end_sec - start_sec, 3)

        seg_dir = segments_dir / f"segment_{idx:04d}"
        seg_dir.mkdir(exist_ok=True)

        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        segment_waveform = waveform_np[start_sample:end_sample]

        wav_path = seg_dir / "segment.wav"
        sf.write(
            wav_path,
            segment_waveform,
            samplerate=sample_rate,
            subtype="PCM_16",
        )

        meta = {
            "segment_index": idx,
            "speaker": str(speaker),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "wav_path": str(wav_path.relative_to(output_dir)),
        }

        (seg_dir / "segment.json").write_text(json.dumps(meta, indent=2))
        turns.append(meta)

        console.log(
            f"[green]Saved[/] segment_{idx:04d} | {str(speaker):<12} | "
            f"{start_sec:>7.3f}s -> {end_sec:>7.3f}s ({duration_sec:>5.3f}s)"
        )

    # ──────────────────────────────────────────────────────────────
    # Optional: raw segmentation scores
    # ──────────────────────────────────────────────────────────────
    scores: SlidingWindowFeature | None = None

    if return_scores:
        console.log("Extracting raw frame-level segmentation scores...")
        pipeline.instantiate({})

        try:
            seg_model: Model = pipeline._segmentation.model
            console.log("[dim]Using bundled segmentation model[/]")
        except AttributeError:
            console.log(
                "[red]Warning: Could not access internal model. Falling back to direct load.[/]"
            )
            seg_model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                revision="main",
            )
            seg_model.to(torch.device(device))

        from pyannote.audio import Inference

        duration = seg_model.specifications.duration
        inference = Inference(
            seg_model,
            duration=duration,
            step=0.1 * duration,
        )

        scores = inference(audio_input)
        scores_path = output_dir / "segmentation_scores.npy"
        np.save(scores_path, scores.data)

        console.log(f"[bold blue]Raw scores saved[/] → {scores_path}")

        insights = compute_scores_insights(scores)
        insights_path = output_dir / "segmentation_scores_insights.json"
        insights_path.write_text(json.dumps(insights, indent=2))

        console.log("[bold yellow]Scores insights saved[/] → scores_insights.json")
    else:
        scores_path = None

    # ──────────────────────────────────────────────────────────────
    # Summary + RTTM
    # ──────────────────────────────────────────────────────────────
    summary = {
        "audio_file": str(audio_path),
        "total_duration_sec": round(total_seconds, 3),
        "sample_rate": sample_rate,
        "num_speakers": len(diarization.labels()),
        "speakers": sorted(str(s) for s in diarization.labels()),
        "num_segments": len(turns),
        "segments": turns,
        "segmentation_scores_npy": str(scores_path) if scores_path else None,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.log(f"[bold green]Done![/] Summary → {summary_path}")

    rttm_path = output_dir / "diarization.rttm"
    with rttm_path.open("w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {audio_path.stem} 1 "
                f"{turn.start:.3f} {turn.duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    console.log(f"[bold magenta]RTTM exported[/] → {rttm_path.name}")

    return summary, scores

def diarize(
    audio: AudioInput,
    output_dir: Path | str,
    *,
    pipeline_id: str = "pyannote/speaker-diarization-3.1",
    device: str | None = None,
    num_speakers: int | None = None,
    return_scores: bool = False,
) -> Tuple[Dict[str, Any], SlidingWindowFeature | None]:
    """Perform speaker diarization on flexible audio input and save segments, metadata, RTTM, optional scores, and Plotly timeline."""
    output_dir = Path(output_dir).expanduser().resolve()
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Loading pipeline [bold cyan]{pipeline_id}[/]")
    pipeline: Pipeline = Pipeline.from_pretrained(pipeline_id, revision="main")

    if device is None:
        device = get_devices(needs=1)[0]
    console.log(f"Using device: [bold green]{device}[/]")
    pipeline.to(torch.device(device))

    # Load audio flexibly and get waveform + sample_rate
    waveform_np = load_audio(audio)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, samples)
    total_seconds = waveform.shape[1] / sample_rate

    # Temporary file for pyannote pipeline (which only accepts file paths)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(
            tmp.name,
            waveform_np,
            samplerate=sample_rate,
            subtype="PCM_16",
        )
        temp_path = Path(tmp.name)

    try:
        console.log(f"Running full diarization on audio ({total_seconds:.1f}s)")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Diarizing...", total=None)
            diarization_result = pipeline(
                str(temp_path),
                num_speakers=num_speakers,
            )
            progress.update(task, completed=True)

        diarization: Annotation = (
            diarization_result.speaker_diarization
            if hasattr(diarization_result, "speaker_diarization")
            else diarization_result
        )

        console.log(f"Detected {len(diarization.labels())} speaker(s): {sorted(str(s) for s in diarization.labels())}")

        turns: list[Dict[str, Any]] = []
        for idx, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            start_sec = round(segment.start, 3)
            end_sec = round(segment.end, 3)
            duration_sec = round(end_sec - start_sec, 3)

            seg_dir = segments_dir / f"segment_{idx:04d}"
            seg_dir.mkdir(exist_ok=True)

            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            wav_path = seg_dir / "segment.wav"
            sf.write(
                wav_path,
                segment_waveform.squeeze().T,
                samplerate=sample_rate,
                subtype="PCM_16",
            )

            meta = {
                "segment_index": idx,
                "speaker": str(speaker),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "wav_path": str(wav_path.relative_to(output_dir)),
            }
            (seg_dir / "segment.json").write_text(json.dumps(meta, indent=2))
            turns.append(meta)

            console.log(
                f"[green]Saved[/] segment_{idx:04d} | {str(speaker):<12} | "
                f"{start_sec:>7.3f}s -> {end_sec:>7.3f}s ({duration_sec:>5.3f}s)"
            )

        scores: SlidingWindowFeature | None = None
        if return_scores:
            console.log("Extracting raw frame-level segmentation scores...")
            pipeline.instantiate({})
            try:
                seg_model: Model = pipeline._segmentation.model
                console.log("[dim]Using bundled segmentation model[/]")
            except AttributeError:
                console.log("[red]Warning: Could not access internal model. Falling back to direct load.[/]")
                seg_model = Model.from_pretrained("pyannote/segmentation-3.0", revision="main")
                seg_model.to(torch.device(device))

            from pyannote.audio import Inference
            duration = seg_model.specifications.duration
            inference = Inference(
                seg_model,
                duration=duration,
                step=0.1 * duration,
            )
            scores = inference(str(temp_path))

            scores_path = output_dir / "segmentation_scores.npy"
            np.save(scores_path, scores.data)
            console.log(f"[bold blue]Raw scores saved[/] → {scores_path}")

            # Timing metadata
            if scores.data.ndim == 3:
                num_chunks = scores.data.shape[0]
                frames_per_chunk = scores.data.shape[1]
                num_speakers = scores.data.shape[2]
                total_frames = num_chunks * frames_per_chunk
            else:
                total_frames, num_speakers = scores.data.shape
                frames_per_chunk = None
                num_chunks = None

            timing = {
                "start": float(scores.sliding_window.start),
                "duration": float(scores.sliding_window.duration),
                "step": float(scores.sliding_window.step),
                "total_frames": int(total_frames),
                "frames_per_chunk": frames_per_chunk,
                "num_chunks": int(num_chunks) if scores.data.ndim == 3 else None,
                "num_speakers": int(num_speakers),
                "frame_rate_hz": round(1 / scores.sliding_window.step, 3),
            }
            (output_dir / "segmentation_scores_timing.json").write_text(json.dumps(timing, indent=2))

            # Insights
            console.log("Analyzing raw speaker probabilities...")
            insights = compute_scores_insights(scores)
            insights_path = output_dir / "segmentation_scores_insights.json"
            insights_path.write_text(json.dumps(insights, indent=2))
            console.log("[bold yellow]Scores insights saved[/] → scores_insights.json")
        else:
            scores_path = None

        # Summary
        summary = {
            "audio_file": "in_memory_audio.wav",  # placeholder since no original path
            "total_duration_sec": round(total_seconds, 3),
            "sample_rate": sample_rate,
            "num_speakers": len(diarization.labels()),
            "speakers": sorted(str(s) for s in diarization.labels()),
            "num_segments": len(turns),
            "segments": turns,
            "segmentation_scores_npy": str(scores_path) if scores_path else None,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        console.log(f"[bold green]Done![/] Summary → {summary_path}")

        # RTTM export
        rttm_path = output_dir / "diarization.rttm"
        with rttm_path.open("w") as f:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                line = (
                    f"SPEAKER audio 1 "
                    f"{turn.start:.3f} {turn.duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )
                f.write(line)
        console.log(f"[bold magenta]RTTM exported[/] → {rttm_path.name}")

        # Per-segment confidence
        if scores is not None:
            if scores.data.ndim == 3:
                flat_scores = scores.data.reshape(-1, scores.data.shape[2])
            else:
                flat_scores = scores.data
            confidences = flat_scores.max(axis=1)
            frame_times = scores.sliding_window.start + np.arange(len(confidences)) * scores.sliding_window.step

            for seg in turns:
                mask = (frame_times >= seg["start_sec"]) & (frame_times < seg["end_sec"])
                seg_conf = float(confidences[mask].mean()) if mask.any() else 0.0
                seg["confidence"] = round(seg_conf, 4)

            summary["segments"] = turns

        # Plotly timeline
        # export_plotly_timeline(
        #     turns=turns,
        #     total_seconds=total_seconds,
        #     audio_name="audio",
        #     output_dir=output_dir,
        # )

        return summary, scores

    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import shutil

    # Default values
    DEFAULT_AUDIO = Path(
        r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    )
    DEFAULT_OUTPUT = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Diarize audio file and save results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Positional (required) argument for audio file
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",                    # makes it optional → uses default if not provided
        default=DEFAULT_AUDIO,
        help="Path to the input audio file (.wav)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where results will be saved"
    )

    args = parser.parse_args()

    # Clean previous results
    shutil.rmtree(args.output, ignore_errors=True)

    diarize_file(
        args.audio,
        output_dir=args.output,
        return_scores=True,
    )
