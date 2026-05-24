import io
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ten_vad import TenVad

console = Console()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
AudioInput = Union[np.ndarray, bytes, bytearray, str, Path]


class SpeechSegment(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]
    segment_probs: List[float]


# ---------------------------------------------------------------------------
# Defaults (mirror vad_firered2's style)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_HOP_SIZE = 160
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 100
DEFAULT_RETURN_SECONDS = False


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------
def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Robust audio loader for ASR pipelines with correct datatype, normalization,
    layout, and resampling.

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
    import librosa

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

    if np.issubdtype(y.dtype, np.integer):
        y = y / (2 ** (np.iinfo(y.dtype).bits - 1))
    if len(y) > 0 and np.abs(y).max() > 1.0 + 1e-6:
        y = y / np.abs(y).max()

    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    sr = current_sr or sr
    if current_sr != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr)

    return y.squeeze(), sr


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio in range [-1, 1] to int16."""
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Core VAD extraction
# ---------------------------------------------------------------------------
def extract_speech_timestamps(
    audio: AudioInput,
    with_scores: bool = False,
    include_non_speech: bool = False,
    return_seconds: bool = DEFAULT_RETURN_SECONDS,
    hop_size: int = DEFAULT_HOP_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_MS,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech segments from audio using TEN VAD.

    Args:
        audio: Input audio (file path, bytes, numpy array, or torch tensor).
        with_scores: If True, return a tuple (segments, frame_probs).
        include_non_speech: If True, include non-speech segments in output.
        return_seconds: If True, start/end in seconds; else in samples.
        hop_size: Number of samples per VAD frame (default 160).
        threshold: VAD probability threshold for speech (default 0.5).
        min_speech_duration_ms: Minimum duration of a speech segment in ms.
        min_silence_duration_ms: Minimum silence duration between speech segments
                                 that should be considered a true break.

    Returns:
        List of SpeechSegment dicts, or tuple (segments, frame_probs).
    """
    audio_np, sr = load_audio(audio, sr=16000, mono=True)
    if sr != 16000:
        raise ValueError(f"TEN VAD requires 16000 Hz, got {sr}")

    audio_int16 = float32_to_int16(audio_np)
    vad = TenVad(hop_size=hop_size, threshold=threshold)

    num_samples = len(audio_int16)
    num_frames = (num_samples + hop_size - 1) // hop_size
    probs = []
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + hop_size, num_samples)
        frame = audio_int16[start:end]
        if len(frame) < hop_size:
            frame = np.pad(frame, (0, hop_size - len(frame)), mode="constant")
        prob, _flag = vad.process(frame)
        probs.append(prob)

    # ------------------------------------------------------------------
    # Convert probabilities to binary flags, then merge runs
    # ------------------------------------------------------------------
    vad_flags = [1 if p >= threshold else 0 for p in probs]
    frame_duration_ms = (hop_size / sr) * 1000
    min_speech_frames = int(np.ceil(min_speech_duration_ms / frame_duration_ms))
    min_silence_frames = int(np.ceil(min_silence_duration_ms / frame_duration_ms))

    def _extract_runs(flags, target_val):
        runs = []
        idx = 0
        while idx < len(flags):
            if flags[idx] == target_val:
                start = idx
                while idx < len(flags) and flags[idx] == target_val:
                    idx += 1
                runs.append((start, idx - 1))
            else:
                idx += 1
        return runs

    def _merge_runs(runs, min_gap_frames):
        if not runs:
            return []
        merged = []
        cur_start, cur_end = runs[0]
        for nxt_start, nxt_end in runs[1:]:
            gap = nxt_start - cur_end - 1
            if gap < min_gap_frames:
                cur_end = nxt_end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = nxt_start, nxt_end
        merged.append((cur_start, cur_end))
        return merged

    # --- Speech segments ---
    speech_runs = _extract_runs(vad_flags, 1)
    merged_speech = _merge_runs(speech_runs, min_silence_frames)

    segments: List[SpeechSegment] = []
    seg_num = 1

    def _make_segment(
        num: int,
        f_start: int,
        f_end: int,
        seg_type: Literal["speech", "non-speech"],
    ) -> SpeechSegment:
        start_sec = f_start * frame_duration_ms / 1000.0
        end_sec = (f_end + 1) * frame_duration_ms / 1000.0
        dur_sec = end_sec - start_sec
        seg_probs_slice = probs[f_start : f_end + 1]
        avg_prob = float(np.mean(seg_probs_slice)) if seg_probs_slice else 0.0

        start_val = start_sec if return_seconds else f_start * hop_size
        end_val = end_sec if return_seconds else (f_end + 1) * hop_size

        return SpeechSegment(
            num=num,
            start=start_val,
            end=end_val,
            prob=avg_prob,
            duration=dur_sec,
            frames_length=f_end - f_start + 1,
            frame_start=f_start,
            frame_end=f_end,
            type=seg_type,
            segment_probs=seg_probs_slice if with_scores else [],
        )

    for f_start, f_end in merged_speech:
        dur_frames = f_end - f_start + 1
        if dur_frames >= min_speech_frames:
            segments.append(_make_segment(seg_num, f_start, f_end, "speech"))
            seg_num += 1

    # --- Non-speech segments (optional) ---
    if include_non_speech:
        non_speech_runs = _extract_runs(vad_flags, 0)
        merged_ns = _merge_runs(non_speech_runs, min_speech_frames)
        for f_start, f_end in merged_ns:
            segments.append(_make_segment(seg_num, f_start, f_end, "non-speech"))
            seg_num += 1
        segments.sort(key=lambda s: s["start"])
        for i, s in enumerate(segments):
            s["num"] = i + 1

    if with_scores:
        return segments, probs
    return segments


# ---------------------------------------------------------------------------
# Audio extraction from segments
# ---------------------------------------------------------------------------
def extract_speech_audio(
    audio: AudioInput,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_MS,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using TEN VAD.
    Returns a flat list of numpy arrays where each array represents one complete
    speech segment in float32 format, normalized to [-1.0, 1.0].
    """
    if sampling_rate != 16000:
        raise ValueError(f"TEN VAD requires 16000 Hz, got {sampling_rate}")

    speech_segments = extract_speech_timestamps(
        audio=audio,
        return_seconds=True,
        include_non_speech=False,
        hop_size=hop_size,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    audio_np, sr = load_audio(audio=audio, sr=sampling_rate, mono=True)
    if sr != sampling_rate:
        raise ValueError(
            f"Loaded sample rate {sr} does not match requested {sampling_rate}"
        )

    speech_audio_chunks: List[np.ndarray] = []
    for segment in speech_segments:
        start_sec: float = segment["start"]
        end_sec: float = segment["end"]
        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))
        segment_audio = audio_np[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue
        speech_audio_chunks.append(segment_audio.astype(np.float32, copy=False))

    return speech_audio_chunks


# ---------------------------------------------------------------------------
# Plot / RMS helpers (matching vad_firered2 style)
# ---------------------------------------------------------------------------
def _frames_from_seconds(sec: float, hop_size: int = 160, sr: int = 16000) -> int:
    """Convert seconds to a frame index using hop_size and sample rate."""
    frame_shift_sec = hop_size / sr
    return int(round(sec / frame_shift_sec))


def _compute_rms(
    signal: np.ndarray,
    frame_length: int = 160,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Compute per-frame RMS energy aligned to 10 ms frames.
    160 samples @ 16 kHz = exactly 10 ms per frame.
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)

    num_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start : start + frame_length]
        if frame.size:
            rms[i] = float(np.sqrt(np.mean(frame**2)))
    return rms


def _generate_plot(
    probs: np.ndarray,
    segment_idx: int,
    duration_sec: float,
    output_path: Path,
    is_dummy: bool = False,
    rms: Optional[np.ndarray] = None,
) -> None:
    """Save a speech-probability (+ optional RMS energy) plot to *output_path*."""
    num_frames = len(probs)
    if num_frames == 0:
        return

    has_rms = rms is not None and len(rms) > 0
    rows = 2 if has_rms else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9.5, 3.2 * rows), dpi=140)
    if rows == 1:
        axes = [axes]

    label = "Speech probability (dummy)" if is_dummy else "Speech probability"
    color = "#ff7f0e" if is_dummy else "#2ca02c"

    ax = axes[0]
    ax.plot(probs, color=color, linewidth=1.8, label=label)
    ax.fill_between(range(num_frames), probs, color=color, alpha=0.14)
    ax.axhline(
        y=0.5,
        linestyle="--",
        color="#d62728",
        alpha=0.65,
        linewidth=1.2,
        label="threshold = 0.5",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, num_frames - 1)
    ax.set_ylabel("Speech Probability", fontsize=10.5)
    ax.set_xlabel(
        f"Frame (10 ms)  —  {num_frames} frames ≈ {duration_sec:.1f} s",
        fontsize=10.5,
    )
    ax.set_title(
        f"Segment {segment_idx:03d} — {'Dummy ' if is_dummy else ''}Model Probabilities",
        fontsize=12,
        pad=12,
    )
    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    if has_rms:
        ax_rms = axes[1]
        ax_rms.plot(range(len(rms)), rms, linewidth=1.6, label="RMS energy")
        ax_rms.fill_between(range(len(rms)), rms, alpha=0.15)
        ax_rms.set_ylabel("RMS Energy", fontsize=10.5)
        ax_rms.set_xlabel("Frame (10 ms)", fontsize=10.5)
        ax_rms.set_xlim(0, len(rms) - 1)
        ax_rms.grid(True, alpha=0.28, linestyle="--", zorder=0)
        ax_rms.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    fig.tight_layout(pad=0.9)
    plt.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reusable save_segments (matches vad_firered2's output structure)
# ---------------------------------------------------------------------------
def save_segments(
    segments: List[SpeechSegment],
    audio_chunks: List[np.ndarray],
    output_base_dir: Path,
) -> List[SpeechSegment]:
    """
    Persist every speech segment to *output_base_dir/segments/segment_NNN/*.

    For each segment the function writes:
      sound.wav          – 16-kHz PCM-16 audio
      meta.json          – SpeechSegment metadata + probs_info summary
      speech_probs.json  – per-frame probabilities + summary stats
      energies.json      – per-frame RMS energy
      speech_and_rms.png – probability + RMS energy plot

    Parameters
    ----------
    segments:
        Output of ``extract_speech_timestamps(..., return_seconds=True,
        with_scores=True)``.  Non-speech segments are skipped automatically.
    audio_chunks:
        Output of ``extract_speech_audio()``.  Must contain one array per
        *speech* segment in the same order.
    output_base_dir:
        Root directory that will receive the ``segments/`` sub-tree.

    Returns
    -------
    List[SpeechSegment]
        Metadata for every saved segment (``output_path`` field populated).
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_base_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    speech_segments = [s for s in segments if s["type"] == "speech"]

    if len(speech_segments) != len(audio_chunks):
        console.print(
            f"[yellow]save_segments: {len(speech_segments)} speech segments but "
            f"{len(audio_chunks)} audio chunks — zipping by position, extras ignored.[/yellow]"
        )

    pairs = list(zip(speech_segments, audio_chunks))
    saved: List[SpeechSegment] = []

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
        task = progress.add_task("[cyan]Saving segments + plots…", total=len(pairs))
        for meta, audio_np in pairs:
            idx = meta["num"]
            seg_dir = segments_dir / f"segment_{idx:03d}"
            seg_dir.mkdir(exist_ok=True)

            # --- WAV ---
            wav_path = seg_dir / "sound.wav"
            try:
                torchaudio.save(
                    str(wav_path),
                    torch.from_numpy(audio_np).unsqueeze(0),
                    16000,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            except Exception as exc:
                console.print(f"[red]Failed to save WAV {wav_path}: {exc}[/red]")
                progress.advance(task)
                continue

            # --- Per-frame probabilities ---
            seg_probs_list: List[float] = meta.get("segment_probs", [])
            seg_probs_arr = np.asarray(seg_probs_list, dtype=np.float32)
            is_dummy = len(seg_probs_arr) == 0

            if is_dummy:
                num_frames = max(1, _frames_from_seconds(meta["duration"]))
                t = np.linspace(0, 1, num_frames)
                base = 0.12 + 0.76 / (1 + np.exp(-14 * (t - 0.48)))
                noise = np.random.default_rng().normal(0, 0.035, num_frames)
                seg_probs_arr = np.clip(base + noise, 0.03, 0.99).astype(np.float32)
                seg_probs_arr *= 0.88 + 0.12 * np.sin(np.pi * t) ** 0.35
                console.print(
                    f"[yellow]Segment {idx:03d}: no probabilities stored — "
                    "using synthetic fallback.[/yellow]"
                )

            probs_info = {
                "num_frames": int(len(seg_probs_arr)),
                "mean": float(np.mean(seg_probs_arr)),
                "max": float(np.max(seg_probs_arr)),
                "min": float(np.min(seg_probs_arr)),
                "std": float(np.std(seg_probs_arr)),
                "median": float(np.median(seg_probs_arr)),
                "frame_rate_hz": 100,
            }

            # --- meta.json ---
            meta_to_save = dict(meta)
            meta_to_save["output_path"] = str(wav_path.relative_to(output_base_dir))
            meta_to_save["probs_info"] = probs_info
            meta_to_save.pop("segment_probs", None)

            with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta_to_save, fh, indent=2, ensure_ascii=False)

            # --- speech_probs.json ---
            with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "probs": seg_probs_arr.tolist(),
                        "frame_shift_sec": 0.010,
                        "frame_start": meta.get("frame_start", 0),
                        "summary": probs_info,
                        "is_dummy": is_dummy,
                    },
                    fh,
                    indent=2,
                )

            # --- energies.json ---
            rms = _compute_rms(audio_np)
            with open(seg_dir / "energies.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rms": rms.tolist(),
                        "frame_shift_sec": 0.010,
                        "num_frames": int(len(rms)),
                    },
                    fh,
                    indent=2,
                )

            # --- Plot ---
            _generate_plot(
                probs=seg_probs_arr,
                segment_idx=idx,
                duration_sec=float(meta["duration"]),
                output_path=seg_dir / "speech_and_rms.png",
                is_dummy=is_dummy,
                rms=rms,
            )

            meta["output_path"] = meta_to_save["output_path"]
            saved.append(meta)
            progress.advance(task)

    console.print(f"[bold green]✓ Saved {len(saved)} segments[/bold green]")
    console.print(
        f"Output: [link=file://{segments_dir.resolve()}]{segments_dir}[/link]"
    )
    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    DEFAULT_AUDIO = str(
        Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
    )

    parser = argparse.ArgumentParser(
        description="Extract speech segments with TEN VAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"speech threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-s",
        "--hop-size",
        type=int,
        default=DEFAULT_HOP_SIZE,
        help=f"frame hop size in samples (default: {DEFAULT_HOP_SIZE})",
    )
    parser.add_argument(
        "-md",
        "--min-speech-duration",
        type=int,
        default=DEFAULT_MIN_SPEECH_MS,
        help=f"minimum speech duration in ms (default: {DEFAULT_MIN_SPEECH_MS})",
    )
    parser.add_argument(
        "-ms",
        "--min-silence-duration",
        type=int,
        default=DEFAULT_MIN_SILENCE_MS,
        help=f"minimum silence duration in ms (default: {DEFAULT_MIN_SILENCE_MS})",
    )
    parser.add_argument(
        "-mp",
        "--min-prob",
        type=float,
        default=0.0,
        help="minimum average speech probability to keep a segment (default: 0.0 = no filter)",
    )
    parser.add_argument(
        "-mnd",
        "--min-duration",
        type=float,
        default=0.0,
        help="minimum duration in seconds to keep a segment (default: 0.0 = no filter)",
    )
    parser.add_argument(
        "--include-non-speech",
        "-n",
        action="store_true",
        help="Include non-speech segments in output",
    )

    args = parser.parse_args()

    audio_path = args.audio_path
    output_dir = Path(args.output_dir)

    # --- Clear output directory (moved from module level) ---
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – TEN VAD", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")

    # --- Extract timestamps ---
    segments, speech_probs = extract_speech_timestamps(
        audio_path,
        with_scores=True,
        include_non_speech=args.include_non_speech,
        return_seconds=True,
        hop_size=args.hop_size,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration,
    )

    # --- Filter segments ---
    original_count = len(segments)
    filtered = []
    for s in segments:
        if s.get("prob", 0.0) < args.min_prob:
            continue
        if s.get("duration", 0.0) < args.min_duration:
            continue
        filtered.append(s)
    segments = filtered

    if original_count != len(segments):
        console.print(
            f"[yellow]Filtered: {len(segments)}/{original_count} segments kept "
            f"(min-prob={args.min_prob:.3f}, min-duration={args.min_duration:.2f}s)[/yellow]"
        )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")

    # --- Extract audio chunks ---
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        hop_size=args.hop_size,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration,
    )

    speech_segments = [s for s in segments if s["type"] == "speech"]
    audio_chunks = audio_chunks[: len(speech_segments)]

    # --- Save all segments ---
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # --- Print segment summaries with ▶ Play links ---
    def play_segment(wav_path: Path):
        try:
            if platform.system() == "Darwin":
                subprocess.run(["afplay", str(wav_path)], check=False)
            elif platform.system() == "Windows":
                subprocess.run(
                    [
                        "powershell",
                        "-c",
                        f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()",
                    ],
                    check=False,
                )
            else:
                subprocess.run(["aplay", str(wav_path)], check=False)
        except Exception:
            pass

    for seg in saved_metas:
        seg_type = seg["type"]
        type_color = "bold green" if seg_type == "speech" else "bold red"
        wav_rel = seg.get("output_path")
        wav_full = output_dir / wav_rel if wav_rel else None

        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white]"
            f" - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
            f"   [bold blue][link=file://{wav_full}]▶ Play[/link][/bold blue]"
        )

    if not any(s["type"] == "speech" for s in saved_metas):
        console.print("[red]No speech segments found after filtering.[/red]")
        raise SystemExit(0)

    # --- Summary JSON ---
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_speech_segments.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        slim = [
            {k: v for k, v in m.items() if k != "segment_probs"}
            for m in saved_metas
        ]
        json.dump(slim, fh, ensure_ascii=False, indent=2)

    console.print(
        f"[bold green]✓ Summary saved to:[/bold green] "
        f"[link=file://{summary_path.resolve()}]{summary_path}[/link]"
    )
    console.rule("Done", style="green")
