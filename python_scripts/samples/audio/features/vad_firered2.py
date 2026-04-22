import io
import json
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from fireredvad.core.constants import SAMPLE_RATE
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from loader import load_audio
from _types import AudioInput, SpeechSegment, WordSegment

console = Console()

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = 0.65,
        min_silence_duration_sec: float = 0.20,
        min_speech_duration_sec: float = 0.15,
        max_speech_duration_sec: float = 12.0,
        smooth_window_size: int = 5,
        pad_start_frame: int = 5,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        console.print(f"[cyan]Loading FireRedVAD (streaming) on {self.device}…[/cyan]")
        frames_per_sec = 100
        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            smooth_window_size=smooth_window_size,
            pad_start_frame=pad_start_frame,
            min_speech_frame=int(min_speech_duration_sec * frames_per_sec),
            max_speech_frame=int(max_speech_duration_sec * frames_per_sec),
            min_silence_frame=int(min_silence_duration_sec * frames_per_sec),
            chunk_max_frame=30000,
        )
        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        self.vad.vad_model.to(self.device)
        console.print("[green]done.[/green]")
        self.sample_rate = SAMPLE_RATE
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0
        self.max_buffer_samples = int(1.2 * self.sample_rate)

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic range compression / gain normalization."""
        if len(chunk) == 0:
            return chunk.astype(np.float32)
        chunk = chunk.astype(np.float32)
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain
        return chunk

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process incoming audio chunk (any length) and return
        the **latest smoothed speech probability**.
        """
        if len(chunk) == 0:
            return self.last_prob
        chunk = self._normalize_chunk(chunk)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        if len(self.audio_buffer) < 4800:
            return self.last_prob
        to_process = self.audio_buffer[-9600:]
        results = self.vad.detect_chunk(to_process)
        self.audio_buffer = self.audio_buffer[-512:]
        if not results:
            return self.last_prob
        last = results[-1]
        prob = last.smoothed_prob
        self.last_prob = prob
        return prob

    def get_latest_result(self) -> Optional[dict]:
        """
        Optional: return more detailed info about the last processed frame
        (useful for debugging or when you need is_speech_start / is_speech_end).
        """
        return None

    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        self.reset()
        frame_results, result = self.vad.detect_full(audio)
        return frame_results, result


def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = 0.5,
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = False,
    with_scores: bool = False,
    include_non_speech: bool = False,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using FireRedVAD.
    When include_non_speech=True, returns both speech and non-speech (silence) segments.
    """
    if max_speech_duration_sec is None:
        max_speech_duration_sec = 15.0

    audio_np, sr = load_audio(audio, sr=16000, mono=True)
    if sr != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sr}")

    vad = FireRedVAD(
        model_dir=SAVE_DIR,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
    )

    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]
    hop_sec = 0.010

    def make_segment(
        num: int,
        start_sec: float,
        end_sec: float,
        seg_type: Literal["speech", "non-speech"],
    ) -> SpeechSegment:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        frame_start = int(start_sec / hop_sec)
        frame_end = int(end_sec / hop_sec)
        segment_probs_slice = probs[frame_start : frame_end + 1]
        avg_prob = float(np.mean(segment_probs_slice)) if segment_probs_slice else 0.0
        duration_sec = end_sec - start_sec
        start_val = start_sec if return_seconds else start_sample
        end_val = end_sec if return_seconds else end_sample
        return SpeechSegment(
            num=num,
            start=start_val,
            end=end_val,
            prob=avg_prob,
            duration=duration_sec,
            frames_length=len(segment_probs_slice),
            frame_start=frame_start,
            frame_end=frame_end,
            type=seg_type,
            segment_probs=segment_probs_slice if with_scores else [],
        )

    enhanced: List[SpeechSegment] = []
    current_time = 0.0
    seg_num = 1

    if include_non_speech and timestamps and timestamps[0][0] > 0.001:
        enhanced.append(make_segment(seg_num, 0.0, timestamps[0][0], "non-speech"))
        seg_num += 1
        current_time = timestamps[0][0]

    for start_sec, end_sec in timestamps:
        if include_non_speech and start_sec > current_time + 0.01:
            enhanced.append(
                make_segment(seg_num, current_time, start_sec, "non-speech")
            )
            seg_num += 1
        enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
        seg_num += 1
        current_time = end_sec

    total_duration = result["dur"]
    if include_non_speech and current_time < total_duration - 0.01:
        enhanced.append(
            make_segment(seg_num, current_time, total_duration, "non-speech")
        )

    if with_scores:
        return enhanced, probs
    return enhanced


def extract_speech_audio(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float | None = None,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using FireRedVAD.
    Returns a flat list of numpy arrays where each array represents one complete
    speech segment in float32 format, normalized to [-1.0, 1.0].
    """
    if sampling_rate != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sampling_rate}")

    speech_segments = extract_speech_timestamps(
        audio=audio,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        include_non_speech=False,
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
# Helpers used by save_segments
# ---------------------------------------------------------------------------

def _frames_from_seconds(sec: float) -> int:
    """Convert seconds to a 10 ms frame index (100 frames per second)."""
    return int(round(sec * 100.0))


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
            rms[i] = float(np.sqrt(np.mean(frame ** 2)))
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
        y=0.4, linestyle="--", color="#d62728", alpha=0.65,
        linewidth=1.2, label="threshold ≈ 0.4",
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
        fontsize=12, pad=12,
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
# save_segments
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

            # ── 1. WAV ────────────────────────────────────────────────────
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

            # ── 2. Probability array ──────────────────────────────────────
            seg_probs_list: List[float] = meta.get("segment_probs", [])
            seg_probs_arr = np.asarray(seg_probs_list, dtype=np.float32)
            is_dummy = len(seg_probs_arr) == 0

            if is_dummy:
                # Synthetic sigmoid fallback so the plot is still meaningful
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

            # ── 3. probs_info summary stats ───────────────────────────────
            probs_info = {
                "num_frames": int(len(seg_probs_arr)),
                "mean":       float(np.mean(seg_probs_arr)),
                "max":        float(np.max(seg_probs_arr)),
                "min":        float(np.min(seg_probs_arr)),
                "std":        float(np.std(seg_probs_arr)),
                "median":     float(np.median(seg_probs_arr)),
                "frame_rate_hz": 100,
            }

            # ── 4. meta.json ──────────────────────────────────────────────
            meta_to_save = dict(meta)
            meta_to_save["output_path"] = str(
                wav_path.relative_to(output_base_dir)
            )
            meta_to_save["probs_info"] = probs_info
            # segment_probs can be large; keep it out of meta.json
            meta_to_save.pop("segment_probs", None)
            with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta_to_save, fh, indent=2, ensure_ascii=False)

            # ── 5. speech_probs.json ──────────────────────────────────────
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

            # ── 6. energies.json ──────────────────────────────────────────
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

            # ── 7. speech_and_rms.png ─────────────────────────────────────
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


if __name__ == "__main__":
    import argparse
    import shutil

    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD"
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
    args = parser.parse_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD2", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")

    # ── Step 1: detect segments (with per-frame probabilities) ────────────
    segments, speech_probs = extract_speech_timestamps(
        audio_path,
        max_speech_duration_sec=8.0,
        return_seconds=True,
        with_scores=True,
        include_non_speech=False,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        seg_type = seg["type"]
        type_color = "bold green" if seg_type == "speech" else "bold red"
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white]"
            f" - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
        )

    if not any(s["type"] == "speech" for s in segments):
        console.print("[red]No speech segments found.[/red]")
        raise SystemExit(0)

    # ── Step 2: extract raw audio for each speech segment ─────────────────
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=16000,
        max_speech_duration_sec=8.0,
    )

    # ── Step 3: save everything to disk ───────────────────────────────────
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # ── Step 4: write summary JSON files ──────────────────────────────────
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

    all_probs_path = output_dir / "speech_probs.json"
    with open(all_probs_path, "w", encoding="utf-8") as fh:
        json.dump(
            speech_probs if isinstance(speech_probs, list) else [],
            fh,
            indent=2,
        )
    console.print(
        f"[bold green]✓ Full probs saved to:[/bold green] "
        f"[link=file://{all_probs_path.resolve()}]{all_probs_path}[/link]"
    )

    console.rule("Done", style="green")
