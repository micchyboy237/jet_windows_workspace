import json
import shutil
from pathlib import Path
from typing import List, Literal, TypedDict, Union
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from rich.console import Console
from ten_vad import TenVad
import io
import os
import librosa
import numpy as np
import torch

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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


def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
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
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return audio_int16


def compute_rms_per_frame(
    audio: np.ndarray,
    hop_size: int,
    start_frame: int,
    end_frame: int,
) -> List[float]:
    """
    Compute RMS energy for each frame in the given frame range.
    Args:
        audio: Float32 audio array (mono).
        hop_size: Number of samples per frame.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
    Returns:
        List of RMS values (one per frame).
    """
    rms_values = []
    for frame_idx in range(start_frame, end_frame + 1):
        start_sample = frame_idx * hop_size
        end_sample = start_sample + hop_size
        frame_audio = audio[start_sample:end_sample]
        if len(frame_audio) == 0:
            rms = 0.0
        else:
            rms = np.sqrt(np.mean(frame_audio**2))
        rms_values.append(float(rms))
    return rms_values


def save_plot(
    probs: List[float],
    rms_values: List[float],
    output_path: Path,
    title: str = "Speech Probabilities and RMS Energy",
) -> None:
    """Create a dual subplot figure and save as PNG."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    frames = np.arange(len(probs))
    ax1.plot(frames, probs, color="blue", linewidth=1)
    ax1.set_ylabel("VAD Probability")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    ax2.plot(frames, rms_values, color="green", linewidth=1)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("RMS Energy")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(data, path: Path) -> None:
    """Save data as JSON with indentation."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def extract_speech_timestamps(
    audio: AudioInput,
    with_scores: bool = False,
    include_non_speech: bool = False,
    hop_size: int = 160,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech segments from audio using TEN VAD.
    Args:
        audio: Input audio (file path, bytes, numpy array, or torch tensor).
        with_scores: If True, return a tuple (segments, frame_probs).
        include_non_speech: If True, include non-speech segments in output.
        hop_size: Number of samples per VAD frame (default 160).
        threshold: VAD probability threshold for speech (default 0.5).
        min_speech_duration_ms: Minimum duration of a speech segment in ms.
        min_silence_duration_ms: Minimum silence duration between speech segments
                                 that should be considered a true break. Silences
                                 shorter than this will be merged into a single
                                 speech segment.
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
        prob, flag = vad.process(frame)
        probs.append(prob)

    vad_flags = [1 if p >= threshold else 0 for p in probs]
    frame_duration_ms = (hop_size / sr) * 1000
    min_speech_frames = int(np.ceil(min_speech_duration_ms / frame_duration_ms))
    min_silence_frames = int(np.ceil(min_silence_duration_ms / frame_duration_ms))

    # ── Extract speech runs ──
    speech_runs = []
    idx = 0
    while idx < len(vad_flags):
        if vad_flags[idx] == 1:
            start = idx
            while idx < len(vad_flags) and vad_flags[idx] == 1:
                idx += 1
            end = idx - 1
            speech_runs.append((start, end))
        else:
            idx += 1

    # ── Merge speech runs separated by short silences ──
    merged_runs = []
    if speech_runs:
        current_start, current_end = speech_runs[0]
        for next_start, next_end in speech_runs[1:]:
            gap_frames = next_start - current_end - 1
            if gap_frames < min_silence_frames:
                current_end = next_end
            else:
                merged_runs.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_runs.append((current_start, current_end))

    segments = []
    segment_counter = 0

    # ── Create speech segments from merged runs ──
    for start_frame, end_frame in merged_runs:
        duration_frames = end_frame - start_frame + 1
        if duration_frames >= min_speech_frames:
            duration_ms = duration_frames * frame_duration_ms
            segment = {
                "num": segment_counter,
                "start": start_frame * frame_duration_ms / 1000.0,
                "end": (end_frame + 1) * frame_duration_ms / 1000.0,
                "prob": float(np.mean(probs[start_frame : end_frame + 1])),
                "duration": duration_ms / 1000.0,
                "frames_length": duration_frames,
                "frame_start": start_frame,
                "frame_end": end_frame,
                "type": "speech",
                "segment_probs": probs[start_frame : end_frame + 1],
            }
            segments.append(segment)
            segment_counter += 1

    # ── Handle non-speech segments with merging ──
    if include_non_speech:
        # STEP 1: Extract raw non-speech runs
        non_speech_runs = []
        idx = 0
        while idx < len(vad_flags):
            if vad_flags[idx] == 0:
                start = idx
                while idx < len(vad_flags) and vad_flags[idx] == 0:
                    idx += 1
                end = idx - 1
                non_speech_runs.append((start, end))
            else:
                # Skip speech frames while collecting non-speech runs
                while idx < len(vad_flags) and vad_flags[idx] == 1:
                    idx += 1
        
        # STEP 2: Merge non-speech runs separated by short speech gaps
        merged_non_speech_runs = []
        if non_speech_runs:
            current_start, current_end = non_speech_runs[0]
            for next_start, next_end in non_speech_runs[1:]:
                # Calculate speech frames between two non-speech runs
                gap_frames = next_start - current_end - 1
                # Merge if the speech gap is shorter than min_speech_duration
                if gap_frames < min_speech_frames:
                    current_end = next_end  # Extend current run
                else:
                    # Gap is long enough: finalize current run
                    merged_non_speech_runs.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            # Don't forget the last run
            merged_non_speech_runs.append((current_start, current_end))
        
        # STEP 3: Create segments from merged non-speech runs
        for start_frame, end_frame in merged_non_speech_runs:
            duration_frames = end_frame - start_frame + 1
            duration_ms = duration_frames * frame_duration_ms
            segment = {
                "num": segment_counter,
                "start": start_frame * frame_duration_ms / 1000.0,
                "end": (end_frame + 1) * frame_duration_ms / 1000.0,
                "prob": float(np.mean(probs[start_frame : end_frame + 1])),
                "duration": duration_ms / 1000.0,
                "frames_length": duration_frames,
                "frame_start": start_frame,
                "frame_end": end_frame,
                "type": "non-speech",
                "segment_probs": probs[start_frame : end_frame + 1],
            }
            segments.append(segment)
            segment_counter += 1
        
        # STEP 4: Sort all segments by time and renumber sequentially
        segments.sort(key=lambda s: s["start"])
        for i, seg in enumerate(segments):
            seg["num"] = i

    if with_scores:
        return segments, probs
    return segments


if __name__ == "__main__":
    import argparse

    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Input audio file path (default: {DEFAULT_AUDIO})"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output results dir (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="VAD probability threshold"
    )
    parser.add_argument(
        "-s", "--hop-size", type=int, default=160, help="Frame hop size in samples"
    )
    parser.add_argument(
        "--min-speech-duration",
        "-d",
        type=int,
        default=250,
        help="Minimum speech segment duration in ms",
    )
    parser.add_argument(
        "--min-silence-duration",
        "-g",
        type=int,
        default=100,
        help="Minimum silence duration in ms",
    )
    parser.add_argument(
        "--include-non-speech",
        "-n",
        action="store_true",
        help="Include non-speech segments",
    )
    args = parser.parse_args()

    segments, scores = extract_speech_timestamps(
        audio=args.input,
        include_non_speech=args.include_non_speech,
        hop_size=args.hop_size,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration,
        with_scores=True,
    )

    audio_np, sr = load_audio(args.input, sr=16000, mono=True)
    audio_int16 = float32_to_int16(audio_np)

    output_dir = Path(args.output)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    speech_segments_json = output_dir / "speech_segments.json"
    speech_probs_json = output_dir / "speech_probs.json"

    with open(speech_segments_json, "w") as f:
        json.dump(
            {
                "count": len(segments),
                "segments": segments,
            },
            f,
            indent=2,
        )
    segments_url = f"file://{speech_segments_json.resolve()}"
    console.print(
        f"[green]{len(segments)} Segments saved to [link={segments_url}]{speech_segments_json}[/link][/green]"
    )

    with open(speech_probs_json, "w") as f:
        json.dump(scores, f, indent=2)
    probs_url = f"file://{speech_probs_json.resolve()}"
    console.print(
        f"[green]{len(scores)} probabilities saved to [link={probs_url}]{speech_probs_json}[/link][/green]"
    )

    console.print(
        f"\n[bold]Generating {len(segments)} per‑segment files for {len(segments)} speech segments...[/bold]"
    )
    for seg in segments:
        num = seg["num"]
        seg_subdir = segments_dir / f"segment_{num:03d}"
        seg_subdir.mkdir(exist_ok=True)

        start_frame = seg["frame_start"]
        end_frame = seg["frame_end"]
        hop_size = args.hop_size
        start_sample = start_frame * hop_size
        end_sample = (end_frame + 1) * hop_size
        segment_audio_int16 = audio_int16[start_sample:end_sample]

        wav_path = seg_subdir / "sound.wav"
        wavfile.write(wav_path, sr, segment_audio_int16)

        segment_probs = seg["segment_probs"]
        probs_path = seg_subdir / "speech_probs.json"
        save_json(segment_probs, probs_path)

        rms_values = compute_rms_per_frame(audio_np, hop_size, start_frame, end_frame)
        energies_path = seg_subdir / "energies.json"
        save_json(rms_values, energies_path)

        segment_json_path = seg_subdir / "segment.json"
        save_json(seg, segment_json_path)

        plot_path = seg_subdir / "speech_and_rms.png"
        save_plot(segment_probs, rms_values, plot_path, title=f"Segment {num:03d}")

        color = "green" if seg["type"] == "speech" else "red"
        type_label = seg["type"].replace("-", "_")
        console.print(
            f"  [cyan]✓[/cyan] segment_{num:03d}: [bold cyan]{len(segment_probs)}[/bold cyan] frames,  [bold cyan]{seg['duration']:.2f}[/bold cyan]s, avg_prob={seg['prob']:.3f}, avg_rms={np.mean(rms_values):.3f}, [{color}]{type_label}[/{color}]"
        )

    console.print(
        f"\n[bold green]All {len(segments)} per‑segment files saved under: {segments_dir}[/bold green]"
    )
