import io
import os
from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import librosa
import numpy as np
import numpy.typing as npt
import torch
from fireredvad.core.constants import SAMPLE_RATE
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from rich.console import Console

# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]


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
    # ─────── FIX 1: In-memory arrays/tensors have unknown original sr ───────

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

    if len(y) > 0 and np.abs(y).max() > 1.0 + 1e-6:
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

    sr = current_sr or sr

    # ─────── FIX 4: ALWAYS resample if current_sr is None or wrong ───────
    if current_sr != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr)

    return y.squeeze(), sr


# assuming SAVE_DIR is defined somewhere; adjust if needed

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

console = Console()


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = 0.65,
        min_silence_duration_sec: float = 0.20,  # 200 ms
        min_speech_duration_sec: float = 0.15,  # 150 ms
        max_speech_duration_sec: float = 12.0,
        smooth_window_size: int = 5,
        pad_start_frame: int = 5,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        print(
            f"Loading FireRedVAD (streaming) on {self.device}... ", end="", flush=True
        )

        # Convert durations → frame counts (100 frames = 1 second)
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
        print("done.")

        self.sample_rate = SAMPLE_RATE  # 16000
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0

        # Minimal look-back — just enough for model right-context + smoothing
        self.max_buffer_samples = int(1.2 * self.sample_rate)  # ~1.2 seconds max

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic range compression / gain normalization"""
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

        # Normalize level (helps a lot with real mic input)
        chunk = self._normalize_chunk(chunk)

        # Append new audio
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
        (useful for debugging or when you need is_speech_start / is_speech_end)
        """
        # This would require keeping the last result object — omitted for simplicity
        # You can extend the class to store self.last_result = results[-1] if needed
        return None

    # Optional: full-file processing (unchanged from original)
    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        self.reset()
        frame_results, result = self.vad.detect_full(audio)
        return frame_results, result


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


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]


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
    # Convert input audio to numpy array
    audio_np, sr = load_audio(
        audio,
        sr=16000,  # FireRedVAD expects 16000 Hz
        mono=True,
    )
    if sr != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sr}")

    # Initialize FireRedVAD
    vad = FireRedVAD(
        model_dir=SAVE_DIR,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
    )

    # Run VAD inference
    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    # Extract timestamps
    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]
    hop_sec = 0.010  # FireRedVAD frame shift (10ms)

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
        avg_prob = np.mean(segment_probs_slice) if segment_probs_slice else 0.0
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

    # Handle initial non-speech segment
    if include_non_speech and timestamps and timestamps[0][0] > 0.001:
        enhanced.append(make_segment(seg_num, 0.0, timestamps[0][0], "non-speech"))
        seg_num += 1
        current_time = timestamps[0][0]

    # Process speech segments
    for start_sec, end_sec in timestamps:
        if include_non_speech and start_sec > current_time + 0.01:
            enhanced.append(
                make_segment(seg_num, current_time, start_sec, "non-speech")
            )
            seg_num += 1
        enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
        seg_num += 1
        current_time = end_sec

    # Handle final non-speech segment
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

    audio_np, sr = load_audio(
        audio=audio,
        sr=sampling_rate,
        mono=True,
    )
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
        segment_audio = segment_audio.astype(np.float32, copy=False)
        speech_audio_chunks.append(segment_audio)

    return speech_audio_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav",
        help="Path to audio file (default: demo file)",
    )
    args = parser.parse_args()
    audio_file = args.audio_file

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments, all_probs = extract_speech_timestamps(
        audio_file,
        max_speech_duration_sec=8.0,
        return_seconds=True,
        time_resolution=2,
        with_scores=True,
        include_non_speech=True,
    )
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        seg_type = seg["type"]
        if seg_type == "speech":
            type_color = "bold green"
        else:
            type_color = "bold red"
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
        )
