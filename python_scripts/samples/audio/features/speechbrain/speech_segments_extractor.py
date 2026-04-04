from __future__ import annotations
import contextlib
import os
from pathlib import Path
from typing import List, Literal, TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from rich.console import Console
from speechbrain.inference.VAD import VAD

import io
import librosa
import numpy.typing as npt

console = Console()

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/vad-crdnn-libriparty").expanduser().resolve()
)

TARGET_RMS_DBFS: float = -20.0      # Recommended for speech VAD/ASR (-18 to -23 range)
SAFETY_HEADROOM_DB: float = 0.5     # Prevent hard clipping (~ -0.5 dBFS max peak)


def _normalize_loudness(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """
    Best-practice loudness normalization for speech/VAD pipelines (zero extra deps).
    1. Peak normalize (prevent clipping).
    2. RMS gain to target dBFS (consistent perceived volume).
    3. Final safety limiter.
    Works reliably on short segments unlike full integrated LUFS.
    """
    if audio_np.size == 0:
        return audio_np.astype(np.float32)

    # 1. Peak normalization
    peak = np.max(np.abs(audio_np))
    if peak > 1e-8:
        audio_np = audio_np / peak

    # 2. RMS-based loudness normalization
    rms = np.sqrt(np.mean(audio_np.astype(np.float64)**2) + 1e-10)
    if rms > 1e-8:
        target_rms = 10 ** (TARGET_RMS_DBFS / 20.0)
        gain = target_rms / rms
        audio_np = audio_np * gain

    # 3. Safety peak limiter
    current_peak = np.max(np.abs(audio_np))
    if current_peak > (1.0 - 10 ** (-SAFETY_HEADROOM_DB / 20.0)):
        limiter = (1.0 - 10 ** (-SAFETY_HEADROOM_DB / 20.0)) / current_peak
        audio_np = audio_np * limiter

    return np.clip(audio_np, -1.0, 1.0).astype(np.float32)


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
    normalize_loudness: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Robust audio loader for ASR/VAD pipelines.

    Handles:
      - File paths
      - In-memory audio bytes (container OR raw PCM)
      - NumPy arrays
      - Torch tensors

    Returns:
        (audio: np.ndarray [samples], sr: int)
    """

    def _decode_raw_pcm(
        data: bytes,
        expected_sr: int,
        channels: int = 1,
        dtype: npt.DTypeLike = np.int16,
    ) -> tuple[np.ndarray, int]:
        """Decode raw PCM bytes into numpy array."""
        itemsize = np.dtype(dtype).itemsize

        if len(data) % (channels * itemsize) != 0:
            raise ValueError(
                f"Invalid raw PCM buffer: {len(data)} bytes not divisible by "
                f"(channels={channels} × itemsize={itemsize})"
            )

        arr = np.frombuffer(data, dtype=dtype)

        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)

        # Normalize if integer
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)

        return arr, expected_sr

    current_sr: int | None = None

    # ─────── Input handling ───────
    if isinstance(audio, (str, os.PathLike)):
        y, current_sr = librosa.load(audio, sr=None, mono=False)

    elif isinstance(audio, bytes):
        y = None
        # Attempt container decode (wav, flac, etc.)
        try:
            y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
        except Exception:
            # Fallback → raw PCM
            y, current_sr = _decode_raw_pcm(
                data=audio,
                expected_sr=sr,
                channels=1,
                dtype=np.int16,
            )

    elif isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = None

    elif isinstance(audio, torch.Tensor):
        y = audio.detach().float().cpu().numpy()
        current_sr = None

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # ─────── Normalize (safety) ───────
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    if y.size > 0:
        max_val = np.abs(y).max()
        if max_val > 1.0 + 1e-6:
            y = y / max_val

    # ─────── Ensure (channels, time) ───────
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
        # Force mono early if requested
        if mono and y.shape[0] > 1:
            y = np.mean(y, axis=0, keepdims=True)
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    # ─────── Sample rate handling ───────
    effective_sr = current_sr or sr

    # Resample ONLY if needed
    if effective_sr != sr:
        y = librosa.resample(y, orig_sr=effective_sr, target_sr=sr)
        effective_sr = sr
        # After resampling we may need to re-mono (rare)
        if mono and y.shape[0] > 1:
            y = np.mean(y, axis=0, keepdims=True)

    # ─────── Loudness Normalization (optional inside loader) ───────
    if normalize_loudness:
        y = _normalize_loudness(y.squeeze(), effective_sr)
        if y.ndim == 1:
            y = y[None, :]

    return y.squeeze().astype(np.float32), effective_sr


def _load_speechbrain_vad() -> VAD:
    """Lazily load the SpeechBrain CRDNN VAD model."""
    with console.status("[bold green]Loading SpeechBrain VAD model...[/bold green]"):
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir=SAVE_DIR,
        )
    console.print("✅ SpeechBrain VAD model ready")
    return vad


vad = _load_speechbrain_vad()


@contextlib.contextmanager
def _vad_tensor_context(waveform: torch.Tensor, sample_rate: int, audio_np: np.ndarray):
    r"""
    Context manager that forces SpeechBrain VAD to use in-memory data.
    Patches both torchaudio.load and speechbrain.dataio.audio_io.load.
    """
    original_load = torchaudio.load
    original_info = None
    original_audio_io_load = None
    try:
        from speechbrain.dataio import audio_io
        original_info = audio_io.info
        original_audio_io_load = getattr(audio_io, 'load', None)
    except Exception:
        pass

    def _tensor_load(*args, **kwargs):
        w = waveform.clone()
        if hasattr(vad, 'device') and vad.device is not None:
            w = w.to(vad.device)
        elif torch.cuda.is_available():
            w = w.to('cuda')
        return w, sample_rate

    def _tensor_info(_path):
        num_samples = waveform.numel() if waveform.dim() == 1 else waveform.shape[1]
        duration = num_samples / sample_rate
        return type(
            "AudioInfo",
            (),
            {
                "frames": num_samples,
                "num_frames": num_samples,
                "sample_rate": sample_rate,
                "samplerate": sample_rate,
                "duration": duration,
                "n_frames": num_samples,
                "length": num_samples,
            },
        )()

    def _tensor_audio_io_load(path, *args, **kwargs):
        tensor = torch.from_numpy(audio_np.copy()).float()
        if hasattr(vad, 'device') and vad.device is not None:
            tensor = tensor.to(vad.device)
        elif torch.cuda.is_available():
            tensor = tensor.to('cuda')

        # Robust shape handling — VAD expects [batch, time] (mono)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)          # [time] → [1, time]
        elif tensor.dim() == 3 and tensor.shape[0] == 1 and tensor.shape[2] == 1:
            tensor = tensor.squeeze(-1)           # rare [1, time, 1] → [1, time]
        elif tensor.dim() == 2 and tensor.shape[0] != 1:
            # stereo case — should not reach here because load_audio already made mono
            tensor = tensor.mean(dim=0, keepdim=True)

        return tensor, sample_rate

    # Apply patches
    torchaudio.load = _tensor_load
    if original_audio_io_load is not None:
        audio_io.load = _tensor_audio_io_load
    if original_info is not None:
        audio_io.info = _tensor_info

    try:
        yield
    finally:
        torchaudio.load = original_load
        if original_audio_io_load is not None:
            audio_io.load = original_audio_io_load
        if original_info is not None:
            audio_io.info = original_info


@torch.no_grad()
def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = 0.5,
    neg_threshold: float = 0.25,
    sampling_rate: int = 16000,
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = False,
    time_resolution: int = 2,
    with_scores: bool = False,
    normalize_loudness: bool = False,
    include_non_speech: bool = False,
    large_chunk_size: int = 30,
    small_chunk_size: int = 10,
    double_check: bool = True,
    apply_energy_VAD: bool = False,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using SpeechBrain VAD (vad-crdnn-libriparty).
    When include_non_speech=True, returns both speech and non-speech (silence) segments.
    """

    if max_speech_duration_sec is None:
        max_speech_duration_sec = 15.0

    audio_np, sr = load_audio(
        audio,
        sr=sampling_rate,
        normalize_loudness=normalize_loudness,
    )

    # ─────── Loudness Normalization (applied here - best location) ───────
    if normalize_loudness:
        with console.status("[bold yellow]Normalizing loudness...[/bold yellow]"):
            audio_np = _normalize_loudness(audio_np, sr)

    # Shape: (1, time) for VAD compatibility
    waveform = torch.from_numpy(audio_np).unsqueeze(0).clamp(-1.0, 1.0)

    # Use in-memory tensor context → no temp file, no Windows path bug
    with _vad_tensor_context(waveform, sr, audio_np):
        with console.status(
            "[bold blue]Running SpeechBrain VAD inference...[/bold blue]"
        ):
            boundaries_sec = vad.get_speech_segments(
                "dummy.wav",  # Path is ignored inside the context
                large_chunk_size=large_chunk_size,
                small_chunk_size=small_chunk_size,
                activation_th=threshold,
                deactivation_th=neg_threshold,
                double_check=double_check,
                apply_energy_VAD=apply_energy_VAD,
                close_th=min_silence_duration_sec,
                len_th=min_speech_duration_sec,
            )

        boundaries_sec = boundaries_sec.view(-1).tolist()
        speech_pairs = list(zip(boundaries_sec[::2], boundaries_sec[1::2]))

        prob_tensor = vad.get_speech_prob_file(
            "dummy.wav",  # Path is ignored inside the context
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
        )
        probs = prob_tensor.squeeze().cpu().tolist()

        hop_samples = 160
        hop_sec = hop_samples / sr

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

        def split_long_speech_segment(
            seg: SpeechSegment,
            temp_audio_path: str,  # kept for signature compatibility (unused now)
            sr: int,
            max_dur_sec: float,
            min_silence_sec: float = 0.45,
            win_sec: float = 0.30,
            energy_th_rel: float = 0.13,
        ) -> List[SpeechSegment]:
            """
            Iteratively split long speech segments.
            Avoids recursion to prevent maximum recursion depth errors.
            """
            segments_to_process = [seg]
            final_segments: List[SpeechSegment] = []

            while segments_to_process:
                current = segments_to_process.pop(0)
                start_s = float(current["start"])
                end_s = float(current["end"])
                duration = end_s - start_s

                if duration <= max_dur_sec:
                    final_segments.append(current)
                    continue

                beg_sample = int(start_s * sr)
                num_frames = int(duration * sr)

                # Load sub-segment from original waveform (in-memory)
                sub_waveform = waveform[:, beg_sample : beg_sample + num_frames]

                if sub_waveform.shape[0] > 1:
                    sub_waveform = sub_waveform.mean(0, keepdim=True)
                sub_waveform = sub_waveform.squeeze(0)

                if sub_waveform.numel() < 100:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                win_samples = int(win_sec * sr)
                hop_samples = win_samples // 2
                pad_total = win_samples - hop_samples
                waveform_padded = F.pad(sub_waveform, (0, pad_total))
                windows = waveform_padded.unfold(0, win_samples, hop_samples)

                rms = torch.sqrt((windows**2).mean(dim=1) + 1e-10)

                if rms.numel() <= 1:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                norm_rms = rms / (rms.max() + 1e-10)
                is_silence = norm_rms < energy_th_rel

                best_len = 0
                best_start_idx = -1
                current_len = 0
                for i, silent in enumerate(is_silence.tolist() + [False]):
                    if silent:
                        current_len += 1
                    else:
                        if current_len > best_len:
                            best_len = current_len
                            best_start_idx = i - current_len
                        current_len = 0

                silence_duration = best_len * (hop_samples / sr)

                if silence_duration < min_silence_sec or best_start_idx < 0:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                split_idx = best_start_idx + best_len // 2
                split_sample = beg_sample + split_idx * hop_samples
                split_sec = split_sample / sr

                if abs(split_sec - start_s) < 0.05 or abs(end_s - split_sec) < 0.05:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                left_seg = make_segment(
                    current["num"], start_s, split_sec, current["type"]
                )
                right_seg = make_segment(
                    current["num"] + 1, split_sec, end_s, current["type"]
                )

                segments_to_process.append(left_seg)
                segments_to_process.append(right_seg)

            return final_segments

        def _force_split_segment(
            seg: SpeechSegment, max_dur_sec: float
        ) -> List[SpeechSegment]:
            """Fallback: hard split with small overlap when no good silence found."""
            parts = []
            cur_start = float(seg["start"])
            target_end = float(seg["end"])

            while cur_start + max_dur_sec <= target_end:
                parts.append(
                    make_segment(
                        seg["num"],
                        cur_start,
                        round(cur_start + max_dur_sec, 3),
                        seg["type"],
                    )
                )
                cur_start += max_dur_sec - 0.25
            parts.append(
                make_segment(seg["num"], round(cur_start, 3), target_end, seg["type"])
            )
            return parts

        # Build initial segments list
        enhanced: List[SpeechSegment] = []
        current_time = 0.0
        seg_num = 1

        if include_non_speech and speech_pairs:
            first_start = speech_pairs[0][0]
            if first_start > 0.001:
                enhanced.append(make_segment(seg_num, 0.0, first_start, "non-speech"))
                seg_num += 1
            current_time = first_start

        for start_sec, end_sec in speech_pairs:
            if include_non_speech and (start_sec > current_time + 0.01):
                enhanced.append(
                    make_segment(seg_num, current_time, start_sec, "non-speech")
                )
                seg_num += 1

            enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
            seg_num += 1
            current_time = end_sec

        # Split long speech segments if requested
        if max_speech_duration_sec < float("inf"):
            final_segments: List[SpeechSegment] = []
            current_num = 1
            for seg in enhanced:
                if seg["type"] != "speech":
                    seg["num"] = current_num
                    final_segments.append(seg)
                    current_num += 1
                    continue

                sub_segments = split_long_speech_segment(
                    seg, "dummy", sr, max_speech_duration_sec  # temp_audio_path unused
                )
                for sub in sub_segments:
                    sub["num"] = current_num
                    final_segments.append(sub)
                    current_num += 1
            enhanced = final_segments

        if include_non_speech:
            total_duration = len(probs) * hop_sec
            if current_time < total_duration - 0.01:
                enhanced.append(
                    make_segment(seg_num, current_time, total_duration, "non-speech")
                )
                seg_num += 1

        if with_scores:
            return enhanced, probs
        return enhanced



if __name__ == "__main__":
    import argparse

    DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\video\generated\extract_video_segment_short\video_segment.mp4"

    parser = argparse.ArgumentParser(description="Extract speech timestamps from audio")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=DEFAULT_AUDIO_PATH,
        help=f"Path to audio file (default: {DEFAULT_AUDIO_PATH})",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Speech threshold (default: 0.5)",
    )
    parser.add_argument(
        "-n",
        "--neg-threshold",
        type=float,
        default=0.25,
        help="Negative noise threshold (default: 0.25)",
    )
    parser.add_argument(
        "-m",
        "--max-speech-duration-sec",
        type=float,
        default=8.0,
        help="Maximum speech segment duration in seconds (default: 8.0)",
    )
    parser.add_argument(
        "-s",
        "--return-seconds",
        action="store_true",
        default=True,
        help="Return timestamps in seconds (default: True)",
    )
    parser.add_argument(
        "-r",
        "--time-resolution",
        type=int,
        default=2,
        help="Time resolution in ms (default: 2)",
    )
    parser.add_argument(
        "-w",
        "--with-scores",
        action="store_true",
        default=True,
        help="Return probability scores along with segments (default: True)",
    )

    args = parser.parse_args()

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(args.audio_file).name}")
    segments, speech_probs = extract_speech_timestamps(
        args.audio_file,
        threshold=args.threshold,
        neg_threshold=args.neg_threshold,
        max_speech_duration_sec=args.max_speech_duration_sec,
        return_seconds=args.return_seconds,
        time_resolution=args.time_resolution,
        with_scores=args.with_scores,
        include_non_speech=True,
        normalize_loudness=True,
    )
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] "
            f"[bold white]{seg['start']:.2f}[/bold white] - "
            f"[bold white]{seg['end']:.2f}[/bold white] "
            f"[yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[bold {'green' if seg['type']=='speech' else 'red'}]{seg['type']}[/bold {'green' if seg['type']=='speech' else 'red'}]"
        )
