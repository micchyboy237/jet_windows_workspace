import os
import time
import tempfile
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from rich.console import Console
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy
from typing import List, Literal, TypedDict
from audio_utils import convert_audio_to_tensor, load_audio

console = Console()


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


def _load_speechbrain_vad() -> VAD:
    """Lazily load the SpeechBrain CRDNN VAD model."""
    with console.status("[bold green]Loading SpeechBrain VAD model...[/bold green]"):
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir=Path("~/.cache/pretrained_models/vad-crdnn-libriparty").expanduser().resolve(),
            local_strategy=LocalStrategy.COPY,
        )
    console.print("✅ SpeechBrain VAD model ready")
    return vad


vad = _load_speechbrain_vad()


@torch.no_grad()
def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = 0.5,
    neg_threshold: float = 0.25,
    sampling_rate: int = 16000,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = False,
    time_resolution: int = 2,
    with_scores: bool = False,
    normalize_loudness: bool = False,
    include_non_speech: bool = False,
    large_chunk_size: int = 30,
    small_chunk_size: int = 10,
    double_check: bool = True,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using SpeechBrain VAD (vad-crdnn-libriparty).
    When include_non_speech=True, returns both speech and non-speech (silence) segments.
    """
    if max_speech_duration_sec is None:
        max_speech_duration_sec = 15.0

    if isinstance(audio, list) and all(isinstance(x, np.ndarray) for x in audio):
        audio = convert_audio_to_tensor(audio)

    audio_np = load_audio(
        audio,
        sr=sampling_rate,
        # normalize_loudness=normalize_loudness,
    )
    waveform = torch.from_numpy(audio_np).unsqueeze(0).clamp(-1.0, 1.0)

    # ── Create temporary WAV file (required by this SpeechBrain VAD implementation) ────────
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name).resolve()
            torchaudio.save(str(temp_path), waveform, sampling_rate)

        # Critical on Windows: give OS time to release write lock
        time.sleep(0.2)

        # === SPEECHBRAIN WINDOWS FIX ===
        # Force POSIX path (forward slashes) so SpeechBrain's internal
        # split_path/fetcher doesn't prepend the wrong drive/root again
        vad_input_path = temp_path.as_posix()

        with console.status("[bold blue]Running SpeechBrain VAD inference...[/bold blue]"):
            boundaries_sec = vad.get_speech_segments(
                vad_input_path,  # ← fixed
                large_chunk_size=large_chunk_size,
                small_chunk_size=small_chunk_size,
                activation_th=threshold,
                deactivation_th=neg_threshold,
                double_check=double_check,
            )

        boundaries_sec = boundaries_sec.view(-1).tolist()
        speech_pairs = list(zip(boundaries_sec[::2], boundaries_sec[1::2]))

        prob_tensor = vad.get_speech_prob_file(
            vad_input_path,  # ← fixed
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
        )
        probs = prob_tensor.squeeze().cpu().tolist()

        hop_samples = 160
        hop_sec = hop_samples / sampling_rate

        def make_segment(
            num: int,
            start_sec: float,
            end_sec: float,
            seg_type: Literal["speech", "non-speech"],
        ) -> SpeechSegment:
            start_sample = int(start_sec * sampling_rate)
            end_sample = int(end_sec * sampling_rate)
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
            temp_audio_path: str,
            sr: int,
            max_dur_sec: float,
            min_silence_sec: float = 0.45,
            win_sec: float = 0.30,
            energy_th_rel: float = 0.13,
        ) -> List[SpeechSegment]:
            """Split one long speech segment at the longest suitable silence gap."""
            start_s = float(seg["start"])
            end_s = float(seg["end"])
            duration = end_s - start_s

            if duration <= max_dur_sec:
                return [seg]
            # ... waveform loading unchanged ...1

            beg_sample = int(start_s * sr)
            num_frames = int(duration * sr)

            waveform, _ = torchaudio.load(
                temp_audio_path,
                frame_offset=beg_sample,
                num_frames=num_frames,
            )
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            waveform = waveform.squeeze(0)  # (time,)

            if waveform.numel() < 100:  # too short → no point splitting
                return [seg]

            win_samples = int(win_sec * sr)
            hop_samples = win_samples // 2

            # ── 1D short-time RMS ────────────────────────────────────────
            # Pad so last window is full
            pad_total = win_samples - hop_samples
            waveform_padded = F.pad(waveform, (0, pad_total))

            # Unfold to (num_windows, win_samples)
            windows = waveform_padded.unfold(0, win_samples, hop_samples)

            # RMS per window
            rms = torch.sqrt((windows**2).mean(dim=1) + 1e-10)

            if rms.numel() <= 1:
                return _force_split_segment(seg, max_dur_sec)

            max_rms = rms.max()
            norm_rms = rms / max_rms
            is_silence = norm_rms < energy_th_rel

            # Find longest consecutive silence region
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

            if silence_duration < min_silence_sec:
                return _force_split_segment(seg, max_dur_sec)

            # Split roughly in the middle of the silence region
            split_idx = best_start_idx + best_len // 2
            split_sample = beg_sample + split_idx * hop_samples
            split_sec = split_sample / sr

            # Create proper new segments with recalculated metadata
            left_seg = make_segment(
                num=seg["num"],  # temporary – will be renumbered later
                start_sec=start_s,
                end_sec=split_sec,
                seg_type=seg["type"],
            )
            right_seg = make_segment(
                num=seg["num"] + 1,  # temporary
                start_sec=split_sec,
                end_sec=end_s,
                seg_type=seg["type"],
            )

            # Recurse
            return split_long_speech_segment(
                left_seg,
                temp_audio_path,
                sr,
                max_dur_sec,
                min_silence_sec,
                win_sec,
                energy_th_rel,
            ) + split_long_speech_segment(
                right_seg,
                temp_audio_path,
                sr,
                max_dur_sec,
                min_silence_sec,
                win_sec,
                energy_th_rel,
            )

        def _force_split_segment(
            seg: SpeechSegment, max_dur_sec: float
        ) -> List[SpeechSegment]:
            """Fallback: hard split with small overlap when no good silence found."""
            parts = []
            cur_start = float(seg["start"])
            target_end = float(seg["end"])

            while cur_start + max_dur_sec <= target_end:  # <= to avoid tiny tail
                parts.append(
                    make_segment(
                        seg["num"],
                        cur_start,
                        round(cur_start + max_dur_sec, 3),
                        seg["type"],
                    )
                )
                cur_start += max_dur_sec - 0.25  # overlap
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
                    seg, temp_path, sampling_rate, max_speech_duration_sec
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

    except Exception as e:
        console.print(f"[red]Error during VAD inference: {str(e)}[/red]")
        raise

    finally:
        if temp_path is not None:
            time.sleep(0.2)  # Give time for file handles to release
            for _ in range(3):  # Retry cleanup
                try:
                    os.remove(temp_path)
                    break
                except PermissionError:
                    time.sleep(0.3)
                except FileNotFoundError:
                    break
            else:
                console.print("[yellow]Warning: Could not delete temp WAV file (still locked).[/yellow]")


if __name__ == "__main__":
    audio_file = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
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
            f"duration=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
            f"type=[{type_color}]{seg_type}[/{type_color}]"
        )
