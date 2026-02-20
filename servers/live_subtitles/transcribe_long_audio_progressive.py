from __future__ import annotations

from typing import Generator, Optional, Tuple, Union
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from rich.console import Console
from rich.live import Live
from rich.text import Text
from tqdm import tqdm

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

# from translators.translate_llm import translate_text
from utils import split_sentences_ja
from audio_utils import AudioInput, split_audio

console = Console()


def normalize_audio_to_16khz_mono_float32(
    audio: AudioInput,
    sr: Optional[int] = None,
    target_sr: int = 16000,
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Convert any supported AudioInput to mono float32 numpy array at 16 kHz.
    Returns (normalized_audio, used_sample_rate)
    """
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
        except Exception:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(target_sr).set_channels(1)
            data = np.array(seg.get_array_of_samples(), dtype=np.float32)
            data /= 32768.0
            sr = target_sr

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

    elif isinstance(audio, bytes):
        if sr is None:
            raise ValueError("sr required for raw bytes input")
        data = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        sr = sr

    elif isinstance(audio, np.ndarray):
        data = audio
        if sr is None:
            raise ValueError("sr required for numpy array input")
        sr = sr
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)
        # Normalize if it looks like integer PCM
        if np.max(np.abs(data)) > 1.5:
            data /= 32768.0

    elif isinstance(audio, torch.Tensor):
        data = audio.cpu().numpy()
        return normalize_audio_to_16khz_mono_float32(
            data, sr=sr, target_sr=target_sr
        )

    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(data) * target_sr / sr)
        data = resample(data, num_samples)

    return data.astype(np.float32), target_sr


def transcribe_progressive(
    model: WhisperModel,
    audio: AudioInput,
    sr: Optional[int] = None,
    language: Optional[str] = None,
    task: str = "transcribe",
    vad_filter: bool = True,
    condition_on_previous_text: bool = True,
    beam_size: int = 5,
    word_timestamps: bool = False,
    show_progress: bool = False,
    chunk_duration_s: float = 15.0,
    overlap_s: float = 3.0,
    **transcribe_kwargs,
) -> Generator[Tuple[Segment, str], None, None]:
    """
    Progressive transcription using overlapping audio chunks.
    Supports flexible AudioInput types.

    Yields:
        (latest_segment, current_full_transcript_text)
    """
    # Normalize audio once at the beginning
    audio_np, sr = normalize_audio_to_16khz_mono_float32(
        audio,
        sr=sr,
        target_sr=16000,
    )

    full_text_parts: list[str] = []
    last_printed_length = 0

    audio_chunks = list(
        split_audio(
            audio=audio_np,
            sr=sr,
            chunk_duration_s=chunk_duration_s,
            overlap_s=overlap_s,
        )
    )

    with tqdm(
        total=len(audio_chunks),
        desc="Transcribing chunks...",
        unit="chunk",
        disable=not show_progress,
    ) as pbar:

        for chunk_audio, chunk_start_time in audio_chunks:
            segments, _ = model.transcribe(
                chunk_audio,
                language=language,
                task=task,
                vad_filter=vad_filter,
                condition_on_previous_text=condition_on_previous_text,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                **transcribe_kwargs,
            )

            for segment in segments:
                # Offset timestamps to global time
                segment.start += chunk_start_time
                segment.end += chunk_start_time

                full_text_parts.append(segment.text.strip())
                current_text = " ".join(full_text_parts).strip()

                yield segment, current_text

                if len(current_text) > last_printed_length:
                    console.print(
                        Text.from_markup(
                            f"[dim]{segment.start:5.1f}s → {segment.end:5.1f}s[/dim]  "
                            f"[bold]{segment.text.strip()}[/bold]"
                        )
                    )
                    last_printed_length = len(current_text)

            pbar.update(1)


# ──────────────────────────────────────────────────────────────────────────────
#   Usage example
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Option 1: file path
    audio_input: AudioInput = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    # Option 2: numpy array (uncomment to test)
    # from scipy.io import wavfile
    # sr, audio_raw = wavfile.read(audio_input)
    # if len(audio_raw.shape) > 1:
    #     audio_raw = audio_raw.mean(axis=1)
    # audio_input = audio_raw.astype(np.float32) / 32768.0
    # original_sr = sr

    # Load model once
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    with Live(console=console, refresh_per_second=4) as live:
        full_transcript = ""

        for segment, current_text in transcribe_progressive(
            model=model,
            audio=audio_input,
            # sr=44100,  # only needed if passing raw numpy/bytes
            language="ja",
            beam_size=5,
            show_progress=True,
            chunk_duration_s=15.0,
            overlap_s=3.0,
        ):
            ja_text = current_text.strip()
            ja_sents = split_sentences_ja(ja_text)

            if ja_sents:
                sent_count = len(ja_sents)
                display_sents = ja_sents[-3:] if sent_count > 3 else ja_sents
                if sent_count > 3:
                    display_sents = ["…"] + display_sents
                sentences_display = "\n".join(
                    f"  {i+1}. {sent}" for i, sent in enumerate(display_sents)
                )
            else:
                sent_count = 0
                sentences_display = "[dim](no complete sentences yet)[/dim]"

            live.update(
                f"[bold green]Live Japanese Transcript[/bold green]  "
                f"[cyan]({sent_count} sentence{'s' if sent_count != 1 else ''})[/cyan]\n\n"
                f"{sentences_display}\n\n"
                f"[dim]Latest segment: {segment.start:.1f}–{segment.end:.1f}s[/dim]\n"
                f"[dim]Raw accumulating text:[/dim] {ja_text}"
            )

# ──────────────────────────────────────────────────────────────────────────────
#   Usage example
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

    # Load model once (reuse for multiple calls)
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )
    chunk_duration_s: float = 15.0
    overlap_s: float = 3.0

    # Example: load your 20-second (or longer) audio
    from scipy.io import wavfile
    sr, audio = wavfile.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)          # to mono
    audio = audio.astype(np.float32) / 32768.0

    # Real-time-like progressive display
    with Live(console=console, refresh_per_second=4) as live:
        full_transcript = ""

        for segment, current_text in transcribe_progressive(
            model,
            audio,
            sr=sr,
            language="ja",
            beam_size=5,
            show_progress=True,
        ):
            ja_text = current_text.strip()
            ja_sents = split_sentences_ja(ja_text)

            # Build nicely formatted sentence display
            if ja_sents:
                sent_count = len(ja_sents)
                # Show numbered sentences (last few for readability)
                display_sents = []
                if sent_count <= 3:
                    display_sents = ja_sents
                else:
                    # Show last 3 sentences + "..." if there are more
                    display_sents = ["…"] + ja_sents[-3:] if sent_count > 3 else ja_sents

                sentences_display = "\n".join(
                    f"  {i+1}. {sent}" for i, sent in enumerate(display_sents)
                )
            else:
                sent_count = 0
                sentences_display = "[dim](no complete sentences yet)[/dim]"

            live.update(
                f"[bold green]Live Japanese Transcript[/bold green]  "
                f"[cyan]({sent_count} sentence{'s' if sent_count != 1 else ''})[/cyan]\n\n"
                f"{sentences_display}\n\n"
                f"[dim]Latest segment: {segment.start:.1f}–{segment.end:.1f}s[/dim]\n"
                f"[dim]Raw accumulating text:[/dim] {ja_text}"
            )
