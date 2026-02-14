from __future__ import annotations

from typing import Generator, Optional, Tuple, Union, List
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.live import Live
from rich.text import Text
from tqdm import tqdm

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from utils.audio_utils import AudioInput, split_audio
from translators.utils import split_sentences_ja
from translators.translate_jp_en_opus import translate_japanese_to_english

console = Console()


def normalize_audio_to_16khz_mono_float32(
    audio: AudioInput,
    sr: Optional[int] = None,
    target_sr: int = 16000,
) -> tuple[npt.NDArray[np.float32], int]:
    # ── (unchanged — your original implementation) ──
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
        if np.max(np.abs(data)) > 1.5:
            data /= 32768.0

    elif isinstance(audio, torch.Tensor):
        data = audio.cpu().numpy()
        return normalize_audio_to_16khz_mono_float32(data, sr=sr, target_sr=target_sr)

    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

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
    # ── (unchanged — your original implementation) ──
    audio_np, sr = normalize_audio_to_16khz_mono_float32(
        audio, sr=sr, target_sr=16000
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
if __name__ == "__main__":
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    # State for sentence-level translation
    translated_sentences: List[str] = []
    ja_sentences_so_far: List[str] = []

    # Full translation cache — we'll keep the last full translation
    full_en_translation: str = ""
    last_full_ja_for_translation: str = ""

    with Live(console=console, refresh_per_second=3) as live:
        for segment, current_text in transcribe_progressive(
            model=model,
            audio=audio_path,
            language="ja",
            beam_size=5,
            show_progress=True,
            chunk_duration_s=15.0,
            overlap_s=3.0,
        ):
            ja_text = current_text.strip()

            # ── Sentence-level translation (incremental) ──
            new_ja_sents = split_sentences_ja(ja_text)

            if len(new_ja_sents) > len(ja_sentences_so_far):
                new_sentences = new_ja_sents[len(ja_sentences_so_far) :]
                for sent in new_sentences:
                    if sent.strip():
                        try:
                            en = translate_japanese_to_english(sent.strip())
                            translated_sentences.append(en.strip())
                        except Exception as e:
                            translated_sentences.append(f"[err: {str(e)}]")
                ja_sentences_so_far = new_ja_sents[:]

            # ── Full text translation (only when meaningfully changed) ──
            if len(ja_text) > len(last_full_ja_for_translation) + 8:
                try:
                    full_en_translation = translate_japanese_to_english(ja_text)
                except Exception as e:
                    full_en_translation = f"[full translation error: {str(e)}]"
                last_full_ja_for_translation = ja_text

            # ── Prepare display ──
            display_count = 5
            ja_to_show = ja_sentences_so_far[-display_count:]
            en_to_show = translated_sentences[-display_count:]

            # Align lengths
            max_lines = max(len(ja_to_show), len(en_to_show))
            ja_to_show = [""] * (max_lines - len(ja_to_show)) + ja_to_show
            en_to_show = [""] * (max_lines - len(en_to_show)) + en_to_show

            lines = []
            for ja, en in zip(ja_to_show, en_to_show):
                ja_line = ja.strip() or "…"
                en_line = en.strip() or "…"
                lines.append(f"  JA: {ja_line}")
                lines.append(f"  EN: {en_line}")
                lines.append("")

            sentences_block = "\n".join(lines).rstrip()

            if not sentences_block.strip():
                sentences_block = "[dim](まだ完全な文がありません)[/dim]"

            live.update(
                f"[bold green]Live Transcription — JA + EN[/bold green]  "
                f"[cyan]({len(ja_sentences_so_far)} 文)[/cyan]\n\n"
                f"{sentences_block}\n"
                f"{'─' * 60}\n"
                f"[dim]Latest segment: {segment.start:.1f}–{segment.end:.1f}s[/dim]   "
                f"[dim]Full JA:[/dim]\n{ja_text}\n"
                f"[dim]Full EN:[/dim]\n{full_en_translation}"
            )