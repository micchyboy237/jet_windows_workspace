# python_scripts\helpers\audio\transcribe_long_audio_progressive_funasr.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple, Union, List, Dict, Any

import numpy as np
import numpy.typing as npt
import scipy.io.wavfile as wavfile
from rich.console import Console
from rich.live import Live
from rich.text import Text
from tqdm import tqdm

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utils.audio_utils import AudioInput, split_audio
from translators.utils import split_sentences_ja
from translators.translate_jp_en_opus import translate_japanese_to_english

console = Console()


def normalize_audio_to_16khz_mono_float32(
    audio: AudioInput,
    sr: Optional[int] = None,
    target_sr: int = 16000,
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Convert any audio input to 16 kHz mono float32 numpy array.
    Supports file path, bytes, numpy array, or torch Tensor.
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
    elif isinstance(audio, np.ndarray):
        data = audio
        if sr is None:
            raise ValueError("sr required for numpy array input")
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


def transcribe_progressive_funasr(
    model: AutoModel,
    audio: AudioInput,
    sr: Optional[int] = None,
    language: str = "ja",
    chunk_duration_s: float = 15.0,
    overlap_s: float = 3.0,
    show_progress: bool = False,
    **generate_kwargs,
) -> Generator[Tuple[Dict[str, Any], str], None, None]:
    """
    Transcribe a long audio file progressively using FunASR.

    Args:
        model: Loaded FunASR AutoModel instance.
        audio: Audio input (file path, bytes, numpy array).
        sr: Sample rate (required for raw bytes/numpy without sr).
        language: Language code (default "ja").
        chunk_duration_s: Length of each audio chunk in seconds.
        overlap_s: Overlap between consecutive chunks in seconds.
        show_progress: Whether to display a tqdm progress bar.
        **generate_kwargs: Additional arguments passed to model.generate().

    Yields:
        Tuple of (segment_dict, accumulated_full_text)
        segment_dict contains keys: "start", "end", "text"
    """
    # Normalise audio to 16 kHz mono float32
    audio_np, sr = normalize_audio_to_16khz_mono_float32(
        audio, sr=sr, target_sr=16000
    )

    full_text_parts: List[str] = []
    last_printed_length = 0

    # Split into overlapping chunks
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
            # Write chunk to temporary WAV file (FunASR works reliably with file paths)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                wavfile.write(str(tmp_path), sr, chunk_audio.astype(np.float32))

            try:
                # Default parameters for Japanese ASR with timestamps
                gen_kwargs = {
                    "language": language,
                    "use_itn": True,
                    "batch_size": 1,
                    "output_timestamp": True,
                    "merge_vad": False,
                }
                gen_kwargs.update(generate_kwargs)

                results = model.generate(input=str(tmp_path), **gen_kwargs)

                if results and len(results) > 0:
                    result = results[0]
                    raw_text = result.get("text", "").strip()
                    text = rich_transcription_postprocess(raw_text)
                    timestamps = result.get("timestamp", [])

                    if timestamps:
                        # Each timestamp entry is [start_ms, end_ms]
                        for idx, ts in enumerate(timestamps):
                            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                                start_ms, end_ms = ts[0], ts[1]
                                start_sec = (start_ms / 1000.0) + chunk_start_time
                                end_sec = (end_ms / 1000.0) + chunk_start_time
                                segment_text = text  # whole chunk text

                                segment = {
                                    "start": round(start_sec, 3),
                                    "end": round(end_sec, 3),
                                    "text": segment_text,
                                }
                                full_text_parts.append(segment_text)
                                current_text = "".join(full_text_parts).strip()
                                yield segment, current_text

                                if len(current_text) > last_printed_length:
                                    console.print(
                                        Text.from_markup(
                                            f"[dim]{segment['start']:5.1f}s → {segment['end']:5.1f}s[/dim]  "
                                            f"[bold]{segment_text}[/bold]"
                                        )
                                    )
                                    last_printed_length = len(current_text)
                    else:
                        # No timestamps – treat the whole chunk as one segment
                        segment = {
                            "start": chunk_start_time,
                            "end": chunk_start_time + chunk_duration_s,
                            "text": text,
                        }
                        full_text_parts.append(text)
                        current_text = "".join(full_text_parts).strip()
                        yield segment, current_text

                        if len(current_text) > last_printed_length:
                            console.print(
                                Text.from_markup(
                                    f"[dim]{segment['start']:5.1f}s → {segment['end']:5.1f}s[/dim]  "
                                    f"[bold]{text}[/bold]"
                                )
                            )
                            last_printed_length = len(current_text)
            finally:
                # Clean up temporary WAV file
                tmp_path.unlink(missing_ok=True)

            pbar.update(1)


if __name__ == "__main__":
    # Example usage: progressive transcription with live translation display
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    # Load FunASR model once
    model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        disable_update=True,
        device="cuda:0",  # or "cpu"
        hub="hf",
    )

    translated_sentences: List[str] = []
    ja_sentences_so_far: List[str] = []
    full_en_translation: str = ""
    last_full_ja_for_translation: str = ""

    with Live(console=console, refresh_per_second=3) as live:
        for segment, current_text in transcribe_progressive_funasr(
            model=model,
            audio=audio_path,
            language="ja",
            chunk_duration_s=15.0,
            overlap_s=3.0,
            show_progress=True,
        ):
            ja_text = current_text.strip()

            # Split into sentences and translate new ones
            new_ja_sents = split_sentences_ja(ja_text)
            if len(new_ja_sents) > len(ja_sentences_so_far):
                new_sentences = new_ja_sents[len(ja_sentences_so_far):]
                for sent in new_sentences:
                    if sent.strip():
                        try:
                            en = translate_japanese_to_english(sent.strip())
                            translated_sentences.append(en.strip())
                        except Exception as e:
                            translated_sentences.append(f"[err: {str(e)}]")
                ja_sentences_so_far = new_ja_sents[:]

            # Occasionally translate the whole accumulated text
            if len(ja_text) > len(last_full_ja_for_translation) + 8:
                try:
                    full_en_translation = translate_japanese_to_english(ja_text)
                except Exception as e:
                    full_en_translation = f"[full translation error: {str(e)}]"
                last_full_ja_for_translation = ja_text

            # Prepare live display
            display_count = 5
            ja_to_show = ja_sentences_so_far[-display_count:]
            en_to_show = translated_sentences[-display_count:]
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
                f"[bold green]Live Transcription — JA + EN (FunASR)[/bold green]  "
                f"[cyan]({len(ja_sentences_so_far)} 文)[/cyan]\n\n"
                f"{sentences_block}\n"
                f"{'─' * 60}\n"
                f"[dim]Latest segment: {segment['start']:.1f}–{segment['end']:.1f}s[/dim]   "
                f"[dim]Full JA:[/dim]\n{ja_text}\n"
                f"[dim]Full EN:[/dim]\n{full_en_translation}"
            )
