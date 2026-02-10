from __future__ import annotations

import os
from pathlib import Path
from typing import Union, List, Tuple, Optional, cast

import numpy as np
import numpy.typing as npt
import torch
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, Word
from pydub import AudioSegment
from rich.console import Console
from rich.logging import RichHandler
import logging
import soundfile as sf  # lightweight fallback for some formats
from audio_utils import AudioInput, split_audio
from faster_whisper.transcribe import TranscriptionInfo  # Added for language info typing

# Setup rich logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("audio_transcriber")
console = Console()


def normalize_audio_to_float32_16khz_mono(
    audio: AudioInput,
    sr: int | None = None,
    target_sr: int = 16000,
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Load / convert any supported AudioInput to mono float32 numpy array at target_sr.
    Returns (audio_array, actual_sample_rate_used)
    """
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)
        try:
            # Prefer pydub for broad format support
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(target_sr).set_channels(1)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # pydub int16 -> float [-1,1]
            return samples, target_sr
        except Exception as e:
            logger.warning(f"pydub failed: {e}, trying soundfile fallback")
            data, sr = sf.read(path, dtype="float32")
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)  # to mono
            if sr != target_sr:
                from scipy.signal import resample
                data = resample(data, int(len(data) * target_sr / sr))
            return data.astype(np.float32), target_sr

    elif isinstance(audio, bytes):
        # Raw bytes - assume PCM 16-bit little-endian mono if no sr given
        if sr is None:
            raise ValueError("sr required when input is raw bytes")
        arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != target_sr:
            from scipy.signal import resample
            arr = resample(arr, int(len(arr) * target_sr / sr))
        return arr, target_sr

    elif isinstance(audio, np.ndarray):
        arr = audio
        if sr is None:
            raise ValueError("sr required when input is numpy array")
        if len(arr.shape) > 1:
            arr = np.mean(arr, axis=1)  # to mono
        arr = arr.astype(np.float32)
        if np.max(np.abs(arr)) > 1.0 + 1e-6:  # probably integer type
            arr = arr / np.iinfo(np.int16).max if arr.dtype.kind == "i" else arr
        if sr != target_sr:
            from scipy.signal import resample
            arr = resample(arr, int(len(arr) * target_sr / sr))
        return arr, target_sr

    elif isinstance(audio, torch.Tensor):
        arr = audio.cpu().numpy()
        return normalize_audio_to_float32_16khz_mono(
            arr, sr=sr, target_sr=target_sr
        )[0], target_sr

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")


def chunk_audio_numpy(
    audio: npt.NDArray[np.float32],
    chunk_length_sec: float = 25.0,
    overlap_sec: float = 4.0,
    sample_rate: int = 16000,
) -> List[Tuple[npt.NDArray[np.float32], float]]:
    """Split numpy audio array into overlapping chunks."""
    chunk_length_samples = int(chunk_length_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step_samples = chunk_length_samples - overlap_samples

    chunks = []
    start_sample = 0
    total_samples = len(audio)

    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_length_samples, total_samples)
        chunk = audio[start_sample:end_sample]

        # Pad last chunk if too short
        if len(chunk) < chunk_length_samples // 2:
            logger.warning(f"Last chunk short ({len(chunk)} samples), padding with zeros")
            padding = np.zeros(chunk_length_samples - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, padding])

        chunks.append((chunk, start_sample / sample_rate))  # start time in seconds
        start_sample += step_samples

    logger.info(f"Split audio into {len(chunks)} overlapping chunks")
    return chunks


def extract_context_prompt(
    previous_segments: List[Segment],
    max_words: int = 25,
) -> Optional[str]:
    """Extract last few words from previous transcription as context prompt."""
    if not previous_segments:
        return None

    words: List[str] = []
    for seg in reversed(previous_segments):
        if hasattr(seg, "words") and seg.words:
            for word in reversed(seg.words):
                words.append(word.word.strip())
                if len(words) >= max_words:
                    break
        else:
            # Fallback: split text
            seg_words = seg.text.strip().split()
            words.extend(reversed(seg_words))
        if len(words) >= max_words:
            break

    if not words:
        return None

    prompt = " ".join(reversed(words)).strip()
    if prompt:
        logger.debug(f"Context prompt (last ~{len(words)} words): {prompt[:80]}...")
    return prompt


def transcribe_chunk(
    model: WhisperModel,
    audio_chunk: npt.NDArray[np.float32],
    start_time_sec: float,
    initial_prompt: Optional[str] = None,
    language: Optional[str] = None,
    **transcribe_kwargs,
) -> Tuple[List[Segment], TranscriptionInfo]:
    """Transcribe one numpy audio chunk and offset timestamps."""
    segments, info = model.transcribe(
        audio_chunk,
        language=language,
        initial_prompt=initial_prompt,
        **transcribe_kwargs
    )

    # Offset timestamps to global time
    adjusted_segments = []
    for seg in segments:
        adjusted = Segment(
            id=seg.id,
            seek=seg.seek,
            start=seg.start + start_time_sec,
            end=seg.end + start_time_sec,
            text=seg.text,
            tokens=seg.tokens,
            temperature=seg.temperature,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
            words=[  # offset word timestamps too
                Word(
                    start=w.start + start_time_sec,
                    end=w.end + start_time_sec,
                    word=w.word,
                    probability=w.probability,
                )
                for w in (seg.words or [])
            ],
        )
        adjusted_segments.append(adjusted)

    return adjusted_segments, info


def merge_transcriptions(
    all_segments_lists: List[List[Segment]],
    overlap_sec: float = 4.0,
) -> List[Segment]:
    """Simple merge: keep earliest segment when overlapping."""
    merged: List[Segment] = []
    last_end = 0.0

    for segments in all_segments_lists:
        for seg in segments:
            if seg.end <= last_end:
                continue
            if seg.start < last_end:
                seg = Segment(
                    id=seg.id,
                    seek=seg.seek,
                    start=last_end,
                    end=seg.end,
                    text=seg.text,
                    tokens=seg.tokens,
                    temperature=seg.temperature,
                    avg_logprob=seg.avg_logprob,
                    compression_ratio=seg.compression_ratio,
                    no_speech_prob=seg.no_speech_prob,
                    words=[w for w in (seg.words or []) if w.start >= last_end],
                )
            merged.append(seg)
            last_end = max(last_end, seg.end)

    logger.info(f"Merged into {len(merged)} final segments")
    return merged


def transcribe_long_audio(
    audio: AudioInput,
    model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    chunk_length_sec: float = 25.0,
    overlap_sec: float = 4.0,
    sr: int | None = None,  # required for np/torch/bytes
    language: Optional[str] = None,
    **transcribe_options,
) -> Tuple[List[Segment], str]:
    """Main entry point: supports flexible AudioInput types."""
    model = WhisperModel(
        model,
        device=device,
        compute_type=compute_type,
        cpu_threads=0 if "cuda" in device else 8,
    )

    # Normalize input to 16kHz mono float32 numpy
    audio_np, sr = normalize_audio_to_float32_16khz_mono(
        audio,
        sr=sr,
        target_sr=16000,
    )

    chunks = chunk_audio_numpy(
        audio_np,
        chunk_length_sec=chunk_length_sec,
        overlap_sec=overlap_sec,
        sample_rate=sr,
    )

    all_segments: List[List[Segment]] = []
    previous_segments: List[Segment] = []
    context_prompt: Optional[str] = None
    
    # Track best detected language (when auto-detecting)
    best_lang: str = language if language is not None else "unknown"
    best_confidence: float = 1.0 if language is not None else 0.0

    for i, (chunk, start_sec) in enumerate(chunks, 1):
        logger.info(f"Transcribing chunk {i}/{len(chunks)} @ {start_sec:.1f}s")

        segments, info = transcribe_chunk(
            model,
            chunk,
            start_time_sec=start_sec,
            initial_prompt=context_prompt,
            language=language if i == 1 else best_lang,
            **transcribe_options
        )

        # Update best language if auto-detecting and this chunk is more confident
        if language is None and info.language_probability > best_confidence:
            best_confidence = info.language_probability
            best_lang = info.language
            logger.info(f"Updated best language: {best_lang} (conf: {best_confidence:.3f})")

        all_segments.append(segments)
        previous_segments = segments
        context_prompt = extract_context_prompt(previous_segments)

    merged_segments = merge_transcriptions(all_segments, overlap_sec=overlap_sec)
    merged_segments.sort(key=lambda s: s.start)

    final_lang = best_lang if language is None else language
    return merged_segments, final_lang


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    audio_file = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    # File path
    segments, lang = transcribe_long_audio(
        audio_file,
        model="kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
        vad_filter=False,
        beam_size=5,
        language="ja",
    )

    # Or numpy array (e.g. from librosa.load, torchaudio.load, etc.)
    # import librosa
    # y, sr = librosa.load("file.wav", sr=None, mono=True)
    # segments, lang = transcribe_long_audio(
    #     y,
    #     sr=sr,
    #     ...
    # )

    console.rule("Final Transcription")
    full_text = " ".join(s.text.strip() for s in segments)
    console.print(f"[bold cyan]Language:[/bold cyan] {lang}")
    console.print(full_text)