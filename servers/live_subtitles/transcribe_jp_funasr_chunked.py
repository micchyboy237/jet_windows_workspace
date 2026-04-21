from __future__ import annotations

import re
import tempfile
import torch
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import scipy.io.wavfile as wavfile
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from sentence_utils import SYMBOL_RANGE, split_sentences_ja
from ja_punctuator import add_punctuation

TimestampPair = Tuple[int, int]


class WordSegment(TypedDict):
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    word: str


class PhraseSegment(TypedDict):
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    phrase: str
    word_segments: List[WordSegment]


class TranscriptionMetadata(TypedDict, total=False):
    processing_duration_sec: float
    model: str
    audio_duration_sec: float
    transcribed_duration_sec: float
    transcribed_duration_pctg: float
    coverage_label: str
    num_chunks: Optional[int]


class TranscriptionResult(TypedDict):
    text_ja: str
    confidence: float
    quality_label: str
    avg_logprob: Optional[float]
    word_segments: list[WordSegment]
    phrase_segments: list[PhraseSegment]
    metadata: TranscriptionMetadata


# =========================
# MODEL INITIALIZATION
# =========================

model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    disable_update=True,
    device="cuda:0",
    hub="hf",
)


# =========================
# CONFIG FOR LONG-FORM CHUNKING (faster-whisper style)
# =========================

CHUNK_LENGTH_SEC = 30.0      # Target chunk size (matches Whisper default)
OVERLAP_SEC = 3.0            # Small overlap to reduce word cuts at boundaries (set to 0.0 for pure non-overlap)
SAMPLING_RATE = 16000        # SenseVoice expects 16kHz


# =========================
# NORMALIZATION HELPERS (unchanged)
# =========================

SYMBOL_PATTERN = re.compile(rf"[{SYMBOL_RANGE}]+")

MAX_WORD_DURATION_SEC = 1.0
MIN_WORD_DURATION_SEC = 0.02
PUNCTUATION_SET = set("。、！？.,!?")


def _postprocess_word_timing(
    word: Optional[str],
    start_sec: Optional[float],
    end_sec: Optional[float],
    duration_sec: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if word and word in PUNCTUATION_SET:
        duration_sec = MIN_WORD_DURATION_SEC
        if start_sec is not None:
            end_sec = start_sec + duration_sec

    if duration_sec is not None:
        if duration_sec > MAX_WORD_DURATION_SEC:
            duration_sec = MAX_WORD_DURATION_SEC
            if start_sec is not None:
                end_sec = start_sec + duration_sec
        elif duration_sec < MIN_WORD_DURATION_SEC:
            duration_sec = MIN_WORD_DURATION_SEC
            if start_sec is not None:
                end_sec = start_sec + duration_sec

    return start_sec, end_sec, duration_sec


def _normalize_phrase(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = SYMBOL_PATTERN.sub("", text)
    return text


def _normalize_word(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


# =========================
# ALIGNMENT CORE (unchanged)
# =========================

def _build_phrase_segments(
    phrases: List[str],
    segments: List[WordSegment],
) -> List[PhraseSegment]:
    phrase_segments: List[PhraseSegment] = []

    seg_idx = 0
    total_segments = len(segments)

    for p_idx, phrase in enumerate(phrases):
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase:
            continue

        normalized_phrase = _normalize_phrase(cleaned_phrase)
        phrase_len = len(normalized_phrase)

        if phrase_len == 0:
            continue

        collected: List[WordSegment] = []
        matched_chars = ""

        while seg_idx < total_segments and len(matched_chars) < phrase_len:
            seg = segments[seg_idx]
            word = seg.get("word") or ""

            normalized_word = _normalize_word(word)

            matched_chars += normalized_word
            collected.append(seg)

            seg_idx += 1

        # Trim overflow
        if len(matched_chars) > phrase_len:
            overflow = len(matched_chars) - phrase_len
            while overflow > 0 and collected:
                last = collected[-1]
                last_word = _normalize_word(last.get("word") or "")
                if len(last_word) <= overflow:
                    overflow -= len(last_word)
                    collected.pop()
                else:
                    break

        if collected:
            start_sec = collected[0]["start_sec"]
            end_sec = collected[-1]["end_sec"]
            duration_sec = (end_sec - start_sec) if start_sec is not None and end_sec is not None else 0.0

            phrase_segments.append(
                {
                    "index": p_idx,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": round(duration_sec, 3),
                    "phrase": cleaned_phrase,
                    "word_segments": collected,
                }
            )

    return phrase_segments


# =========================
# HELPERS (unchanged)
# =========================

def calculate_transcribed_duration_percentage(
    segments: list[WordSegment], total_audio_duration_seconds: float
) -> float:
    if total_audio_duration_seconds <= 0:
        return 0.0
    total_transcribed_sec = sum(seg.get("duration_sec") or 0.0 for seg in segments)
    return (total_transcribed_sec / total_audio_duration_seconds) * 100


def get_audio_duration_seconds(audio_path: Path) -> float:
    sample_rate, data = wavfile.read(str(audio_path))
    return len(data) / sample_rate


def get_coverage_quality_label(pct: float) -> str:
    if pct >= 92.0: return "excellent (very clean)"
    if pct >= 82.0: return "good"
    if pct >= 65.0: return "fair (some noise/BGM)"
    if pct >= 40.0: return "sparse speech"
    if pct >= 15.0: return "very sparse"
    return "almost no speech"


# =========================
# CHUNKING LOGIC (faster-whisper inspired)
# =========================

def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int = SAMPLING_RATE,
    chunk_length_sec: float = CHUNK_LENGTH_SEC,
    overlap_sec: float = OVERLAP_SEC,
) -> List[Tuple[np.ndarray, float]]:
    """Split waveform into chunks with optional overlap."""
    if len(audio) == 0:
        return []

    chunk_samples = int(chunk_length_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step = max(chunk_samples - overlap_samples, chunk_samples // 2)  # ensure reasonable step

    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        # Skip tiny trailing chunks
        if len(chunk) < sample_rate * 2:  # < 2 seconds
            break

        chunks.append((chunk, start / sample_rate))
        start += step

    return chunks


def _transcribe_chunk(
    chunk: np.ndarray,
    hotwords: Optional[str | list] = None,
) -> Dict[str, Any]:
    """Transcribe one chunk and return raw FunASR result."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        wavfile.write(str(tmp_path), SAMPLING_RATE, chunk.astype(np.int16))

    try:
        results = model.generate(
            input=str(tmp_path),
            cache={},
            language="ja",
            use_itn=True,
            batch_size=1,
            output_timestamp=True,
            hotwords=hotwords,
            merge_vad=False,   # We control chunking manually
        )
        return results[0] if results else {}
    finally:
        tmp_path.unlink(missing_ok=True)


# =========================
# LONG-FORM TRANSCRIPTION (NEW - faster-whisper style)
# =========================

def transcribe_japanese_llm_long_audio(
    audio_path: Path,
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
) -> TranscriptionResult:
    started = datetime.now(timezone.utc)

    # Load full audio
    sample_rate, audio_data = wavfile.read(str(audio_path))
    if sample_rate != SAMPLING_RATE:
        raise ValueError(f"Expected {SAMPLING_RATE}Hz audio, got {sample_rate}Hz")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)  # stereo -> mono

    audio_duration = len(audio_data) / sample_rate

    # Split into chunks
    chunks = split_audio_into_chunks(audio_data, sample_rate=SAMPLING_RATE)

    all_word_segments: List[WordSegment] = []
    full_text_parts: List[str] = []
    context = context_prompt or ""

    for i, (chunk, offset_sec) in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} @ {offset_sec:.1f}s")

        # Pass accumulated context as hotwords for better continuity across chunks
        current_hotwords = hotwords
        if context:
            if isinstance(current_hotwords, list):
                current_hotwords = current_hotwords + [context]
            elif current_hotwords:
                current_hotwords = [current_hotwords, context]
            else:
                current_hotwords = context

        result = _transcribe_chunk(chunk, hotwords=current_hotwords)

        if not result:
            continue

        ja_text = rich_transcription_postprocess(result.get("text", ""))

        timestamps = result.get("timestamp", [])
        words = result.get("words", [])

        for idx, ts in enumerate(timestamps):
            if not (isinstance(ts, (list, tuple)) and len(ts) >= 2):
                continue
            start_ms, end_ms = ts[0], ts[1]
            start_sec = (start_ms / 1000.0) + offset_sec if start_ms is not None else None
            end_sec = (end_ms / 1000.0) + offset_sec if end_ms is not None else None
            duration_sec = (end_sec - start_sec) if start_sec is not None and end_sec is not None else None

            word = words[idx] if idx < len(words) else None
            start_sec, end_sec, duration_sec = _postprocess_word_timing(
                word, start_sec, end_sec, duration_sec
            )

            all_word_segments.append(
                {
                    "index": len(all_word_segments),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
                    "word": word,
                }
            )

        if ja_text.strip():
            full_text_parts.append(ja_text)
            # Stronger context carry-over for next chunk
            context = " ".join(full_text_parts[-3:])[-450:]

    # Final assembly
    ja_text = " ".join(full_text_parts).strip()
    ja_text = add_punctuation(ja_text) if ja_text else ""

    transcribed_percentage = calculate_transcribed_duration_percentage(
        all_word_segments, audio_duration
    )
    transcribed_duration_sec = (transcribed_percentage / 100) * audio_duration
    coverage_label = get_coverage_quality_label(transcribed_percentage)

    metadata: TranscriptionMetadata = {
        "model": "SenseVoiceSmall",
        "processing_duration_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
        "audio_duration_sec": round(audio_duration, 3),
        "transcribed_duration_sec": round(transcribed_duration_sec, 3),
        "transcribed_duration_pctg": round(transcribed_percentage, 2),
        "coverage_label": coverage_label,
        "num_chunks": len(chunks),
    }

    phrase_segments: List[PhraseSegment] = []
    if all_word_segments and ja_text:
        phrases = split_sentences_ja(ja_text)
        phrase_segments = _build_phrase_segments(phrases, all_word_segments)
        ja_text = "".join(phrases)

    return {
        "text_ja": ja_text,
        "confidence": None,
        "quality_label": None,
        "avg_logprob": None,
        "word_segments": all_word_segments,
        "phrase_segments": phrase_segments,
        "metadata": metadata,
    }


# =========================
# SHORT-FORM TRANSCRIPTION (original, unchanged)
# =========================

def transcribe_japanese_llm_from_file(
    audio_path: Path,
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
) -> TranscriptionResult:
    started = datetime.now(timezone.utc)
    raw_results = _transcribe_file(audio_path, hotwords=hotwords)

    if not raw_results:
        return TranscriptionResult(
            text_ja="",
            confidence=None,
            quality_label="N/A",
            avg_logprob=None,
            word_segments=[],
            phrase_segments=[],
            metadata={},
        )

    first = raw_results[0]
    ja_text = rich_transcription_postprocess(first["text"])

    segments: list[WordSegment] = []
    timestamps = first.get("timestamp", [])
    words = first.get("words", [])

    for idx, ts in enumerate(timestamps):
        start_sec = end_sec = duration_sec = None
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start_ms, end_ms = ts[0], ts[1]
            start_sec = start_ms / 1000.0 if start_ms is not None else None
            end_sec = end_ms / 1000.0 if end_ms is not None else None
            duration_sec = end_sec - start_sec if start_sec is not None and end_sec is not None else None

        word = words[idx] if idx < len(words) else None
        start_sec, end_sec, duration_sec = _postprocess_word_timing(
            word, start_sec, end_sec, duration_sec
        )

        segments.append({
            "index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
            "word": word,
        })

    audio_duration = get_audio_duration_seconds(audio_path)
    transcribed_percentage = calculate_transcribed_duration_percentage(segments, audio_duration)
    transcribed_duration_sec = (transcribed_percentage / 100) * audio_duration
    coverage_label = get_coverage_quality_label(transcribed_percentage)

    metadata: TranscriptionMetadata = {
        "model": "SenseVoiceSmall",
        "processing_duration_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
        "audio_duration_sec": round(audio_duration, 3),
        "transcribed_duration_sec": round(transcribed_duration_sec, 3),
        "transcribed_duration_pctg": round(transcribed_percentage, 2),
        "coverage_label": coverage_label,
    }

    phrase_segments: list[PhraseSegment] = []
    if segments and ja_text.strip():
        ja_text = add_punctuation(ja_text)
        phrases = split_sentences_ja(ja_text)
        phrase_segments = _build_phrase_segments(phrases, segments)
        ja_text = "".join(phrases)

    return {
        "text_ja": ja_text,
        "confidence": None,
        "quality_label": None,
        "avg_logprob": None,
        "word_segments": segments,
        "phrase_segments": phrase_segments,
        "metadata": metadata,
    }


def _transcribe_file(
    audio_path: Path,
    *,
    hotwords: str | list[str] | None = None,
) -> List[Dict[str, Any]]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        results = model.generate(
            input=str(audio_path),
            cache={},
            language="ja",
            use_itn=True,
            batch_size=1,
            output_timestamp=True,
            hotwords=hotwords,
            merge_vad=False,
        )
        return results
    except Exception as e:
        from rich.console import Console
        console = Console()
        console.print(f"[bold red]FunASR error:[/bold red] {e}")
        console.print(traceback.format_exc())
        return []


# =========================
# PUBLIC API (auto-routes short vs long)
# =========================

def transcribe_japanese(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
    save_temp_wav: Path | None = None,
) -> TranscriptionResult:
    processing_started = datetime.now(timezone.utc)

    if save_temp_wav:
        audio_path = save_temp_wav
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)

    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sample_rate, arr)

    # Auto-route: short audio uses single call, long audio uses chunked pipeline
    duration = get_audio_duration_seconds(audio_path)
    print(f"Audio duration: {duration:.2f}s")
    print(f"Max chunk duration: {CHUNK_LENGTH_SEC}s")
    if duration > CHUNK_LENGTH_SEC * 1.1:   # added small buffer
        result = transcribe_japanese_llm_long_audio(
            audio_path, hotwords=hotwords, context_prompt=context_prompt
        )
    else:
        result = transcribe_japanese_llm_from_file(
            audio_path, hotwords=hotwords, context_prompt=context_prompt
        )

    if not save_temp_wav:
        audio_path.unlink(missing_ok=True)

    return result


# =========================
# MAIN (for testing)
# =========================

if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from rich.console import Console
    from rich.pretty import pprint
    from translate_jp_en_llm import translate_japanese_to_english  # assuming this exists

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\start_15s_recording_1_speaker.wav"
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", default=default_audio_path)
    args = parser.parse_args()

    audio_path = Path(args.audio_path)

    console = Console()
    console.print("[bold green]Starting Japanese transcription...[/bold green]")

    # Use the unified public API (automatically routes short vs long audio)
    result: TranscriptionResult = transcribe_japanese(
        audio_bytes=open(audio_path, "rb").read(),
        sample_rate=16000,                    # SenseVoice expects 16kHz
        hotwords=None,
        context_prompt=None,
        save_temp_wav=None,                   # Set to audio_path if you want to keep the file
    )

    ja_text = result.pop("text_ja")
    word_segments = result.pop("word_segments")
    phrase_segments = result.pop("phrase_segments")
    metadata = result.pop("metadata")

    pprint(result, expand_all=True)

    scores_json_path = OUTPUT_DIR / "scores.json"
    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved scores to:[/bold green] [link=file://{scores_json_path.resolve()}]{scores_json_path}[/link]"
    )

    ja_text_path = OUTPUT_DIR / "ja_text.md"
    with open(ja_text_path, "w", encoding="utf-8") as f:
        f.write(ja_text)
    console.print(
        f"[bold green]Saved ja_text to:[/bold green] [link=file://{ja_text_path.resolve()}]{ja_text_path}[/link]"
    )

    word_segments_path = OUTPUT_DIR / "word_segments.json"
    with open(word_segments_path, "w", encoding="utf-8") as f:
        json.dump(word_segments, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved word_segments to:[/bold green] [link=file://{word_segments_path.resolve()}]{word_segments_path}[/link]"
    )

    phrase_segments_path = OUTPUT_DIR / "phrase_segments.json"
    with open(phrase_segments_path, "w", encoding="utf-8") as f:
        json.dump(phrase_segments, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved phrase_segments to:[/bold green] [link=file://{phrase_segments_path.resolve()}]{phrase_segments_path}[/link]"
    )

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved metadata to:[/bold green] [link=file://{metadata_path.resolve()}]{metadata_path}[/link]"
    )

    # === Per-phrase sub-directories with meta.json + sound.wav ===
    phrases_dir = OUTPUT_DIR / "phrases"
    phrases_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold green]Created phrases directory:[/bold green] [link=file://{phrases_dir.resolve()}]{phrases_dir}[/link]"
    )

    # Read full audio once for slicing
    sample_rate, full_audio_data = wavfile.read(str(audio_path))
    for phrase in phrase_segments:
        phrase_num = phrase["index"]
        phrase_dir = phrases_dir / f"phrase_{phrase_num}"
        phrase_dir.mkdir(parents=True, exist_ok=True)

        # meta.json
        meta_path = phrase_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(phrase, f, ensure_ascii=False, indent=2)
        console.print(
            f"[bold green]Saved meta.json to:[/bold green] [link=file://{meta_path.resolve()}]{meta_path}[/link]"
        )

        # sound.wav = timestamp-sliced audio clip
        start_sec = phrase.get("start_sec")
        end_sec = phrase.get("end_sec")
        if start_sec is not None and end_sec is not None:
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            sliced_data = full_audio_data[start_sample:end_sample]

            sound_path = phrase_dir / "sound.wav"
            wavfile.write(str(sound_path), sample_rate, sliced_data)
            console.print(
                f"[bold green]Saved sound.wav to:[/bold green] [link=file://{sound_path.resolve()}]{sound_path}[/link]"
            )
        else:
            console.print(
                f"[yellow]Skipping sound.wav for phrase_{phrase_num} (no timestamps)[/yellow]"
            )

    console.print(f"JA:\n[bold cyan]{ja_text}[/bold cyan]")
    en_text = translate_japanese_to_english(ja_text)["text"]
    console.print(f"EN:\n[bold cyan]{en_text}[/bold cyan]")

    console.print("[bold green]✅ Transcription complete![/bold green]")
