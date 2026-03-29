from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, TypedDict

import numpy as np
import scipy.io.wavfile as wavfile
from reazonspeech.espnet.asr import (
    audio_from_numpy,
    audio_from_path,
    load_model,
    transcribe,
)


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


class TranscriptionResult(TypedDict):
    text_ja: str
    confidence: float
    quality_label: str
    avg_logprob: Optional[float]
    word_segments: list[WordSegment]
    phrase_segments: list[PhraseSegment]
    metadata: TranscriptionMetadata


# Global ReazonSpeech model (loaded once, exactly like FunASR)
model = load_model()


def calculate_transcribed_duration_percentage(
    segments: list[WordSegment], total_audio_duration_seconds: float
) -> float:
    if total_audio_duration_seconds <= 0:
        return 0.0
    total_transcribed_sec = sum(seg.get("duration_sec") or 0.0 for seg in segments)
    percentage = (total_transcribed_sec / total_audio_duration_seconds) * 100
    return percentage


def get_coverage_quality_label(pct: float) -> str:
    if pct >= 92.0:
        return "excellent (very clean)"
    if pct >= 82.0:
        return "good"
    if pct >= 65.0:
        return "fair (some noise/BGM)"
    if pct >= 40.0:
        return "sparse speech"
    if pct >= 15.0:
        return "very sparse"
    return "almost no speech"


def get_audio_duration_seconds(audio_path: Path) -> float:
    sample_rate, data = wavfile.read(str(audio_path))
    return len(data) / sample_rate


def _process_reazon_result(
    ret: Any,
    audio_duration: float,
    started: datetime,
    model_name: str = "reazonspeech-espnet-v2",
) -> TranscriptionResult:
    """Shared post-processing that turns a ReazonSpeech TranscribeResult into the exact FunASR-style dict."""
    if not ret.segments or not ret.text:
        return {
            "text_ja": "",
            "confidence": None,
            "quality_label": "N/A",
            "avg_logprob": None,
            "word_segments": [],
            "phrase_segments": [],
            "metadata": {},
        }

    text_ja = ret.text

    # Map each Reazon segment → one WordSegment (word = full segment text)
    word_segments: list[WordSegment] = []
    for idx, seg in enumerate(ret.segments):
        start_sec = float(seg.start_seconds)
        end_sec = float(seg.end_seconds)
        duration_sec = end_sec - start_sec
        word_segments.append(
            {
                "index": idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(duration_sec, 3),
                "word": seg.text,
            }
        )

    transcribed_percentage = calculate_transcribed_duration_percentage(
        word_segments, audio_duration
    )
    transcribed_duration_sec = (transcribed_percentage / 100.0) * audio_duration
    coverage_label = get_coverage_quality_label(transcribed_percentage)

    metadata: TranscriptionMetadata = {
        "model": model_name,
        "processing_duration_sec": round(
            (datetime.now(timezone.utc) - started).total_seconds(), 3
        ),
        "audio_duration_sec": round(audio_duration, 3),
        "transcribed_duration_sec": round(transcribed_duration_sec, 3),
        "transcribed_duration_pctg": round(transcribed_percentage, 2),
        "coverage_label": coverage_label,
    }

    # 1:1 phrase segments (Reazon already produces nicely punctuated segments)
    phrase_segments: list[PhraseSegment] = []
    for p_idx, wseg in enumerate(word_segments):
        phrase_segments.append(
            {
                "index": p_idx,
                "start_sec": wseg["start_sec"],
                "end_sec": wseg["end_sec"],
                "duration_sec": wseg["duration_sec"],
                "phrase": wseg["word"],
                "word_segments": [wseg],
            }
        )

    return {
        "text_ja": text_ja,
        "confidence": None,
        "quality_label": None,
        "avg_logprob": None,
        "word_segments": word_segments,
        "phrase_segments": phrase_segments,
        "metadata": metadata,
    }


def transcribe_japanese_from_file(
    audio_path: Path,
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
) -> TranscriptionResult:
    """Path-based transcription (mirrors FunASR naming/API for easy swapping)."""
    started = datetime.now(timezone.utc)
    audio = audio_from_path(audio_path)
    ret = transcribe(model, audio)
    audio_duration = get_audio_duration_seconds(audio_path)
    return _process_reazon_result(ret, audio_duration, started)


def transcribe_japanese(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
    save_temp_wav: Path | None = None,
) -> TranscriptionResult:
    """
    Transcribe raw PCM int16 bytes.
    Designed for live server usage (in-memory path, no temp file unless requested).
    """
    started = datetime.now(timezone.utc)

    audio_bytes_np = np.frombuffer(audio_bytes, dtype=np.int16)

    if save_temp_wav:
        audio_path = save_temp_wav
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(audio_path), sample_rate, audio_bytes_np)

    # In-memory conversion for ReazonSpeech
    waveform = audio_bytes_np.astype(np.float32) / 32768.0
    audio = audio_from_numpy(waveform, sample_rate)

    ret = transcribe(model, audio)
    audio_duration = len(audio_bytes_np) / sample_rate

    return _process_reazon_result(ret, audio_duration, started)


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path

    from rich.console import Console
    from rich.pretty import pprint

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_audio_path = (
        r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
    )

    parser = argparse.ArgumentParser(description="Japanese ASR demo (ReazonSpeech).")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=default_audio_path,
        help="Japanese audio to transcribe (optional, defaults to sample audio path)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    console = Console()

    console.print("[bold green]Japanese (ReazonSpeech):[/bold green]")
    result: TranscriptionResult = transcribe_japanese_from_file(
        audio_path,
    )

    ja_text = result.pop("text_ja")
    word_segments = result.pop("word_segments")
    phrase_segments = result.pop("phrase_segments")
    metadata = result.pop("metadata")

    pprint(result, expand_all=True)
    print(f"\nJA:\n{ja_text}")

    # (rest of the test output / per-phrase WAV export is identical to FunASR)
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

    phrases_dir = OUTPUT_DIR / "phrases"
    phrases_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold green]Created phrases directory:[/bold green] [link=file://{phrases_dir.resolve()}]{phrases_dir}[/link]"
    )

    sample_rate, full_audio_data = wavfile.read(str(audio_path))
    for phrase in phrase_segments:
        phrase_num = phrase["index"]
        phrase_dir = phrases_dir / f"phrase_{phrase_num}"
        phrase_dir.mkdir(parents=True, exist_ok=True)
        meta_path = phrase_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(phrase, f, ensure_ascii=False, indent=2)
        console.print(
            f"[bold green]Saved meta.json to:[/bold green] [link=file://{meta_path.resolve()}]{meta_path}[/link]"
        )

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

    console.print(
        "[bold green]✅ Per-phrase audio + meta export complete![/bold green]"
    )
