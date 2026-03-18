from __future__ import annotations
import nagisa
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import numpy as np
import scipy.io.wavfile as wavfile
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

TimestampPair = Tuple[int, int]

class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]

class TranscriptionMetadata(TypedDict, total=False):
    processing_duration_seconds: float
    model: str
    # You can add more optional / future fields here
    # version: str
    # hostname: str
    # input_duration_seconds: float
    # hotwords_used: bool | list[str]
    # error: str                  # in case of partial failure

class TranscriptionResult(TypedDict):
    text_ja: str
    confidence: float
    quality_label: str
    avg_logprob: Optional[float]
    segments: list[WordSegment]
    metadata: TranscriptionMetadata

model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    disable_update=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
    # trust_remote_code=True,
)

def _transcribe_file(
    audio_path: Path,
    *,
    hotwords: str | list[str] | None = None,
) -> List[Dict[str, Any]]:
    results = model.generate(
        input=str(audio_path),
        cache={},
        language="ja",
        use_itn=True,
        batch_size=32, 
        output_timestamp=True,
        hotwords=hotwords,
        merge_vad=False,
        # merge_length_s=15,
    )
    return results


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
            segments=[],
            metadata={},
        )

    first = raw_results[0]
    ja_text = rich_transcription_postprocess(first["text"])

    segments = []
    timestamps = first.get("timestamp", [])
    words = first.get("words", [])

    for idx, ts in enumerate(timestamps):
        start_ms = None
        end_ms = None
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start_ms = ts[0]
            end_ms = ts[1]
        segments.append(
            {
                "index": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "word": words[idx] if idx < len(words) else None,
            }
        )

    duration = (datetime.now(timezone.utc) - started).total_seconds()
    metadata = {
        "processing_duration_seconds": round(duration, 3),
        "model": "SenseVoiceSmall",
    }

    return {
        "text_ja": ja_text,
        "confidence": None,
        "quality_label": "N/A",
        "avg_logprob": None,
        "segments": segments,
        "metadata": metadata,
    }


def transcribe_japanese(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
    save_temp_wav: Path | None = None,       # ← now actually used
) -> TranscriptionResult:
    """
    Transcribe raw PCM int16 bytes.
    Designed for live server usage.
    """
    processing_started = datetime.now(timezone.utc)

    if save_temp_wav:
        audio_path = save_temp_wav
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            audio_path = Path(tmp.name)

    # Write audio
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sample_rate, arr)

    # Transcribe
    result = transcribe_japanese_llm_from_file(
        audio_path,
        hotwords=hotwords,
        context_prompt=context_prompt,
    )

    # Clean up temporary file if we created one
    if not save_temp_wav:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result


# ────────────────────────────────────────────────
# Quick Demo
# ────────────────────────────────────────────────

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

    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\start_15s_recording_1_speaker.wav"

    parser = argparse.ArgumentParser(
        description="Japanese ASR demo."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=audio_path,
        help="Japanese audio to transcribe (optional, defaults to sample audio path)",
    )
    args = parser.parse_args()

    console = Console()

    console.print("[bold green]Japanese:[/bold green]")
    result: TranscriptionResult = transcribe_japanese_llm_from_file(
        args.audio_path,
    )

    pprint(result, expand_all=True)
    print(f"\nJA:\n{result["text_ja"]}")

    result_json_path = OUTPUT_DIR / "transcription_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]Saved JSON result to:[/bold green] {result_json_path}")
