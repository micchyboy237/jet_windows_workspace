from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import scipy.io.wavfile as wavfile
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

TimestampPair = Tuple[int, int]  # (start_ms, end_ms)


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]


@dataclass
class TranscriptionResult:
    text_ja: str
    confidence: Optional[float]
    quality_label: str
    avg_logprob: Optional[float]
    segments: List[WordSegment]
    metadata: Dict[str, Any]


model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    disable_update=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

# ---------------------------------------------------------------------
# Core Transcription Logic
# ---------------------------------------------------------------------

def _transcribe_file(
    audio_path: Path,
    *,
    language: str = "ja",
) -> List[Dict[str, Any]]:
    results = model.generate(
        input=str(audio_path),
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        output_timestamp=True,
    )

    return results


def transcribe_japanese_llm_from_file(
    audio_path: Path,
    *,
    client_id: str = "unknown",
    utterance_id: str = "unknown",
    segment_num: int = 0,
) -> TranscriptionResult:

    started = datetime.now(timezone.utc)

    raw_results = _transcribe_file(audio_path)

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

        # SenseVoice format: [start, end]
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

    duration = (
        datetime.now(timezone.utc) - started
    ).total_seconds()

    metadata = {
        "client_id": client_id,
        "utterance_id": utterance_id,
        "segment_num": segment_num,
        "processing_duration_seconds": round(duration, 3),
        "model": "SenseVoiceSmall",
    }

    return TranscriptionResult(
        text_ja=ja_text,
        confidence=None,           # SenseVoice does not provide logprobs
        quality_label="N/A",
        avg_logprob=None,
        segments=segments,
        metadata=metadata,
    )


def transcribe_japanese_llm_from_bytes(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    client_id: str = "unknown",
    utterance_id: str = "unknown",
    segment_num: int = 0,
) -> TranscriptionResult:
    """
    Transcribe raw PCM int16 bytes.
    Designed for live server usage.
    """

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:

        arr = np.frombuffer(audio_bytes, dtype=np.int16)
        wavfile.write(tmp.name, sample_rate, arr)

        result = transcribe_japanese_llm_from_file(
            Path(tmp.name),
            client_id=client_id,
            utterance_id=utterance_id,
            segment_num=segment_num,
        )

    try:
        Path(tmp.name).unlink(missing_ok=True)
    except Exception:
        pass

    return result
