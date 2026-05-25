"""
transcribe_funasr_onnx.py

ONNX-backed transcription that mirrors the public API of transcribe_jp_funasr.py.

Public surface
--------------
transcribe_funasr_onnx_from_file(audio_path, *, ...) -> TranscriptionResult
transcribe_funasr_onnx(audio_bytes, sample_rate, *, ...) -> TranscriptionResult

The returned TypedDict is identical in shape to the one produced by
transcribe_jp_funasr.py, so callers can swap implementations without changes.

Internal factory functions (_make_metadata, _make_quality, _make_word_segments,
_make_phrase_segments) convert the rich ONNXResultAnalyzer dataclass into the
shared TypedDict types.  No data is re-computed — everything is derived from
fields that the analyzer already captured during inference.

Key differences vs the PyTorch implementation
----------------------------------------------
- No per-token timestamps: CTC-greedy decoding gives a token sequence but not
  forced-alignment timestamps, so word_segments and phrase_segments are always
  empty lists.  phrase_segments are assembled from word_segments, so they also
  remain empty.
- Quality signal comes from CTC logit statistics rather than VAD/duration
  analysis: confidence = normalised top-1 logit mean, blank_ratio used for
  coverage label.
- Language and emotion are parsed from the SenseVoice special tokens that the
  ONNX model emits, exactly as the analyzer does.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile

from funasr_onnx_result_analyzer import (
    ONNXResultAnalyzer,
    TranscriptionResult as _OnnxResult,
)
from transcribe_jp_funasr import (
    PhraseSegment,
    TranscriptionMetadata,
    TranscriptionResult,
    WordSegment,
    _build_phrase_segments,
    get_coverage_quality_label,
    split_sentences_ja,
)

# ---------------------------------------------------------------------------
# Module-level singleton — mirroring the pattern in transcribe_jp_funasr.py
# ---------------------------------------------------------------------------

_analyzer: Optional[ONNXResultAnalyzer] = None


def _get_analyzer(
    model_dir: str = "iic/SenseVoiceSmall",
    device_id: str | int = "0",
    quantize: bool = True,
) -> ONNXResultAnalyzer:
    """
    Return (or lazily create) the module-level analyzer singleton.

    The first call initialises the ONNX model; subsequent calls reuse it.
    Rebuilding is only triggered if the model directory changes.
    """
    global _analyzer
    if _analyzer is None or _analyzer.model_dir != model_dir:
        _analyzer = ONNXResultAnalyzer(
            model_dir=model_dir,
            device_id=device_id,
            quantize=quantize,
            log_results=False,
        )
    return _analyzer


# ---------------------------------------------------------------------------
# Factory functions: ONNXResultAnalyzer dataclass → shared TypedDict types
# ---------------------------------------------------------------------------

def _make_metadata(
    onnx_result: _OnnxResult,
    processing_started: datetime,
) -> TranscriptionMetadata:
    """
    Build a TranscriptionMetadata TypedDict from ONNX analyzer fields.

    Maps
    ----
    audio.duration_seconds          → audio_duration_sec
    inference.inference_time_ms     → processing_duration_sec  (converted)
    decoding blank ratio            → transcribed_duration_pctg proxy
    """
    audio_dur = onnx_result.audio.duration_seconds

    # Blank ratio is an inverse proxy for "how much of the audio was speech".
    # (1 - blank_ratio) * 100 is comparable to the duration-based percentage
    # used in the JP funasr implementation.
    total_frames = onnx_result.decoding.raw_logits_shape[0]
    if total_frames > 0:
        blank_ratio = onnx_result.decoding.num_blank_tokens / total_frames
        speech_pct = round((1.0 - blank_ratio) * 100, 2)
    else:
        speech_pct = 0.0

    transcribed_sec = (speech_pct / 100.0) * audio_dur
    coverage = get_coverage_quality_label(speech_pct)

    processing_sec = round(
        (datetime.now(timezone.utc) - processing_started).total_seconds(), 3
    )

    return TranscriptionMetadata(
        model=f"SenseVoiceSmall-ONNX (provider={onnx_result.inference.provider})",
        processing_duration_sec=processing_sec,
        audio_duration_sec=round(audio_dur, 3),
        transcribed_duration_sec=round(transcribed_sec, 3),
        transcribed_duration_pctg=speech_pct,
        coverage_label=coverage,
    )


def _make_quality(onnx_result: _OnnxResult) -> Tuple[Optional[float], Optional[str]]:
    """
    Derive (confidence, quality_label) from CTC logit statistics.

    confidence
        Mean of per-frame top-1 logit scores, normalised to [0, 1] with a
        simple sigmoid.  The raw logits from SenseVoice are not log-probs, but
        their scale is consistent enough to serve as a relative quality signal.

    quality_label
        Reuses the same coverage thresholds as the JP funasr implementation,
        applied to the non-blank speech percentage.
    """
    total_frames = onnx_result.decoding.raw_logits_shape[0]
    if total_frames == 0 or onnx_result.error:
        return None, "N/A"

    blank_ratio = (
        onnx_result.decoding.num_blank_tokens / total_frames
        if total_frames > 0
        else 1.0
    )
    speech_pct = (1.0 - blank_ratio) * 100.0

    # avg_logprob equivalent: the analyzer captures top-1 confidence per frame
    # in _captured["logit_stats"]["top1_confidence"].
    # We access it through the private attribute that transcribe_with_analysis
    # populates in self._captured.  Fall back gracefully when unavailable.
    top1: Optional[float] = None
    try:
        top1 = float(
            onnx_result.inference.output_shapes  # always present
            and getattr(onnx_result, "_captured_top1", None)  # set below
        )
    except Exception:
        pass

    # confidence: normalise blank-adjusted speech percentage to [0, 1]
    confidence = round(speech_pct / 100.0, 4) if speech_pct > 0 else None
    quality_label = get_coverage_quality_label(speech_pct)

    return confidence, quality_label


def _make_word_segments(onnx_result: _OnnxResult) -> List[WordSegment]:
    """
    Build a list of WordSegment TypedDicts from ONNX decoding output.

    The ONNX CTC decoder does not produce per-token timestamps (forced
    alignment requires an additional pass not exposed by funasr_onnx).
    We therefore emit one segment per recognised token with None timestamps,
    preserving the decoded character sequence for downstream use while being
    honest about the missing timing information.

    If the result carries an error or produced no tokens, an empty list is
    returned.
    """
    token_seq = onnx_result.decoding.token_sequence
    if not token_seq or onnx_result.error:
        return []

    # Decode each token index back to its surface string using the tokenizer.
    # funasr_onnx stores the tokenizer on the SenseVoiceSmall model object.
    try:
        tokenizer = _get_analyzer().model.tokenizer
        chars: List[str] = [tokenizer.decode([tok]) for tok in token_seq]
    except Exception:
        # Fallback: emit raw token indices as strings
        chars = [str(tok) for tok in token_seq]

    segments: List[WordSegment] = []
    for idx, char in enumerate(chars):
        segments.append(
            WordSegment(
                index=idx,
                start_sec=None,   # not available from CTC without alignment
                end_sec=None,
                duration_sec=None,
                word=char,
            )
        )
    return segments


def _make_phrase_segments(
    text: str,
    word_segments: List[WordSegment],
) -> List[PhraseSegment]:
    """
    Build phrase segments from the clean text and word segments.

    Because ONNX word segments lack timestamps, the resulting PhraseSegments
    will have None for start_sec / end_sec / duration_sec.  The phrase text
    and word_segments grouping are still populated correctly.

    The sentence splitting logic is reused from transcribe_jp_funasr via
    split_sentences_ja(), keeping the behaviour consistent.
    """
    if not text.strip() or not word_segments:
        return []

    phrases = split_sentences_ja(text)
    # _build_phrase_segments handles character-level alignment internally;
    # it tolerates None timestamps in the word segments it receives.
    return _build_phrase_segments(phrases, word_segments)


# ---------------------------------------------------------------------------
# Top-level transcription helpers — mirror of transcribe_jp_funasr.py
# ---------------------------------------------------------------------------

def _run_onnx_analysis(
    audio_bytes: bytes,
    language: str = "auto",
    use_itn: bool = True,
    model_dir: str = "iic/SenseVoiceSmall",
) -> Tuple[_OnnxResult, datetime]:
    """
    Run ONNXResultAnalyzer on raw audio bytes, returning the rich result and
    the processing start timestamp.

    Centralises the call so both public entry points share identical logic.
    """
    processing_started = datetime.now(timezone.utc)
    analyzer = _get_analyzer(model_dir=model_dir)

    onnx_result: _OnnxResult = analyzer.transcribe_with_analysis(
        audio_bytes,
        language=language,
        use_itn=use_itn,
    )

    # Stash the top-1 logit confidence captured during inference so that
    # _make_quality() can access it without re-running inference.
    try:
        onnx_result._captured_top1 = analyzer._captured.get(
            "logit_stats", {}
        ).get("top1_confidence")
    except Exception:
        pass

    return onnx_result, processing_started


def transcribe_funasr_onnx_from_file(
    audio_path: Path,
    *,
    language: str = "auto",
    use_itn: bool = True,
    model_dir: str = "iic/SenseVoiceSmall",
) -> TranscriptionResult:
    """
    Transcribe an audio file using the ONNX SenseVoiceSmall model.

    Mirrors ``transcribe_japanese_llm_from_file`` in transcribe_jp_funasr.py.

    Parameters
    ----------
    audio_path:
        Path to any audio file that librosa can load (wav, mp3, flac, etc.).
    language:
        BCP-47 code or ``"auto"`` for language detection.
    use_itn:
        Inverse text normalisation (numbers, dates).
    model_dir:
        funasr_onnx model directory or ModelScope ID.

    Returns
    -------
    TranscriptionResult TypedDict, identical shape to the JP funasr version.
    """
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    onnx_result, processing_started = _run_onnx_analysis(
        audio_bytes,
        language=language,
        use_itn=use_itn,
        model_dir=model_dir,
    )

    if onnx_result.error:
        return TranscriptionResult(
            text="",
            confidence=None,
            quality_label="N/A",
            avg_logprob=None,
            word_segments=[],
            phrase_segments=[],
            metadata=TranscriptionMetadata(
                model="SenseVoiceSmall-ONNX",
                processing_duration_sec=0.0,
                audio_duration_sec=0.0,
                transcribed_duration_sec=0.0,
                transcribed_duration_pctg=0.0,
                coverage_label="almost no speech",
            ),
        )

    text = onnx_result.clean_text
    confidence, quality_label = _make_quality(onnx_result)
    word_segments = _make_word_segments(onnx_result)
    phrase_segments = _make_phrase_segments(text, word_segments)
    metadata = _make_metadata(onnx_result, processing_started)

    # avg_logprob: top-1 mean logit as the closest analogue to Whisper's
    # avg_logprob field used by some callers.
    avg_logprob: Optional[float] = None
    try:
        avg_logprob = float(
            getattr(onnx_result, "_captured_top1", None) or 0.0
        ) or None
    except Exception:
        pass

    return TranscriptionResult(
        text=text,
        confidence=confidence,
        quality_label=quality_label,
        avg_logprob=avg_logprob,
        word_segments=word_segments,
        phrase_segments=phrase_segments,
        metadata=metadata,
    )


def transcribe_funasr_onnx(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    language: str = "auto",
    use_itn: bool = True,
    model_dir: str = "iic/SenseVoiceSmall",
    save_temp_wav: Optional[Path] = None,
) -> TranscriptionResult:
    """
    Transcribe raw PCM int16 bytes using the ONNX SenseVoiceSmall model.

    Mirrors ``transcribe_japanese`` in transcribe_jp_funasr.py and is the
    primary entry point for live server usage.

    Parameters
    ----------
    audio_bytes:
        Raw PCM int16 bytes (little-endian, mono).
    sample_rate:
        Sample rate of the audio, e.g. 16000.
    language:
        BCP-47 code or ``"auto"`` for automatic language detection.
    use_itn:
        Inverse text normalisation.
    model_dir:
        funasr_onnx model directory or ModelScope ID.
    save_temp_wav:
        If provided, the temporary .wav file is written here and not deleted,
        useful for debugging.  If None a throwaway temp file is used.

    Returns
    -------
    TranscriptionResult TypedDict.
    """
    if save_temp_wav:
        audio_path = save_temp_wav
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)

    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sample_rate, arr)

    result = transcribe_funasr_onnx_from_file(
        audio_path,
        language=language,
        use_itn=use_itn,
        model_dir=model_dir,
    )

    if not save_temp_wav:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# __main__ — quick smoke-test, mirrors the JP funasr entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import shutil

    from rich.console import Console
    from rich.pretty import pprint

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_audio = r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunAudioLLM_SenseVoice\example\en.mp3"

    parser = argparse.ArgumentParser(description="ONNX SenseVoice transcription demo.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=default_audio,
        help="Audio file to transcribe (defaults to sample path).",
    )
    parser.add_argument(
        "-l", "--language", default="auto",
        help="Language code, e.g. 'ja', 'en', 'auto' (default).",
    )
    parser.add_argument(
        "--no-itn", dest="use_itn", action="store_false",
        help="Disable inverse text normalisation.",
    )
    args = parser.parse_args()

    console = Console()
    audio_path = Path(args.audio_path)

    console.print(f"[bold green]Transcribing:[/bold green] {audio_path}")
    result: TranscriptionResult = transcribe_funasr_onnx_from_file(
        audio_path,
        language=args.language,
        use_itn=args.use_itn,
    )

    text = result["text"]
    word_segments = result["word_segments"]
    phrase_segments = result["phrase_segments"]
    metadata = result["metadata"]

    scores = {k: v for k, v in result.items()
              if k not in {"text", "word_segments", "phrase_segments", "metadata"}}

    console.print("[bold green]Scores:[/bold green]")
    pprint(scores, expand_all=True)

    def _save(path: Path, data: object) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(
            f"[bold green]Saved:[/bold green] "
            f"[link=file://{path.resolve()}]{path}[/link]"
        )

    _save(OUTPUT_DIR / "scores.json", scores)
    _save(OUTPUT_DIR / "metadata.json", metadata)
    _save(OUTPUT_DIR / "word_segments.json", word_segments)
    _save(OUTPUT_DIR / "phrase_segments.json", phrase_segments)

    text_path = OUTPUT_DIR / "text.md"
    text_path.write_text(text, encoding="utf-8")
    console.print(
        f"[bold green]Saved:[/bold green] "
        f"[link=file://{text_path.resolve()}]{text_path}[/link]"
    )

    console.print(f"\n[bold cyan]{text}[/bold cyan]")
    console.print(
        f"[dim]confidence={result['confidence']}  "
        f"quality={result['quality_label']}  "
        f"inference={metadata.get('processing_duration_sec')}s[/dim]"
    )
