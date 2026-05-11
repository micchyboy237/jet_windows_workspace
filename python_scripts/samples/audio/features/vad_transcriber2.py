"""
vad_transcriber2.py
=========================
Transcribes speech segments produced by extract_speech_timestamps()
(from vad_firered_hybrid.py) using FunAudioLLM/SenseVoiceSmall.

Japanese-pipeline improvements
--------------------------------
* Default language is ``"ja"`` – auto-detect is unreliable because
  Japanese and Chinese share the same Unicode code-plane, causing the
  model to occasionally emit Chinese characters.
* Default ``use_itn=False`` – ITN (Inverse Text Normalisation) rewrites
  numbers and dates; this can corrupt Japanese kanji counters.
* Default ``ban_emo_unk=True`` – suppresses the ``<EMO_UNKNOWN>`` tag
  so it never reaches the downstream translation model.
* Language-mismatch warning when the model chose ``<|zh|>`` / ``<|yue|>``
  instead of ``<|ja|>``.
* ``stream_transcribe_all()`` generator yields one TranscribedSegment at
  a time for pipeline fan-out (transcribe seg N while translating seg N-1).
* Overlap word-drop has a minimum floor of 1 to prevent boundary-word
  duplication in Japanese where WPS estimates tend to be very low.

Key design decisions
--------------------
* SenseVoiceSmall hard-caps input at 30 s.  Any VAD segment longer than
  MAX_CHUNK_SEC (28 s, leaving 2 s headroom) is split into overlapping
  windows of that size with OVERLAP_SEC (4 s) of acoustic context shared
  between adjacent windows.
* Audio is passed as a raw float32 numpy array – no temporary WAV files.
* Overlap words are estimated and dropped from the front of every
  non-first chunk so the joined transcript reads naturally.
* Per-chunk raw text is preserved in TranscribedSegment.chunks_meta for
  callers that want to do their own re-alignment.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Union

import numpy as np
import torch

from funasr.utils.postprocess_utils import rich_transcription_postprocess
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from loader import load_audio
from _types import SpeechSegment
from vad_firered_hybrid import extract_speech_timestamps

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "FunAudioLLM/SenseVoiceSmall"
MAX_CHUNK_SEC: float = 28.0   # hard model limit is 30 s; keep 2 s headroom
OVERLAP_SEC: float = 4.0      # acoustic context shared between adjacent chunks
SAMPLE_RATE: int = 16_000     # SenseVoiceSmall requires 16 kHz mono


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class ChunkMeta:
    """Metadata for one 28-s (or shorter) inference call."""
    chunk_index: int
    start_sec: float           # absolute start in the original audio
    end_sec: float             # absolute end in the original audio
    raw_text: str              # text straight from rich_transcription_postprocess
    is_overlap_trimmed: bool   # True when overlap words were dropped from front


@dataclass
class TranscribedSegment:
    """Final transcription result aligned to one VAD speech segment."""
    num: int                   # matches SpeechSegment.num
    start: float               # segment start (seconds)
    end: float                 # segment end (seconds)
    duration: float
    text: str                  # clean joined transcript (Japanese)
    language: str              # detected language tag (should be "ja")
    chunks_meta: List[ChunkMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SenseVoiceTranscriber:
    """
    Thin wrapper around SenseVoiceSmall optimised for Japanese ASR.

    Japanese-specific defaults vs the base model's inference API:
      language    = "ja"    (not "auto" – avoids zh/ja confusion)
      use_itn     = False   (ITN rewrites kanji counters incorrectly)
      ban_emo_unk = True    (drop <EMO_UNKNOWN> so translator stays clean)
    """

    # Language codes that sit adjacent to "ja" in Unicode / training data.
    # If SenseVoice returns one of these for a Japanese segment it usually
    # means the audio had enough Chinese-adjacent phonemes to confuse it.
    _JP_ADJACENT = frozenset({"zh", "yue"})

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        device: Optional[str] = None,
        language: str = "ja",       # explicit beats "auto" for Japanese
        use_itn: bool = False,      # ITN can corrupt JP kanji counters
        ban_emo_unk: bool = True,   # suppress <EMO_UNKNOWN> tag
        hub: str = "hf",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.language = language
        self.use_itn = use_itn
        self.ban_emo_unk = ban_emo_unk

        console.print(
            f"[cyan]Loading SenseVoiceSmall from '{model_dir}' on {device}…[/cyan]"
        )
        from funasr.models.sense_voice.model import SenseVoiceSmall as _SVS  # type: ignore[import]

        self._model, self._kwargs = _SVS.from_pretrained(
            model=model_dir,
            device=device,
            hub=hub,
        )
        self._model.eval()
        console.print("[green]SenseVoiceSmall ready.[/green]")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _infer_chunk(self, audio_np: np.ndarray) -> tuple[str, str]:
        """
        Run one inference call on a ≤30 s float32 numpy array.

        Returns
        -------
        (clean_text, raw_text)
            clean_text : emotion/language tags stripped by
                         rich_transcription_postprocess.
            raw_text   : original model output string – retained so
                         callers can inspect the ``<|ja|>`` language token.
        """
        res = self._model.inference(
            data_in=audio_np.astype(np.float32),
            language=self.language,
            use_itn=self.use_itn,
            ban_emo_unk=self.ban_emo_unk,
            **self._kwargs,
        )
        raw_text: str = res[0][0]["text"]
        clean_text: str = rich_transcription_postprocess(raw_text)
        return clean_text, raw_text

    def _warn_if_wrong_language(self, raw_text: str, seg_num: int) -> None:
        """
        Emit a Rich warning when SenseVoice chose a Chinese variant
        instead of Japanese.  This usually means the audio had enough
        sino-phonemes to confuse the detector; the transcript may contain
        Chinese characters, which would degrade the EN translation.
        """
        m = re.search(r"<\|([a-z]{2,})\|>", raw_text)
        if m and m.group(1) in self._JP_ADJACENT:
            console.print(
                f"[yellow]⚠  Segment {seg_num:03d}: SenseVoice chose language "
                f"'[bold]{m.group(1)}[/bold]' (expected 'ja'). "
                "The transcript may contain Chinese characters — "
                "consider forcing language='ja'.[/yellow]"
            )

    @staticmethod
    def _estimate_overlap_words(
        text: str,
        chunk_dur_sec: float,
        min_drop: int = 1,
    ) -> int:
        """
        Estimate how many words at the *front* of *text* fall inside the
        acoustic overlap region so they can be dropped from chunk N (N > 0).

        Japanese words are typically short (1–2 characters), so the raw
        WPS estimate often rounds to 0.  We therefore apply a *min_drop*
        floor (default 1) to guarantee at least one word is removed on
        every non-first chunk, preventing the boundary word from appearing
        twice in the joined transcript.
        """
        words = text.split()
        if not words or chunk_dur_sec <= 0:
            return 0
        wps = len(words) / chunk_dur_sec
        estimated = math.floor(wps * OVERLAP_SEC)
        # Never drop more than half; never drop fewer than min_drop (unless
        # the chunk itself is only one word long).
        cap = max(1, len(words) // 2)
        return max(min_drop, min(estimated, cap))

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    def transcribe_segment(
        self,
        audio_np: np.ndarray,
        segment: SpeechSegment,
    ) -> TranscribedSegment:
        """
        Transcribe one VAD ``SpeechSegment``.

        Parameters
        ----------
        audio_np:
            Full original audio at 16 kHz (float32).  The segment's
            start/end times are used to slice it — do **not** pre-slice.
        segment:
            A ``SpeechSegment`` dict from ``extract_speech_timestamps``
            called with ``return_seconds=True``.

        Returns
        -------
        TranscribedSegment
            If the segment is ≤ MAX_CHUNK_SEC (28 s) it is transcribed
            in a single call.  Longer segments are split into overlapping
            28-s windows and the results joined with estimated-overlap
            trimming to avoid duplicated words at chunk boundaries.
        """
        start_sec: float = float(segment["start"])
        end_sec: float = float(segment["end"])
        duration: float = end_sec - start_sec

        start_sample = int(round(start_sec * SAMPLE_RATE))
        end_sample = int(round(end_sec * SAMPLE_RATE))
        seg_audio = audio_np[start_sample:end_sample].astype(np.float32)

        chunks_meta: List[ChunkMeta] = []

        if duration <= MAX_CHUNK_SEC:
            # ── Fast path: single inference call ──────────────────────
            text, raw = self._infer_chunk(seg_audio)
            self._warn_if_wrong_language(raw, segment["num"])
            chunks_meta.append(ChunkMeta(
                chunk_index=0,
                start_sec=start_sec,
                end_sec=end_sec,
                raw_text=text,
                is_overlap_trimmed=False,
            ))
            joined_text = text

        else:
            # ── Slow path: sliding window with overlap ─────────────────
            #
            # Window layout (samples within the segment):
            #   chunk 0: [0,           MAX_CHUNK_SAMPLES]
            #   chunk 1: [STEP_SAMPLES, STEP+MAX_CHUNK]   (4 s overlap)
            #   ...
            max_chunk_samples = int(MAX_CHUNK_SEC * SAMPLE_RATE)
            step_samples = int((MAX_CHUNK_SEC - OVERLAP_SEC) * SAMPLE_RATE)

            text_parts: List[str] = []
            chunk_start_sample = 0
            chunk_index = 0

            while chunk_start_sample < len(seg_audio):
                chunk_end_sample = min(
                    chunk_start_sample + max_chunk_samples, len(seg_audio)
                )
                chunk_audio = seg_audio[chunk_start_sample:chunk_end_sample]
                chunk_dur = len(chunk_audio) / SAMPLE_RATE

                raw_text, raw_full = self._infer_chunk(chunk_audio)
                if chunk_index == 0:
                    self._warn_if_wrong_language(raw_full, segment["num"])

                is_trimmed = False
                trimmed_text = raw_text

                if chunk_index > 0 and raw_text.strip():
                    drop = self._estimate_overlap_words(raw_text, chunk_dur)
                    if drop > 0:
                        words = raw_text.split()
                        trimmed_text = " ".join(words[drop:])
                        is_trimmed = True

                abs_start = start_sec + chunk_start_sample / SAMPLE_RATE
                abs_end = start_sec + chunk_end_sample / SAMPLE_RATE

                chunks_meta.append(ChunkMeta(
                    chunk_index=chunk_index,
                    start_sec=abs_start,
                    end_sec=abs_end,
                    raw_text=raw_text,
                    is_overlap_trimmed=is_trimmed,
                ))

                if trimmed_text.strip():
                    text_parts.append(trimmed_text.strip())

                if chunk_end_sample >= len(seg_audio):
                    break
                chunk_start_sample += step_samples
                chunk_index += 1

            joined_text = " ".join(text_parts)

        language_tag = _extract_language_tag(
            chunks_meta[0].raw_text if chunks_meta else ""
        )

        return TranscribedSegment(
            num=segment["num"],
            start=start_sec,
            end=end_sec,
            duration=duration,
            text=joined_text.strip(),
            language=language_tag,
            chunks_meta=chunks_meta,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def transcribe_all(
        self,
        audio_np: np.ndarray,
        segments: List[SpeechSegment],
    ) -> List[TranscribedSegment]:
        """
        Transcribe every speech segment and return a flat list.

        Non-speech segments (type == 'non-speech') are silently skipped.
        A Rich progress bar is shown during processing.
        """
        return list(self.stream_transcribe_all(audio_np, segments))

    def stream_transcribe_all(
        self,
        audio_np: np.ndarray,
        segments: List[SpeechSegment],
    ) -> Generator[TranscribedSegment, None, None]:
        """
        Generator that yields one ``TranscribedSegment`` as soon as it is
        ready, allowing a downstream consumer (e.g. a translator) to start
        working on segment N while segment N+1 is still being transcribed.

        Non-speech segments are silently skipped.

        Usage
        -----
        >>> for seg in transcriber.stream_transcribe_all(audio_np, vad_segs):
        ...     translation = translator.translate(seg.text)
        ...     print(f"[{seg.start:.1f}s] {translation}")
        """
        speech_segs = [s for s in segments if s["type"] == "speech"]
        total = len(speech_segs)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        with progress:
            task = progress.add_task(
                "[cyan]Transcribing segments…", total=total
            )
            for seg in speech_segs:
                result = self.transcribe_segment(audio_np, seg)

                n_chunks = len(result.chunks_meta)
                chunk_tag = (
                    f"[dim]({n_chunks} chunks)[/dim]" if n_chunks > 1 else ""
                )
                console.print(
                    f"[yellow][{result.start:.2f}–{result.end:.2f}s][/yellow] "
                    f"[magenta]{result.duration:.1f}s[/magenta] {chunk_tag}\n"
                    f"  [white]{result.text or '[no speech detected]'}[/white]"
                )

                progress.advance(task)
                yield result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_language_tag(raw_text: str) -> str:
    """Pull the first <|xx|> language token out of a SenseVoice raw string."""
    m = re.search(r"<\|([a-z]{2,})\|>", raw_text)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Convenience end-to-end function
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio: Union[str, Path, np.ndarray],
    *,
    vad_threshold: float = 0.5,
    min_silence_duration_sec: float = 0.25,
    min_speech_duration_sec: float = 0.25,
    max_speech_duration_sec: Optional[float] = None,
    model_dir: str = MODEL_DIR,
    device: Optional[str] = None,
    language: str = "ja",
    use_itn: bool = False,
    hub: str = "hf",
) -> List[TranscribedSegment]:
    """
    One-shot helper: load audio → VAD → transcribe → return results.

    Parameters
    ----------
    audio:
        Path to an audio file *or* a float32 numpy array at 16 kHz.
    language:
        SenseVoice language code.  Use ``"ja"`` for Japanese (default).
        Other options: ``"auto"``, ``"en"``, ``"zh"``, ``"yue"``,
        ``"ko"``, ``"nospeech"``.
    """
    audio_np, sr = load_audio(audio, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")

    console.rule("Step 1/2 — Voice Activity Detection", style="blue")
    segments = extract_speech_timestamps(
        audio_np,
        threshold=vad_threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        include_non_speech=False,
    )
    console.print(f"[green]VAD found {len(segments)} speech segment(s).[/green]\n")

    console.rule("Step 2/2 — SenseVoice Transcription", style="blue")
    transcriber = SenseVoiceTranscriber(
        model_dir=model_dir,
        device=device,
        language=language,
        use_itn=use_itn,
        hub=hub,
    )
    return transcriber.transcribe_all(audio_np, segments)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Transcribe Japanese audio with FireRedVAD + SenseVoiceSmall"
    )
    parser.add_argument("audio_path", help="Input audio file")
    parser.add_argument(
        "-l", "--language", default="ja",
        help="SenseVoice language code (default: ja)",
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-itn", action="store_true",
                        help="Enable Inverse Text Normalisation (off by default for JP)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-speech", type=float, default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    results = transcribe_audio(
        args.audio_path,
        vad_threshold=args.threshold,
        max_speech_duration_sec=args.max_speech,
        model_dir=args.model_dir,
        device=args.device,
        language=args.language,
        use_itn=args.use_itn,
    )

    console.rule("Transcript", style="green")
    for r in results:
        console.print(
            f"[yellow][{r.start:.2f}–{r.end:.2f}s][/yellow]  {r.text}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "num": r.num, "start": r.start, "end": r.end,
                "duration": r.duration, "language": r.language,
                "text": r.text,
                "chunks": [
                    {"chunk_index": c.chunk_index, "start_sec": c.start_sec,
                     "end_sec": c.end_sec, "raw_text": c.raw_text,
                     "is_overlap_trimmed": c.is_overlap_trimmed}
                    for c in r.chunks_meta
                ],
            }
            for r in results
        ]
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[bold green]✓ JSON saved → {out_path}[/bold green]")
    console.rule("Done", style="green")
