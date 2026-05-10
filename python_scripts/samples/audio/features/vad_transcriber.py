"""
vad_transcriber.py
=========================
Transcribes speech segments produced by extract_speech_timestamps()
(from vad_firered_hybrid.py) using FunAudioLLM/SenseVoiceSmall.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

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

# Re-use loaders / types from the existing codebase
from loader import load_audio
from _types import AudioInput, SpeechSegment

# Local VAD module (for the convenience wrapper)
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
    text: str                  # clean joined transcript
    language: str              # detected / forced language tag
    chunks_meta: List[ChunkMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
class SenseVoiceTranscriber:
    """
    Thin wrapper around SenseVoiceSmall that owns the model lifetime
    and exposes a single ``transcribe_segment`` method.
    """

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        device: str | None = None,
        language: str = "auto",
        use_itn: bool = True,
        ban_emo_unk: bool = False,
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
        # Import here so the module is usable even without funasr installed
        # when only the type stubs / helpers are needed.
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
    def _infer_chunk(self, audio_np: np.ndarray) -> str:
        """
        Run one inference call on a ≤30 s float32 numpy array.

        Returns the clean transcript string (emotion tags removed).
        """
        # data_in accepts a float32 ndarray at 16 kHz directly
        res = self._model.inference(
            data_in=audio_np.astype(np.float32),
            language=self.language,
            use_itn=self.use_itn,
            ban_emo_unk=self.ban_emo_unk,
            **self._kwargs,
        )
        raw = res[0][0]["text"]
        return rich_transcription_postprocess(raw)

    @staticmethod
    def _estimate_overlap_words(text: str, chunk_dur_sec: float) -> int:
        """
        Estimate how many words at the *front* of *text* fall inside the
        acoustic overlap region so they can be dropped.

        Strategy: words-per-second from the whole chunk → scale by OVERLAP_SEC.
        We err on the side of keeping more words (floor, min 0).
        """
        words = text.split()
        if not words or chunk_dur_sec <= 0:
            return 0
        wps = len(words) / chunk_dur_sec          # words per second
        estimated = math.floor(wps * OVERLAP_SEC)
        # Never drop more than half the words
        return min(estimated, len(words) // 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_segment(
        self,
        audio_np: np.ndarray,
        segment: SpeechSegment,
    ) -> TranscribedSegment:
        """
        Transcribe one VAD ``SpeechSegment``.

        *audio_np* must be the **full** original audio at 16 kHz (float32).
        The segment's start/end times are used to slice it.

        If the segment is ≤ MAX_CHUNK_SEC it is transcribed in a single call.
        Longer segments are split into overlapping windows and the results
        joined with estimated-overlap trimming.
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
            text = self._infer_chunk(seg_audio)
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
            # Window layout (absolute seconds within the segment):
            #   chunk 0: [0,          MAX_CHUNK_SEC]
            #   chunk 1: [MAX_CHUNK_SEC - OVERLAP_SEC,  2*MAX_CHUNK_SEC - OVERLAP_SEC]
            #   ...
            #
            step_sec = MAX_CHUNK_SEC - OVERLAP_SEC   # advance per chunk = 24 s
            max_chunk_samples = int(MAX_CHUNK_SEC * SAMPLE_RATE)
            step_samples = int(step_sec * SAMPLE_RATE)

            text_parts: List[str] = []
            chunk_start_sample = 0
            chunk_index = 0

            while chunk_start_sample < len(seg_audio):
                chunk_end_sample = min(
                    chunk_start_sample + max_chunk_samples, len(seg_audio)
                )
                chunk_audio = seg_audio[chunk_start_sample:chunk_end_sample]
                chunk_dur = len(chunk_audio) / SAMPLE_RATE

                raw_text = self._infer_chunk(chunk_audio)

                is_trimmed = False
                trimmed_text = raw_text

                if chunk_index > 0 and raw_text.strip():
                    # Drop estimated overlap words from the front
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

                # Advance; stop if we've reached the end
                if chunk_end_sample >= len(seg_audio):
                    break
                chunk_start_sample += step_samples
                chunk_index += 1

            joined_text = " ".join(text_parts)

        # Detect language from first chunk's raw tag if possible
        language_tag = _extract_language_tag(chunks_meta[0].raw_text if chunks_meta else "")

        return TranscribedSegment(
            num=segment["num"],
            start=start_sec,
            end=end_sec,
            duration=duration,
            text=joined_text.strip(),
            language=language_tag,
            chunks_meta=chunks_meta,
        )

    def transcribe_all(
        self,
        audio_np: np.ndarray,
        segments: List[SpeechSegment],
    ) -> List[TranscribedSegment]:
        """
        Transcribe a list of VAD segments with a Rich progress bar.

        Only *speech* segments are processed; non-speech segments are skipped.
        """
        speech_segs = [s for s in segments if s["type"] == "speech"]
        results: List[TranscribedSegment] = []

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
                "[cyan]Transcribing segments…", total=len(speech_segs)
            )
            for seg in speech_segs:
                result = self.transcribe_segment(audio_np, seg)
                results.append(result)

                # Pretty-print each result as it arrives
                dur_tag = f"[magenta]{result.duration:.1f}s[/magenta]"
                n_chunks = len(result.chunks_meta)
                chunk_tag = (
                    f"[dim]({n_chunks} chunks)[/dim]" if n_chunks > 1 else ""
                )
                console.print(
                    f"[yellow][{result.start:.2f}–{result.end:.2f}][/yellow] "
                    f"{dur_tag} {chunk_tag}\n"
                    f"  [white]{result.text or '[no speech detected]'}[/white]"
                )
                progress.advance(task)

        return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_language_tag(raw_text: str) -> str:
    """Pull the first <|xx|> language token out of a SenseVoice raw string."""
    import re
    m = re.search(r"<\|([a-z]{2,})\|>", raw_text)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Convenience end-to-end function
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio: Union[str, Path, np.ndarray],
    *,
    # VAD options (forwarded to extract_speech_timestamps)
    vad_threshold: float = 0.5,
    min_silence_duration_sec: float = 0.25,
    min_speech_duration_sec: float = 0.25,
    max_speech_duration_sec: float | None = None,
    # SenseVoice options
    model_dir: str = MODEL_DIR,
    device: str | None = None,
    language: str = "auto",
    use_itn: bool = True,
    hub: str = "hf",
) -> List[TranscribedSegment]:
    """
    One-shot helper: VAD → segment → transcribe.

    Parameters
    ----------
    audio:
        Path to an audio file *or* a float32 numpy array at 16 kHz.
    vad_threshold:
        FireRedVAD speech probability threshold (0–1).
    language:
        SenseVoice language code: ``"auto"``, ``"en"``, ``"zh"``,
        ``"yue"``, ``"ja"``, ``"ko"``, ``"nospeech"``.

    Returns
    -------
    List of :class:`TranscribedSegment`, one per VAD speech segment.
    """
    # 1. Load audio once (we need it both for VAD and for slicing)
    audio_np, sr = load_audio(audio, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")

    # 2. Run VAD
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

    # 3. Transcribe
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
        description="Transcribe audio with FireRedVAD + SenseVoiceSmall"
    )
    parser.add_argument("audio_path", help="Input audio file")
    parser.add_argument(
        "-l", "--language", default="auto",
        help="Language code: auto, en, zh, yue, ja, ko (default: auto)",
    )
    parser.add_argument(
        "--model-dir", default=MODEL_DIR,
        help=f"SenseVoiceSmall model dir or HF repo (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: cpu / cuda / cuda:0 (default: auto-detect)",
    )
    parser.add_argument(
        "--no-itn", action="store_true",
        help="Disable Inverse Text Normalisation (numbers, punctuation)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="VAD speech threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-speech", type=float, default=None,
        help="Max VAD segment duration in seconds before forced split",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Optional path to write results as JSON",
    )
    args = parser.parse_args()

    results = transcribe_audio(
        args.audio_path,
        vad_threshold=args.threshold,
        max_speech_duration_sec=args.max_speech,
        model_dir=args.model_dir,
        device=args.device,
        language=args.language,
        use_itn=not args.no_itn,
    )

    console.rule("Transcript", style="green")
    full_transcript = " ".join(r.text for r in results if r.text)
    console.print(f"\n[bold white]{full_transcript}[/bold white]\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "num": r.num,
                "start": r.start,
                "end": r.end,
                "duration": r.duration,
                "language": r.language,
                "text": r.text,
                "chunks": [
                    {
                        "chunk_index": c.chunk_index,
                        "start_sec": c.start_sec,
                        "end_sec": c.end_sec,
                        "raw_text": c.raw_text,
                        "is_overlap_trimmed": c.is_overlap_trimmed,
                    }
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
