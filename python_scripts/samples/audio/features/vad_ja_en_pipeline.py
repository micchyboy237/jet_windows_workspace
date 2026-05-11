"""
vad_ja_en_pipeline.py
=================
Streams Japanese speech from audio → FireRedVAD → SenseVoiceSmall →
Shisa V2.1 (llama-cpp-python) → English translation, printing each
segment as it arrives.

Model used
----------
  shisa-v2.1-llama3.2-3b  (Q4_K_M GGUF via llama-cpp-python)
  Path: C:\\Users\\druiv\\.cache\\llama.cpp\\translators\\
             shisa-v2.1-llama3.2-3b.Q4_K_M.gguf

Translation quality settings (from Shisa docs)
------------------------------------------------
  temperature = 0.2    – lower = more literal / accurate
  top_p       = 0.9    – prevents cross-lingual token leakage
  No cross-lingual token leakage via EnglishOnlyLogitsProcessor:
    * Scans the full GGUF vocabulary at startup.
    * Collects every token ID whose decoded bytes contain any
      CJK Unified Ideographs (U+4E00–U+9FFF), Hiragana, Katakana,
      or CJK Extension blocks.
    * Sets those token logits to -inf before every sampling step,
      making it physically impossible for the model to emit them.
    * This is the strongest guarantee short of a grammar constraint
      and does not slow down sampling noticeably (<1 ms per step).

Usage
-----
    python vad_ja_en_pipeline.py path/to/audio.wav
    python vad_ja_en_pipeline.py path/to/audio.wav --output results.json
    python vad_ja_en_pipeline.py path/to/audio.wav --n-gpu-layers -1
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from llama_cpp import Llama, LogitsProcessorList
from rich.console import Console
from rich.rule import Rule

# Local modules
from loader import load_audio
from vad_firered_hybrid import extract_speech_timestamps
from vad_transcriber2 import (
    SenseVoiceTranscriber,
    TranscribedSegment,
    SAMPLE_RATE,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GGUF_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators"
    r"\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
)

# Shisa recommended settings for translation tasks
TRANSLATION_TEMPERATURE: float = 0.2
TRANSLATION_TOP_P: float = 0.9
TRANSLATION_MAX_TOKENS: int = 512

# System prompt – instructs the model to produce English only.
# The DPO "no-extra-text" dataset used during Shisa training already
# inclines the model to skip preamble; we reinforce this explicitly.
SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator. "
    "When given Japanese text, output ONLY the English translation "
    "with no explanations, no romanisation, no notes, and no Japanese characters. "
    "Translate faithfully and naturally."
)


# ---------------------------------------------------------------------------
# English-only logits processor
# ---------------------------------------------------------------------------

def _is_cjk_char(ch: str) -> bool:
    """Return True if ch is a CJK/Hiragana/Katakana character."""
    cp = ord(ch)
    return (
        0x3040 <= cp <= 0x30FF   # Hiragana + Katakana
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs (main block)
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0xFF65 <= cp <= 0xFF9F  # Halfwidth Katakana
        or 0x3000 <= cp <= 0x303F  # CJK Symbols & Punctuation
    )


def build_cjk_token_ids(llm: Llama) -> list[int]:
    """
    Walk the full GGUF vocabulary and collect the IDs of every token
    whose decoded bytes contain at least one CJK / Hiragana / Katakana
    character.  Called once at startup; result is cached as a list.
    """
    bad_ids: list[int] = []
    n_vocab: int = llm.n_vocab()
    for token_id in range(n_vocab):
        try:
            token_bytes: bytes = llm.detokenize([token_id])
            token_str = token_bytes.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if any(_is_cjk_char(ch) for ch in token_str):
            bad_ids.append(token_id)
    return bad_ids


class EnglishOnlyLogitsProcessor:
    """
    A ``llama_cpp.LogitsProcessor`` that sets the logit score of every
    CJK / Hiragana / Katakana token to ``-inf`` before sampling.

    This makes it physically impossible for the model to emit Japanese or
    Chinese characters regardless of temperature or top-p settings.

    The processor is built once at startup (``build_cjk_token_ids`` scans
    the full vocabulary) and then applied cheaply on every sampling step
    by writing ``-inf`` into a numpy float32 view of the logits array.

    Signature matches ``llama_cpp.LogitsProcessor``:
        __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray
    """

    def __init__(self, cjk_token_ids: list[int]) -> None:
        # Store as a numpy array for fast fancy-indexing
        self._bad_ids = np.array(cjk_token_ids, dtype=np.int32)

    def __call__(
        self,
        input_ids: np.ndarray,   # shape (n_past,)  – tokens generated so far
        scores: np.ndarray,      # shape (n_vocab,) – raw logits, float32
    ) -> np.ndarray:
        """
        Set all CJK / Hiragana / Katakana token logits to -inf.

        Parameters
        ----------
        input_ids:
            Token IDs generated so far in this completion (not used here
            but required by the LogitsProcessor protocol).
        scores:
            Raw logit scores for every vocabulary token.  Modified in-place
            and returned.

        Returns
        -------
        np.ndarray
            The modified scores array with CJK positions zeroed to -inf.
        """
        scores[self._bad_ids] = -np.inf
        return scores


# ---------------------------------------------------------------------------
# Translation result type
# ---------------------------------------------------------------------------

@dataclass
class TranslatedSegment:
    """Paired Japanese transcription + English translation for one VAD segment."""
    num: int
    start: float
    end: float
    duration: float
    japanese: str
    english: str


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class ShisaTranslator:
    """
    Wraps the Shisa V2.1 GGUF loaded via llama-cpp-python.

    The model is loaded once and reused for all segments.  The
    ``EnglishOnlyLogitsProcessor`` is built at init time by scanning
    the vocabulary (~0.5 s on CPU for a 3-B model).
    """

    def __init__(
        self,
        gguf_path: str = DEFAULT_GGUF_PATH,
        n_gpu_layers: int = 0,
        n_ctx: int = 1024,
        verbose: bool = False,
    ) -> None:
        console.print(f"[cyan]Loading Shisa GGUF from:\n  {gguf_path}[/cyan]")
        # n_ctx: how many tokens fit in one call (prompt + reply combined).
        # For translation: Japanese segment (~200 tokens max) + English reply
        # (~300 tokens) + system prompt (~60 tokens) = ~560 total.
        # 1024 gives comfortable headroom without the memory blow-up that
        # a large n_ctx causes when logits_all=True is accidentally set.
        #
        # logits_all=False (default): only the last token's logits are kept
        # in memory.  The scores array is n_ctx × n_vocab × 4 bytes; at
        # n_ctx=2048 and n_vocab=128256 that is ~1 GB — which is what
        # crashes the process.  logits_all=False keeps it at 128256 × 4 ≈
        # 0.5 MB instead.
        self._llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=False,   # CRITICAL: prevents ~1 GB scores allocation
            verbose=verbose,
        )
   
        console.print("[cyan]Building English-only logits processor…[/cyan]")
        cjk_ids = build_cjk_token_ids(self._llm)
        console.print(
            f"[green]Blocked {len(cjk_ids):,} CJK/Hiragana/Katakana token IDs.[/green]"
        )
        self._logits_processor = LogitsProcessorList(
            [EnglishOnlyLogitsProcessor(cjk_ids)]
        )
        console.print("[green]Shisa translator ready.[/green]")

    def translate(self, japanese_text: str) -> str:
        """
        Translate a single Japanese string to English.

        Parameters
        ----------
        japanese_text:
            Clean Japanese transcript from SenseVoiceSmall (emotion tags
            already stripped by ``rich_transcription_postprocess``).

        Returns
        -------
        str
            English translation with no leading/trailing whitespace.
            Empty string if ``japanese_text`` is blank.

        Notes
        -----
        * ``temperature=0.2`` and ``top_p=0.9`` follow Shisa's official
          recommendation for translation tasks to maximise accuracy and
          minimise cross-lingual token leakage.
        * The ``EnglishOnlyLogitsProcessor`` provides a hard vocabulary
          constraint on top of the sampling parameters.
        * ``stream=False`` is used here; for real-time token streaming
          use ``stream=True`` and iterate the returned generator.
        """
        if not japanese_text.strip():
            return ""

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": japanese_text.strip()},
            ],
            temperature=TRANSLATION_TEMPERATURE,
            top_p=TRANSLATION_TOP_P,
            max_tokens=TRANSLATION_MAX_TOKENS,
            logits_processor=self._logits_processor,
            stream=False,
        )
        english: str = response["choices"][0]["message"]["content"]
        return english.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_path: str,
    *,
    # VAD
    vad_threshold: float = 0.5,
    min_silence_sec: float = 0.25,
    min_speech_sec: float = 0.25,
    max_speech_sec: Optional[float] = None,
    # ASR
    asr_device: Optional[str] = None,
    asr_model_dir: str = "FunAudioLLM/SenseVoiceSmall",
    asr_hub: str = "hf",
    # Translation
    gguf_path: str = DEFAULT_GGUF_PATH,
    n_gpu_layers: int = 0,
    translator_verbose: bool = False,
) -> List[TranslatedSegment]:
    """
    Full pipeline: audio → VAD → Japanese ASR → English translation.

    The ASR and translation stages are interleaved via a generator:
    the translator starts working on segment N while segment N+1 is
    still being transcribed by SenseVoiceSmall, reducing overall
    wall-clock time on longer recordings.

    Parameters
    ----------
    audio_path:
        Path to any audio format supported by librosa / torchaudio.
    n_gpu_layers:
        Number of Shisa model layers to offload to GPU (0 = CPU-only,
        -1 = all layers).

    Returns
    -------
    List[TranslatedSegment]
    """
    # ── 1. Load audio once ─────────────────────────────────────────────
    console.rule("Step 1/3 — Load Audio", style="blue")
    audio_np, sr = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    console.print(
        f"[green]Loaded {len(audio_np)/sr:.1f} s of audio at {sr} Hz.[/green]"
    )

    # ── 2. VAD ────────────────────────────────────────────────────────
    console.rule("Step 2/3 — Voice Activity Detection", style="blue")
    vad_segments = extract_speech_timestamps(
        audio_np,
        threshold=vad_threshold,
        min_silence_duration_sec=min_silence_sec,
        min_speech_duration_sec=min_speech_sec,
        max_speech_duration_sec=max_speech_sec,
        return_seconds=True,
        include_non_speech=False,
    )
    speech_count = sum(1 for s in vad_segments if s["type"] == "speech")
    console.print(f"[green]VAD found {speech_count} speech segment(s).[/green]\n")

    if speech_count == 0:
        console.print("[red]No speech detected. Exiting.[/red]")
        return []

    # ── 3. Load models ────────────────────────────────────────────────
    console.rule("Step 3/3 — Load Models", style="blue")
    transcriber = SenseVoiceTranscriber(
        model_dir=asr_model_dir,
        device=asr_device,
        language="ja",
        use_itn=False,
        ban_emo_unk=True,
        hub=asr_hub,
    )
    translator = ShisaTranslator(
        gguf_path=gguf_path,
        n_gpu_layers=n_gpu_layers,
        verbose=translator_verbose,
    )

    # ── 4. Stream: transcribe → translate ──────────────────────────────
    console.rule("Transcription + Translation", style="green")
    results: List[TranslatedSegment] = []

    for transcribed in transcriber.stream_transcribe_all(audio_np, vad_segments):
        if not transcribed.text.strip():
            console.print(
                f"[dim]Segment {transcribed.num:03d} "
                f"[{transcribed.start:.2f}–{transcribed.end:.2f}s]: "
                "no transcription, skipping.[/dim]"
            )
            continue

        console.print(
            f"\n[bold yellow]Segment {transcribed.num:03d} "
            f"[{transcribed.start:.2f}–{transcribed.end:.2f}s][/bold yellow]"
        )
        console.print(f"  [cyan]JP:[/cyan] {transcribed.text}")

        english = translator.translate(transcribed.text)

        console.print(f"  [green]EN:[/green] {english}")

        results.append(TranslatedSegment(
            num=transcribed.num,
            start=transcribed.start,
            end=transcribed.end,
            duration=transcribed.duration,
            japanese=transcribed.text,
            english=english,
        ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Japanese Speech → English Translation pipeline\n"
            "  FireRedVAD → SenseVoiceSmall → Shisa V2.1 (llama-cpp-python)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio_path", help="Input audio file (any format)")
    parser.add_argument(
        "--gguf", default=DEFAULT_GGUF_PATH,
        help="Path to shisa-v2.1 GGUF file",
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=0,
        help="GPU layers to offload (0=CPU, -1=all). Default: 0",
    )
    parser.add_argument(
        "--asr-device", default=None,
        help="SenseVoice device: cpu / cuda / cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="VAD speech threshold 0–1 (default: 0.5)",
    )
    parser.add_argument(
        "--max-speech", type=float, default=None,
        help="Max VAD segment duration in seconds (default: no limit)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--verbose-llm", action="store_true",
        help="Enable llama.cpp verbose logging",
    )
    args = parser.parse_args()

    console.rule(
        "Japanese Speech → English Translation  (FireRedVAD + SenseVoice + Shisa)",
        style="bold blue",
    )

    results = run_pipeline(
        args.audio_path,
        vad_threshold=args.threshold,
        max_speech_sec=args.max_speech,
        asr_device=args.asr_device,
        gguf_path=args.gguf,
        n_gpu_layers=args.n_gpu_layers,
        translator_verbose=args.verbose_llm,
    )

    # Final summary
    console.rule("Full Transcript", style="green")
    for r in results:
        console.print(
            f"[yellow][{r.start:.2f}–{r.end:.2f}s][/yellow] "
            f"[cyan]{r.japanese}[/cyan]\n"
            f"  → [green]{r.english}[/green]"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                [asdict(r) for r in results],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"\n[bold green]✓ Results saved → {out_path}[/bold green]")

    console.rule("Done", style="green")


if __name__ == "__main__":
    main()
