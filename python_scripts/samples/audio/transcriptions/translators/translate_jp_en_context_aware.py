"""
Option 1 — Long Audio Translation
ASR → Semantic Chunking → Rolling Context Translation

Single-file reference implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Iterable, Deque, Union
from collections import deque
from threading import Lock

from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage

from rich import print
from rich.pretty import pprint
import json

# ────────────────────────────────────────────────
# LLM CONFIGURATION (reuse / adjust freely)
# ────────────────────────────────────────────────

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf"

MODEL_SETTINGS = {
    "n_ctx": 1024,
    "n_gpu_layers": -1,
    "flash_attn": True,
    "logits_all": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "tokenizer_kwargs": {"add_bos_token": False},
    "n_batch": 128,
    "n_threads": 6,
    "n_threads_batch": 6,
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
}

TRANSLATION_DEFAULTS = {
    "temperature": 0.5,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "max_tokens": 512,
}

_llm: Llama | None = None
_llm_lock = Lock()


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)
    return _llm


# ────────────────────────────────────────────────
# ASR DATA MODEL (engine-agnostic)
# ────────────────────────────────────────────────

@dataclass(frozen=True)
class ASRSegment:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


# ────────────────────────────────────────────────
# SEMANTIC CHUNKER
# ────────────────────────────────────────────────

class SemanticChunker:
    """
    Groups ASR segments into sentence-safe chunks
    using pauses and size thresholds.
    """

    def __init__(
        self,
        max_chars: int = 800,
        min_pause: float = 0.4,
    ) -> None:
        self.max_chars = max_chars
        self.min_pause = min_pause

    def chunk(self, segments: Iterable[Union[ASRSegment, str]]) -> List[List[ASRSegment]]:
        chunks: List[List[ASRSegment]] = []
        current: List[ASRSegment] = []
        char_count = 0
        prev_end: Optional[float] = None

        for seg in segments:
            if isinstance(seg, str):
                segment = ASRSegment(text=seg, start=0.0, end=0.0)
            else:
                segment = seg

            pause = (segment.start - prev_end) if prev_end is not None else None
            prev_end = segment.end

            if current and (
                char_count + len(segment.text) > self.max_chars
                or (pause is not None and pause >= self.min_pause)
            ):
                chunks.append(current)
                current = []
                char_count = 0

            current.append(segment)
            char_count += len(segment.text)

        if current:
            chunks.append(current)

        return chunks


# ────────────────────────────────────────────────
# ROLLING TRANSLATION CONTEXT
# ────────────────────────────────────────────────

class TranslationContext:
    """
    Maintains short-term translated context
    to preserve discourse continuity.
    """

    def __init__(self, max_items: int = 4) -> None:
        self._history: Deque[str] = deque(maxlen=max_items)

    def add(self, text: str) -> None:
        self._history.append(text.strip())

    def render(self) -> str:
        if not self._history:
            return ""
        return "Previous translated context:" + "\n".join(
            f"- {t}" for t in self._history
        )


# ────────────────────────────────────────────────
# CONTEXT-AWARE TRANSLATOR
# ────────────────────────────────────────────────

class ContextAwareTranslator:
    def __init__(self) -> None:
        self._context = TranslationContext()

    def translate(self, japanese_text: str) -> str:
        messages: List[ChatCompletionRequestMessage] = [
            {
                "role": "system",
                "content": (
                    "You are a professional Japanese-to-English translator. "
                    "Translate accurately while preserving references and producing "
                    "natural, fluent English."
                ),
            }
        ]

        context_text = self._context.render()
        if context_text:
            messages.append({
                "role": "user",
                "content": (
                    f"{context_text}\n\n"
                    "Using the above previous English translation for context if needed, "
                    "now translate the following Japanese text to natural, fluent English:"
                ),
            })

        messages.append(
            {
                "role": "user",
                "content": japanese_text.strip(),
            }
        )

        llm = get_llm()
        response = llm.create_chat_completion(
            messages=messages,
            **TRANSLATION_DEFAULTS,
        )

        translated = response["choices"][0]["message"]["content"]
        self._context.add(translated)
        llm.reset()

        return translated


# ────────────────────────────────────────────────
# END-TO-END PIPELINE
# ────────────────────────────────────────────────

class LongAudioTranslator:
    def __init__(
        self,
        chunker: SemanticChunker,
        translator: ContextAwareTranslator,
    ) -> None:
        self.chunker = chunker
        self.translator = translator

    def translate(self, segments: Union[List[ASRSegment], List[str]]) -> str:
        # Normalize input to List[ASRSegment]
        normalized: List[ASRSegment]
        if segments and isinstance(segments[0], str):
            normalized = [ASRSegment(text=s, start=0.0, end=0.0) for s in segments]
        else:
            normalized = segments  # type: ignore[assignment]

        chunks = self.chunker.chunk(normalized)
        results: List[str] = []

        for chunk in chunks:
            jp_text = " ".join(seg.text for seg in chunk)
            en_text = self.translator.translate(jp_text)
            results.append(en_text)

        return "\n\n".join(results)


def translate_text(segments: Union[List[ASRSegment], List[str]]) -> str:
    chunker = SemanticChunker(max_chars=40)
    translator = ContextAwareTranslator()
    pipeline = LongAudioTranslator(chunker, translator)

    english = pipeline.translate(segments)
    return english


# ────────────────────────────────────────────────
# USAGE EXAMPLE
# ────────────────────────────────────────────────

def example_texts_only(asr_segments=None):
    print("\n[bold yellow]Running example_texts_only (texts only)...[/bold yellow]")
    # Simulated ASR output (replace with Whisper / faster-whisper)
    asr_segments = [
        "本商品は",
        "30日経過後の",
        "返品・交換は",
        "お受けできません。",
        "ご了承ください。",
    ]

    english = translate_text(asr_segments)

    print(f"\n[bold cyan]Translation:[/bold cyan]")
    pprint(english)

    print()


def example_with_timing(asr_segments=None):
    print("\n[bold yellow]Running example_with_timing (with timing)...[/bold yellow]")
    # Simulated ASR output (replace with Whisper / faster-whisper)
    asr_segments = [
        ASRSegment("本商品は", 0.0, 0.4),
        ASRSegment("30日経過後の", 0.5, 1.0),
        ASRSegment("返品・交換は", 1.1, 1.7),
        ASRSegment("お受けできません。", 1.8, 2.4),
        ASRSegment("ご了承ください。", 3.0, 3.5),
    ]

    english = translate_text(asr_segments)

    print(f"\n[bold cyan]Translation:[/bold cyan]")
    pprint(english)

    print()


if __name__ == "__main__":
    example_texts_only()
    example_with_timing()