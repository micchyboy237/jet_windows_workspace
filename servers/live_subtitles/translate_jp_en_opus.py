# servers/live_subtitles/translate_jp_en.py
"""
Standalone Japanese → English translation module using CTranslate2 + Helsinki-NLP/opus-mt-ja-en
"""

import math
from typing import List, Optional, TypedDict

from transformers import AutoTokenizer
from translator_types import Translator

from utils import split_sentences_ja


# ────────────────────────────────────────────────
# Types
# ────────────────────────────────────────────────

class TranslationResult(TypedDict):
    text: str
    log_prob: Optional[float]
    confidence: Optional[float]
    quality: str


# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER_NAME = "Helsinki-NLP/opus-mt-ja-en"

tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER_NAME)

translator = Translator(
    TRANSLATOR_MODEL_PATH,
    device="cpu",
    compute_type="int8",
)


# ────────────────────────────────────────────────
# Confidence Helpers
# ────────────────────────────────────────────────

def translation_quality_label(log_prob: Optional[float]) -> str:
    """
    Quality label for translation confidence based on cumulative log-probability.
    """
    if log_prob is None or not math.isfinite(log_prob):
        return "N/A"
    if log_prob > -0.4:
        return "High"
    elif log_prob > -1.0:
        return "Good"
    elif log_prob > -2.0:
        return "Medium"
    else:
        return "Low"


def translation_confidence_score(
    log_prob: Optional[float],
    num_tokens: Optional[int] = None,
    min_tokens: int = 1,
    fallback: float = 0.0
) -> float:
    """
    Convert cumulative translation log-prob to normalized confidence score [0.0, 1.0].
    Uses length-normalized per-token probability (geometric mean).
    """
    if log_prob is None or num_tokens is None or num_tokens <= 0:
        return fallback

    effective_tokens = max(min_tokens, num_tokens)
    per_token_prob = math.exp(log_prob / effective_tokens)

    return float(min(1.0, max(0.0, per_token_prob)))


# ────────────────────────────────────────────────
# Main Translation Function
# ────────────────────────────────────────────────

def translate_japanese_to_english(
    text_ja: str,
    beam_size: int = 4,
    max_decoding_length: int = 512,
    min_tokens_for_confidence: int = 3,
    enable_scoring: bool = False,
) -> TranslationResult:
    """
    Translate Japanese text to English using OPUS-MT model via CTranslate2.
    """

    if not text_ja.strip():
        return {
            "text": "",
            "log_prob": None,
            "confidence": None,
            "quality": "N/A",
        }

    sentences_ja: List[str] = split_sentences_ja(text_ja)

    if not sentences_ja:
        return {
            "text": "",
            "log_prob": None,
            "confidence": None,
            "quality": "N/A",
        }

    batch_src_tokens = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(sent.strip()))
        for sent in sentences_ja if sent.strip()
    ]

    if not batch_src_tokens:
        return {
            "text": "",
            "log_prob": None,
            "confidence": None,
            "quality": "N/A",
        }

    results = translator.translate_batch(
        batch_src_tokens,
        return_scores=enable_scoring,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
    )

    en_sentences: List[str] = []
    translation_logprob: Optional[float] = None
    translation_confidence: Optional[float] = None

    for result in results:
        if not result.hypotheses:
            continue

        hyp = result.hypotheses[0]

        en_sent = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(hyp),
            skip_special_tokens=True
        ).strip()

        if en_sent:
            en_sentences.append(en_sent)

        if enable_scoring and hasattr(result, "scores") and result.scores:
            translation_logprob = result.scores[0]
            num_output_tokens = len(hyp)

            translation_confidence = translation_confidence_score(
                translation_logprob,
                num_output_tokens,
                min_tokens=min_tokens_for_confidence,
            )

    en_text = "\n".join(en_sentences).strip()

    quality_label = (
        translation_quality_label(translation_logprob)
        if enable_scoring
        else "N/A"
    )

    return {
        "text": en_text,
        "log_prob": translation_logprob,
        "confidence": translation_confidence,
        "quality": quality_label,
    }


# ────────────────────────────────────────────────
# Quick Demo
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.pretty import pprint
    from rich.console import Console

    console = Console()

    examples = [
        "本商品は30日経過後の返品・交換はお受けできませんのでご了承ください。",
    ]

    for i, jp_text in enumerate(examples, 1):
        console.rule(f"Example {i}")

        console.print("[dim]Japanese:[/dim]")
        console.print(jp_text, style="italic cyan")
        console.print()

        console.print("[bold green]English:[/bold green]")
        result = translate_japanese_to_english(
            jp_text,
            enable_scoring=True,
        )

        pprint(result, expand_all=True)
