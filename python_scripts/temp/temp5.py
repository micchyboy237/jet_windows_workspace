from __future__ import annotations

from typing import List

from rich.console import Console
from rich.progress import track
import re
from fast_bunkai import FastBunkai
import jaconv

# SudachiPy imports
from sudachipy import tokenizer, dictionary

console = Console()

# ── Global SudachiPy initialization (do once) ──
_SUDACHI_TOKENIZER = dictionary.Dictionary().create()


def split_sentences_ja(text: str) -> List[str]:
    """
    Split Japanese text into sentences with basic space-to-period heuristic.
    """
    text = re.sub(
        r'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F])[ ]+([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F])',
        r'\1。\2',
        text,
    )
    splitter = FastBunkai()
    sentences = list(splitter(text))
    return [s.strip() for s in sentences if s.strip()]


def tokenize_japanese(
    text: str,
    mode: tokenizer.Tokenizer.SplitMode = tokenizer.Tokenizer.SplitMode.B,
    use_dictionary_form: bool = False
) -> List[str]:
    """
    Tokenize Japanese text using SudachiPy.
    """
    tokens = _SUDACHI_TOKENIZER.tokenize(text, mode)

    if use_dictionary_form:
        return [t.dictionary_form() for t in tokens if t.dictionary_form() != "*"]
    else:
        return [t.surface() for t in tokens]


def clean_asr_text(
    text: str,
    *,
    split_mode: tokenizer.Tokenizer.SplitMode = tokenizer.Tokenizer.SplitMode.B,
    use_lemma: bool = False,
    show_progress: bool = True
) -> str:
    """
    Clean ASR-generated Japanese text for better translation quality.
    """
    # 1. Basic normalization
    text = jaconv.normalize(text)

    # 2. Remove common fillers & reduce repetitions
    text = re.sub(r'(えっと|あのー|あのね|えー|まぁ|そのー|じゃなくて|っていうか)\s*', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 3. Sentence boundary recovery
    sentences = split_sentences_ja(text)

    # 4. Tokenization & cleaning
    cleaned_sentences: List[str] = []

    iterator = track(
        sentences,
        description="Cleaning sentences...",
        disable=not show_progress
    )

    for sent in iterator:
        # Debug line - remove when confident it's working
        # console.print(f"[dim]Tokenizing with mode: {split_mode}[/dim]")

        tokens = tokenize_japanese(
            sent,
            mode=split_mode,
            use_dictionary_form=use_lemma
        )

        cleaned = jaconv.h2z(' '.join(tokens))
        cleaned = jaconv.normalize(cleaned)
        cleaned_sentences.append(cleaned)

    # 5. Final output format
    final_text = ' '.join(cleaned_sentences)

    # Visual feedback
    preview_len = 200
    console.print("[bold green]Before:[/]", text[:preview_len] + ("..." if len(text) > preview_len else ""))
    console.print("[bold cyan]After :[/]", final_text[:preview_len] + ("..." if len(final_text) > preview_len else ""))

    return final_text


# ── Example usage ──
if __name__ == "__main__":
    raw_asr = (
        "あのーえっと今日はですねえー晴れてますあのー昨日じゃなくて一昨日は雨だったんですけど"
    )

    modes = [
        ("A", tokenizer.Tokenizer.SplitMode.A),
        ("B", tokenizer.Tokenizer.SplitMode.B),
        ("C", tokenizer.Tokenizer.SplitMode.C),
    ]

    for mode_name, mode in modes:
        console.rule(f"SplitMode.{mode_name} ＋ Surface form")
        clean_asr_text(raw_asr, split_mode=mode, use_lemma=False)
        
        console.rule(f"SplitMode.{mode_name} ＋ Dictionary form (lemma)")
        clean_asr_text(raw_asr, split_mode=mode, use_lemma=True)