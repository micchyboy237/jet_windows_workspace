import re
from typing import Union, Sequence, List
from pathlib import Path
from fast_bunkai import FastBunkai

# Unicode ranges
JA_RANGE = r"\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F"
SYMBOL_RANGE = (
    r"\u2600-\u26FF"
    r"\u2700-\u27BF"
    r"\U0001F300-\U0001FAFF"
)


def split_sentences_ja(
    text: str,
    punctuations: str = "、…・",
) -> List[str]:
    if not text.strip():
        return []

    splitter = FastBunkai()
    chunks = list(splitter(text))  # First pass: respect 。！？ properly

    if not punctuations:
        return [s.strip() for s in chunks if s.strip()]

    # Pattern: split *after* each extra punctuation, keeping it with the left side
    extra_punc_escaped = re.escape(punctuations)
    pattern = f"(?<=[{extra_punc_escaped}])\\s*(?![{extra_punc_escaped}])"

    result = []

    for chunk in chunks:
        # If no extra punctuations in this chunk → keep as-is
        if not re.search(f"[{extra_punc_escaped}]", chunk):
            cleaned = chunk.strip()
            if cleaned:
                result.append(cleaned)
            continue

        # Split after extra punctuation (lookbehind ensures punctuation stays left)
        pieces = re.split(pattern, chunk)

        for piece in pieces:
            cleaned = piece.strip()
            if cleaned:
                result.append(cleaned)

    return result


def split_symbols_ja(text: str) -> List[str]:
    """
    Split Japanese text into sentences, using symbol clusters (🎼 etc.) as section dividers.
    - Symbol clusters are completely removed/ignored
    - Text between symbols is treated as separate blocks
    - Each block is sentence-split independently using FastBunkai
    - Result: sentences stay grouped by their original sections (no unwanted merging)
    """
    if not text:
        return []

    # Split on symbol clusters → get pure text blocks
    blocks = re.split(rf"[{SYMBOL_RANGE}]+", text)

    # Remove empty/blank blocks (leading/trailing symbols, consecutive symbols)
    blocks = [block.strip() for block in blocks if block.strip()]

    if not blocks:
        return []

    splitter = FastBunkai()
    sentences: List[str] = []

    # Process each block independently → preserves section boundaries
    for block in blocks:
        # Optional: normalize internal whitespace if needed
        block = re.sub(r"\s+", " ", block).strip()
        
        if not block:
            continue
            
        # Split this block into proper sentences
        block_sentences = [s.strip() for s in splitter(block) if s.strip()]
        sentences.extend(block_sentences)

    return sentences


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split Japanese text into sentences using different strategies."
    )

    parser.add_argument(
        "ja_text",
        nargs="?",
        default=(
            "🎼作戦を担うスゴーデエージェント黄昏れ00の顔を使い分ける彼の任務は家族を作ること"
            "父ロイドフォージャー精神科医正体スパイコードネーム黄昏れ母ヨルフォージャー市役所職員"
            "🎼正体殺しやコードネーム茨姫娘アーニャフォージャー正体た戦を担うスゴーデエージェント黄昏れ0の顔を使い分ける"
            "彼の任務は家族を作ること父ロイドフォージャー精神科医正体スパイコードネーム黄昏れ"
            "母ヨルフォージャー市役所職員🎼正体殺しやコードネーム茨姫娘アーニャフォージャー正体心を読むことができるエスパー"
        ),
        help="Japanese text to split into sentences (if omitted, uses built-in example)",
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["sentences", "symbols"],
        default="sentences",
        help="Splitting mode: 'sentences' = split_sentences_ja (default), 'symbols' = split_symbols_ja"
    )

    args = parser.parse_args()

    # Select splitter function
    if args.mode == "symbols":
        splitter_func = split_symbols_ja
        mode_name = "split_symbols_ja"
    else:
        splitter_func = split_sentences_ja
        mode_name = "split_sentences_ja (default)"

    print(f"Using mode: {mode_name}\n")

    sentences = splitter_func(args.ja_text)

    if not sentences:
        print("No sentences extracted.")
    else:
        for i, sent in enumerate(sentences, 1):
            print(f"{i:2d}. {sent}")
