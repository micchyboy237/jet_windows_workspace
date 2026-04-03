from difflib import SequenceMatcher
from typing import List, Tuple
import argparse
from rich.console import Console
from rich.text import Text
from sudachipy import tokenizer
from sudachipy import dictionary
import unicodedata

console = Console()
tokenizer_obj = dictionary.Dictionary(dict_type="core").create()
sudachi_mode = tokenizer.Tokenizer.SplitMode.C


def _is_any_punctuation(char: str) -> bool:
    """Return True if character is any kind of punctuation."""
    return unicodedata.category(char).startswith("P")


def _strip_punctuation_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Remove punctuation while keeping a mapping to original indices.

    Returns:
        cleaned_text, index_map (clean_idx -> original_idx)
    """
    cleaned_chars: List[str] = []
    index_map: List[int] = []

    for idx, ch in enumerate(text):
        if not _is_any_punctuation(ch):
            cleaned_chars.append(ch)
            index_map.append(idx)

    return "".join(cleaned_chars), index_map


def split_tokens(text: str) -> List[str]:
    """Tokenize Japanese text using SudachiPy."""
    if not text or not text.strip():
        return []
    try:
        morphemes = tokenizer_obj.tokenize(text, sudachi_mode)
        return [m.surface() for m in morphemes]
    except Exception as e:
        console.print(f"[red]SudachiPy error:[/red] {e}", style="bold")
        return []


def _is_punctuation(token: str) -> bool:
    """Return True if token is a sentence-terminating punctuation (not counted as a word)."""
    return token in {"。", "！", "？", "…"}


def _get_appended_text(a: str, b: str) -> str:
    """
    Return the text that was newly appended at the end of b.

    Hybrid logic:
      - If b starts with a (exact prefix, common in live subtitles), use fast slice.
      - Otherwise fall back to original token-based matcher (preserves "ignore earlier changes").
    """
    if not b or a == b:
        return ""

    if b.startswith(a):
        return b[len(a):]

    tokens_a: List[str] = split_tokens(a)
    tokens_b: List[str] = split_tokens(b)

    if not tokens_b:
        return b
    if not tokens_a:
        return b

    matcher = SequenceMatcher(None, tokens_a, tokens_b)

    covered = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            covered = max(covered, j2)
        elif tag == "replace":
            a_len = i2 - i1
            covered = max(covered, j1 + a_len)

    if covered >= len(tokens_b):
        return ""

    prefix_len = sum(len(t) for t in tokens_b[:covered])
    return b[prefix_len:]


def count_newly_appended_words(a: str, b: str) -> int:
    """
    Count how many tokens were newly appended at the *end* of b.

    Uses SudachiPy for accurate Japanese tokenization.
    Ignores earlier changes/replacements/deletions.

    Preprocessing:
        - Removes all punctuation before diffing.

    Punctuation (sentence terminators) are excluded from the count.
    """
    if not b or a == b:
        return 0

    # Preprocess: remove punctuation
    clean_a, _ = _strip_punctuation_with_map(a)
    clean_b, _ = _strip_punctuation_with_map(b)

    appended_text = _get_appended_text(clean_a, clean_b)

    if not appended_text:
        return 0

    tokens = split_tokens(appended_text)
    appended_count = sum(1 for t in tokens if not _is_punctuation(t) and t.strip())

    return max(0, appended_count)


def extract_newly_appended_text(a: str, b: str) -> str:
    """
    Extract the newly appended text at the *end* of b.

    Preprocessing:
        - Removes punctuation for matching robustness.

    Postprocessing:
        - Restores original punctuation using index mapping.
    """
    if not b or a == b:
        return ""

    # Preprocess: strip punctuation but keep mapping
    clean_a, _ = _strip_punctuation_with_map(a)
    clean_b, map_b = _strip_punctuation_with_map(b)

    appended_clean = _get_appended_text(clean_a, clean_b)

    if not appended_clean:
        return ""

    # Find where appended text starts in cleaned string
    start_idx = clean_b.rfind(appended_clean)
    if start_idx == -1:
        return ""

    end_idx = start_idx + len(appended_clean)

    # Map back to original indices
    orig_start = map_b[start_idx]
    orig_end = map_b[end_idx - 1] + 1

    return b[orig_start:orig_end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count newly appended Japanese tokens using SudachiPy "
                    "or extract the newly appended text (including punctuation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("base", type=str, help="Original/base Japanese string")
    parser.add_argument("new", type=str, help="New/updated Japanese string")
    parser.add_argument(
        "-x",
        "--extract",
        action="store_true",
        help="Instead of counting, extract and print the newly appended text "
             "(includes all punctuation and whitespace).",
    )

    args = parser.parse_args()

    console.print(f"[bold]Base:[/bold] {args.base}")
    console.print(f"[bold]New :[/bold] {args.new}")
    console.print("[dim]Tokenizer:[/dim] SudachiPy (core dictionary, SplitMode.C)")

    if args.extract:
        extracted = extract_newly_appended_text(args.base, args.new)

        console.rule()
        console.print("[bold green]Extracted newly appended text:[/bold green]")
        console.print(f"[bold cyan]{extracted}[/bold cyan]")

        if not extracted:
            console.print("[dim]→ No new text was appended at the end.[/dim]")
        else:
            console.print(f"[dim]→ Length: {len(extracted)} chars[/dim]")

    else:
        count = count_newly_appended_words(args.base, args.new)
        extracted = extract_newly_appended_text(args.base, args.new)

        new_tokens = split_tokens(extracted) if extracted else []
        tokens_a = split_tokens(args.base)

        console.print(f"[dim]Base tokens:[/dim] {tokens_a}")
        console.print("[dim]New tokens (appended only):[/dim] ", end="")

        if new_tokens:
            token_text = Text()
            for i, token in enumerate(new_tokens):
                if i > 0:
                    token_text.append(" ")
                token_text.append(token, style="bold cyan")
            console.print(token_text)
        else:
            console.print("[]")

        console.rule()
        console.print(
            f"[bold green]Newly appended tokens:[/bold green] [bold cyan]{count}[/bold cyan]"
        )

        if count > 0:
            console.print("[dim]→ These tokens were appended at the end.[/dim]")
        else:
            console.print("[dim]→ No new tokens appended at the end.[/dim]")
