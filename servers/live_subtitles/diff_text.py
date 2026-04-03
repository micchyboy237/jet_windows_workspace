from difflib import SequenceMatcher
from typing import List
import argparse

from rich.console import Console
from rich.text import Text

# SudachiPy (required)
from sudachipy import tokenizer
from sudachipy import dictionary

console = Console()

# Initialize Sudachi tokenizer (SplitMode.C = shortest units)
tokenizer_obj = dictionary.Dictionary(dict_type="core").create()
sudachi_mode = tokenizer.Tokenizer.SplitMode.C


def split_tokens(text: str) -> List[str]:
    """Tokenize Japanese text using SudachiPy."""
    if not text or not text.strip():
        return []
    try:
        morphemes = tokenizer_obj.tokenize(text, sudachi_mode)
        return [m.surface() for m in morphemes]  # include ALL tokens (including whitespace) for exact reconstruction
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
    Punctuation (sentence terminators) are now excluded from the count.
    Uses hybrid prefix logic for correctness on compound words.
    """
    if not b or a == b:
        return 0
    appended_text = _get_appended_text(a, b)
    if not appended_text:
        return 0
    tokens = split_tokens(appended_text)
    appended_count = sum(1 for t in tokens if not _is_punctuation(t) and t.strip())
    return max(0, appended_count)

def extract_newly_appended_text(a: str, b: str) -> str:
    """
    Extract the newly appended text at the *end* of b.
    """
    appended = _get_appended_text(a, b).rstrip()
    if _is_punctuation(appended):
        return ""
    return appended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count newly appended Japanese tokens using SudachiPy "
                    "or extract the newly appended text (strips trailing whitespace "
                    "and sentence-terminating punctuation).",
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
        # Updated counting behavior using hybrid logic
        count = count_newly_appended_words(args.base, args.new)

        # Use appended text and tokens for pretty output
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
            console.print(f"[dim]→ These tokens were appended at the end.[/dim]")
        else:
            console.print(f"[dim]→ No new tokens appended at the end.[/dim]")
