from difflib import SequenceMatcher
from typing import List
import argparse

from rich.console import Console

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
        return [m.surface() for m in morphemes if m.surface().strip()]
    except Exception as e:
        console.print(f"[red]SudachiPy error:[/red] {e}", style="bold")
        return []


def count_newly_appended_words(a: str, b: str) -> int:
    """
    Count how many tokens were newly appended at the *end* of b.
    
    Uses SudachiPy for accurate Japanese tokenization.
    Ignores earlier changes/replacements/deletions.
    """
    if not b or a == b:
        return 0

    tokens_a: List[str] = split_tokens(a)
    tokens_b: List[str] = split_tokens(b)

    if not tokens_b:
        return 0
    if not tokens_a:
        return len(tokens_b)

    matcher = SequenceMatcher(None, tokens_a, tokens_b)
    covered = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            covered = max(covered, j2)
        elif tag == "replace":
            a_len = i2 - i1
            covered = max(covered, j1 + a_len)
        # insert/delete ignored

    appended_count = len(tokens_b) - covered
    return max(0, appended_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count newly appended Japanese tokens using SudachiPy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("base", type=str, help="Original/base Japanese string")
    parser.add_argument("new", type=str, help="New/updated Japanese string")
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show tokenized lists for debugging.",
    )

    args = parser.parse_args()

    count = count_newly_appended_words(args.base, args.new)

    console.print(f"[bold]Base:[/bold] {args.base}")
    console.print(f"[bold]New :[/bold] {args.new}")
    console.print("[dim]Tokenizer:[/dim] SudachiPy (core dictionary, SplitMode.C)")

    if args.show_tokens:
        tokens_a = split_tokens(args.base)
        tokens_b = split_tokens(args.new)
        console.print(f"[dim]Base tokens:[/dim] {tokens_a}")
        console.print(f"[dim]New  tokens:[/dim] {tokens_b}")

    console.rule()
    console.print(
        f"[bold green]Newly appended tokens:[/bold green] [bold cyan]{count}[/bold cyan]"
    )
    if count > 0:
        console.print(f"[dim]→ These tokens were appended at the end.[/dim]")
    else:
        console.print(f"[dim]→ No new tokens appended at the end.[/dim]")
