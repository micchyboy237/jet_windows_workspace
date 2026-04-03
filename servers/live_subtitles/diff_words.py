# diff_words.py
from difflib import SequenceMatcher
from typing import List

def count_newly_appended_words(a: str, b: str, word_sep: str = " ") -> int:
    """
    Count how many whole words were newly appended at the *end* of b.
    Ignores all earlier changes, replacements, deletions, or inserts by
    finding the farthest point in b that can be aligned to a via equal
    or replace operations (treating replaced words as "covered" old content).
    Only pure trailing inserts or the excess words in a trailing replace
    block are considered newly appended.
    """
    if not b or a == b:
        return 0
    words_a: List[str] = [w for w in a.split(word_sep) if w] if a.strip() else []
    words_b: List[str] = [w for w in b.split(word_sep) if w] if b.strip() else []
    if not words_b:
        return 0
    if not words_a:
        return len(words_b)
    matcher = SequenceMatcher(None, words_a, words_b)
    covered = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Full equal block is covered old content
            covered = max(covered, j2)
        elif tag == "replace":
            # Only the portion of the replace block in b that aligns to a is "covered"
            # (the replacement itself). Any excess words in the b side of the replace
            # are treated as newly appended (part of the trailing change).
            a_len = i2 - i1
            covered = max(covered, j1 + a_len)
        # insert/delete blocks are ignored (earlier changes)
    appended_count = len(words_b) - covered
    return max(0, appended_count)


if __name__ == "__main__":
    import argparse

    from rich.console import Console

    console = Console()

    parser = argparse.ArgumentParser(
        description="Count how many whole words were newly appended at the end of the new string.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "base",
        type=str,
        help="The original/base string (a)",
    )
    parser.add_argument(
        "new",
        type=str,
        help="The new/updated string (b)",
    )
    parser.add_argument(
        "-s",
        "--sep",
        type=str,
        default=" ",
        help="Word separator (default: space)",
    )
    parser.add_argument(
        "--show-words",
        action="store_true",
        help="Also display the split words for base and new",
    )

    args = parser.parse_args()

    count = count_newly_appended_words(args.base, args.new, args.sep)

    console.print(f"[bold]Base:[/bold] {args.base}")
    console.print(f"[bold]New :[/bold] {args.new}")
    console.print(f"[bold]Sep :[/bold] {repr(args.sep)}")

    if args.show_words:
        words_a = [w for w in args.base.split(args.sep) if w] if args.base.strip() else []
        words_b = [w for w in args.new.split(args.sep) if w] if args.new.strip() else []
        console.print(f"[dim]Base words:[/dim] {words_a}")
        console.print(f"[dim]New  words:[/dim] {words_b}")

    console.rule()
    console.print(
        f"[bold green]Newly appended words:[/bold green] [bold cyan]{count}[/bold cyan]"
    )
    if count > 0:
        console.print(f"[dim]→ These are the trailing words added at the end.[/dim]")

