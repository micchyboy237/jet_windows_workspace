from difflib import SequenceMatcher
from typing import TypedDict, Optional

from sentence_utils import split_sentences_ja

from rich.console import Console
from rich.style import Style
from rich.text import Text

console = Console()


class ExtractedNewText(TypedDict):
    """
    TypedDict for the result of extract_new_text and extract_new_ja_text.
    """
    new_text: str
    similarity: float
    unchanged_text: str
    start_index: int


def strip_trailing_punctuation(text: str) -> str:
    """Strip trailing punctuation and whitespace from the end of text.
    
    Handles both English/Latin punctuation and common Japanese punctuation.
    Used to make similarity computation robust to ending punctuation differences
    in live subtitles.
    """
    if not text:
        return text
    # Common punctuation + whitespace (including Japanese)
    punct = r'.,!?;:\s。！？…、　'
    # Remove trailing characters from the set
    return text.rstrip(punct)


def _similarity(a: str, b: str) -> float:
    """Compute similarity ignoring trailing punctuation on both strings."""
    a_clean = strip_trailing_punctuation(a)
    b_clean = strip_trailing_punctuation(b)
    if not a_clean or not b_clean:
        # Fallback to original if cleaning removes everything
        matcher = SequenceMatcher(None, a, b)
        return matcher.ratio()
    matcher = SequenceMatcher(None, a_clean, b_clean)
    return matcher.ratio()


def console_diff_highlight(
    a: str,
    b: str,
    label_a: str = "Original",
    label_b: str = "Modified",
    context: bool = True,
    label_width: Optional[int] = None,
) -> None:
    """
    Print two strings with inline labels and diff highlighting.

    - Red strikethrough: deleted text (from a)
    - Green underline: inserted text (from b)
    - Normal text: unchanged parts
    """

    matcher = SequenceMatcher(None, a, b)

    # Build labels
    def build_label(label: str) -> str:
        label_text = f"{label}:"
        if label_width is not None:
            return f"{label_text:<{label_width}}"
        return f"{label_text} "

    text_a = Text(build_label(label_a), style="bold")
    text_b = Text(build_label(label_b), style="bold")

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text_a.append(a[i1:i2])
            text_b.append(b[j1:j2])

        elif tag == "delete":
            text_a.append(a[i1:i2], style=Style(color="red", strike=True))
            if context:
                text_b.append(" " * (i2 - i1))

        elif tag == "insert":
            text_a.append(" " * (j2 - j1))
            text_b.append(b[j1:j2], style=Style(color="green", underline=True))

        elif tag == "replace":
            text_a.append(a[i1:i2], style=Style(color="red", strike=True))
            text_b.append(b[j1:j2], style=Style(color="green", underline=True))

    console.print(text_a)
    console.print(text_b)

    console.print(f"[italic][bold]Similarity:[/bold] [bold cyan]{_similarity(a, b):.1%}[/bold cyan][/italic]")


def extract_new_text(a: str, b: str) -> ExtractedNewText:
    """
    Extracts ONLY the text that was added at the END of string `b`
    compared to string `a`.
    - Handles both pure suffix additions and cases where the last characters
      of `a` are replaced/continued (common in live streaming subtitles).
    - Uses character-level common prefix for robustness on Japanese text.
    - Ignores all changes that are not at the very end.
    """
    if not b or a == b:
        return {
            "new_text": "",
            "similarity": 1.0,
            "unchanged_text": b,
            "start_index": len(b),
        }

    matcher = SequenceMatcher(None, a, b)

    if b.startswith(a):
        common = len(a)
        new_text = b[common:]
        start_index = common
    else:
        min_len = min(len(a), len(b))
        common = 0
        for i in range(min_len):
            if a[i] == b[i]:
                common += 1
            else:
                break
        while common > 0 and a[common - 1].isspace():
            common -= 1
        candidate = b[common:]
        new_text = candidate.lstrip()
        leading_ws_len = len(candidate) - len(new_text)
        start_index = common + leading_ws_len

    unchanged_text = b[:start_index]

    return {
        "new_text": new_text,
        "similarity": _similarity(a, b),
        "unchanged_text": unchanged_text,
        "start_index": start_index,
    }


def extract_new_ja_text(a: str, b: str) -> ExtractedNewText:
    """
    Japanese-specific version of extract_new_text.

    Extracts ONLY the text that was added at the END of string `b`
    compared to string `a`, but uses sentence-level matching via
    split_sentences_ja for complete context.

    This fixes texts that are cut off at the start (a common issue in
    live subtitle streams where the buffer or incremental update may
    begin mid-sentence). The returned new_text always begins at a
    sentence boundary and may include the beginning of the current
    sentence (providing full sentence context) even if part of it was
    previously present in `a`.

    Also returns:
      - unchanged_text: prefix of b before the new_text
      - start_index: exact index in b where new_text begins
    """
    if not b or a == b:
        return {
            "new_text": "",
            "similarity": 1.0,
            "unchanged_text": b,
            "start_index": len(b),
        }

    sentences_a = split_sentences_ja(a)
    sentences_b = split_sentences_ja(b)

    # Find common sentence prefix ignoring trailing punctuation
    common_count = 0
    for sa, sb in zip(sentences_a, sentences_b):
        # Ignore trailing punctuation differences (。 vs 、 etc.) which are
        # extremely common in live Japanese subtitle streams. This uses the
        # exact same helper already used for similarity scoring.
        if strip_trailing_punctuation(sa) == strip_trailing_punctuation(sb):
            common_count += 1
        else:
            break

    # Locate the start index of the first new sentence in b
    if common_count >= len(sentences_b):
        start_index = len(b)
    else:
        # Advance past all common sentences
        pos = 0
        for i in range(common_count):
            sent = sentences_b[i]
            idx = b.find(sent, pos)
            if idx == -1:
                # Extremely rare fallback
                pos = len(b)
                break
            pos = idx + len(sent)

        # Start of the first new sentence
        next_sent = sentences_b[common_count]
        start_index = b.find(next_sent, pos)
        if start_index == -1:
            start_index = len(b)

    new_text = b[start_index:].lstrip() if start_index < len(b) else ""
    matcher = SequenceMatcher(None, a, b)
    unchanged_text = b[:start_index]

    return {
        "new_text": new_text,
        "similarity": _similarity(a, b),
        "unchanged_text": unchanged_text,
        "start_index": start_index,
    }


# ==================== Example Usage ====================

if __name__ == "__main__":
    import argparse
    from rich.console import Console

    console = Console()

    parser = argparse.ArgumentParser(
        description="Show diff highlight or extract only ending added text."
    )
    parser.add_argument("s1", help="First (original) string")
    parser.add_argument("s2", help="Second (modified) string")
    parser.add_argument("-l1", "--label1", default="Original", help="Label for first string")
    parser.add_argument("-l2", "--label2", default="Modified", help="Label for second string")
    parser.add_argument(
        "-x", "--extract",
        action="store_true",
        help="Extract only the ending appended text from b",
    )
    args = parser.parse_args()

    if args.extract:
        console.print(f"[bold white]{args.label1}:[/bold white]\n[white]{args.s1}[/white]")
        console.print(f"[bold white]{args.label2}:[/bold white]\n[bold magenta]{args.s2}[/bold magenta]\n")

        result = extract_new_ja_text(args.s1, args.s2)

        console.print(f"[bold white]Unchanged text:[/bold white]\n[white]{result['unchanged_text'] or '[dim](none)[/dim]'}[/white]")
        console.print(f"[bold white]Ending new text:[/bold white]\n[bold cyan]{result['new_text'] or '[dim](none)[/dim]'}[/bold cyan]")
        console.print(f"[bold yellow]Length:[/bold yellow] {len(result['new_text'])} chars")
        console.print(
            f"[bold yellow]Similarity:[/bold yellow] [bold cyan]{result['similarity']:.1%}[/bold cyan]"
        )
        console.print(
            f"[bold yellow]Start index:[/bold yellow] [bold cyan]{result['start_index']}[/bold cyan]"
        )
    else:
        console_diff_highlight(args.s1, args.s2, args.label1, args.label2)
