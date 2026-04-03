from difflib import SequenceMatcher
from typing import Optional

from rich.console import Console
from rich.style import Style
from rich.text import Text

console = Console()


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

    console.print(f"[italic][bold]Similarity:[/bold] [bold cyan]{matcher.ratio():.1%}[/bold cyan][/italic]")


# ==================== Example Usage ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Show diff highlight between two strings with optional labels."
    )
    parser.add_argument("s1", help="First (original) string")
    parser.add_argument("s2", help="Second (modified) string")
    parser.add_argument("-l1", "--label1", default="Original", help="Label for first string")
    parser.add_argument("-l2", "--label2", default="Modified", help="Label for second string")
    args = parser.parse_args()

    console_diff_highlight(args.s1, args.s2, args.label1, args.label2)
