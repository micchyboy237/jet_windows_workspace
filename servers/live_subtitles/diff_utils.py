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
    # s1 = "The quick brown fox jumps over the lazy dog"
    # s2 = "The quick blue fox jumps over the lazy cat"

    # console_diff_highlight(s1, s2)

    # # Another example with longer changes
    # print("\n" + "=" * 60)
    # s3 = "Python is an excellent programming language for beginners and experts"
    # s4 = "Python is a powerful programming language used by data scientists"

    # console_diff_highlight(s3, s4)

    s1 = "えあうめ楽しか"
    s2 = "えあうん楽しか"

    label_a = "Query"
    label_b = "Match"

    console_diff_highlight(s1, s2, label_a, label_b)
