from typing import List

from rich.console import Console
from sudachipy import dictionary, tokenizer
import re

console = Console()
tokenizer_obj = dictionary.Dictionary().create()  # dict_type deprecated
sudachi_mode = tokenizer.Tokenizer.SplitMode.C


def _split_punctuation(tokens: List[str]) -> List[str]:
    """Split consecutive punctuation into individual tokens."""
    result = []
    punct_pattern = re.compile(r'[！？、。：；「」『』()[\].,!?;:"\'「」『』]')
    for token in tokens:
        if punct_pattern.search(token) and len(token) > 1:
            # Split each punctuation character
            for char in token:
                if punct_pattern.match(char) or char.isspace():
                    result.append(char)
                else:
                    result.append(char)  # fallback
        else:
            result.append(token)
    return result


def split_tokens(text: str) -> List[str]:
    """Tokenize Japanese text using SudachiPy."""
    if text is None:
        raise TypeError("Expected string, got None")
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text).__name__}")
    if not text or not text.strip():
        return []

    try:
        morphemes = tokenizer_obj.tokenize(text, sudachi_mode)
        tokens = [m.surface() for m in morphemes]
        return _split_punctuation(tokens)
    except Exception as e:
        console.print(f"[red]SudachiPy error:[/red] {e}", style="bold")
        return []
