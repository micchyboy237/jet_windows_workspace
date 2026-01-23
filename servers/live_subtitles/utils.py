from typing import List

from fast_bunkai import FastBunkai

from typing import Union, Sequence
from pathlib import Path


def split_sentences_ja(text: str) -> List[str]:
    """
    Split Japanese text into sentences using FastBunkai.
    
    FastBunkai provides excellent speed and accuracy for punctuated or emoji-rich text.
    For casual/spoken-style text with spaces instead of periods (common in transcripts or chats),
    we apply a lightweight preprocessing step: replace single spaces surrounded by Japanese characters
    with a period (。) to guide the splitter toward natural clause boundaries.
    
    This keeps the implementation generic, reusable, and minimal—no heavy dependencies beyond fast_bunkai.
    
    Args:
        text: The Japanese text to split.
    
    Returns:
        A list of sentences as strings (stripped of whitespace).
    
    Example:
        >>> text = "3人の先生から電話があった 近地なんか心当たりある?"
        >>> split_sentences_ja(text)
        ['3人の先生から電話があった', '近地なんか心当たりある?']
        
        >>> text = "羽田から✈️出発して、友だちと🍣食べました。最高！また行きたいな😂でも、予算は大丈夫かな…?"
        >>> split_sentences_ja(text)
        ['羽田から✈️出発して、友だちと🍣食べました。', '最高！', 'また行きたいな😂', 'でも、予算は大丈夫かな…?']
    """
    import re
    
    # Preprocess: treat isolated spaces (common in informal text) as potential sentence breaks
    # Only replace spaces that are between Japanese chars (hiragana, katakana, kanji, some punctuation)
    text = re.sub(r'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F])[ ]+([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F])',
                  r'\1。\2', text)
    
    splitter = FastBunkai()
    sentences = list(splitter(text))
    return [s.strip() for s in sentences if s.strip()]

# Supported audio extensions
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma",
    ".webm", ".mp4", ".mkv", ".avi"
}

AudioPathsInput = Union[str, Path, Sequence[Union[str, Path]]]

def resolve_audio_paths(audio_inputs: AudioPathsInput, recursive: bool = False) -> list[str]:
    """
    Resolve single file, list, or directory into a sorted list of absolute audio file paths as strings.
    """
    inputs = [audio_inputs] if isinstance(audio_inputs, (str, Path)) else audio_inputs
    resolved_paths: list[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for p in path.glob(pattern):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                    resolved_paths.append(p.resolve())
        elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            resolved_paths.append(path.resolve())
        elif path.exists():
            print(f"Skipping non-audio file: {path}")
        else:
            print(f"Path not found: {path}")

    if not resolved_paths:
        raise ValueError("No valid audio files found from provided inputs.")

    # Return sorted list of absolute path strings
    return sorted(str(p) for p in resolved_paths)
