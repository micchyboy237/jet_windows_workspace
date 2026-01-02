from typing import List

from fast_bunkai import FastBunkai


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
