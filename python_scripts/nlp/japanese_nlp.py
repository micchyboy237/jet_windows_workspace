from typing import List, Union

import nagisa


def extract_japanese_nouns(
    text: Union[str, List[str]],
    pos_tags: List[str] = None,
    lowercase: bool = False,
    deduplicate: bool = False,
) -> List[str]:
    """
    Extract nouns from Japanese text using nagisa.

    Parameters
    ----------
    text : str or list of str
        Japanese text (single string or list of strings/paragraphs)
    pos_tags : list of str, optional
        POS tags to extract (default: ['名詞'])
        You can add e.g. ['名詞', '固有名詞'] if needed
    lowercase : bool, default False
        Whether to lowercase romaji/alphabet words (usually not recommended for Japanese)
    deduplicate : bool, default False
        Return unique nouns only (preserves first appearance order)

    Returns
    -------
    list of str
        List of extracted noun tokens
        If input was list → returns flat list of all nouns
    """
    if pos_tags is None:
        pos_tags = ["名詞"]

    # Handle single string or list of strings uniformly
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    all_nouns = []

    for t in texts:
        if not t.strip():
            continue
        try:
            tagged = nagisa.extract(t, extract_postags=pos_tags)
            nouns = tagged.words
            if lowercase:
                nouns = [w.lower() for w in nouns]
            all_nouns.extend(nouns)
        except Exception as e:
            print(f"Warning: failed to process text: {t[:30]}... → {e}")

    if deduplicate:
        seen = set()
        unique = []
        for n in all_nouns:
            if n not in seen:
                unique.append(n)
                seen.add(n)
        return unique

    return all_nouns
