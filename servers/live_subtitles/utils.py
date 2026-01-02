from fast_bunkai import FastBunkai

def split_sentences_ja(text: str) -> list[str]:
    """
    Split Japanese text into sentences using FastBunkai.

    This function is reusable, generic, and follows DRY principles.
    It initializes FastBunkai internally with default settings (fast and accurate for modern Japanese).

    Args:
        text: The Japanese text to split.

    Returns:
        A list of sentences as strings.

    Example:
        >>> text = "羽田から✈️出発して、友だちと🍣食べました。最高！また行きたいな😂でも、予算は大丈夫かな…?"
        >>> split_sentences_ja(text)
        ['羽田から✈️出発して、友だちと🍣食べました。', '最高！', 'また行きたいな😂', 'でも、予算は大丈夫かな…?']
    """
    splitter = FastBunkai()
    return list(splitter(text))
