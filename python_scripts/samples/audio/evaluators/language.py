from typing import TypedDict
from fast_langdetect import detect
from string_utils import remove_non_alpha_numeric

class DetectLangResult(TypedDict):
    lang: str
    score: float

def detect_lang(text: str) -> DetectLangResult:
    """
    Detect language using fast-langdetect (modern FastText wrapper, NumPy 2.0+ compatible).
    Returns ISO-like code ("ja"/"en"/"unknown") with confidence score (max prob from top-1).
    """
    cleaned = remove_non_alpha_numeric(text)
    if not cleaned:
        return {"lang": "unknown", "score": 0.0}

    # result = detect_language(cleaned)  # {'lang': 'en', 'probs': {'en': 0.999, ...}}
    res_list = detect(cleaned, model="lite", k=1)
    result: DetectLangResult = {'lang': res_list[0]["lang"].lower(), 'score': res_list[0]["score"]}

    return result