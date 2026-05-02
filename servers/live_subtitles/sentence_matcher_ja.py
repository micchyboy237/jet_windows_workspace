from __future__ import annotations
import argparse
import re
from typing import List, Optional, TypedDict

from rapidfuzz import fuzz, process, utils


class FuzzyMatchInput(TypedDict):
    """Input parameters used for the fuzzy matching operation."""
    query: str
    text: str
    level: str
    score_cutoff: int
    max_extra_chars: int
    max_extra_words: int


class FuzzyMatchContainsResult(TypedDict):
    """TypedDict for the return value of fuzzy_shortest_best_match."""

    match: str
    score: float
    start: int
    end: int
    text: str  # the original text from which this match was taken


class FuzzyMatchResult(TypedDict):
    """Complete fuzzy match result."""
    input: FuzzyMatchInput
    match: str
    score: float
    start: int
    end: int
    text: str
    remaining: str
    passed: bool


class PrefixTexts(TypedDict):
    prev_ja: str
    prev_en: str
    full_ja: str
    full_en: str


class FuzzyPrefixMatchResult(TypedDict):
    new_ja: str
    prev_ja: str
    new_en: str
    prev_en: str
    full_ja: str
    full_en: str
    is_continuation: bool


def _split_words(text: str) -> List[str]:
    """Smart word splitter with CJK support."""
    if not text:
        return []
    
    # Detect CJK
    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text):
        return list(text)  # character level for Japanese
    
    return text.split()


def _preprocess(text: str) -> str:
    """Consistent preprocessing for better matching."""
    return utils.default_process(text)  # lower + strip punctuation etc.



def fuzzy_shortest_best_match_contains(
    query: str,
    texts: str | List[str],
    score_cutoff: int = 75,
    max_extra_chars: int = 20,
) -> FuzzyMatchContainsResult:
    """
    Find the shortest contiguous substring with the highest score across one or more texts.
    When multiple texts are provided, only the best result (highest score) is returned.
    using fuzzy matching (WRatio), preferring higher score then shorter length.

    Args:
        query: The string to search for.
        texts: A single string or list of strings to search within.
        score_cutoff: Minimum acceptable score (default: 75).
        max_extra_chars: Maximum extra characters allowed in the match window.

    Returns:
        FuzzyMatchContainsResult containing match, score, start, end, and the original text.
    """
    if not query:
        return {"match": "", "score": 0.0, "start": -1, "end": -1, "text": ""}

    # Normalize input to list
    if isinstance(texts, str):
        text_list: List[str] = [texts]
    else:
        text_list = [t for t in texts if t]  # remove empty strings

    if not text_list:
        return {"match": "", "score": 0.0, "start": -1, "end": -1, "text": ""}

    best_result: Optional[FuzzyMatchResult] = None
    best_score: float = -1.0

    for text in text_list:
        # Quick candidate search per text
        candidates = process.extract(
            query, [text], scorer=fuzz.partial_ratio, limit=3, score_cutoff=score_cutoff
        )
        if not candidates:
            continue

        # Detailed search for shortest best window in this text
        local_best_score: float = -1.0
        local_best_start: int = -1
        local_best_end: int = -1
        local_best_match: str = ""
        local_best_length: int = float("inf")

        query_len = len(query)
        max_len = query_len + max_extra_chars

        for length in range(query_len, max_len + 1):
            for i in range(len(text) - length + 1):
                window = text[i : i + length]
                score = fuzz.WRatio(query, window)

                if score > local_best_score or (
                    score == local_best_score and length < local_best_length
                ):
                    local_best_score = score
                    local_best_start = i
                    local_best_end = i + length
                    local_best_match = window
                    local_best_length = length

        # Fallback for this text
        if local_best_score < score_cutoff and candidates:
            local_best_match = candidates[0][0]
            local_best_score = float(candidates[0][1])
            local_best_start = text.find(local_best_match)
            local_best_end = local_best_start + len(local_best_match)

        # Compare with global best
        if local_best_score > best_score:
            best_score = local_best_score
            best_result = {
                "match": local_best_match,
                "score": local_best_score,
                "start": local_best_start,
                "end": local_best_end,
                "text": text,
            }

    if best_result is None:
        return {"match": "", "score": 0.0, "start": -1, "end": -1, "text": ""}

    return best_result


def fuzzy_shortest_best_match(
    query: str,
    text: str | List[str],
    score_cutoff: int = 80,           # raised a bit
    max_extra_chars: int = 25,
    max_extra_words: int = 4,         # tightened
    level: str = "word",
) -> FuzzyMatchResult:
    """
    Improved fuzzy sentence matcher.
    Better suited for live subtitles / sentence alignment.
    """
    if not query:
        return _empty_result(query, "", level, score_cutoff, max_extra_chars, max_extra_words)

    # Normalize input
    if isinstance(text, str):
        text_list = [text]
    else:
        text_list = [t for t in text if t]

    if not text_list:
        return _empty_result(query, "", level, score_cutoff, max_extra_chars, max_extra_words)

    best_result: Optional[FuzzyMatchResult] = None
    best_score: float = -1.0

    processed_query = _preprocess(query)
    q_len = len(query)

    for t in text_list:
        if not t:
            continue

        if level == "word":
            query_tokens = _split_words(query)
            text_tokens = _split_words(t)
            q_word_len = len(query_tokens)

            if q_word_len == 0:
                continue

            local_best_score = -1.0
            local_best_start = -1
            local_best_end = -1
            local_best_match = ""
            local_best_length = float("inf")

            max_w_len = q_word_len + max_extra_words

            for w_len in range(q_word_len, max_w_len + 1):
                for i in range(len(text_tokens) - w_len + 1):
                    window_tokens = text_tokens[i : i + w_len]

                    # Join properly (no spaces for Japanese)
                    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', "".join(window_tokens)):
                        window = "".join(window_tokens)
                    else:
                        window = " ".join(window_tokens)

                    # Use stricter + preprocessed scoring
                    score = fuzz.token_set_ratio(processed_query, _preprocess(window))

                    win_len = len(window)
                    if (score > local_best_score or 
                        (abs(score - local_best_score) < 0.01 and win_len < local_best_length)):
                        local_best_score = score
                        local_best_match = window
                        local_best_start = t.find(window)
                        local_best_end = local_best_start + len(window) if local_best_start != -1 else -1
                        local_best_length = win_len

            # Strong fallback only if needed
            if local_best_score < score_cutoff and t:
                candidates = process.extract(
                    processed_query,
                    [t],
                    scorer=fuzz.token_set_ratio,
                    limit=3,
                    score_cutoff=score_cutoff - 10
                )
                if candidates:
                    best_cand = candidates[0]
                    local_best_match = best_cand[0]
                    local_best_score = float(best_cand[1])
                    local_best_start = t.find(local_best_match)
                    local_best_end = local_best_start + len(local_best_match) if local_best_start != -1 else -1

        else:
            # Character level (mostly unchanged but improved scoring)
            local_best_score = -1.0
            local_best_start = -1
            local_best_end = -1
            local_best_match = ""
            local_best_length = float("inf")

            query_len = len(query)
            max_len = query_len + max_extra_chars

            for length in range(query_len, max_len + 1):
                for i in range(len(t) - length + 1):
                    window = t[i : i + length]
                    score = fuzz.token_set_ratio(processed_query, _preprocess(window))

                    if (score > local_best_score or 
                        (abs(score - local_best_score) < 0.01 and length < local_best_length)):
                        local_best_score = score
                        local_best_start = i
                        local_best_end = i + length
                        local_best_match = window
                        local_best_length = length

            if local_best_score < score_cutoff and t:
                candidates = process.extract(
                    processed_query, [t],
                    scorer=fuzz.token_set_ratio,
                    limit=3,
                    score_cutoff=score_cutoff - 10
                )
                if candidates:
                    best_cand = candidates[0]
                    local_best_match = best_cand[0]
                    local_best_score = float(best_cand[1])
                    local_best_start = t.find(local_best_match)
                    local_best_end = local_best_start + len(local_best_match)

        # Update global best
        if local_best_score > best_score:
            best_score = local_best_score
            best_result = {
                "input": {
                    "query": query,
                    "text": t,
                    "level": level,
                    "score_cutoff": score_cutoff,
                    "max_extra_chars": max_extra_chars,
                    "max_extra_words": max_extra_words,
                },
                "match": local_best_match,
                "score": local_best_score,
                "start": local_best_start,
                "end": local_best_end,
                "text": t,
                "remaining": t[local_best_end:] if local_best_end != -1 else "",
                "passed": local_best_score >= score_cutoff,
            }

    return best_result or _empty_result(query, "", level, score_cutoff, max_extra_chars, max_extra_words)


def _empty_result(query: str, text: str, level: str, score_cutoff: int,
                  max_extra_chars: int, max_extra_words: int) -> FuzzyMatchResult:
    return {
        "input": {
            "query": query, "text": text, "level": level,
            "score_cutoff": score_cutoff, "max_extra_chars": max_extra_chars,
            "max_extra_words": max_extra_words
        },
        "match": "", "score": 0.0, "start": -1, "end": -1,
        "text": "", "remaining": "", "passed": False
    }


def fuzzy_match_prefix_texts(texts_dict: PrefixTexts) -> FuzzyPrefixMatchResult:
    prev_ja = texts_dict["prev_ja"]
    prev_en = texts_dict["prev_en"]
    full_ja = texts_dict["full_ja"]
    full_en = texts_dict["full_en"]

    # === Incremental "New" parts using fuzzy matching ===
    is_continuation = False
    new_ja = full_ja
    if prev_ja:
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            query=prev_ja, text=full_ja
        )
        print("\nFuzzy Result JA:")
        log_fuzzy_result(result)
        if result["passed"]:
            new_ja = result["remaining"]


    new_en = full_en
    if prev_en:
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            query=prev_en, text=full_en   # ← Fixed: use the new translation
        )
        print("\nFuzzy Result EN:")
        log_fuzzy_result(result)
        if result["passed"]:
            new_en = result["remaining"]
            is_continuation = True


    return {
        "prev_en": prev_en,
        "new_en": new_en,
        "prev_ja": prev_ja,
        "new_ja": new_ja,
        "full_ja": full_ja,
        "full_en": full_en,
        "is_continuation": is_continuation,
    }



def log_fuzzy_result(result: FuzzyMatchResult):
    inp = result["input"]
    print(f"Query     : {inp['query']}")
    print(f"Text      : {inp['text']}")
    print(f"Level     : {inp['level']}")
    print(f"Score     : {result['score']:.1f}")
    print(f"Cutoff    : {inp['score_cutoff']}")
    print(f"Passed    : {result['passed']}")
    print(f"Match     : {result['match']}")
    print(f"Slice     : [{result['start']}:{result['end']}]")
    print(f"Remaining : {result['remaining']}")

    highlighted = (
        result["text"][: result["start"]]
        + f"\033[1;33m{result['text'][result['start'] : result['end']]}\033[0m"
        + result["text"][result["end"] :]
    )
    print("\nHighlighted in text:")
    print(highlighted)

    print("✅ Accepted" if result["passed"] else "❌ Below threshold")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved Fuzzy sentence matcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument("text", nargs="+", type=str, help="Text(s) to search in")
    parser.add_argument("-c", "--score-cutoff", type=int, default=82, help="Minimum score")
    parser.add_argument("-e", "--max-extra-chars", type=int, default=25, help="Max extra chars")
    parser.add_argument("--max-extra-words", type=int, default=4, help="Max extra words")
    parser.add_argument("-l", "--level", type=str, choices=["char", "word"], default="word",
                        help="Matching level")
    args = parser.parse_args()

    result: FuzzyMatchResult = fuzzy_shortest_best_match(
        query=args.query,
        text=args.text,
        score_cutoff=args.score_cutoff,
        max_extra_chars=args.max_extra_chars,
        max_extra_words=args.max_extra_words,
        level=args.level,
    )

    inp = result["input"]
    print(f"Query : {inp['query']}")
    print(f"Text : {inp['text']}")
    print(f"Level : {inp['level']}")
    print(f"Score : {result['score']:.1f}")
    print(f"Cutoff : {inp['score_cutoff']}")
    print(f"Passed : {result['passed']}")
    print(f"Match : {result['match']}")
    print(f"Slice : [{result['start']}:{result['end']}]")
    print(f"Remaining : {result['remaining']}")

    # Highlighting
    highlighted = (
        result["text"][: result["start"]]
        + f"\033[1;33m{result['text'][result['start']:result['end']]}\033[0m"
        + result["text"][result["end"]:]
    )
    print("\nHighlighted:")
    print(highlighted)
    print("✅ Accepted" if result["passed"] else "❌ Below threshold")


if __name__ == "__main__":
    main()
