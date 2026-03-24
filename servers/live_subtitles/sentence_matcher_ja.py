from __future__ import annotations

import argparse
from typing import List, Optional, TypedDict

from rapidfuzz import fuzz, process


class FuzzyMatchResult(TypedDict):
    """TypedDict for the return value of fuzzy_shortest_best_match."""

    match: str
    score: float
    start: int
    end: int
    text: str  # the original text from which this match was taken


def fuzzy_shortest_best_match(
    query: str,
    texts: str | List[str],
    score_cutoff: int = 80,
    max_extra_chars: int = 20,
) -> FuzzyMatchResult:
    """
    Find the shortest contiguous substring with the highest score across one or more texts.
    When multiple texts are provided, only the best result (highest score) is returned.
    using fuzzy matching (WRatio), preferring higher score then shorter length.

    Args:
        query: The string to search for.
        texts: A single string or list of strings to search within.
        score_cutoff: Minimum acceptable score (default: 80).
        max_extra_chars: Maximum extra characters allowed in the match window.

    Returns:
        FuzzyMatchResult containing match, score, start, end, and the original text.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the shortest best fuzzy match of a query inside one or more texts. "
        "When multiple texts are given, returns only the highest-scoring result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional required arguments
    parser.add_argument(
        "query",
        type=str,
        help="The query string to search for",
    )
    parser.add_argument(
        "texts",
        nargs="+",
        type=str,
        help="One or more texts to search within",
    )

    parser.add_argument(
        "-c",
        "--score-cutoff",
        type=int,
        default=80,
        help="Minimum score to accept (0-100)",
    )
    parser.add_argument(
        "-e",
        "--max-extra-chars",
        type=int,
        default=20,
        help="Maximum extra characters allowed beyond query length",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show highlighted match in the original text",
    )

    args = parser.parse_args()

    result: FuzzyMatchResult = fuzzy_shortest_best_match(
        query=args.query,
        texts=args.texts,
        score_cutoff=args.score_cutoff,
        max_extra_chars=args.max_extra_chars,
    )

    print(f"Match : {result['match']}")
    print(f"Score : {result['score']:.1f}")
    print(f"Slice : [{result['start']}:{result['end']}]")
    print(f"Length: {result['end'] - result['start']}")
    print(
        f"From  : {result.get('text', '')[:80]}{'...' if len(result.get('text', '')) > 80 else ''}"
    )

    if result["score"] >= args.score_cutoff:
        print("✅ Accepted")
    else:
        print("❌ Below threshold")

    if args.verbose and result["start"] != -1 and result.get("text"):
        highlighted = (
            result["text"][: result["start"]]
            + f"\033[1;33m{result['text'][result['start'] : result['end']]}\033[0m"
            + result["text"][result["end"] :]
        )
        print("\nHighlighted in text:")
        print(highlighted)


if __name__ == "__main__":
    main()
