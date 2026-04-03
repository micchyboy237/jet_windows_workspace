import pytest
from diff_text import count_newly_appended_words


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Given basic append with punctuation variations
        ("えあうめ楽しか", "えあうん楽しか 追加の単語です", 4),

        # Given long real-world paragraph mutation
        (
            "世界各国が水面下で熾烈な情報戦を繰り広げる時代、睨み合う2つの国、東のオスタニア、西のウスタリス戦争を企てるオスタニア政府用人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏れ00の顔を使い分ける彼の任務は家族を作ること.",
            "世界各国が水面下で熾烈な情報戦を繰り広げる時代、睨み合う2つの国、東のオスタニア、西のウスタリス戦争を企てるオスタニア政府用人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏れ0の顔を使い分ける彼の任務は家族を作ること父・ロイドフォージャー、精神科医正体・スパイ・コードネーム黄昏れ、母、ヨルフォージャー・市役所職員、正体。",
            20,
        ),

        # Given punctuation at boundary
        ("テストです。", "テストです。追加の文章です。", 4),

        # Given number mutation + append
        ("黄昏れ00", "黄昏れ0の顔を使い分ける父・ロイドフォージャー", 8),

        # Given replacement (no append)
        ("家族を作ること。", "家族を作る任務。", 0),

        # Given append after stable prefix
        ("スゴーデエージェント黄昏れ", "スゴーデエージェント黄昏れの任務は家族を作ること。", 7),
    ],
)
def test_count_newly_appended_words(a: str, b: str, expected: int):
    # When counting appended tokens
    result = count_newly_appended_words(a, b)

    # Then exact match expected
    assert result == expected


def test_japanese_punctuation_handling():
    """SudachiPy punctuation and name handling."""

    # Given punctuation boundary
    result = count_newly_appended_words(
        "テストです。",
        "テストです。追加の文章です。",
    )
    assert result == 4

    # Given mixed numeric + punctuation mutation
    result = count_newly_appended_words(
        "黄昏れ00の顔",
        "黄昏れ0の顔を使い分ける父・ロイドフォージャー",
    )
    assert result > 0


def test_extract_newly_appended_text():
    """Unit tests for extract function (punctuation restored)."""

    from diff_text import extract_newly_appended_text

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            # Given whitespace append
            ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " 追加の単語です"),

            # Given punctuation preserved
            ("テストです。", "テストです。追加の文章です。", "追加の文章です。"),

            # Given empty base
            ("", "新しいテキストです。", "新しいテキストです。"),

            # Given identical strings
            ("同じテキスト", "同じテキスト", ""),

            # Given replacement only
            ("家族を作ること。", "家族を作る任務。", ""),

            # Given appended clause
            (
                "スゴーデエージェント黄昏れ",
                "スゴーデエージェント黄昏れの任務は家族を作ること。",
                "の任務は家族を作ること。",
            ),

            # Given numeric mutation + append
            (
                "黄昏れ00",
                "黄昏れ0の顔を使い分ける父・ロイドフォージャー",
                "の顔を使い分ける父・ロイドフォージャー",
            ),

            # Given partial prefix match
            (
                "オペ",
                "オペレーションストリックスを発動作戦を担うスゴーデエージェント注昏れ。",
                "レーションストリックスを発動作戦を担うスゴーデエージェント注昏れ。",
            ),
        ],
    )
    def _inner(a: str, b: str, expected: str):
        # When extracting appended text
        result = extract_newly_appended_text(a, b)

        # Then exact match expected
        assert result == expected

    _inner


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])