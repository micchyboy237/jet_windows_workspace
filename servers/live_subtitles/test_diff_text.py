import pytest

from diff_text import count_newly_appended_words


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Simple replacement + append (Sudachi tokenizes "追加の単語です" into 4 tokens)
        ("えあうめ楽しか", "えあうん楽しか 追加の単語です", 4),

        # Long real-world subtitle example
        (
            "世界各国が水面下で熾烈な情報戦を繰り広げる時代、睨み合う2つの国、東のオスタニア、西のウスタリス戦争を企てるオスタニア政府用人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏れ00の顔を使い分ける彼の任務は家族を作ること.",
            "世界各国が水面下で熾烈な情報戦を繰り広げる時代、睨み合う2つの国、東のオスタニア、西のウスタリス戦争を企てるオスタニア政府用人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏れ0の顔を使い分ける彼の任務は家族を作ること父・ロイドフォージャー、精神科医正体・スパイ・コードネーム黄昏れ、母、ヨルフォージャー・市役所職員、正体。",
            21,
        ),

        # Punctuation handling
        ("テストです。", "テストです。追加の文章です。", 5),

        # Name change + append
        ("黄昏れ00", "黄昏れ0の顔を使い分ける父・ロイドフォージャー", 8),

        # Only replacement, no append
        ("家族を作ること。", "家族を作る任務。", 0),

        # Pure append after identical prefix
        ("スゴーデエージェント黄昏れ", "スゴーデエージェント黄昏れの任務は家族を作ること。", 8),
    ],
)
def test_count_newly_appended_words(a: str, b: str, expected: int):
    assert count_newly_appended_words(a, b) == expected


def test_japanese_punctuation_handling():
    """SudachiPy punctuation and name handling."""
    assert count_newly_appended_words(
        "テストです。",
        "テストです。追加の文章です。",
    ) == 5

    assert count_newly_appended_words(
        "黄昏れ00の顔",
        "黄昏れ0の顔を使い分ける父・ロイドフォージャー",
    ) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
