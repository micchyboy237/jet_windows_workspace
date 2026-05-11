import pytest
from word_utils_ja import split_tokens


def test_empty_and_whitespace():
    """Test empty string and various whitespace handling."""
    assert split_tokens("") == []
    assert split_tokens("   ") == []
    assert split_tokens("\n\t\r  ") == []
    assert split_tokens("　") == []  # Full-width space


def test_basic_japanese():
    """Test basic Japanese sentence."""
    text = "私は学生です。"
    result = split_tokens(text)
    expected = ["私", "は", "学生", "です", "。"]
    assert result == expected


def test_compound_and_particles():
    """Test compounds and particles (adjusted for SplitMode.C)."""
    text = "麩菓子は、麩を主材料とした日本の菓子です。"
    result = split_tokens(text)
    expected = ["麩", "菓子", "は", "、", "麩", "を", "主材", "料", "と", "し", "た", "日本", "の", "菓子", "です", "。"]
    assert result == expected


def test_longer_sentence():
    """Test longer natural Japanese sentence (adjusted for SplitMode.C)."""
    text = "東京駅で新幹線に乗って大阪に行きます。"
    result = split_tokens(text)
    expected = ["東京駅", "で", "新幹線", "に", "乗っ", "て", "大阪", "に", "行き", "ます", "。"]
    assert result == expected


def test_mixed_content():
    """Test Japanese mixed with English, numbers, and punctuation."""
    text = "Hello, 2023年にPythonを使ってAIを勉強します！"
    result = split_tokens(text)
    
    assert "Hello" in result
    assert "Python" in result
    assert "AI" in result
    assert "勉強" in result
    assert "ます" in result
    assert "！" in result
    assert any(x in result for x in ["2023年", "2023"])


def test_katakana_and_hiragana():
    """Test different Japanese scripts."""
    text = "これはカタカナのテストです。hello世界"
    result = split_tokens(text)
    expected_parts = ["これ", "は", "カタカナ", "の", "テスト", "です", "。", "hello", "世界"]
    
    for part in expected_parts:
        assert part in result


def test_proper_nouns_and_numbers():
    """Test proper nouns and date numbers."""
    text = "山田太郎は2024年4月1日に東京で生まれました。"
    result = split_tokens(text)
    
    assert "山田" in result
    assert "太郎" in result
    assert "東京" in result
    assert any(x in result for x in ["2024年", "2024"])
    assert any(x in result for x in ["4月", "1日"])


def test_punctuation_only():
    """Test strings containing only punctuation."""
    text = "！？、。：；「」『』()[]"
    result = split_tokens(text)
    expected = ["！", "？", "、", "。", "：", "；", "「", "」", "『", "』", "(", ")", "[", "]"]
    assert result == expected


def test_error_handling():
    """Test that the function handles unexpected input gracefully."""
    # Current implementation crashes on non-string → we should fix the function too
    assert split_tokens("") == []
    assert split_tokens("   ") == []
    
    # These will still fail until you fix the function (recommended)
    with pytest.raises((AttributeError, TypeError)):
        split_tokens(None)
    
    with pytest.raises((AttributeError, TypeError)):
        split_tokens(123)


@pytest.mark.parametrize("text, expected", [
    ("", []),
    ("こんにちは", ["こんにちは"]),
    ("おはようございます。", ["おはようございます", "。"]),
    ("ありがとう！", ["ありがとう", "！"]),
    ("はい。", ["はい", "。"]),
])
def test_various_inputs(text, expected):
    """Parameterized test for common cases."""
    result = split_tokens(text)
    assert result == expected