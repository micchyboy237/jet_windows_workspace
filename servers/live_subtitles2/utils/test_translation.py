"""Tests for Japanese to English translation utility."""

import pytest
from .translation import JapaneseToEnglishTranslator


class TestJapaneseToEnglishTranslation:
    """Test class for translation behaviors."""

    @pytest.fixture(scope="class")
    def translator(self):
        return JapaneseToEnglishTranslator()

    def test_translate_japanese_to_english_hello(self, translator):
        # Given: common Japanese greeting
        japanese_text = "こんにちは、世界"

        # When: translate
        result = translator.translate_japanese_to_english(japanese_text)

        # Then
        expected_contains = ["hello", "world"]
        assert any(word in result.lower() for word in expected_contains)

    def test_translate_japanese_to_english_empty(self, translator):
        # Given: empty input
        japanese_text = ""

        # When
        result = translator.translate_japanese_to_english(japanese_text)

        # Then
        expected = ""
        assert result == expected

    def test_translate_japanese_to_english_long_sentence(self, translator):
        # Given: real world sentence
        japanese_text = "これはテスト文です。ライブ字幕システムを構築しています。"

        # When
        result = translator.translate_japanese_to_english(japanese_text)

        # Then
        assert isinstance(result, str)
        assert len(result) > 5  # meaningful translation
