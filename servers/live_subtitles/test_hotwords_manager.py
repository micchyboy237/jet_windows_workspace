import pytest

from hotword_manager import HotwordManager


class TestHotwordManagerInitialization:
    def test_default_initialization(self) -> None:
        # Given
        manager = HotwordManager()

        # When
        result = manager.get_hotwords()

        # Then
        expected = []
        assert result == expected


class TestHotwordManagerUpdate:
    def test_single_token_update(self) -> None:
        # Given
        manager = HotwordManager()

        # When
        manager.update_from_tokens(["東京"])

        # Then
        result = manager.get_hotwords()
        expected = ["東京"]
        assert result == expected

    def test_multiple_tokens_accumulate_frequency(self) -> None:
        # Given
        manager = HotwordManager()

        # When
        manager.update_from_tokens(["東京", "大阪", "東京"])

        # Then
        result = manager.get_hotwords()
        expected = ["東京", "大阪"]
        assert result == expected

    def test_ignores_single_character_tokens(self) -> None:
        # Given
        manager = HotwordManager()

        # When
        manager.update_from_tokens(["あ", "東京"])

        # Then
        result = manager.get_hotwords()
        expected = ["東京"]
        assert result == expected

    def test_ignores_numeric_tokens(self) -> None:
        # Given
        manager = HotwordManager()

        # When
        manager.update_from_tokens(["1234", "東京"])

        # Then
        result = manager.get_hotwords()
        expected = ["東京"]
        assert result == expected


class TestHotwordManagerRanking:
    def test_frequency_ranking(self) -> None:
        # Given
        manager = HotwordManager()

        # When
        manager.update_from_tokens(["東京"])
        manager.update_from_tokens(["大阪"])
        manager.update_from_tokens(["東京"])

        # Then
        result = manager.get_hotwords()
        expected = ["東京", "大阪"]
        assert result == expected

    def test_max_hotwords_limit(self) -> None:
        # Given
        manager = HotwordManager(max_hotwords=2)

        # When
        manager.update_from_tokens(["東京", "大阪", "京都"])

        # Then
        result = manager.get_hotwords()
        expected = ["東京", "大阪"]
        assert result == expected


class TestHotwordManagerDecay:
    def test_decay_reduces_weight(self) -> None:
        # Given
        manager = HotwordManager(decay=0.5)
        manager.update_from_tokens(["東京"])

        # When
        manager.decay()

        # Then
        result = manager.get_hotwords()
        expected = ["東京"]
        assert result == expected

    def test_decay_removes_low_weight_words(self) -> None:
        # Given
        manager = HotwordManager(decay=0.4)
        manager.update_from_tokens(["東京"])

        # When
        manager.decay()
        manager.decay()

        # Then
        result = manager.get_hotwords()
        expected = []
        assert result == expected


class TestHotwordManagerClear:
    def test_clear_removes_all_hotwords(self) -> None:
        # Given
        manager = HotwordManager()
        manager.update_from_tokens(["東京", "大阪"])

        # When
        manager.clear()

        # Then
        result = manager.get_hotwords()
        expected = []
        assert result == expected


class TestHotwordManagerComplexScenario:
    def test_realistic_session_flow(self) -> None:
        # Given
        manager = HotwordManager(max_hotwords=3, decay=0.9)

        # When
        # Utterance 1
        manager.update_from_tokens(["東京", "彼女", "東京"])

        # Utterance 2
        manager.update_from_tokens(["大阪", "彼女"])

        # Apply decay
        manager.decay()

        # Then
        result = manager.get_hotwords()

        # Frequency logic:
        # 東京 = 2 * 0.9 = 1.8
        # 彼女 = 2 * 0.9 = 1.8
        # 大阪 = 1 * 0.9 = 0.9
        expected = ["東京", "彼女", "大阪"]

        assert result == expected
