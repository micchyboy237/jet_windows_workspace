import numpy as np
import pytest
from norm_speech_loudness import normalize_speech_loudness


@pytest.fixture
def sample_rate() -> int:
    return 16_000


@pytest.fixture
def pure_silence(sample_rate: int) -> np.ndarray:
    # 1 second of silence
    return np.zeros(sample_rate, dtype=np.float32)


@pytest.fixture
def simple_speech_like_signal(sample_rate: int) -> np.ndarray:
    """
    Simulated speech:
    - 200 Hz sine wave
    - Amplitude typical for spoken voice
    """
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


@pytest.fixture
def mixed_speech_and_silence(sample_rate: int) -> np.ndarray:
    """
    0.5s silence + 0.5s speech
    """
    silence = np.zeros(sample_rate // 2, dtype=np.float32)
    t = np.linspace(0, 0.5, sample_rate // 2, endpoint=False)
    speech = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    return np.concatenate([silence, speech])


@pytest.fixture
def mock_silero_all_speech(monkeypatch):
    """
    Mocks Silero VAD to return probability=1.0 everywhere
    """

    def fake_probs(audio, sample_rate):
        return np.ones(len(audio), dtype=np.float32)

    monkeypatch.setattr(
        "norm_speech_loudness._speech_probability",
        fake_probs,
    )


@pytest.fixture
def mock_silero_no_speech(monkeypatch):
    """
    Mocks Silero VAD to return probability=0.0 everywhere
    """

    def fake_probs(audio, sample_rate):
        return np.zeros(len(audio), dtype=np.float32)

    monkeypatch.setattr(
        "norm_speech_loudness._speech_probability",
        fake_probs,
    )


@pytest.fixture
def mock_silero_partial_speech(monkeypatch):
    """
    Mocks Silero VAD to return:
    - silence: 0.0
    - speech: 1.0
    """

    def fake_probs(audio, sample_rate):
        probs = np.zeros(len(audio), dtype=np.float32)
        midpoint = len(audio) // 2
        probs[midpoint:] = 1.0
        return probs

    monkeypatch.setattr(
        "norm_speech_loudness._speech_probability",
        fake_probs,
    )


class TestNormalizeSpeechLoudness:
    def test_returns_original_audio_when_no_speech_detected(
        self,
        pure_silence,
        sample_rate,
        mock_silero_no_speech,
    ):
        """
        Given audio with no detected speech
        When normalize_speech_loudness is called
        Then the original audio is returned unchanged
        """
        result = normalize_speech_loudness(
            audio=pure_silence,
            sample_rate=sample_rate,
        )

        expected = pure_silence

        assert np.array_equal(result, expected)

    def test_normalizes_clear_speech_to_target_peak(
        self,
        simple_speech_like_signal,
        sample_rate,
        mock_silero_all_speech,
    ):
        """
        Given clear speech audio
        When normalize_speech_loudness is applied
        Then the output peak is close to the speech peak target
        """
        result = normalize_speech_loudness(
            audio=simple_speech_like_signal,
            sample_rate=sample_rate,
            peak_target=0.99,
        )

        peak = np.max(np.abs(result))

        # Exact bound check (not approximate loudness)
        assert peak <= 0.99
        assert peak > 0.95

    def test_silence_does_not_affect_speech_loudness(
        self,
        mixed_speech_and_silence,
        sample_rate,
        mock_silero_partial_speech,
    ):
        """
        Given audio with silence followed by speech
        When speech-weighted loudness normalization is applied
        Then the silence remains near zero and speech is normalized
        """
        result = normalize_speech_loudness(
            audio=mixed_speech_and_silence,
            sample_rate=sample_rate,
            peak_target=0.99,
        )

        first_half = result[: len(result) // 2]
        second_half = result[len(result) // 2 :]

        # Silence should remain silence
        assert np.max(np.abs(first_half)) == 0.0

        # Speech should be clearly present and normalized
        speech_peak = np.max(np.abs(second_half))
        assert speech_peak <= 0.99
        assert speech_peak > 0.9

    def test_short_speech_clip_does_not_crash(
        self,
        sample_rate,
        mock_silero_all_speech,
    ):
        """
        Given a very short speech clip (< 0.4s)
        When normalize_speech_loudness is called
        Then it returns valid normalized audio without error
        """
        short_clip = np.random.uniform(-0.1, 0.1, int(sample_rate * 0.1)).astype(
            np.float32
        )

        result = normalize_speech_loudness(
            audio=short_clip,
            sample_rate=sample_rate,
        )

        assert result.shape == short_clip.shape
        assert np.max(np.abs(result)) <= 1.0

    def test_return_dtype_is_respected(
        self,
        simple_speech_like_signal,
        sample_rate,
        mock_silero_all_speech,
    ):
        """
        Given a requested return dtype
        When normalize_speech_loudness is called
        Then the output dtype matches exactly
        """
        result = normalize_speech_loudness(
            audio=simple_speech_like_signal,
            sample_rate=sample_rate,
            return_dtype=np.float64,
        )

        assert result.dtype == np.float64
