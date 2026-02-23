"""Tests for Japanese ASR utility."""

import numpy as np
from utils.asr import ASRTranscriber, TranscriptionSegment


class TestJapaneseASR:
    """Test class to separate behaviors for ASR."""

    def setup_method(self):
        self.transcriber = ASRTranscriber(model_size="tiny")  # tiny for fast test

    def test_transcribe_japanese_asr_short_audio(self):
        # Given: short silent audio as real world test input
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        sample_rate = 16000

        # When: transcribe
        result_text, result_segments = self.transcriber.transcribe_japanese_asr(
            audio, sample_rate
        )

        # Then
        expected_text = ""
        expected_segments: list[TranscriptionSegment] = []
        assert result_text == expected_text
        assert result_segments == expected_segments

    def test_transcribe_japanese_asr_signature(self):
        # Given: valid audio
        audio = np.random.randn(32000).astype(np.float32)

        # When
        result = self.transcriber.transcribe_japanese_asr(audio)

        # Then: check types and structure
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, segments = result
        assert isinstance(text, str)
        assert isinstance(segments, list)
