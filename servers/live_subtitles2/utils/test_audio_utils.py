# utils/test_audio_utils.py
import numpy as np
import pytest

from .audio_utils import AudioStreamProcessor


@pytest.fixture
def dummy_asr():
    class DummyASR:
        def transcribe_japanese_asr(self, audio, sr):
            return "テスト", []

    return DummyASR()


@pytest.fixture
def dummy_translator():
    class DummyTrans:
        def translate_japanese_to_english(self, t):
            return "test"

    return DummyTrans()


def test_vad_processor_initialization(dummy_asr, dummy_translator):
    p = AudioStreamProcessor(dummy_asr, dummy_translator)
    assert p.vad_iterator is not None


def test_process_chunk_silence(dummy_asr, dummy_translator):
    p = AudioStreamProcessor(dummy_asr, dummy_translator)
    silence = np.zeros(512, dtype=np.float32)
    for _ in range(20):
        assert p.process_chunk(silence) is None
