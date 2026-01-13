# tests/test_transcribe_jonatasgrosman_wav2vec2.py
"""
Unit & integration-ish tests for JapaneseASR class

Note: These tests mostly use mocking because:
- We don't want to download 1.2GB+ model in CI
- Real inference is too slow/expensive for unit tests
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from typing import Any

from transcribe_jonatasgrosman_wav2vec2 import (
    JapaneseASR,
    TranscriptionResult,
    AudioInput,
    TARGET_SR,
    DEFAULT_MAX_CHUNK_SEC,
)


# ==================== Fixtures ====================

@pytest.fixture
def mock_processor():
    from transformers import BatchEncoding
    processor = MagicMock()
    fake_encoding = BatchEncoding(
        {
            "input_values": torch.randn(1, 32000),
            "attention_mask": torch.ones(1, 32000, dtype=torch.long)
        },
        tensor_type="pt"
    )
    processor.return_value = fake_encoding
    processor.batch_decode.return_value = ["これはテストです"]
    return processor


@pytest.fixture
def mock_model(mock_processor):
    model = MagicMock()
    model.device = "cpu"

    # For normal forward pass
    model.return_value = MagicMock(
        logits=torch.randn(1, 200, 10000)  # time, vocab
    )

    # For generate() call (beam search / long audio path)
    generate_out = MagicMock()
    generate_out.sequences = torch.tensor([[0, 1234, 5678, 2]])  # dummy token ids
    model.generate.return_value = generate_out

    return model


@pytest.fixture
def asr_instance(mock_processor, mock_model):
    with patch("transcribe_jonatasgrosman_wav2vec2.Wav2Vec2Processor.from_pretrained", return_value=mock_processor), \
         patch("transcribe_jonatasgrosman_wav2vec2.Wav2Vec2ForCTC.from_pretrained", return_value=mock_model), \
         patch("transcribe_jonatasgrosman_wav2vec2.torch.cuda.is_available", return_value=False):

        asr = JapaneseASR(device="cpu")
        yield asr


@pytest.fixture
def short_audio_16kHz():
    """~2 seconds @ 16kHz"""
    return np.sin(np.linspace(0, 2 * np.pi * 440 * 2, int(TARGET_SR * 2)), dtype=np.float32)


@pytest.fixture
def long_audio_16kHz():
    """~65 seconds → should trigger chunking"""
    return np.random.randn(int(TARGET_SR * 65)).astype(np.float32)


# ==================== Tests ====================

class TestJapaneseASRInitialization:

    def test_init_default_device(self):
        """GIVEN no device specified
        WHEN creating JapaneseASR
        THEN should pick cpu when cuda not available"""
        with patch("torch.cuda.is_available", return_value=False):
            asr = JapaneseASR()
            assert asr.device == "cpu"

    def test_init_respects_passed_device(self):
        asr = JapaneseASR(device="cpu")  # even if cuda would be available
        assert asr.device == "cpu"


class TestAudioLoadingAndPreprocessing:

    def test_loads_numpy_array_correctly(self, asr_instance):
        """GIVEN numpy float32 array + sample rate
        WHEN transcribe is called
        THEN should not crash on preprocessing"""
        arr = np.random.randn(16000).astype(np.float32) * 0.3

        result = asr_instance.transcribe(
            arr,
            input_sample_rate=TARGET_SR,
            return_confidence=False
        )

        assert isinstance(result, dict)
        assert "text" in result

    def test_raises_when_numpy_without_sr(self, asr_instance):
        arr = np.random.randn(16000).astype(np.float32)

        with pytest.raises(ValueError, match="input_sample_rate"):
            asr_instance.transcribe(arr)


class TestShortAudioTranscription:

    def test_short_audio_single_chunk_path(self, asr_instance, mock_processor, short_audio_16kHz):
        """GIVEN short audio (< max_chunk_sec)
        WHEN transcribe called
        THEN uses single forward pass, not generate()
        AND chunks_info = 'single chunk'"""

        result = asr_instance.transcribe(
            short_audio_16kHz,
            input_sample_rate=TARGET_SR,
            num_beams=1
        )

        assert result["chunks_info"] == "single chunk"
        assert result["text"] == "これはテストです"
        assert mock_processor.batch_decode.called
        assert not asr_instance.model.generate.called  # ← important!


class TestLongAudioChunking:

    def test_long_audio_triggers_chunking(self, asr_instance, long_audio_16kHz):
        """GIVEN long audio (> max_chunk_sec)
        WHEN transcribe()
        THEN should use chunking strategy"""

        result = asr_instance.transcribe(
            long_audio_16kHz,
            input_sample_rate=TARGET_SR,
            max_chunk_seconds=30.0,
            chunk_overlap_seconds=2.0
        )

        assert "chunks" in result["chunks_info"]
        assert asr_instance.model.generate.called  # long path uses generate()


class TestConfidenceAndLogprobs:

    def test_returns_avg_logprob_when_requested(self, asr_instance, short_audio_16kHz):
        result = asr_instance.transcribe(
            short_audio_16kHz,
            input_sample_rate=TARGET_SR,
            return_confidence=True,
            return_logprobs=False
        )

        assert "avg_logprob" in result
        assert "avg_confidence" in result
        assert isinstance(result["avg_logprob"], float)
        assert -10 < result["avg_logprob"] < 0  # rough sanity


    def test_quality_categories_assigned(self, asr_instance, short_audio_16kHz):
        # We force very good logprob for test
        with patch.object(asr_instance.model, "forward") as mock_forward:
            logits = torch.full((1, 200, 10000), -0.01)  # very confident
            logits[0, :, 1234] = 5.0  # make one token very likely
            mock_forward.return_value = MagicMock(logits=logits)

            result = asr_instance.transcribe(
                short_audio_16kHz,
                input_sample_rate=TARGET_SR,
                return_confidence=True
            )

            assert result["quality_avg_logprob"] in {"very_high", "high", "medium", "low", "very_low"}
            # most probably "very_high" in this artificial case


class TestEdgeCases:

    def test_empty_audio_raises_clear_error(self, asr_instance):
        empty_array = np.array([], dtype=np.float32)

        result = asr_instance.transcribe(
            empty_array,
            input_sample_rate=TARGET_SR
        )

        assert result["text"] == ""
        assert result["duration_sec"] == 0.0

    def test_very_short_audio(self, asr_instance):
        very_short = np.sin(np.linspace(0, 0.1, 1600), dtype=np.float32)  # ~0.1s

        result = asr_instance.transcribe(very_short, TARGET_SR)
        assert result["text"] != ""
        assert result["duration_sec"] == pytest.approx(0.1, abs=0.05)


class TestQualityThresholds:

    @pytest.mark.parametrize("value, expected_category", [
        (-0.3,  "very_high"),
        (-0.8,  "high"),
        (-1.2,  "medium"),
        (-2.1,  "low"),
        (-4.5,  "very_low"),
        (-0.95, "high"),     # exactly on boundary → should take >=
    ])
    def test_avg_logprob_quality_classification(self, value, expected_category):
        label = JapaneseASR._get_quality_label(
            value,
            JapaneseASR._get_quality_label.__globals__["QUALITY_THRESHOLDS_AVG_LOGPROB"]
        )
        assert label == expected_category