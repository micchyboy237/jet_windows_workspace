# servers/live_subtitles/tests/test_normalize_loudness.py
"""
Unit tests for the standalone loudness normalization function, including BDD and
I/O/input cases: numpy, torch.Tensor, file paths, WAV bytes.
"""

from __future__ import annotations

import pytest
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from io import BytesIO

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from preprocessors import normalize_loudness


# Helper to generate simple test tones
def sine_wave(freq: float, duration: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestNormalizeLoudness:
    SR = 44100

    @pytest.fixture
    def sine_audio(self):
        """Generate a standard sine wave used across multiple tests."""
        return sine_wave(freq=440, duration=5.0, sr=self.SR, amplitude=0.5)

    @pytest.fixture
    def temp_wav_path(self, tmp_path: Path, sine_audio):
        """Create a temporary WAV file with sine audio."""
        wav_path = tmp_path / "test_sine.wav"
        sf.write(wav_path, sine_audio, self.SR)
        return wav_path

    @pytest.fixture
    def wav_bytes(self, sine_audio):
        """Generate in-memory WAV bytes."""
        buffer = BytesIO()
        sf.write(buffer, sine_audio, self.SR, format='WAV')
        return buffer.getvalue()

    def test_normalizes_to_target_lufs(self):
        # Given: a moderate-level sine wave (should measure around -20 LUFS)
        audio = sine_wave(freq=440, duration=10.0, sr=self.SR, amplitude=0.5)

        # When: normalizing to -14 LUFS
        normalized = normalize_loudness(audio, self.SR, target_lufs=-14.0)

        # Then: resulting loudness should be close to target (±0.5 LUFS tolerance)
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5

    def test_file_path_input(self, temp_wav_path):
        """Given a file path to a WAV file
           When normalize_loudness is called with the path
           Then it loads the audio, infers sample rate, and normalizes correctly"""
        normalized = normalize_loudness(str(temp_wav_path), sample_rate=None)  # sample_rate ignored for path
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5
        assert normalized.shape == (self.SR * 5,)

    def test_bytes_input(self, wav_bytes):
        """Given raw WAV bytes
           When normalize_loudness is called with the bytes
           Then it loads the audio in-memory, infers sample rate, and normalizes"""
        normalized = normalize_loudness(wav_bytes, sample_rate=None)
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_torch_tensor_input(self, sine_audio):
        """Given a torch.Tensor
           When normalize_loudness is called with the tensor
           Then it converts to numpy and normalizes correctly"""
        tensor = torch.from_numpy(sine_audio)
        normalized = normalize_loudness(tensor, sample_rate=self.SR)
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5
        assert isinstance(normalized, np.ndarray)

    def test_preserves_silent_audio(self):
        # Given: nearly silent audio
        audio = np.random.uniform(-1e-6, 1e-6, size=44100 * 5).astype(np.float32)

        # When: attempting normalization
        normalized = normalize_loudness(audio, self.SR)

        # Then: output should be identical (no amplification of noise)
        np.testing.assert_array_equal(audio, normalized)

    def test_prevents_clipping(self):
        # Given: loud audio that would clip after normalization
        audio = sine_wave(freq=440, duration=5.0, sr=self.SR, amplitude=0.95)

        # When: normalizing (will require gain > 1.0)
        normalized = normalize_loudness(audio, self.SR, target_lufs=-14.0)

        # Then: peak should stay safely below 1.0
        peak = np.max(np.abs(normalized))
        assert peak <= 0.96  # headroom ensures ~0.95 max

    def test_handles_stereo_input(self):
        # Given: stereo sine waves (same signal on both channels)
        mono = sine_wave(freq=440, duration=3.0, sr=self.SR, amplitude=0.4)
        stereo = np.stack([mono, mono], axis=1)

        # When: normalizing stereo
        normalized = normalize_loudness(stereo, self.SR)

        # Then: shape preserved and loudness correct
        assert normalized.shape == stereo.shape
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5

    def test_custom_target_lufs(self):
        # Given: moderate audio
        audio = sine_wave(freq=1000, duration=8.0, sr=self.SR, amplitude=0.3)

        # When: normalizing to a different target (-20 LUFS)
        normalized = normalize_loudness(audio, self.SR, target_lufs=-20.0)

        # Then: result should match the custom target
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -20.5 <= result_lufs <= -19.5