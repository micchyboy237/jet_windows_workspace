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


class TestNormalizeLoudnessReturnDtype:
    """Test all supported return_dtype options for normalize_loudness."""

    SR = 44100

    @pytest.fixture
    def sine_audio(self):
        return sine_wave(freq=440, duration=5.0, sr=self.SR, amplitude=0.5)

    @pytest.fixture
    def silent_audio(self):
        return np.zeros(44100 * 2, dtype=np.float32)

    def test_return_float32(self, sine_audio):
        """Given return_dtype='float32'
           When normalize_loudness is called
           Then output is float32, within [-1, 1], and reaches target loudness"""
        normalized = normalize_loudness(sine_audio, self.SR, return_dtype="float32")
        expected_dtype = np.float32
        result_dtype = normalized.dtype
        assert result_dtype == expected_dtype
        assert np.max(np.abs(normalized)) <= 1.0
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5

    def test_return_float64(self, sine_audio):
        """Given return_dtype='float64'
           When normalize_loudness is called
           Then output is float64 with higher precision"""
        normalized = normalize_loudness(sine_audio, self.SR, return_dtype="float64")
        expected_dtype = np.float64
        result_dtype = normalized.dtype
        assert result_dtype == expected_dtype
        assert np.max(np.abs(normalized)) <= 1.0
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.5 <= result_lufs <= -13.5

    def test_return_int16(self, sine_audio):
        """Given return_dtype='int16'
           When normalize_loudness is called
           Then output is int16, scaled correctly, no overflow, and loudness close to target"""
        normalized = normalize_loudness(sine_audio, self.SR, return_dtype="int16")
        expected_dtype = np.int16
        result_dtype = normalized.dtype
        assert result_dtype == expected_dtype
        assert normalized.min() >= -32768
        assert normalized.max() <= 32767
        # Convert back to float for loudness measurement
        float_audio = normalized.astype(np.float32) / 32767.0
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(float_audio)
        assert -14.5 <= result_lufs <= -13.5

    def test_return_int32(self, sine_audio):
        """Given return_dtype='int32'
           When normalize_loudness is called
           Then output is int32, scaled correctly, no overflow, and loudness close to target"""
        normalized = normalize_loudness(sine_audio, self.SR, return_dtype="int32")
        expected_dtype = np.int32
        result_dtype = normalized.dtype
        assert result_dtype == expected_dtype
        assert normalized.min() >= -2147483648
        assert normalized.max() <= 2147483647
        # Convert back to float for loudness measurement
        float_audio = normalized.astype(np.float64) / 2147483647.0
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(float_audio)
        assert -14.5 <= result_lufs <= -13.5

    def test_return_dtype_none_defaults_to_float32(self, sine_audio):
        """Given return_dtype=None (default)
           When normalize_loudness is called
           Then output remains float32 (backward compatibility)"""
        normalized = normalize_loudness(sine_audio, self.SR)
        expected_dtype = np.float32
        result_dtype = normalized.dtype
        assert result_dtype == expected_dtype

    def test_silent_audio_preserved_across_dtypes(self, silent_audio):
        """Given nearly silent audio
           When normalized with any return_dtype
           Then output values remain zero (no amplification of noise)"""
        for dtype_str in ["float32", "float64", "int16", "int32"]:
            normalized = normalize_loudness(silent_audio, self.SR, return_dtype=dtype_str)
            expected_all_zero = np.zeros_like(normalized)
            np.testing.assert_array_equal(normalized, expected_all_zero)


class TestNormalizeLoudnessSpeechMode:
    """Comprehensive tests for speech mode: ensures consistent loudness and higher allowed peaks."""

    SR = 44100

    @pytest.fixture
    def moderate_speech_like_audio(self):
        """
        Typical unprocessed speech recording level.
        Amplitude 0.5 → measured loudness ≈ -16 to -17 LUFS.
        Represents clear, well-recorded speech (common in podcasts/videos before normalization).
        """
        return sine_wave(freq=440, duration=8.0, sr=self.SR, amplitude=0.5)

    @pytest.fixture
    def loud_speech_like_audio(self):
        """
        Overly loud speech recording (e.g., close-mic shouting).
        Amplitude 0.85 → measured loudness ≈ -5 LUFS → will be attenuated to target.
        """
        return sine_wave(freq=440, duration=6.0, sr=self.SR, amplitude=0.85)

    @pytest.fixture
    def low_speech_like_audio(self):
        """
        Very quiet distant speech recording requiring significant gain.
        Amplitude 0.15 → measured loudness ≈ -27 to -28 LUFS → post-gain peak > 1.0 → tests headroom.
        """
        return sine_wave(freq=440, duration=7.0, sr=self.SR, amplitude=0.15)

    @pytest.fixture
    def mixed_speech_like_audio(self):
        """
        Real-world speech simulation: silence → normal speech → loud burst → silence.
        Tests dynamic behavior and ensures silence isn't amplified.
        """
        sr = self.SR
        total_samples = sr * 10
        audio = np.zeros(total_samples, dtype=np.float32)

        # 0–2s: silence
        # 2–6s: moderate speech (amplitude 0.5)
        segment = sine_wave(freq=440, duration=4.0, sr=sr, amplitude=0.5)
        audio[2*sr:6*sr] = segment

        # 6–7s: loud burst (amplitude 0.8)
        burst = sine_wave(freq=440, duration=1.0, sr=sr, amplitude=0.8)
        audio[6*sr:7*sr] = burst

        # 7–10s: silence
        return audio

    def test_speech_mode_makes_all_inputs_consistently_loud(
        self,
        moderate_speech_like_audio,
        loud_speech_like_audio,
        low_speech_like_audio,
        mixed_speech_like_audio,
    ):
        """Given various speech recordings (quiet, normal, loud, mixed)
           When normalized with mode='speech'
           Then all result in integrated loudness very close to -13.0 LUFS"""
        inputs = [
            ("moderate", moderate_speech_like_audio),
            ("loud", loud_speech_like_audio),
            ("low", low_speech_like_audio),
            ("mixed", mixed_speech_like_audio),
        ]

        meter = pyln.Meter(self.SR)

        for name, audio in inputs:
            normalized = normalize_loudness(audio, self.SR, mode="speech")
            result_lufs = meter.integrated_loudness(normalized)
            assert -13.6 <= result_lufs <= -12.4, f"{name} speech audio failed: {result_lufs:.2f} LUFS"

    def test_speech_mode_allows_higher_peaks_than_general(
        self,
        low_speech_like_audio,
    ):
        """Given very quiet speech requiring strong gain
           When normalized in speech vs general mode
           Then speech mode allows significantly higher peak level"""
        normalized_speech = normalize_loudness(low_speech_like_audio, self.SR, mode="speech")
        normalized_general = normalize_loudness(low_speech_like_audio, self.SR, mode="general")

        peak_speech = np.max(np.abs(normalized_speech))
        peak_general = np.max(np.abs(normalized_general))

        # Speech mode: headroom 1.0 → peak very close to 1.0
        assert peak_speech >= 0.98
        assert peak_speech <= 1.0

        # General mode: headroom 1.05 → peak noticeably lower
        assert peak_general <= 0.97

        # Clear difference
        assert peak_speech > peak_general + 0.03

    def test_speech_mode_respects_explicit_target_lufs(
        self,
        moderate_speech_like_audio,
    ):
        """Given mode='speech' but user sets custom target_lufs
           When normalized
           Then custom target is used, not the speech preset"""
        normalized = normalize_loudness(
            moderate_speech_like_audio,
            self.SR,
            target_lufs=-16.0,
            mode="speech"
        )
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -16.6 <= result_lufs <= -15.4

    def test_general_mode_preserves_original_behavior(
        self,
        moderate_speech_like_audio,
    ):
        """Given explicit mode='general'
           When normalized
           Then uses -14.0 LUFS and conservative headroom"""
        normalized = normalize_loudness(moderate_speech_like_audio, self.SR, mode="general")
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -14.6 <= result_lufs <= -13.4

        peak = np.max(np.abs(normalized))
        assert peak <= 0.97  # conservative headroom

    def test_invalid_mode_raises_error(
        self,
        moderate_speech_like_audio,
    ):
        """Given invalid mode value
           When calling normalize_loudness
           Then ValueError is raised"""
        with pytest.raises(ValueError, match="Invalid mode"):
            normalize_loudness(moderate_speech_like_audio, self.SR, mode="music")

    def test_mixed_audio_preserves_silence_in_speech_mode(
        self,
        mixed_speech_like_audio,
    ):
        """Given speech with silent sections in speech mode
           When normalized
           Then silent parts remain near zero (no noise amplification)"""
        normalized = normalize_loudness(mixed_speech_like_audio, self.SR, mode="speech")

        # First 1 second should still be near silent
        silent_segment = normalized[:self.SR]
        assert np.max(np.abs(silent_segment)) < 1e-5

        # Last 2 seconds should be silent
        tail_segment = normalized[-2*self.SR:]
        assert np.max(np.abs(tail_segment)) < 1e-5


        # INSERT_YOUR_CODE

class TestNormalizeLoudnessMaxThreshold:
    """Dedicated tests for the optional max_loudness_threshold parameter."""

    SR = 44100

    @pytest.fixture
    def moderate_speech_like_audio(self):
        """Typical speech level (~ -16 to -17 LUFS with amplitude 0.5)."""
        return sine_wave(freq=440, duration=8.0, sr=self.SR, amplitude=0.5)

    @pytest.fixture
    def loud_speech_like_audio(self):
        """Overly loud speech (~ -5 LUFS)."""
        return sine_wave(freq=440, duration=6.0, sr=self.SR, amplitude=0.85)

    @pytest.fixture
    def low_speech_like_audio(self):
        """Very quiet speech (~ -27 LUFS)."""
        return sine_wave(freq=440, duration=7.0, sr=self.SR, amplitude=0.15)

    def test_max_threshold_prevents_amplification_of_loud_content(
        self,
        loud_speech_like_audio,
    ):
        """Given already loud audio and max_loudness_threshold set below measured
           When normalized
           Then no amplification occurs – output loudness remains close to input"""
        # Set threshold to -10.0 (well below measured ~ -5 LUFS)
        normalized = normalize_loudness(
            loud_speech_like_audio,
            self.SR,
            max_loudness_threshold=-10.0,
            mode="speech"  # speech mode targets -13.0
        )
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)

        # Should not be boosted toward -13 → stays near original (~ -5 LUFS)
        # But still attenuated slightly if needed (here original is louder than target)
        assert result_lufs < -8.0  # clearly not amplified

    def test_max_threshold_allows_normal_amplification_when_below(
        self,
        low_speech_like_audio,
    ):
        """Given quiet audio and max_loudness_threshold above measured
           When normalized
           Then full amplification to target occurs"""
        normalized = normalize_loudness(
            low_speech_like_audio,
            self.SR,
            max_loudness_threshold=-10.0,  # way above measured ~ -27
            mode="speech"
        )
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -13.6 <= result_lufs <= -12.4

    def test_max_threshold_still_allows_attenuation(
        self,
        loud_speech_like_audio,
    ):
        """Given very loud audio exceeding both threshold and target
           When normalized with threshold
           Then attenuation toward target still occurs"""
        normalized = normalize_loudness(
            loud_speech_like_audio,
            self.SR,
            max_loudness_threshold=-8.0,  # between original (~-5) and target (-13)
            target_lufs=-14.0  # general mode for clearer attenuation
        )
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        # Should be attenuated close to -14 (not capped at -8)
        assert -14.6 <= result_lufs <= -13.4

    def test_max_threshold_none_behaves_as_before(
        self,
        low_speech_like_audio,
    ):
        """Given max_loudness_threshold=None (default)
           When normalized
           Then full bidirectional normalization occurs"""
        normalized = normalize_loudness(
            low_speech_like_audio,
            self.SR,
            mode="speech"
        )
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        assert -13.6 <= result_lufs <= -12.4

    def test_max_threshold_with_moderate_input_and_speech_mode(
        self,
        moderate_speech_like_audio,
    ):
        """Given moderate input (~ -16 LUFS) and threshold slightly above measured
           When in speech mode (target -13)
           Then amplification is blocked if threshold prevents it"""
        normalized = normalize_loudness(
            moderate_speech_like_audio,
            self.SR,
            max_loudness_threshold=-15.0,  # just above measured
            mode="speech"
        )
        meter = pyln.Meter(self.SR)
        result_lufs = meter.integrated_loudness(normalized)
        # Should stay near original, not boosted to -13
        assert result_lufs < -14.0
