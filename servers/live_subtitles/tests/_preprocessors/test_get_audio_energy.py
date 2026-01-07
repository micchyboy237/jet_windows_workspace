# test_get_audio_energy.py
import numpy as np
import pytest
from pathlib import Path
from io import BytesIO
import soundfile as sf

from preprocessors import get_audio_energy

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@pytest.fixture
def sample_rate() -> int:
    return 44100


@pytest.fixture
def sine_wave_mono(sample_rate) -> np.ndarray:
    """A 1-second sine wave at 440 Hz, amplitude 1.0 (common test tone)"""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def silent_audio() -> np.ndarray:
    return np.zeros(10000, dtype=np.float32)


@pytest.fixture
def full_scale_audio() -> np.ndarray:
    return np.ones(10000, dtype=np.float32)


@pytest.fixture
def temp_wav_file(sine_wave_mono, tmp_path: Path) -> str:
    """Create a temporary WAV file with the sine wave"""
    wav_path = tmp_path / "test_sine.wav"
    sf.write(str(wav_path), sine_wave_mono, 44100)
    return str(wav_path)


@pytest.fixture
def wav_bytes(sine_wave_mono) -> bytes:
    """In-memory WAV bytes of the sine wave"""
    buffer = BytesIO()
    sf.write(buffer, sine_wave_mono, 44100, format='WAV')
    return buffer.getvalue()


def test_silent_audio_numpy(silent_audio):
    # Given a silent NumPy array (mono)
    # When computing energy
    result = get_audio_energy(silent_audio)

    # Then it should be exactly zero
    expected = 0.0
    assert result == expected


def test_full_scale_dc_numpy(full_scale_audio):
    # Given a full-scale constant signal (all samples = 1.0)
    # When computing energy
    result = get_audio_energy(full_scale_audio)

    # Then RMS should be 1.0
    expected = 1.0
    assert result == expected


def test_sine_wave_numpy(sine_wave_mono):
    # Given a normalized sine wave (peak amplitude 1.0)
    # When computing energy
    result = get_audio_energy(sine_wave_mono)

    # Then RMS should be approximately 0.7071 (1 / sqrt(2))
    expected = np.sqrt(0.5)  # Exact theoretical value
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_multi_channel_same_content(sine_wave_mono):
    # Given a stereo signal where both channels are identical sine waves
    stereo = np.stack([sine_wave_mono, sine_wave_mono], axis=1)

    # When computing energy
    result = get_audio_energy(stereo)

    # Then global RMS remains the same as mono
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_multi_channel_opposite_phase(sine_wave_mono):
    # Given a stereo signal with opposite phases (L = sine, R = -sine)
    stereo_opp = np.stack([sine_wave_mono, -sine_wave_mono], axis=1)

    # When computing energy
    result = get_audio_energy(stereo_opp)

    # Then energy is preserved (same RMS as mono)
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_empty_array():
    # Given an empty NumPy array
    empty = np.array([], dtype=np.float32)

    # When computing energy
    result = get_audio_energy(empty)

    # Then return 0.0
    expected = 0.0
    assert result == expected


def test_empty_2d_array():
    # Given a 2D empty array (0 frames, 2 channels)
    empty_2d = np.zeros((0, 2), dtype=np.float32)

    # When computing energy
    result = get_audio_energy(empty_2d)

    # Then return 0.0
    expected = 0.0
    assert result == expected


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_tensor(sine_wave_mono):
    # Given a torch.Tensor version of the sine wave
    tensor = torch.from_numpy(sine_wave_mono)

    # When computing energy
    result = get_audio_energy(tensor)

    # Then same RMS as NumPy version
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_file_path_input(temp_wav_file, sine_wave_mono):
    # Given a path to a valid WAV file containing the sine wave
    # When computing energy
    result = get_audio_energy(temp_wav_file)

    # Then matches the direct NumPy result
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_bytes_input(wav_bytes):
    # Given raw WAV bytes in memory
    # When computing energy
    result = get_audio_energy(wav_bytes)

    # Then matches the expected RMS
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_unsupported_input():
    # Given an unsupported input type (e.g., list)
    bad_input = [1.0, 2.0, 3.0]

    # When calling the function
    # Then it should raise ValueError
    with pytest.raises(ValueError, match="Unsupported audio input type"):
        get_audio_energy(bad_input)