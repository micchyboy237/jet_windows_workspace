# tests/test_load_audio.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Import the function — replace with your actual module name
from audio_utils import load_audio  # <-- update this import

# Optional torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def sample_rate() -> int:
    return 16_000


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audio"
    d.mkdir()
    return d


# Given: A 16kHz mono WAV file saved on disk
def test_load_from_file_path(audio_dir: Path, sample_rate: int):
    # Create a simple 16kHz tone
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    path = audio_dir / "a440.wav"
    import soundfile as sf
    sf.write(path, signal, sample_rate)

    # When
    y = load_audio(str(path), sr=sample_rate, mono=True)

    # Then
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) == sample_rate
    assert -1.0 <= y.min() < y.max() <= 1.0
    assert_allclose(y.max(), 0.5, atol=1e-2)  # peak preserved


# Given: A stereo 48kHz WAV file
def test_resampling_and_mono_conversion(audio_dir: Path, sample_rate: int):
    t = np.linspace(0, 0.5, 48_000, endpoint=False)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([left, right])  # shape (2, 24000)

    path = audio_dir / "stereo_48k.wav"
    import soundfile as sf
    sf.write(path, stereo.T, 48_000)  # soundfile expects (frames, channels)

    # When
    y = load_audio(path, sr=sample_rate, mono=True)

    # Then
    assert y.shape == (sample_rate // 2,)  # 0.5s at 16kHz
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0


# Given: Raw int16 numpy array at 44.1kHz, shape (time,)
def test_numpy_int16_array_resampling_and_normalization(sample_rate: int):
    # Simulate 44.1kHz int16 recording
    t = np.linspace(0, 1.0, 44_100, endpoint=False)
    signal_int16 = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # When
    y = load_audio(signal_int16, sr=sample_rate, mono=True)

    # Then
    expected_len = sample_rate
    assert y.shape == (expected_len,)
    assert y.dtype == np.float32
    assert_allclose(np.abs(y).max(), 1.0, atol=1e-3)


# Given: Float numpy array in [-1, 1], wrong sample rate assumed
def test_numpy_float_array_already_normalized(sample_rate: int):
    clean = np.random.uniform(-0.8, 0.8, size=32000).astype(np.float32)

    # When
    y = load_audio(clean, sr=sample_rate)

    # Then
    assert_array_equal(y, clean)  # should not be re-normalized
    assert y.shape == (32000,)


# Given: Numpy array with channels last → (time, channels)
def test_numpy_channels_last_is_handled_correctly(sample_rate: int):
    stereo_time_first = np.random.randn(8000, 2).astype(np.float32)  # (time, ch)

    # When
    y = load_audio(stereo_time_first, sr=sample_rate, mono=True)

    # Then
    assert y.shape == (8000,)
    # Mono mix should be average of both channels
    expected = stereo_time_first.mean(axis=1)
    assert_allclose(y, expected, atol=1e-6)


# Given: Torch tensor (channels first)
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_torch_tensor_channels_first(sample_rate: int):
    if not HAS_TORCH:
        return
    tensor = torch.randn(2, 16000) * 0.5  # (ch, time)

    # When
    y = load_audio(tensor, sr=sample_rate, mono=True)

    # Then
    assert y.shape == (16000,)
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 0.5


# Given: In-memory WAV bytes (full file)
def test_bytes_input_full_wav(tmp_path: Path, sample_rate: int):
    import soundfile as sf
    t = np.linspace(0, 0.1, sample_rate // 10, endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)

    buf = tmp_path / "tone.wav"
    sf.write(buf, signal, sample_rate)
    wav_bytes = buf.read_bytes()

    y = load_audio(wav_bytes, sr=sample_rate, mono=True)

    assert y.shape == (len(signal),)
    assert y.dtype == np.float32
    assert_allclose(np.abs(y).max(), 0.3, atol=1e-3)


# Given: Invalid input type
def test_raises_on_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported audio input type"):
        load_audio(12345)  # int → not allowed