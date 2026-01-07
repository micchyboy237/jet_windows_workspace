# test_has_sound.py
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

# Optional import for torch tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from preprocessors import has_sound


@pytest.fixture
def temp_wav_dir(tmp_path: Path):
    """Create a temporary directory with silent and noisy WAV files."""
    dir_path = tmp_path / "audio"
    dir_path.mkdir()

    # Silent audio: 1 second of zeros at 44100 Hz, mono
    sample_rate = 44100
    duration = 1.0
    silent = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Noisy audio: sine wave at 440 Hz, amplitude 0.5 (RMS ≈ 0.353 → well above 0.01)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noisy = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    silent_path = dir_path / "silent.wav"
    noisy_path = dir_path / "noisy.wav"

    sf.write(str(silent_path), silent, sample_rate)
    sf.write(str(noisy_path), noisy, sample_rate)

    return {
        "silent_path": str(silent_path),
        "noisy_path": str(noisy_path),
        "sample_rate": sample_rate,
    }


def test_has_sound_with_numpy_array_silent():
    # Given: a completely silent NumPy array
    silent_audio = np.zeros(44100, dtype=np.float32)  # 1 second at 44.1kHz

    # When: checking if it has sound
    result = has_sound(silent_audio)

    # Then: it should be considered silent
    expected = False
    assert result == expected


def test_has_sound_with_numpy_array_has_sound():
    # Given: a NumPy array with a clear sine wave (RMS ≈ 0.353)
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # When: checking if it has sound
    result = has_sound(audio)

    # Then: it should be detected as having sound
    expected = True
    assert result == expected


def test_has_sound_with_numpy_array_near_threshold():
    # Given: very quiet audio just below default threshold (RMS ≈ 0.007 < 0.01)
    # Constant value array: RMS = absolute value of the constant
    quiet_audio = np.full(44100, 0.01, dtype=np.float32)        # RMS = 0.01
    quieter_audio = quiet_audio * 0.7                           # RMS = 0.007 exactly
    slightly_louder = quiet_audio * 0.85                        # RMS = 0.0085 exactly

    # When: checking with default threshold
    result = has_sound(quieter_audio)

    # Then: should be False
    expected = False
    assert result == expected

    # When: checking with lower custom threshold
    result_custom = has_sound(slightly_louder, threshold=0.008)

    # Then: should be True
    expected_custom = True
    assert result_custom == expected_custom


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_has_sound_with_torch_tensor():
    # Given: a torch tensor with clear sound
    audio_np = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 44100, endpoint=False))
    audio_tensor = torch.from_numpy(audio_np.astype(np.float32))

    # When: checking if it has sound
    result = has_sound(audio_tensor)

    # Then: should detect sound
    expected = True
    assert result == expected


def test_has_sound_with_file_path(temp_wav_dir):
    # Given: paths to silent and noisy WAV files
    silent_path = temp_wav_dir["silent_path"]
    noisy_path = temp_wav_dir["noisy_path"]

    # When / Then: silent file should return False
    result_silent = has_sound(silent_path)
    expected_silent = False
    assert result_silent == expected_silent

    # When / Then: noisy file should return True
    result_noisy = has_sound(noisy_path)
    expected_noisy = True
    assert result_noisy == expected_noisy


def test_has_sound_with_bytes_io(temp_wav_dir):
    # Given: in-memory bytes of a noisy WAV
    noisy_path = temp_wav_dir["noisy_path"]
    with open(noisy_path, "rb") as f:
        wav_bytes = f.read()

    # When: passing bytes directly
    result = has_sound(wav_bytes)

    # Then: should detect sound
    expected = True
    assert result == expected


def test_has_sound_empty_input():
    # Given: empty NumPy array
    empty_audio = np.array([], dtype=np.float32)

    # When: checking for sound
    result = has_sound(empty_audio)

    # Then: should return False (handled gracefully by get_audio_energy)
    expected = False
    assert result == expected