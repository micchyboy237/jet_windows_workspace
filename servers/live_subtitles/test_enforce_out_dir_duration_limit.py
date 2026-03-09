from __future__ import annotations

import wave
from pathlib import Path

import pytest

from ws_server_subtitles_utils import (
    enforce_out_dir_duration_limit,
    MAX_TOTAL_AUDIO_SECONDS,
)


# -----------------------------
# Test utilities
# -----------------------------

def create_wav(path: Path, duration_sec: float, sr: int = 16000) -> None:
    """
    Create a silent WAV file with the given duration.
    """
    frames = int(duration_sec * sr)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * frames)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def tmp_audio_dir(tmp_path: Path):
    """
    Given a temporary audio directory
    """
    yield tmp_path

    # cleanup
    for f in tmp_path.glob("*"):
        f.unlink(missing_ok=True)


# ------------------------------------------------
# Tests
# ------------------------------------------------

class TestDurationRetention:

    def test_keeps_all_files_when_under_limit(self, tmp_audio_dir: Path):
        """
        Given total audio duration below retention limit
        When retention cleanup runs
        Then no files are deleted
        """

        create_wav(tmp_audio_dir / "a.wav", 10)
        create_wav(tmp_audio_dir / "b.wav", 20)
        create_wav(tmp_audio_dir / "c.wav", 30)

        enforce_out_dir_duration_limit(tmp_audio_dir)

        result = sorted(p.name for p in tmp_audio_dir.glob("*.wav"))
        expected = ["a.wav", "b.wav", "c.wav"]

        assert result == expected

    def test_removes_oldest_files_when_over_limit(self, tmp_audio_dir: Path):
        """
        Given several wav files exceeding 5 minutes total
        When retention cleanup runs
        Then the oldest files are removed
        """

        # create files totaling 360s
        create_wav(tmp_audio_dir / "1.wav", 120)
        create_wav(tmp_audio_dir / "2.wav", 120)
        create_wav(tmp_audio_dir / "3.wav", 120)

        enforce_out_dir_duration_limit(tmp_audio_dir)

        result = sorted(p.name for p in tmp_audio_dir.glob("*.wav"))

        # newest two should remain (240 sec <= 300)
        expected = ["2.wav", "3.wav"]

        assert result == expected

    def test_preserves_newest_audio_within_budget(self, tmp_audio_dir: Path):
        """
        Given multiple utterances
        When cleanup runs
        Then the newest files are prioritized
        """

        create_wav(tmp_audio_dir / "a.wav", 100)
        create_wav(tmp_audio_dir / "b.wav", 100)
        create_wav(tmp_audio_dir / "c.wav", 100)
        create_wav(tmp_audio_dir / "d.wav", 100)

        enforce_out_dir_duration_limit(tmp_audio_dir)

        result = sorted(p.name for p in tmp_audio_dir.glob("*.wav"))

        # newest three = 300 seconds
        expected = ["b.wav", "c.wav", "d.wav"]

        assert result == expected

    def test_corrupt_wav_does_not_break_cleanup(self, tmp_audio_dir: Path):
        """
        Given a corrupt wav file
        When cleanup runs
        Then it does not crash
        """

        create_wav(tmp_audio_dir / "valid.wav", 10)

        corrupt = tmp_audio_dir / "corrupt.wav"
        corrupt.write_bytes(b"not-a-wav")

        enforce_out_dir_duration_limit(tmp_audio_dir)

        result = sorted(p.name for p in tmp_audio_dir.glob("*"))

        expected = ["corrupt.wav", "valid.wav"]

        assert result == expected