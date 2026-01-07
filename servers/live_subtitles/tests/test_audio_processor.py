# tests/test_audio_processor.py

import os
import json
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from live_subtitles_client_with_overlay import Config
from audio_processor import AudioProcessor


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        segments = os.path.join(tmp, "segments")
        non_speech = os.path.join(tmp, "non_speech_segments")
        os.makedirs(segments)
        os.makedirs(non_speech)
        yield segments, non_speech


def generate_chunk(samples: int = 512, amplitude: float = 1000.0):
    """Generate a 512-sample int16 chunk (default VAD chunk size)."""
    data = np.sin(np.linspace(0, 2 * np.pi, samples)) * amplitude
    return (data.astype(np.int16)).tobytes()


@pytest.mark.asyncio
async def test_speech_segment_long_enough_is_saved(config, temp_dirs):
    segments_dir, non_speech_dir = temp_dirs

    vad_mock = MagicMock()
    vad_mock.chunk_samples.return_value = 512
    vad_mock.chunk_bytes.return_value = 1024
    vad_mock.return_value = 0.9  # high probability → speech

    # Controllable clock for tests
    test_time = 1000.0  # arbitrary start

    def mono():
        nonlocal test_time
        return test_time

    def wall():
        nonlocal test_time
        return test_time + 100.0  # wallclock offset irrelevant

    processor = AudioProcessor(
        config=config,
        vad=vad_mock,
        segments_dir=segments_dir,
        non_speech_dir=non_speech_dir,
        stream_start_time_ref={"value": None},
        segment_start_wallclock={},
        non_speech_wallclock={},
        current_time_mono=mono,
        current_time_wall=wall,
    )

    ws = AsyncMock()

    # Given: 20 chunks of speech (≈0.64s at 16kHz, > min 0.25s)
    for _ in range(20):
        test_time += 512 / 16000  # advance time by one chunk duration
        chunk = generate_chunk(amplitude=5000.0)
        rms = 5000.0 / np.sqrt(2)  # approx RMS of sine
        await processor.process_chunk(chunk, rms, 0.9, ws)

    # When: trigger silence long enough to end segment
    test_time += 0.2  # jump forward > 0.1s silence
    for _ in range(10):
        test_time += 512 / 16000
        await processor.process_chunk(generate_chunk(amplitude=10.0), 10.0, 0.1, ws)

    # Then: a speech segment was saved
    saved_dirs = [d for d in os.listdir(segments_dir) if d.startswith("segment_")]
    assert len(saved_dirs) == 1

    seg_dir = os.path.join(segments_dir, saved_dirs[0])
    assert os.path.exists(os.path.join(seg_dir, "sound.wav"))
    assert os.path.exists(os.path.join(seg_dir, "metadata.json"))
    assert os.path.exists(os.path.join(seg_dir, "speech_probabilities.json"))

    with open(os.path.join(seg_dir, "metadata.json")) as f:
        meta = json.load(f)
    expected_duration = round(20 * 512 / 16000, 3)
    assert meta["duration_sec"] == expected_duration
    assert meta["num_chunks"] == 20


@pytest.mark.asyncio
async def test_short_speech_burst_is_discarded(config, temp_dirs):
    segments_dir, _ = temp_dirs

    vad_mock = MagicMock()
    vad_mock.chunk_samples.return_value = 512
    vad_mock.chunk_bytes.return_value = 1024
    vad_mock.return_value = 0.9

    processor = AudioProcessor(
        config=config,
        vad=vad_mock,
        segments_dir=segments_dir,
        non_speech_dir=temp_dirs[1],
        stream_start_time_ref={"value": None},
        segment_start_wallclock={},
        non_speech_wallclock={},
    )

    ws = AsyncMock()

    # Given: only 3 chunks of speech → ~0.096s < 0.25s
    for _ in range(3):
        await processor.process_chunk(generate_chunk(amplitude=5000.0), 3500.0, 0.9, ws)

    # When: silence follows
    for _ in range(10):
        await processor.process_chunk(generate_chunk(amplitude=10.0), 10.0, 0.1, ws)

    # Then: no speech segment saved
    assert not any(d.startswith("segment_") for d in os.listdir(segments_dir))


@pytest.mark.asyncio
async def test_non_speech_long_or_loud_is_saved(config, temp_dirs):
    _, non_speech_dir = temp_dirs

    vad_mock = MagicMock()
    vad_mock.chunk_samples.return_value = 512
    vad_mock.chunk_bytes.return_value = 1024
    vad_mock.return_value = 0.1

    # Controllable clock for tests
    test_time = 2000.0

    def mono():
        nonlocal test_time
        return test_time

    def wall():
        nonlocal test_time
        return test_time + 100.0

    processor = AudioProcessor(
        config=config,
        vad=vad_mock,
        segments_dir=temp_dirs[0],
        non_speech_dir=non_speech_dir,
        stream_start_time_ref={"value": time.time() - 10.0},  # ensure base_time is set early
        segment_start_wallclock={},
        non_speech_wallclock={},
        current_time_mono=mono,
        current_time_wall=wall,
    )

    ws = AsyncMock()

    # Given: many low-energy silence chunks → will hit 3s threshold
    for _ in range(150):
        test_time += 512 / 16000
        await processor.process_chunk(generate_chunk(amplitude=20.0), 20.0, 0.1, ws)

    # Then: one non-speech segment saved (even if quiet, because duration >=3s)
    saved = [d for d in os.listdir(non_speech_dir) if d.startswith("segment_")]
    assert len(saved) == 1