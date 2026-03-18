# tests/test_audio_context_buffer.py
"""
Unit tests for the pure sample-based AudioContextBuffer (timestamp-ignoring version).
Run with: pytest tests/test_audio_context_buffer.py -v
"""

import pytest
import numpy as np
from audio_context_buffer import AudioContextBuffer


class TestAudioContextBuffer:
    def test_initialization(self):
        buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)
        assert buffer.max_samples == 480_000
        assert len(buffer.segments) == 0
        assert buffer.total_samples == 0
        assert buffer.get_total_duration() == 0.0

    def test_add_single_chunk_below_limit(self):
        buffer = AudioContextBuffer(max_duration_sec=5.0, sample_rate=16000)
        chunk_duration_sec = 2.5
        samples = int(chunk_duration_sec * 16000)
        # Nice deterministic test signal
        chunk = np.sin(2 * np.pi * np.arange(samples) / 16000 * 440).astype(np.float32)

        buffer.add_audio_segment(999.99, chunk)  # timestamp is deliberately ignored

        assert buffer.get_total_duration() == pytest.approx(chunk_duration_sec)
        context = buffer.get_context_audio()
        assert len(context) == samples
        assert context.dtype == np.int16

        # Verify exact round-trip conversion (float32 → int16)
        expected = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
        assert np.array_equal(context, expected)

    def test_pruning_when_exceeding_max_duration(self):
        """Add 4 × 1-second chunks → should keep only the last 3 seconds."""
        sr = 100  # tiny sample rate for exact math
        buffer = AudioContextBuffer(max_duration_sec=3.0, sample_rate=sr)
        chunk_samples = 100  # 1.0 s each

        chunks = []
        for i in range(4):
            amp = 0.1 * (i + 1)
            chunk = np.full(chunk_samples, amp, dtype=np.float32)
            chunks.append(chunk)
            buffer.add_audio_segment(0.0, chunk)

        # After 4th chunk: first chunk must be pruned
        assert buffer.get_total_duration() == pytest.approx(3.0)
        assert len(buffer.segments) == 3
        assert buffer.total_samples == 300

        context = buffer.get_context_audio()
        assert len(context) == 300

        expected_float = np.concatenate(chunks[1:])          # chunks 2, 3, 4
        expected_int16 = np.clip(expected_float * 32768.0, -32768, 32767).astype(np.int16)
        assert np.array_equal(context, expected_int16)

    def test_single_chunk_exactly_max_duration(self):
        buffer = AudioContextBuffer(max_duration_sec=2.0, sample_rate=100)
        chunk = np.zeros(200, dtype=np.float32)   # exactly 2 s
        buffer.add_audio_segment(0, chunk)

        assert buffer.get_total_duration() == 2.0
        assert len(buffer.segments) == 1
        assert len(buffer.get_context_audio()) == 200

    def test_chunk_larger_than_max_raises(self):
        buffer = AudioContextBuffer(max_duration_sec=5.0, sample_rate=16000)
        too_big = np.zeros(int(5.1 * 16000), dtype=np.float32)

        with pytest.raises(ValueError) as exc:
            buffer.add_audio_segment(0, too_big)
        assert "exceeds max_duration_sec" in str(exc.value)

    def test_empty_buffer(self):
        buffer = AudioContextBuffer(max_duration_sec=30.0)
        assert buffer.get_total_duration() == 0.0
        context = buffer.get_context_audio()
        assert len(context) == 0
        assert context.dtype == np.int16

    def test_many_small_chunks_prune_correctly(self):
        """150 × 0.1 s chunks = 15 s total → buffer must stay exactly at 10 s."""
        buffer = AudioContextBuffer(max_duration_sec=10.0, sample_rate=16000)
        chunk_samples = 1600  # 0.1 s

        for i in range(150):
            chunk = np.full(chunk_samples, 0.05, dtype=np.float32)
            buffer.add_audio_segment(0.0, chunk)

        assert buffer.get_total_duration() == pytest.approx(10.0, abs=1e-6)
        assert buffer.total_samples == 160_000
        assert len(buffer.get_context_audio()) == 160_000

    def test_timestamps_completely_ignored(self):
        """Even crazy timestamps must not affect behavior."""
        buffer = AudioContextBuffer(max_duration_sec=2.0, sample_rate=100)
        chunk = np.full(100, 0.5, dtype=np.float32)

        buffer.add_audio_segment(999999.0, chunk)   # nonsense start time
        buffer.add_audio_segment(-100.0, chunk)     # negative start time

        assert buffer.get_total_duration() == pytest.approx(2.0)
        assert len(buffer.segments) == 2
        # No exception, no weird gaps — timestamps are ignored

    def test_prune_after_exactly_max_samples(self):
        """Edge case: adding a chunk that lands exactly on the limit."""
        sr = 100
        buffer = AudioContextBuffer(max_duration_sec=3.0, sample_rate=sr)
        chunk = np.zeros(300, dtype=np.float32)  # exactly 3 s
        buffer.add_audio_segment(0, chunk)
        assert buffer.get_total_duration() == 3.0
        assert buffer.total_samples == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])