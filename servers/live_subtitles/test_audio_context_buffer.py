import numpy as np
import pytest
from audio_context_buffer import AudioContextBuffer

SAMPLE_RATE = 16000


def generate_audio(duration_sec: float, value: float = 1.0) -> np.ndarray:
    samples = int(duration_sec * SAMPLE_RATE)
    return np.full(samples, value, dtype=np.float32)


class TestAudioContextBuffer:
    def test_gap_inserts_silence(self):
        """
        Given two segments with a gap
        When reconstructing context audio
        Then silence is inserted between them
        """
        buffer = AudioContextBuffer(sample_rate=SAMPLE_RATE)
        seg1 = generate_audio(1.0)
        seg2 = generate_audio(1.0)

        buffer.add_audio_segment(0.0, seg1)
        buffer.add_audio_segment(2.0, seg2)

        result = buffer.get_context_audio()
        expected = np.concatenate(
            [
                seg1,
                np.zeros(SAMPLE_RATE, dtype=np.float32),  # 1 second silence
                seg2,
            ]
        )
        assert np.array_equal(result, expected)

    def test_no_gap_no_silence(self):
        """
        Given contiguous segments
        When reconstructing
        Then no silence is inserted
        """
        buffer = AudioContextBuffer(sample_rate=SAMPLE_RATE)
        seg1 = generate_audio(1.0)
        seg2 = generate_audio(1.0)

        buffer.add_audio_segment(0.0, seg1)
        buffer.add_audio_segment(1.0, seg2)

        result = buffer.get_context_audio()
        expected = np.concatenate([seg1, seg2])
        assert np.array_equal(result, expected)

    def test_overlap_no_silence(self):
        """
        Given overlapping segments
        When reconstructing
        Then no silence is inserted between overlap
        """
        buffer = AudioContextBuffer(sample_rate=SAMPLE_RATE)
        seg1 = generate_audio(2.0)
        seg2 = generate_audio(1.0)

        buffer.add_audio_segment(0.0, seg1)
        buffer.add_audio_segment(1.5, seg2)

        result = buffer.get_context_audio()
        expected = np.concatenate([seg1, seg2])
        assert np.array_equal(result, expected)

    def test_total_duration_with_gap(self):
        """
        Given segments with a gap
        When calculating total duration
        Then duration includes silence gap
        """
        buffer = AudioContextBuffer(sample_rate=SAMPLE_RATE)
        buffer.add_audio_segment(0.0, generate_audio(1.0))
        buffer.add_audio_segment(2.0, generate_audio(1.0))
        result = buffer.get_total_duration()
        expected = 3.0
        assert result == expected

    def test_total_duration_no_gap(self):
        """
        Given contiguous segments
        When calculating duration
        Then no extra time is added
        """
        buffer = AudioContextBuffer(sample_rate=SAMPLE_RATE)
        buffer.add_audio_segment(0.0, generate_audio(1.0))
        buffer.add_audio_segment(1.0, generate_audio(1.0))
        result = buffer.get_total_duration()
        expected = 2.0
        assert result == expected

    def test_empty_buffer(self):
        """
        Given no segments
        When retrieving audio and duration
        Then outputs are empty/zero
        """
        buffer = AudioContextBuffer(sample_rate=SAMPLE_RATE)
        audio = buffer.get_context_audio()
        duration = buffer.get_total_duration()
        assert audio.size == 0
        assert duration == 0.0

    def test_pruning_old_segments(self):
        """
        Given segments exceeding max_duration window
        When adding new segment
        Then old segments are pruned
        """
        buffer = AudioContextBuffer(max_duration_sec=2.0, sample_rate=SAMPLE_RATE)
        buffer.add_audio_segment(0.0, generate_audio(1.0))
        buffer.add_audio_segment(1.0, generate_audio(1.0))
        buffer.add_audio_segment(3.0, generate_audio(2.0))

        remaining = list(buffer.segments)
        assert len(remaining) == 1
        assert remaining[0]["start_sec"] == 3.0
        assert remaining[0]["end_sec"] == 5.0


    def test_segment_exceeds_max_duration_raises_error(self):
        """
        Given a segment longer than max_duration
        When adding to buffer
        Then a ValueError is raised
        """
        buffer = AudioContextBuffer(max_duration_sec=2.0, sample_rate=SAMPLE_RATE)

        seg = generate_audio(3.0)  # 3 seconds > 2 seconds

        with pytest.raises(ValueError) as exc_info:
            buffer.add_audio_segment(0.0, seg)

        result = str(exc_info.value)
        expected = "Segment duration"

        assert expected in result