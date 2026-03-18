from collections import deque
from typing import Deque, Dict, List

import numpy as np


class AudioSegment(Dict):
    start_sec: float
    end_sec: float
    audio: np.ndarray


class AudioContextBuffer:
    """Circular buffer that accumulates timestamped audio segments and reconstructs
    a continuous audio stream with silence filling real gaps between segments.
    """

    def __init__(
        self,
        max_duration_sec: float = 30.0,
        sample_rate: int = 16000,
    ):
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.segments: Deque[AudioSegment] = deque()
        self.latest_end_sec: float = 0.0

    def add_audio_segment(
        self,
        start_sec: float,
        audio_np: np.ndarray,
    ) -> None:
        """Add segment with only start time. End time is computed from audio length.

        Fixes/validates duration using actual array length.
        """
        audio_np = audio_np.copy()
        seg_duration = len(audio_np) / self.sample_rate
        end_sec = start_sec + seg_duration

        # Enforce invariant: segment duration must not exceed max_duration_sec
        if seg_duration - self.max_duration_sec > 1e-6:
            raise ValueError(
                f"Segment duration ({seg_duration:.2f}s) exceeds "
                f"max_duration_sec ({self.max_duration_sec:.2f}s). "
                "Split the audio before adding."
            )

        self.segments.append(
            {"start_sec": start_sec, "end_sec": end_sec, "audio": audio_np}
        )
        self.latest_end_sec = max(self.latest_end_sec, end_sec)
        self._prune_old_segments()

    def _prune_old_segments(self) -> None:
        """Remove segments outside rolling window."""
        min_allowed = self.latest_end_sec - self.max_duration_sec
        while self.segments and self.segments[0]["end_sec"] <= min_allowed:
            self.segments.popleft()

    def get_context_audio(self) -> np.ndarray:
        """Reconstruct full audio timeline including real silence gaps."""
        if not self.segments:
            return np.array([], dtype=np.int16)

        segments: List[AudioSegment] = list(self.segments)
        output_audio: List[np.ndarray] = []
        current_time = segments[0]["start_sec"]

        for seg in segments:
            gap = seg["start_sec"] - current_time
            if gap > 0:
                silence_samples = int(gap * self.sample_rate)
                if silence_samples > 0:
                    output_audio.append(np.zeros(silence_samples, dtype=np.int16))
            output_audio.append(seg["audio"])
            current_time = seg["end_sec"]

        context_audio_np = np.concatenate(output_audio) if output_audio else np.array([])

        # Convert buffer's float32 [-1,1] back to proper int16 PCM bytes
        # so the ASR receives valid audio instead of garbage.
        context_audio_int16 = np.clip(
            context_audio_np * 32768.0, -32768, 32767
        ).astype(np.int16)
        return context_audio_int16

    def get_total_duration(self) -> float:
        """Total duration including real silence gaps."""
        if not self.segments:
            return 0.0

        segments: List[AudioSegment] = list(self.segments)
        total = 0.0
        current_time = segments[0]["start_sec"]

        for seg in segments:
            gap = seg["start_sec"] - current_time
            if gap > 0:
                total += gap
            seg_duration = seg["end_sec"] - seg["start_sec"]
            total += seg_duration
            current_time = seg["end_sec"]

        return total
