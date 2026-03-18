from collections import deque
from typing import Deque, List
import numpy as np


class AudioContextBuffer:
    """Pure sample-based circular buffer (ignores all client timestamps).

    Keeps only the most recent max_duration_sec of audio.
    Chunks are treated as contiguous live audio — no silence gaps are inserted.
    Extremely memory efficient and guaranteed to never exceed the limit.
    """

    def __init__(
        self,
        max_duration_sec: float = 30.0,
        sample_rate: int = 16000,
    ):
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.max_samples: int = int(max_duration_sec * sample_rate)
        self.segments: Deque[np.ndarray] = deque()   # stores float32 normalized [-1, 1]
        self.total_samples: int = 0

    def add_audio_segment(
        self,
        start_sec: float,          # ← ignored (kept only for API compatibility)
        audio_np: np.ndarray,
    ) -> None:
        """Add a new audio chunk. Timestamps are completely ignored."""
        audio_np = audio_np.copy()
        chunk_samples = len(audio_np)

        # Same safety check as before (single chunk must not exceed limit)
        seg_duration = chunk_samples / self.sample_rate
        if seg_duration - self.max_duration_sec > 1e-6:
            raise ValueError(
                f"Segment duration ({seg_duration:.2f}s) exceeds "
                f"max_duration_sec ({self.max_duration_sec:.2f}s). "
                "Split the audio before adding."
            )

        self.segments.append(audio_np)
        self.total_samples += chunk_samples

        self._prune_old_segments()

    def _prune_old_segments(self) -> None:
        """Remove oldest chunks until we are back under max_samples."""
        while self.segments and self.total_samples > self.max_samples:
            oldest = self.segments.popleft()
            self.total_samples -= len(oldest)

    def get_context_audio(self) -> np.ndarray:
        """Return the last N seconds as int16 PCM (exactly what FunASR expects)."""
        if not self.segments:
            return np.array([], dtype=np.int16)

        # All chunks are already float32 normalized → just concatenate
        segments: List[np.ndarray] = list(self.segments)
        context_audio_np: np.ndarray = np.concatenate(segments)

        # Convert back to int16 exactly like the old code did
        context_audio_int16 = np.clip(
            context_audio_np * 32768.0, -32768, 32767
        ).astype(np.int16)

        return context_audio_int16

    def get_total_duration(self) -> float:
        """Returns current buffered duration (will always be ≤ max_duration_sec)."""
        return self.total_samples / self.sample_rate