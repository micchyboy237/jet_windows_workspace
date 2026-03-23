from collections import deque
from typing import Deque, List, Optional, TypedDict
import numpy as np


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    duration_ms: Optional[int]
    word: Optional[str]


class SegmentMeta(TypedDict):
    uuid: str
    forced: bool
    vad_reason: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    started_at: str
    full_word_segments: list[WordSegment]
    full_ja_text: str
    full_en_text: str
    ja_text: str
    en_text: str


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
        self.segments: Deque[tuple[np.ndarray[np.int16], SegmentMeta]] = deque()
        self.total_samples: int = 0

    # ────────────────────────────────────────────────
    def add_audio_segment(
        self,
        audio_np: np.ndarray,
        meta: SegmentMeta,
    ) -> None:
        """Add a new audio chunk. Timestamps are completely ignored."""
        if audio_np.dtype != np.int16:
            raise TypeError("add_audio_segment expects np.int16 array")

        audio_np = audio_np.astype(np.int16, copy=True)   # ensure int16
        chunk_samples = len(audio_np)

        seg_duration = chunk_samples / self.sample_rate
        if seg_duration - self.max_duration_sec > 1e-6:
            raise ValueError(
                f"Segment duration ({seg_duration:.2f}s) exceeds "
                f"max_duration_sec ({self.max_duration_sec:.2f}s). "
                "Split the audio before adding."
            )

        self.segments.append((audio_np, meta))
        self.total_samples += chunk_samples

        self._prune_old_segments()

    # ────────────────────────────────────────────────
    def _prune_old_segments(self) -> None:
        while self.segments and self.total_samples > self.max_samples:
            oldest_audio, _ = self.segments.popleft()
            self.total_samples -= len(oldest_audio)

    def get_context_audio(self) -> np.ndarray:
        """Return the last N seconds as int16 PCM (exactly what FunASR expects)."""
        if not self.segments:
            return np.array([], dtype=np.int16)

        # All segments are already int16 → just concatenate
        return np.concatenate([audio for audio, _ in self.segments]).astype(np.int16)

    def get_total_duration(self) -> float:
        """Returns current buffered duration (will always be ≤ max_duration_sec)."""
        return self.total_samples / self.sample_rate

    def get_latest_segment(self) -> tuple[np.ndarray, SegmentMeta] | tuple[None, None]:
        """Return the most recently added audio segment and its metadata.
        
        Returns:
            tuple (audio_np: np.ndarray, meta: SegmentMeta) if buffer is not empty,
            None otherwise.
        """
        if not self.segments:
            return None, None
        return self.segments[-1]   # most recent is at the right end

    def get_list_metadata(self) -> List[SegmentMeta]:
        """Return list of metadata for all buffered segments.

        Returns:
            List[SegmentMeta]: Ordered metadata list.
        """
        if not self.segments:
            return []

        return [meta for _, meta in self.segments]

    def reset(self) -> None:
        """Clear all buffered audio and metadata, reset total ample count to 0."""
        self.segments.clear()
        self.total_samples = 0
