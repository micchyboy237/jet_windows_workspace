from collections import deque
from typing import Deque, List, Optional, TypedDict
import numpy as np


class SegmentMeta(TypedDict):
    uuid: str
    forced: bool
    vad_reason: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    started_at: str
    matched_pos: int
    matched_sent: str
    old_sents: list[str]
    new_sents: list[str]
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

    def get_prepared_audio_for_transcription(
        self,
        current_audio_np: np.ndarray,
        max_transcribe_sec: float = 30.0,
    ) -> np.ndarray:
        """
        Return context + current audio, trimmed from the oldest part if needed.

        Guarantees the audio passed to ReazonSpeech k2-asr never exceeds the 30 s hard limit.
        Returns the *most recent* possible audio (exactly what the buffer will contain
        after the subsequent add_audio_segment + prune).

        Args:
            current_audio_np: New segment (must be int16 PCM).
            max_transcribe_sec: Hard limit (matches TOO_LONG_SECONDS in ReazonSpeech).

        Returns:
            np.ndarray[int16]: Audio ready for transcription (≤ max_transcribe_sec).
        """
        if current_audio_np.dtype != np.int16:
            current_audio_np = current_audio_np.astype(np.int16, copy=True)

        context_audio = self.get_context_audio()
        if context_audio.size == 0:
            return current_audio_np

        full_audio = np.concatenate([context_audio, current_audio_np])

        max_samples = int(max_transcribe_sec * self.sample_rate)
        if len(full_audio) > max_samples:
            excess_samples = len(full_audio) - max_samples
            full_audio = full_audio[excess_samples:]

        return full_audio


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

    def get_last_segment(self) -> tuple[np.ndarray, SegmentMeta] | tuple[None, None]:
        """Return the most recently added audio segment and its metadata.
        
        Returns:
            tuple (audio_np: np.ndarray, meta: SegmentMeta) if buffer is not empty,
            None otherwise.
        """
        if not self.segments:
            return None, None
        return self.segments[-1]   # most recent is at the right end

    def get_last_sentence(self) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """Return the most recent sentence with its utterance ID and sentence index.

        Returns:
            tuple:
                sentence (Optional[str])
                utt_id (Optional[str])
                sent_idx (Optional[int])
        """
        if not self.segments:
            return None, None, None

        # iterate backwards (most recent first)
        for _, meta in reversed(self.segments):
            new_sents = meta.get("new_sents")
            if not new_sents:
                continue

            # iterate backwards within sentences to ensure non-empty
            for idx in range(len(new_sents) - 1, -1, -1):
                sent = new_sents[idx]
                if sent:
                    sent = sent.strip()
                    if sent:
                        return sent, meta["uuid"], idx

        return None, None, None

    def get_list_metadata(self) -> List[SegmentMeta]:
        """Return list of metadata for all buffered segments.

        Returns:
            List[SegmentMeta]: Ordered metadata list.
        """
        if not self.segments:
            return []

        return [meta for _, meta in self.segments]
    
    def get_context_uuid(self) -> Optional[str]:
        """Return the UUID of the first (oldest) segment in the buffer.

        Returns:
            Optional[str]: UUID if buffer is not empty, otherwise None.
        """
        if not self.segments:
            return None

        _, meta = self.segments[0]
        return meta["uuid"]

    def reset(self) -> None:
        """Clear all buffered audio and metadata, reset total ample count to 0."""
        self.segments.clear()
        self.total_samples = 0
