from collections import deque
from typing import Deque, List, Optional, TypedDict, Literal, Tuple
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
    old_ja_sents: list[str]
    new_ja_sents: list[str]
    old_en_sents: list[str]
    new_en_sents: list[str]
    full_ja_text: str
    full_en_text: str
    ja_text: str
    en_text: str


class TranslationHistoryItem(TypedDict):
    """Structured history item for LLM chat completion."""
    role: Literal["user", "assistant"]
    content: str


class HistoryResult(TypedDict):
    history: List[TranslationHistoryItem]
    included_indices: set[int]
    included_duration_sec: float
    excluded_duration_sec: float
    total_segments: int
    included_segments: int
    excluded_segments: int


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

    def get_context_audio(self, max_segments: Optional[int] = None) -> np.ndarray:
        """Return the last N seconds as int16 PCM (exactly what FunASR expects).

        Args:
            max_segments: If provided, only include audio from the last N segments.
                        If None, includes all buffered segments.
        """
        if not self.segments:
            return np.array([], dtype=np.int16)

        segments = (
            list(self.segments)[-max_segments:]
            if max_segments is not None
            else self.segments
        )

        return np.concatenate([audio for audio, _ in segments]).astype(np.int16)

    def get_context_audio_within_limit(
        self,
        new_audio_duration_sec: float,
    ) -> tuple[np.ndarray, float, int]:
        """Return context audio trimmed at SEGMENT boundaries so (context + new) fits
        within max_duration_sec. Never slices mid-segment audio to avoid word/sentence cutoff.

        Trimming strategy:
        - Drop WHOLE oldest segments until the remaining context + new chunk fits.
        - This means we may use slightly less than the maximum allowed context,
          but we never cut audio mid-word.

        Args:
            new_audio_duration_sec: Duration in seconds of the incoming new audio chunk.

        Returns:
            tuple:
                context_audio (np.ndarray int16): Safe context audio to prepend.
                actual_context_sec (float): Duration of returned context audio.
                segments_used (int): How many buffer segments were included.
        """
        if not self.segments:
            return np.array([], dtype=np.int16), 0.0, 0

        allowed_context_sec = max(0.0, self.max_duration_sec - new_audio_duration_sec)

        # Walk newest → oldest, accumulate WHOLE segments only.
        segments_list = list(self.segments)
        selected: list[np.ndarray] = []
        accumulated_sec = 0.0

        for audio, _ in reversed(segments_list):
            seg_duration = len(audio) / self.sample_rate
            if accumulated_sec + seg_duration > allowed_context_sec:
                # This whole segment would overflow — skip it (don't slice).
                break
            selected.append(audio)
            accumulated_sec += seg_duration

        if not selected:
            return np.array([], dtype=np.int16), 0.0, 0

        selected.reverse()  # restore chronological order (oldest → newest)
        context_audio = np.concatenate(selected).astype(np.int16)
        actual_context_sec = len(context_audio) / self.sample_rate
        return context_audio, actual_context_sec, len(selected)

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

    def get_last_sentence(self) -> tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        """Return the most recent JA/EN sentence pair with its utterance ID and sentence index.

        Returns:
            tuple:
                ja_sentence (Optional[str])
                en_sentence (Optional[str])
                utt_id (Optional[str])
                sent_idx (Optional[int])
        """
        if not self.segments:
            return None, None, None, None

        # iterate backwards (most recent segment first)
        for _, meta in reversed(self.segments):
            new_ja_sents = meta.get("new_ja_sents")
            new_en_sents = meta.get("new_en_sents")

            if not new_ja_sents:
                continue

            # iterate backwards within JA sentences
            for idx in range(len(new_ja_sents) - 1, -1, -1):
                ja_sent = new_ja_sents[idx]
                if not ja_sent:
                    continue

                ja_sent = ja_sent.strip()
                if not ja_sent:
                    continue

                # safely get aligned EN sentence (if exists)
                en_sent: Optional[str] = None
                if new_en_sents and idx < len(new_en_sents):
                    en_candidate = new_en_sents[idx]
                    if en_candidate:
                        en_sent = en_candidate.strip() or None

                return ja_sent, en_sent, meta["uuid"], idx

        return None, None, None, None
   

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

    def get_context_history_by_duration(
        self,
        max_duration_sec: float = 30.0,
        reserved_duration_sec: float = 0.0,
    ) -> HistoryResult:
        """
        Build translation history from buffered segments, filtered by total
        duration instead of segment count.

        Walks segments from NEWEST to OLDEST, accumulating each segment's
        ``duration_sec`` field.  Stops adding segments the moment the running
        total would exceed the effective time budget
        (max_duration_sec - reserved_duration_sec). The accepted segments are
        then reversed so the result is in chronological (oldest-first) order,
        matching what the LLM expects as a conversation history.

        Duration is accumulated for ALL segments in the walk (including those
        with empty text), so the time budget stays consistent with the actual
        audio buffer regardless of which segments have usable text.

        Args:
            max_duration_sec: Maximum total audio duration (in seconds) of
                segments to include.  Defaults to 30.0 s.
            reserved_duration_sec: Duration (in seconds) to reserve from the
                budget before walking segments — e.g. the incoming new audio
                chunk duration. The effective budget becomes
                (max_duration_sec - reserved_duration_sec).

        Returns:
            HistoryResult with history messages, included indices, and
            duration accounting for both included and excluded segments.
        """
        effective_max = max(0.0, max_duration_sec - reserved_duration_sec)
        if not self.segments:
            return HistoryResult(
                history=[],
                included_indices=set(),
                included_duration_sec=0.0,
                excluded_duration_sec=0.0,
                total_segments=0,
                included_segments=0,
                excluded_segments=0,
            )

        selected: List[Tuple[int, str, str]] = []  # (index, ja_text, en_text)
        # Snapshot once — consistent view, safe from concurrent prune calls.
        accumulated_sec = 0.0
        segments_list = list(self.segments)

        # Walk newest → oldest so we always keep the most recent context.
        for i, (_, meta) in reversed(list(enumerate(segments_list))):
            ja = (meta.get("ja_text") or "").strip()
            en = (meta.get("en_text") or "").strip()
            seg_duration = float(meta.get("duration_sec") or 0.0)

            # Stop once adding this segment would exceed the time budget.
            if accumulated_sec + seg_duration > effective_max:
                break

            # Always accumulate duration — even for text-empty segments —
            # so the budget matches the real audio window.
            accumulated_sec += seg_duration

            # Only include segments that have usable text for history.
            if ja and en:
                selected.append((i, ja, en))

        # Restore chronological order before building the message list.
        selected.reverse()

        included_indices = {i for i, _, _ in selected}
        excluded_indices = set(range(len(segments_list))) - included_indices
        excluded_duration_sec = sum(
            float((segments_list[i][1].get("duration_sec") or 0.0))
            for i in excluded_indices
        )

        history: List[TranslationHistoryItem] = []
        for _, ja, en in selected:
            history.append({"role": "user",      "content": ja})
            history.append({"role": "assistant", "content": en})

        return HistoryResult(
            history=history,
            included_indices=included_indices,
            included_duration_sec=accumulated_sec,
            excluded_duration_sec=excluded_duration_sec,
            total_segments=len(segments_list),
            included_segments=len(included_indices),
            excluded_segments=len(excluded_indices),
        )
   

    def reset(self) -> None:
        """Clear all buffered audio and metadata, reset total ample count to 0."""
        self.segments.clear()
        self.total_samples = 0
