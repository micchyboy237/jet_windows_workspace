"""
audio_segments_buffer.py
========================
Manages a list of VAD speech segments and the raw audio array they came from.

AudioSegmentsBuffer.get_context(index)
  Returns the audio that should be fed to the transcriber for segment[index]:
    - Looks backward through previous segments.
    - Accumulates (segment_duration + gap_to_next) until the total would
      exceed MAX_CONTEXT_AUDIO seconds.
    - The returned audio is SPEECH-ONLY (silent gaps are excluded), so the
      transcription model never sees dead air.
    - The returned metadata list tells the caller which segments are included
      (useful for prompt-prefix injection or logging).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from _types import SpeechSegment

MAX_CONTEXT_AUDIO: float = 30.0
SAMPLE_RATE: int = 16_000


@dataclass
class SegmentContext:
    """
    Everything the transcriber needs for one inference call.

    Attributes
    ----------
    current_index : int
        Position of the *target* segment in the original segment list.
    segments : List[SpeechSegment]
        Context segments in chronological order, **including** the target
        segment as the last entry.
    audio : np.ndarray
        Speech-only audio (float32, 16 kHz) formed by concatenating the
        audio of each segment in `segments`.  Silent gaps between segments
        are NOT included — they are counted toward the 30-s budget but
        trimmed from the actual bytes.
    total_duration_sec : float
        Sum of speech durations + inter-segment gaps for the chosen window.
        Will be <= MAX_CONTEXT_AUDIO.
    speech_duration_sec : float
        Sum of speech-only durations (i.e. len(audio) / SAMPLE_RATE).
    """

    current_index: int
    segments: List[SpeechSegment] = field(default_factory=list)
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    total_duration_sec: float = 0.0
    speech_duration_sec: float = 0.0


class AudioSegmentsBuffer:
    """
    Holds the full audio waveform and the VAD segment list.

    Parameters
    ----------
    audio_np : np.ndarray
        The entire recording as a float32 array at `sample_rate` Hz.
    segments : List[SpeechSegment]
        Speech segments produced by extract_speech_timestamps()
        with return_seconds=True.  Non-speech segments are ignored.
    sample_rate : int
        Must match audio_np (default 16 000 Hz).
    max_context_sec : float
        Maximum total duration (speech + gaps) to include as context.
    """

    def __init__(
        self,
        audio_np: np.ndarray,
        segments: List[SpeechSegment],
        sample_rate: int = SAMPLE_RATE,
        max_context_sec: float = MAX_CONTEXT_AUDIO,
    ) -> None:
        self._audio = audio_np.astype(np.float32, copy=False)
        self._sample_rate = sample_rate
        self._max_context_sec = max_context_sec

        # Keep only speech segments, sorted by start time.
        self._segments: List[SpeechSegment] = sorted(
            [s for s in segments if s["type"] == "speech"],
            key=lambda s: float(s["start"]),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def segments(self) -> List[SpeechSegment]:
        """All speech segments in chronological order (read-only view)."""
        return list(self._segments)

    def __len__(self) -> int:
        return len(self._segments)

    def get_context(self, index: int) -> SegmentContext:
        """
        Build the context window for segment at *index*.

        Algorithm
        ---------
        1. Start with just the target segment.
        2. Walk backward one segment at a time.
        3. For each candidate, compute:
               gap      = target_or_prev_start - candidate_end
               addition = candidate_duration + gap
           If accumulated_total + addition <= MAX_CONTEXT_AUDIO → include it.
           Otherwise → stop (do not include it or anything further back).
        4. Concatenate the SPEECH audio of the chosen segments in order.

        The gap between two consecutive included segments is counted toward
        the 30-s budget but is NOT written into the output audio array.

        Parameters
        ----------
        index : int
            Zero-based index into self.segments.

        Returns
        -------
        SegmentContext
        """
        if not (0 <= index < len(self._segments)):
            raise IndexError(
                f"Segment index {index} out of range (0–{len(self._segments) - 1})"
            )

        target = self._segments[index]
        target_start = float(target["start"])
        target_end = float(target["end"])
        target_dur = target_end - target_start

        # We'll collect segments from newest-to-oldest, then reverse.
        chosen: List[SpeechSegment] = [target]
        accumulated_sec = target_dur  # start with just the target's speech

        # Walk backward through preceding segments.
        # `right_boundary` is the start of the segment immediately to the
        # right of the candidate being evaluated — used to measure the gap.
        right_boundary = target_start

        for prev_idx in range(index - 1, -1, -1):
            candidate = self._segments[prev_idx]
            candidate_start = float(candidate["start"])
            candidate_end = float(candidate["end"])
            candidate_dur = candidate_end - candidate_start

            # Gap from end of candidate to start of right-neighbour.
            gap_sec = right_boundary - candidate_end  # always >= 0

            addition = candidate_dur + gap_sec

            if accumulated_sec + addition <= self._max_context_sec:
                chosen.append(candidate)
                accumulated_sec += addition
                right_boundary = candidate_start
            else:
                # Even a partial fit is not allowed — stop here.
                break

        # Restore chronological order (we built newest-first).
        chosen.reverse()

        # Build speech-only audio by slicing and concatenating.
        audio_chunks: List[np.ndarray] = []
        for seg in chosen:
            s = int(round(float(seg["start"]) * self._sample_rate))
            e = int(round(float(seg["end"]) * self._sample_rate))
            chunk = self._audio[s:e]
            if chunk.size > 0:
                audio_chunks.append(chunk)

        context_audio = (
            np.concatenate(audio_chunks).astype(np.float32)
            if audio_chunks
            else np.array([], dtype=np.float32)
        )

        speech_dur = float(len(context_audio)) / self._sample_rate

        return SegmentContext(
            current_index=index,
            segments=chosen,
            audio=context_audio,
            total_duration_sec=accumulated_sec,
            speech_duration_sec=speech_dur,
        )

    def iter_contexts(self):
        """
        Yield a SegmentContext for every segment in chronological order.

        Usage
        -----
        >>> buf = AudioSegmentsBuffer(audio_np, segments)
        >>> for ctx in buf.iter_contexts():
        ...     transcriber.transcribe_with_context(ctx)
        """
        for i in range(len(self._segments)):
            yield self.get_context(i)
