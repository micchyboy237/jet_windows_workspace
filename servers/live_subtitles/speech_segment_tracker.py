# servers\live_subtitles\speech_segment_tracker.py
from dataclasses import dataclass
import numpy as np
from collections import deque
from typing import List, Optional

from fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult


@dataclass
class AccumulatedSpeechSegment:
    start_seconds: float
    end_seconds: float
    audio: np.ndarray
    jp: str = ""
    en: str = ""


class SpeechSegmentTracker:
    def __init__(self, sample_rate: int = 16000, max_context_segments: int = 5):
        self.sample_rate = sample_rate
        self.max_context_segments = max_context_segments
        self.context_buffer: deque[AccumulatedSpeechSegment] = deque(maxlen=max_context_segments)

        self.current_audio_chunks: List[np.ndarray] = []
        self.current_start_frame: int = -1
        self.in_speech: bool = False
        self.accumulated_frames: int = 0
        self.max_accum_frames = 1200  # ~12s forced flush

    def reset_current(self) -> None:
        self.current_audio_chunks = []
        self.current_start_frame = -1
        self.in_speech = False
        self.accumulated_frames = 0

    def process_vad_results(
        self, audio_chunk: np.ndarray, vad_results: List[StreamVadFrameResult]
    ) -> Optional[AccumulatedSpeechSegment]:
        if not vad_results:
            return None

        chunk_frames = len(vad_results)
        first_frame = vad_results[0].frame_idx
        has_start = any(r.is_speech_start for r in vad_results)
        has_end = any(r.is_speech_end for r in vad_results)

        speech_frames = sum(1 for r in vad_results if r.is_speech)
        avg_prob = sum(r.smoothed_prob for r in vad_results) / chunk_frames
        starts = [r.speech_start_frame for r in vad_results if r.is_speech_start]
        ends = [r.speech_end_frame for r in vad_results if r.is_speech_end]

        print(f"[VAD] chunk {first_frame:4d} | speech:{speech_frames:2d}/{chunk_frames} "
              f"avg_p:{avg_prob:.3f} | start:{has_start}({min(starts,default=-1)}) "
              f"end:{has_end}({max(ends,default=-1)}) accum:{self.accumulated_frames}")

        # === SIMPLIFIED START LOGIC (no more fragile potential_start_frame) ===
        new_start = min(starts) if starts else None
        new_end = max(ends) if ends else None

        if has_start:
            if self.in_speech:
                print(f"   → New start while in speech ({new_start}) → continuation")
                if new_start is not None and (self.current_start_frame == -1 or new_start < self.current_start_frame):
                    self.current_start_frame = new_start
            else:
                print(f"   → New speech start detected at {new_start}")
                if new_start is not None:
                    self.current_start_frame = new_start
                self.in_speech = True
                # Clear previous audio buffer for a fresh segment
                self.current_audio_chunks = []
                self.accumulated_frames = 0

        if self.in_speech:
            self.current_audio_chunks.append(audio_chunk.copy())
            self.accumulated_frames += chunk_frames

        force_flush = self.in_speech and self.accumulated_frames >= self.max_accum_frames

        if (has_end or force_flush) and self.in_speech and self.current_audio_chunks:
            self.in_speech = False
            full_audio = (np.concatenate(self.current_audio_chunks)
                          if len(self.current_audio_chunks) > 1 else self.current_audio_chunks[0])
            end_frame = new_end if new_end is not None and new_end >= 0 else -1
            if end_frame < self.current_start_frame or end_frame < 0:
                end_frame = self.current_start_frame + self.accumulated_frames

            if self.current_start_frame <= 0:
                print("   ⚠️  WARNING: start_frame was invalid (still -1), forcing 0.00")
                start_sec = 0.0
            else:
                start_sec = max(0.0, self.current_start_frame / 100.0)

            end_sec = max(start_sec + 0.6, end_frame / 100.0)

            duration = end_sec - start_sec
            audio_dur = len(full_audio) / self.sample_rate

            if duration < 0.6 or audio_dur < 0.5:
                print(f"   ⚠️  Discarding short segment (dur={duration:.2f}s)")
                self.reset_current()
                return None

            if force_flush:
                print(f"   ⚡ Forced flush after ~{self.accumulated_frames/100:.1f}s")

            segment = AccumulatedSpeechSegment(
                start_seconds=round(start_sec, 2),
                end_seconds=round(end_sec, 2),
                audio=full_audio,
            )

            print(f"   ✅ Finalized [{segment.start_seconds:.2f} → {segment.end_seconds:.2f}] "
                  f"({duration:.2f}s, {audio_dur:.1f}s audio)")

            self.reset_current()
            return segment

        return None  # still accumulating