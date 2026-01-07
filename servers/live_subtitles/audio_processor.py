# audio_processor.py

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict, Callable

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

from pysilero_vad import SileroVoiceActivityDetector


class SegmentMetadata(TypedDict):
    segment_id: int
    duration_sec: float
    num_chunks: int
    num_samples: int
    start_sec: float
    end_sec: float


@dataclass
class SegmentState:
    """Mutable state for an active segment (speech or non-speech)."""
    buffer: bytearray = field(default_factory=bytearray)
    start_monotonic: float | None = None
    start_wallclock: float | None = None
    chunk_count: int = 0
    probs: list[float] = field(default_factory=list)
    rms_list: list[float] = field(default_factory=list)
    energy_sum: float = 0.0
    energy_sum_squares: float = 0.0
    max_energy: float = 0.0
    min_energy: float = float("inf")
    peak_rms: float = 0.0
    segment_id: int | None = None   # NEW: store the ID assigned at creation


class AudioProcessor:
    def __init__(
        self,
        config,
        vad: SileroVoiceActivityDetector,
        segments_dir: str,
        non_speech_dir: str,
        stream_start_time_ref: dict,  # mutable dict to hold global stream_start_time
        segment_start_wallclock: dict[int, float],
        non_speech_wallclock: dict[int, float],
        *,
        current_time_mono: Callable[[], float] | None = None,
        current_time_wall: Callable[[], float] | None = None,
    ):
        self.config = config
        self.vad = vad
        self.segments_dir = segments_dir
        self.non_speech_dir = non_speech_dir
        self.stream_start_time_ref = stream_start_time_ref
        self.segment_start_wallclock = segment_start_wallclock
        self.non_speech_wallclock = non_speech_wallclock

        self._time_mono = current_time_mono or time.monotonic
        self._time_wall = current_time_wall or time.time

        # Debug logging
        self.log = logging.getLogger("AudioProcessor")
        self.log.setLevel(logging.DEBUG)

        # Runtime state
        self.speech: SegmentState | None = None
        self.non_speech: SegmentState | None = None
        self.silence_start: float | None = None

    def _next_segment_id(self, kind: Literal["speech", "non_speech"]) -> int:
        dir_path = self.segments_dir if kind == "speech" else self.non_speech_dir
        existing = [d for d in os.listdir(dir_path) if d.startswith("segment_")]
        return len(existing) + 1

    def _save_segment(
        self,
        state: SegmentState,
        segment_id: int,
        kind: Literal["speech", "non_speech"],
        base_time: float,
    ) -> None:
        dir_path = os.path.join(self.segments_dir if kind == "speech" else self.non_speech_dir,
                                f"segment_{segment_id:04d}")
        os.makedirs(dir_path, exist_ok=True)

        wav_path = os.path.join(dir_path, "sound.wav")
        audio_np = np.frombuffer(state.buffer, dtype=np.int16)
        wavfile.write(wav_path, self.config.sample_rate, audio_np)

        num_samples = len(state.buffer) // 2
        duration = num_samples / self.config.sample_rate

        start_sec = round(state.start_wallclock - base_time, 3)
        end_sec = round(start_sec + duration, 3)

        # Energy stats
        chunk_count = state.chunk_count or 1
        avg_rms = state.energy_sum / chunk_count
        rms_std = np.sqrt((state.energy_sum_squares / chunk_count) - (avg_rms ** 2)) if chunk_count > 1 else 0.0

        metadata: dict = {
            "segment_id": segment_id,
            "type": kind,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(duration, 3),
            "num_chunks": state.chunk_count,
            "num_samples": num_samples,
            "audio_energy": {
                "rms_min": round(state.min_energy, 4) if state.min_energy != float("inf") else 0.0,
                "rms_max": round(state.max_energy, 4),
                "rms_ave": round(avg_rms, 4),
                "rms_std": round(rms_std, 4),
                "peak_rms": round(state.peak_rms, 4),
            },
        }

        meta_path = os.path.join(dir_path, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save probabilities and RMS
        probs_path = os.path.join(dir_path, f"{kind}_probabilities.json")
        with open(probs_path, "w", encoding="utf-8") as f:
            json.dump([round(p, 4) for p in state.probs], f, indent=2)

        rms_path = os.path.join(dir_path, f"{kind}_rms.json")
        with open(rms_path, "w", encoding="utf-8") as f:
            json.dump([round(r, 4) for r in state.rms_list], f, indent=2)

        # Optional plots
        if len(state.probs) > 1:
            time_axis = np.arange(len(state.probs)) * (self.vad.chunk_samples() / self.config.sample_rate)
            for data, name, color in [
                (state.probs, "probability", "gray" if kind == "non_speech" else "blue"),
                (state.rms_list, "rms", "red" if kind == "non_speech" else "orange"),
            ]:
                plt.figure(figsize=(8, 3))
                plt.plot(time_axis, data, color=color)
                plt.xlabel("Time (s)")
                plt.ylabel("VAD Probability" if "prob" in name else "RMS")
                plt.title(f"{kind.capitalize()} {name.capitalize()} – segment_{segment_id:04d}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(dir_path, f"{kind}_{name}_plot.png"))
                plt.close()

    async def process_chunk(
        self,
        chunk: bytes,
        rms: float,
        speech_prob: float,
        ws,
    ) -> None:
        # Log per-chunk details only when there is noticeable energy or VAD activity
        # This prevents log spam during long periods of complete silence
        if rms > 10.0 or speech_prob > 0.1:
            self.log.debug("[process_chunk] rms=%.2f speech_prob=%.3f", rms, speech_prob)

        has_sound = rms > 50.0

        if speech_prob >= self.config.vad_threshold:
            # Speech detected
            if self.speech is None:
                # Start new speech segment
                self.log.debug("[speech] Starting new speech segment")
                seg_id = self._next_segment_id("speech")
                now_wall = self._time_wall()
                now_mono = self._time_mono()

                self.segment_start_wallclock[seg_id] = now_wall
                if self.stream_start_time_ref.get("value") is None:
                    self.stream_start_time_ref["value"] = now_wall
                    self.log.debug("[timing] stream_start_time set to %.3f", now_wall)

                self.speech = SegmentState(
                    start_monotonic=now_mono,
                    start_wallclock=now_wall,
                    peak_rms=rms,
                    max_energy=rms,
                    min_energy=rms,
                )

            self.speech.buffer.extend(chunk)
            self.speech.chunk_count += 1
            self.speech.probs.append(speech_prob)
            self.speech.rms_list.append(rms)
            self.speech.energy_sum += rms
            self.speech.energy_sum_squares += rms ** 2
            self.speech.max_energy = max(self.speech.max_energy, rms)
            self.speech.min_energy = min(self.speech.min_energy, rms)
            self.speech.peak_rms = max(self.speech.peak_rms, rms)

            # Reset silence timer and non-speech state
            self.silence_start = None
            self.non_speech = None

            # Send to server
            payload = {
                "type": "audio",
                "sample_rate": self.config.sample_rate,
                "pcm": base64.b64encode(chunk).decode("ascii"),
            }
            await ws.send(json.dumps(payload))

        else:
            # Silence / non-speech
            if self.speech is not None and self.silence_start is None:
                self.silence_start = self._time_mono()
                self.log.debug("[speech] Silence started within speech at %.3f", self.silence_start)

            # Non-speech handling
            if self.speech is None and has_sound:
                if self.non_speech is None:
                    self.log.debug("[non-speech] Starting new non-speech segment")
                    now_wall = self._time_wall()
                    self.non_speech = SegmentState(
                        start_monotonic=self._time_mono(),
                        start_wallclock=now_wall,
                        peak_rms=rms,
                        max_energy=rms,
                        min_energy=rms,
                        segment_id=self._next_segment_id("non_speech"),
                    )
                    self.non_speech_wallclock[self.non_speech.segment_id] = now_wall

                # Append current chunk to active non-speech segment
                self.non_speech.buffer.extend(chunk)
                self.non_speech.chunk_count += 1
                self.non_speech.probs.append(speech_prob)
                self.non_speech.rms_list.append(rms)
                self.non_speech.energy_sum += rms
                self.non_speech.energy_sum_squares += rms ** 2
                self.non_speech.max_energy = max(self.non_speech.max_energy, rms)
                self.non_speech.min_energy = min(self.non_speech.min_energy, rms)
                self.non_speech.peak_rms = max(self.non_speech.peak_rms, rms)

                # Save non-speech segment if:
                # - It is long enough (>= 3 seconds) OR
                # - It contains a loud burst (current RMS > 100 and duration >= 1s) AND
                #   the segment has some audible content (current chunk audible or peak RMS > 50)
                elapsed = self._time_mono() - self.non_speech.start_monotonic
                trigger_save = elapsed >= 3.0 or (rms > 100.0 and elapsed >= 1.0)
                seg_id_to_save = None
                if trigger_save:
                    segment_has_audible_content = has_sound or self.non_speech.peak_rms > 50.0
                    should_save = elapsed >= 3.0 or (rms > 100.0 and segment_has_audible_content)
                    if self.non_speech.chunk_count > 0 and should_save:
                        seg_id_to_save = self.non_speech.segment_id
                        base_time = self.stream_start_time_ref["value"] or self._time_wall()
                        self._save_segment(self.non_speech, seg_id_to_save, "non_speech", base_time)
                    # Always reset the segment after evaluation
                    self.non_speech = None
                    if seg_id_to_save is not None:
                        self.non_speech_wallclock.pop(seg_id_to_save, None)

            # Check for end of speech segment
            if (
                self.speech is not None
                and self.silence_start is not None
                and self._time_mono() - self.silence_start > self.config.min_silence_duration
            ):

                self.log.debug("[speech] Silence duration exceeded, checking segment")
                duration_sec = self._time_mono() - self.speech.start_monotonic
                if duration_sec < self.config.min_speech_duration:
                    self.log.debug("[speech] Segment too short (%.3fs), discarding/merging", duration_sec)
                    # Too short → merge into ongoing non-speech or discard
                    if self.non_speech is not None:
                        self.non_speech.buffer.extend(self.speech.buffer)
                        self.non_speech.chunk_count += self.speech.chunk_count
                        self.non_speech.energy_sum += self.speech.energy_sum
                        self.non_speech.energy_sum_squares += self.speech.energy_sum_squares
                        self.non_speech.max_energy = max(self.non_speech.max_energy, self.speech.max_energy)
                        self.non_speech.min_energy = min(self.non_speech.min_energy, self.speech.min_energy)
                        self.non_speech.peak_rms = max(self.non_speech.peak_rms, self.speech.peak_rms)
                    self.speech = None
                    self.silence_start = None
                else:
                    # Valid speech segment
                    self.log.debug("[speech] Valid segment ended, saving")
                    base_time = self.stream_start_time_ref["value"] or self.speech.start_wallclock
                    current_seg_id = max(self.segment_start_wallclock.keys())  # last created speech id
                    self._save_segment(self.speech, current_seg_id, "speech", base_time)

                    await ws.send(json.dumps({"type": "end_of_utterance"}))

                    self.speech = None
                    self.silence_start = None
                    self.non_speech = None  # fresh start after valid speech