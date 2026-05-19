from typing import Union

import matplotlib
from config import (
    FRAME_PER_SECONDS,
    SAMPLE_RATE,
)
from vad_types import StreamVadFrame

matplotlib.use("Agg")
import numpy as np
import torch
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from vad_config import (
    BUFFER_OVERLAP_SAMPLES,
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_THRESHOLD,
    MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD,
    SAVE_DIR,
    VAD_CONTEXT_WINDOW_SAMPLES,
)
from vad_hybrid_stream_vad_postprocessor import (
    HybridStreamVadPostprocessor,
)
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# FireRedVAD wrapper
# ---------------------------------------------------------------------------


class FireRedVAD:
    """Wrapper for FireRedVAD with simple streaming-like API."""

    def __init__(
        self,
        model_dir: str = SAVE_DIR,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
        max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
        smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
        max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        console.print(f"[cyan]Loading FireRedVAD (streaming) on {self.device}…[/cyan]")

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=threshold,
            smooth_window_size=smooth_window_size,
            min_speech_frame=int(min_speech_duration_sec * FRAME_PER_SECONDS),
            max_speech_frame=int(max_speech_duration_sec * FRAME_PER_SECONDS),
            min_silence_frame=int(min_silence_duration_sec * FRAME_PER_SECONDS),
            chunk_max_frame=30000,
        )

        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)

        self.vad.postprocessor = HybridStreamVadPostprocessor(
            smooth_window_size=config.smooth_window_size,
            speech_threshold=config.speech_threshold,
            pad_start_frame=config.pad_start_frame,
            min_speech_frame=config.min_speech_frame,
            max_speech_frame=config.max_speech_frame,
            min_silence_frame=config.min_silence_frame,
            prob_weight=DEFAULT_PROB_WEIGHT,
            rms_weight=DEFAULT_RMS_WEIGHT,
        )

        self.vad.vad_model.to(self.device)
        console.print("[green]done.[/green]")

        self.sample_rate = SAMPLE_RATE
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_prob: float = 0.0

        # Accumulated probabilities with full frame info
        self._accumulated_probs: list[StreamVadFrame] = []

        self.max_buffer_samples = int(max_buffer_sec * self.sample_rate)

    def reset(self) -> None:
        """Reset internal VAD state and clear audio buffer."""
        self.vad.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0
        self._accumulated_probs.clear()

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Simple dynamic normalization"""
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            return chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            return chunk * gain
        return chunk

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process incoming audio chunk and return the latest smoothed speech probability.
        Also accumulates full frame information including speech_start_frame and speech_end_frame.
        """
        if len(chunk) == 0:
            return self.last_prob

        chunk = self._normalize_chunk(chunk)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        total_samples = len(self.audio_buffer)
        if total_samples < MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD:
            return self.last_prob

        full_buffer = self.audio_buffer
        to_process = full_buffer[-VAD_CONTEXT_WINDOW_SAMPLES:]

        try:
            results = self.vad.detect_chunk(to_process)

            # Keep only overlap for next iteration
            overlap = full_buffer[-BUFFER_OVERLAP_SAMPLES:]
            self.audio_buffer = overlap

            if not results:
                return self.last_prob

            # === Accumulate full frame results ===
            for res in results:
                self._accumulated_probs.append(
                    {
                        "frame_idx": res.frame_idx,
                        "raw_prob": float(res.raw_prob),
                        "smoothed_prob": float(res.smoothed_prob),
                        "is_speech": bool(res.is_speech),
                        "is_speech_start": bool(res.is_speech_start),
                        "is_speech_end": bool(res.is_speech_end),
                        "speech_start_frame": int(res.speech_start_frame),
                        "speech_end_frame": int(res.speech_end_frame),
                    }
                )

            last = results[-1]
            prob = last.smoothed_prob
            self.last_prob = prob
            return prob

        except Exception as e:
            console.print(f"[red]VAD detect_chunk error: {e}[/red]")
            self.audio_buffer = full_buffer[-BUFFER_OVERLAP_SAMPLES:]
            return self.last_prob

    def detect_full(
        self,
        audio: Union[str, np.ndarray],
    ) -> tuple[list, dict]:
        """Full-file detection with accumulated probabilities."""
        self.reset()
        frame_results, result = self.vad.detect_full(audio)

        # Populate accumulator with full information
        self._accumulated_probs = [
            {
                "frame_idx": r.frame_idx,
                "raw_prob": float(r.raw_prob),
                "smoothed_prob": float(r.smoothed_prob),
                "is_speech": bool(r.is_speech),
                "is_speech_start": bool(r.is_speech_start),
                "is_speech_end": bool(r.is_speech_end),
                "speech_start_frame": int(r.speech_start_frame),
                "speech_end_frame": int(r.speech_end_frame),
            }
            for r in frame_results
        ]

        return frame_results, result

    def get_accumulated_probs(self) -> list[StreamVadFrame]:
        """Return copy of all accumulated frame probabilities (including start/end frames)."""
        return self._accumulated_probs.copy()

    def get_prob_history(self, as_numpy: bool = False):
        """Convenience method for numpy arrays (frames + probabilities)."""
        if not self._accumulated_probs:
            if as_numpy:
                return np.array([], dtype=int), np.array([], dtype=float)
            return []

        if not as_numpy:
            return self.get_accumulated_probs()

        frames = np.array([p["frame_idx"] for p in self._accumulated_probs], dtype=int)
        smoothed = np.array(
            [p["smoothed_prob"] for p in self._accumulated_probs], dtype=float
        )
        return frames, smoothed
