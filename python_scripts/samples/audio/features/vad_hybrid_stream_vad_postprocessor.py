# vad_hybrid_stream_vad_postprocessor

from typing import Optional

import numpy as np
from fireredvad.core.stream_vad_postprocessor import (
    StreamVadFrameResult,
    StreamVadPostprocessor,
)
from vad_config import (
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
)
from vad_utils import compute_hybrid_probs
from config import HOP_SIZE
from rich.console import Console

console = Console()


class HybridStreamVadPostprocessor(StreamVadPostprocessor):
    """
    Extends StreamVadPostprocessor with hybrid speech scoring.

    For each frame the raw_prob stored in StreamVadFrameResult is not the
    bare model output but a weighted combination of the model's speech
    probability and the frame's normalised RMS energy:

        hybrid = prob_weight * model_prob + rms_weight * rms_norm

    This hybrid score is then smoothed and thresholded exactly as in the
    parent class, so all state-machine logic is inherited unchanged.
    """

    def __init__(
        self,
        smooth_window_size: int,
        speech_threshold: float,
        pad_start_frame: int,
        min_speech_frame: int,
        max_speech_frame: int,
        min_silence_frame: int,
        prob_weight: float = DEFAULT_PROB_WEIGHT,
        rms_weight: float = DEFAULT_RMS_WEIGHT,
        frame_samples: int = HOP_SIZE,
    ) -> None:
        super().__init__(
            smooth_window_size=smooth_window_size,
            speech_threshold=speech_threshold,
            pad_start_frame=pad_start_frame,
            min_speech_frame=min_speech_frame,
            max_speech_frame=max_speech_frame,
            min_silence_frame=min_silence_frame,
        )
        self.prob_weight = prob_weight
        self.rms_weight = rms_weight
        self.frame_samples = frame_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_one_frame(
        self,
        raw_prob: float,
        audio_frame: Optional[np.ndarray] = None,
    ) -> StreamVadFrameResult:
        """
        Process one VAD frame using a hybrid speech probability.

        Args:
            raw_prob:    Model speech probability in [0, 1].
            audio_frame: Corresponding audio samples for this frame.
                         When provided the hybrid score blends model prob
                         with normalised RMS energy.  When None the model
                         prob is used as-is (same behaviour as the parent).

        Returns:
            StreamVadFrameResult with raw_prob = hybrid score and
            smoothed_prob = window-averaged hybrid score.
        """
        hybrid_prob = self._compute_hybrid_prob(raw_prob, audio_frame)
        # Delegate to parent — it handles smoothing, thresholding, and the
        # full state machine.  The "raw_prob" it receives is our hybrid score.
        return super().process_one_frame(hybrid_prob)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_hybrid_prob(
        self,
        model_prob: float,
        audio_frame: Optional[np.ndarray],
    ) -> float:
        """
        Blend model speech probability with per-frame RMS energy.

        Reuses the batch helper compute_hybrid_probs() with a single-frame
        array so the normalisation and clipping logic stay in one place.
        Falls back to model_prob when no audio is supplied.
        """
        if audio_frame is None or len(audio_frame) == 0:
            return model_prob

        probs_arr = np.array([model_prob], dtype=np.float32)
        hybrid_arr = compute_hybrid_probs(
            probs=probs_arr,
            audio_np=audio_frame,
            prob_weight=self.prob_weight,
            rms_weight=self.rms_weight,
            frame_samples=self.frame_samples,
        )

        if len(hybrid_arr) == 0:
            return model_prob  # fallback: no audio overlap

        # Clamp to [0, 1] — weighted sum can exceed 1.0 if both weights > 0.5
        return float(np.clip(hybrid_arr[0], 0.0, 1.0))
