from typing import List

import numpy as np


def compute_rms_per_frame(
    audio: np.ndarray,
    hop_size: int,
    start_frame: int,
    end_frame: int,
) -> List[float]:
    """
    Compute RMS energy for each frame in the given frame range.
    Args:
        audio: Float32 audio array (mono).
        hop_size: Number of samples per frame.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
    Returns:
        List of RMS values (one per frame).
    """
    rms_values = []
    for frame_idx in range(start_frame, end_frame + 1):
        start_sample = frame_idx * hop_size
        end_sample = start_sample + hop_size
        frame_audio = audio[start_sample:end_sample]
        if len(frame_audio) == 0:
            rms = 0.0
        else:
            rms = np.sqrt(np.mean(frame_audio**2))
        rms_values.append(float(rms))
    return rms_values
