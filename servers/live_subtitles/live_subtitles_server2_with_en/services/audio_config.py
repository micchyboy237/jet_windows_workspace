# audio_config.py

"""
These constants are exactly the same as in fireredvad.core.constants.
Keep them synchronized with FireRedVAD for compatibility.
"""

SILENCE_MAX_THRESHOLD = 0.001
SAMPLE_RATE = 16000

# Frame duration
FRAME_LENGTH_MS = 25
FRAME_LENGTH_S = 0.025

# Hop length
FRAME_SHIFT_MS = 10
HOP_STEP_MS = FRAME_SHIFT_MS
FRAME_SHIFT_S = 0.010
HOP_STEP_S = FRAME_SHIFT_S

# Samples per frame (16000 * 25 / 1000 = 400 samples)
FRAME_LENGTH_SAMPLE = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
# Hop size in samples (16000 * 10 / 1000 = 160 samples)
FRAME_SHIFT_SAMPLE = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)
HOP_SIZE = FRAME_SHIFT_SAMPLE
# Frames per second (1000 / 10 = 100)
FRAME_PER_SECONDS = int(1000 / FRAME_SHIFT_MS)


# New loudness thresholds (RMS values for float audio)
VERY_QUIET_MAX = 0.03  # upper bound for "very_quiet"
NORMAL_MAX = 0.12  # upper bound for "normal" speech
LOUD_MAX = 0.25  # upper bound for "loud"
