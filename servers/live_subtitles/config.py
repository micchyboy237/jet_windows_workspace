SILENCE_MAX_THRESHOLD = 0.001
SAMPLE_RATE = 16000

# Frame duration
FRAME_LENGTH_MS = 25
FRAME_LENGTH_S = 0.025

# Hop length
FRAME_SHIFT_MS = 10
FRAME_SHIFT_S = 0.010

# Samples per frame (16000 * 25 / 1000 = 400)
FRAME_LENGTH_SAMPLE = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
# Hop size in samples (16000 * 10 / 1000 = 160)
FRAME_SHIFT_SAMPLE = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)
# Frames per second (1000 / 10 = 100)
FRAME_PER_SECONDS = int(1000 / FRAME_SHIFT_MS)

# Frame length in samples (16000 * 25 / 1000.0 = 400)
FRAME_LENGTH_SAMPLES = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0)
# Hop size in samples (16000 * 10 / 1000.0 = 160)
HOP_SIZE = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0)

# Example values:
# FRAME_LENGTH_SAMPLES = 16000 * 0.025 = 400 samples
# HOP_SIZE = 160 samples

# New loudness thresholds (RMS values for float audio)
VERY_QUIET_MAX = 0.03  # upper bound for "very_quiet"
NORMAL_MAX = 0.12  # upper bound for "normal" speech
LOUD_MAX = 0.25  # upper bound for "loud"
