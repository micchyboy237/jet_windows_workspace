SILENCE_MAX_THRESHOLD = 0.001
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25.0
HOP_LENGTH_MS = 10.0

# Dynamically compute HOP_SIZE (in samples)
FRAME_LENGTH_SAMPLES = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0)
HOP_SIZE = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000.0)

# Example values:
# FRAME_LENGTH_SAMPLES = 16000 * 0.025 = 400 samples
# HOP_SIZE = 160 samples

# New loudness thresholds (RMS values for float audio)
VERY_QUIET_MAX = 0.03  # upper bound for "very_quiet"
NORMAL_MAX = 0.12  # upper bound for "normal" speech
LOUD_MAX = 0.25  # upper bound for "loud"
