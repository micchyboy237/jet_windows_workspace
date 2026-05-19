from pathlib import Path

# ────────────────────────────────────────────────
# Pretrained Models
# ────────────────────────────────────────────────

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

# ────────────────────────────────────────────────
# General Defaults
# ────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_SILENCE_SEC = 0.8
DEFAULT_MIN_SPEECH_SEC = 0.250
DEFAULT_MAX_SPEECH_SEC = 15.0
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_RETURN_SECONDS = False
DEFAULT_WITH_SCORES = False
DEFAULT_INCLUDE_NON_SPEECH = False

DEFAULT_SMOOTH_WINDOW_SIZE = 5
DEFAULT_MAX_BUFFER_SEC = 1.2

# ────────────────────────────────────────────────
# Streaming Constants (aligned with FireRedVAD)
# ────────────────────────────────────────────────
MIN_BUFFER_SAMPLES_BEFORE_FIRST_VAD = 4800  # ~300 ms

# Context window: multiple of frame shift for clean processing
VAD_CONTEXT_WINDOW_SAMPLES = 9600  # 600 ms (60 frames)

# Overlap must be multiple of FRAME_SHIFT_SAMPLE (160)
BUFFER_OVERLAP_SAMPLES = 640  # 40 frames (~40 ms) — good trade-off
# ────────────────────────────────────────────────


# ────────────────────────────────────────────────
# Pre/Post-roll Settings
# ────────────────────────────────────────────────

# Pre/Post-roll defaults (extension settings)
DEFAULT_PREROLL_MAX_SEC = 0.300  # maximum look-back window
DEFAULT_PREROLL_HYBRID_THRESHOLD = 0.1  # hybrid score below which we stop extending

DEFAULT_POSTROLL_MAX_SEC = 0.300  # maximum look-forward window
DEFAULT_POSTROLL_HYBRID_THRESHOLD = 0.1  # hybrid score below which we stop extending

# Unified weights for speech probability and RMS energy in hybrid scoring (used for both preroll and postroll)
DEFAULT_PROB_WEIGHT = 0.5  # weight for speech probability
DEFAULT_RMS_WEIGHT = 0.5  # weight for normalised RMS energy

# ────────────────────────────────────────────────
# Valley/Trough Detection Settings
# ────────────────────────────────────────────────


# Soft-limit split defaults
DEFAULT_SOFT_LIMIT_SEC = 6.0
DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S = 0.5
DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW = 20
DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE = 0.15
DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S = 3.0

# Soft-limit-high split defaults
DEFAULT_SOFT_LIMIT_SEC_HIGH = 10.0
DEFAULT_SOFT_LIMIT_SEC_HIGH_MIN_VALLEY_DURATION_S = 0.3
DEFAULT_SOFT_LIMIT_SEC_HIGH_SMOOTHING_WINDOW = 20
DEFAULT_SOFT_LIMIT_SEC_HIGH_TROUGH_PROMINENCE = 0.15
DEFAULT_SOFT_LIMIT_SEC_HIGH_MIN_TROUGH_OFFSET_S = 5.0

# Relaxed defaults once past hard_limit
DEFAULT_HARD_LIMIT_SEC = DEFAULT_MAX_SPEECH_SEC
DEFAULT_HARD_LIMIT_MIN_VALLEY_DURATION_S = 0.2
DEFAULT_HARD_LIMIT_SMOOTHING_WINDOW = 20
DEFAULT_HARD_LIMIT_TROUGH_PROMINENCE = 0.15
DEFAULT_HARD_LIMIT_MIN_TROUGH_OFFSET_S = 7.5
