from pathlib import Path

try:
    from services.audio_config import HOP_STEP_MS
except ImportError:
    from audio_config import HOP_STEP_MS

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
AUDIO_TAGGING_MODEL = (
    BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
)
CLASS_LABELS_INDICES_CSV = (
    BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
)

DEFAULT_BASE_DIR: Path = (
    Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
)
DEFAULT_MODEL_PATH: Path = (
    DEFAULT_BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
)
DEFAULT_LABELS_PATH: Path = (
    DEFAULT_BASE_DIR
    / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
)

DEFAULT_TOP_K: int = 5
DEFAULT_NUM_THREADS: int = 1
DEFAULT_PROVIDER: str = "cpu"

DEFAULT_MIN_SPEECH_PROB_THRESHOLD: float = 0.1
DEFAULT_SPEECH_PROB_THRESHOLD: float = 0.2
DEFAULT_SPEECH_TOP_N: int = 3

DEFAULT_CHUNK_DURATION: float = 0.5
DEFAULT_CHUNK_OVERLAP: float = 0.25
DEFAULT_MIN_CHUNK_DURATION: float = 0.5  # Minimum chunk size in seconds

# Speech segment constants
DEFAULT_MIN_SILENCE_DURATION_SEC: float = 1.5
DEFAULT_MIN_SPEECH_DURATION_SEC: float = 0.5
DEFAULT_RESOLUTION_MS: float = float(HOP_STEP_MS)  # hop between timeline cells

# NEW: Confidence tier duration boundary thresholds (seconds)
DEFAULT_CONFIDENCE_VERY_SHORT_MAX: float = 0.5
DEFAULT_CONFIDENCE_SHORT_MAX: float = 1.5
DEFAULT_CONFIDENCE_NORMAL_MAX: float = 5.0

# NEW: High confidence thresholds by duration category
DEFAULT_HIGH_CONFIDENCE_VERY_SHORT: dict = {
    "prob_threshold": 0.75,
    "density_threshold": 0.85,
    "chunk_ratio_threshold": 0.80,
}
DEFAULT_HIGH_CONFIDENCE_SHORT: dict = {
    "prob_threshold": 0.65,
    "density_threshold": 0.75,
    "chunk_ratio_threshold": 0.70,
}
DEFAULT_HIGH_CONFIDENCE_NORMAL: dict = {
    "prob_threshold": 0.60,
    "density_threshold": 0.70,
    "chunk_ratio_threshold": 0.60,
}
DEFAULT_HIGH_CONFIDENCE_LONG: dict = {
    "prob_threshold": 0.55,
    "density_threshold": 0.65,
    "chunk_ratio_threshold": 0.55,
}

# NEW: Medium confidence thresholds by duration category
DEFAULT_MEDIUM_CONFIDENCE_VERY_SHORT: dict = {
    "prob_threshold": 0.55,
    "density_threshold": 0.70,
}
DEFAULT_MEDIUM_CONFIDENCE_SHORT: dict = {
    "prob_threshold": 0.45,
    "density_threshold": 0.60,
}
DEFAULT_MEDIUM_CONFIDENCE_NORMAL: dict = {
    "prob_threshold": 0.40,
    "density_threshold": 0.50,
}
DEFAULT_MEDIUM_CONFIDENCE_LONG: dict = {
    "prob_threshold": 0.35,
    "density_threshold": 0.45,
}

SPEECH_CLASS_NAMES: list[str] = [
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
]
