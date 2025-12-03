from typing import Optional
from faster_whisper import WhisperModel

_STREAMING_MODEL: Optional[WhisperModel] = None

def get_streaming_model() -> WhisperModel:
    global _STREAMING_MODEL
    if _STREAMING_MODEL is None:
        model_size = "large-v2"
        quantized_model_path = f"C:/Users/druiv/.cache/hf_ctranslate2_models/whisper-{model_size}-ct2"
        _STREAMING_MODEL = WhisperModel(
            quantized_model_path,
            device="cpu",
            compute_type="int8",
        )
    return _STREAMING_MODEL