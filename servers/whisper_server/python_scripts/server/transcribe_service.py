from __future__ import annotations
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from helpers.audio.whisper_ct2_transcriber import WhisperCT2Transcriber, QuantizedModelSizes

_MODEL_CACHE: Dict[str, WhisperCT2Transcriber] = {}
_CACHE_LOCK = ThreadPoolExecutor(max_workers=1)

def get_transcriber(
    model_size: QuantizedModelSizes = "large-v2",
    compute_type: str = "int8_float16",
    device: str = "cpu",
) -> WhisperCT2Transcriber:
    key = f"{model_size}|{compute_type}|{device}"
    if key not in _MODEL_CACHE:
        def init():
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = WhisperCT2Transcriber(
                    model_size=model_size,
                    device=device,
                    compute_type=compute_type,
                )
        _CACHE_LOCK.submit(init).result()
    return _MODEL_CACHE[key]
