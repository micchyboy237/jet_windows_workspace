from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# Thread-safe cache + initialization lock (exactly like CT2 cache)
_STREAMING_MODEL_CACHE: Dict[str, WhisperModel] = {}
_CACHE_LOCK = ThreadPoolExecutor(max_workers=1)


def _make_cache_key(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8",
) -> str:
    return f"{model_size}|{device}|{compute_type}"


def get_streaming_model(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8",  # best for GTX 1660
    quantized_model_root: Optional[str] = None,
) -> WhisperModel:
    """
    Return a cached faster-whisper model.
    Remembers the last used configuration → subsequent calls are instant.
    """
    key = _make_cache_key(model_size, device, compute_type)

    if key not in _STREAMING_MODEL_CACHE:

        def _init_model() -> None:
            if key not in _STREAMING_MODEL_CACHE:
                log.info(f"Loading faster-whisper model: {model_size} | {compute_type} | {device}")

                # Optional: allow custom quantized model path (useful for large-v3 int8_float16 etc.)
                model_path = model_size
                if quantized_model_root:
                    import pathlib
                    candidate = pathlib.Path(quantized_model_root) / f"whisper-{model_size}-ct2"
                    if candidate.exists():
                        model_path = str(candidate)

                _STREAMING_MODEL_CACHE[key] = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    download_root=quantized_model_root,  # fallback to HF hub if not local
                )
                log.info(f"faster-whisper model cached → {key}")

        # Block until initialization is done (prevents duplicate loading)
        _CACHE_LOCK.submit(_init_model).result()

    return _STREAMING_MODEL_CACHE[key]