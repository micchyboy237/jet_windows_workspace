# servers/audio_server/python_scripts/server/utils/streaming_model.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from faster_whisper import WhisperModel

from python_scripts.server.utils.logger import get_logger
log = get_logger("streaming_model")

# Thread-safe cache + initialization lock
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
                log.info(
                    f"[bold yellow]Loading faster-whisper model[/] "
                    f"[dim]→[/] [cyan]{model_size}[/] | [green]{compute_type}[/] | [blue]{device}[/]"
                )

                model_path = model_size
                if quantized_model_root:
                    import pathlib
                    candidate = pathlib.Path(quantized_model_root) / f"whisper-{model_size}-ct2"
                    if candidate.exists():
                        model_path = str(candidate)
                        log.info(f"Using local quantized model: {model_path}")

                _STREAMING_MODEL_CACHE[key] = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    download_root=quantized_model_root,
                )

                log.info(
                    f"[bold green]faster-whisper model ready & cached[/] "
                    f"[dim]→[/] [bright_white]{key}[/]"
                )

        _CACHE_LOCK.submit(_init_model).result()

    return _STREAMING_MODEL_CACHE[key]