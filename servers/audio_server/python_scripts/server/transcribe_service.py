# servers/audio_server/python_scripts/server/transcribe_service.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from .whisper_ct2_transcriber import WhisperCT2Transcriber, QuantizedModelSizes
from .utils.logger import get_logger

log = get_logger("ct2_cache")

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
                log.info(
                    f"[bold yellow]Loading CTranslate2 model[/] "
                    f"[dim]→[/] [cyan]{model_size}[/] | [green]{compute_type}[/] | [blue]{device}[/]"
                )
                _MODEL_CACHE[key] = WhisperCT2Transcriber(
                    model_size=model_size,
                    device=device,
                    compute_type=compute_type,
                )
                log.info(
                    f"[bold green]CTranslate2 model ready & cached[/] "
                    f"[dim]→[/] [bright_white]{key}[/]"
                )
        _CACHE_LOCK.submit(init).result()
    else:
        log.debug(f"[dim]CTranslate2 cache hit[/] → {key}")
    return _MODEL_CACHE[key]