# servers/audio_server/python_scripts/server/utils/logger.py

from __future__ import annotations
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console(stderr=True)

def get_logger(name: str = "whisper_server") -> logging.Logger:
    logger = logging.getLogger(name)  # ← WAS MISSING
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger