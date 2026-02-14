# servers/live_subtitles/logger.py
"""
Centralized logging configuration for the live subtitles project.
Uses RichHandler for nice console output.
"""

import logging
from rich.logging import RichHandler


def setup_logger(
    name: str = "live-subtitles",
    level: int = logging.INFO,
    show_time: bool = True,
    show_path: bool = False,
    markup: bool = True,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger with RichHandler.
    
    Returns a configured logger instance.
    Call this once at application entry point (usually in the main server file).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    handler = RichHandler(
        show_time=show_time,
        show_path=show_path,
        markup=markup,
        rich_tracebacks=rich_tracebacks,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger


# Convenience global logger (preferred way to use it)
logger = setup_logger()

# Optional: You can also expose named loggers if needed
def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the root 'live-subtitles' namespace."""
    return logging.getLogger(f"live-subtitles.{name}")