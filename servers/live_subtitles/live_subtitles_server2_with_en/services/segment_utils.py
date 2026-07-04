"""
Segment utility functions for live subtitles server.
Contains reusable audio file management and segment processing utilities.
"""

import logging
import os
from pathlib import Path

from services.config import SEGMENT_AUDIO_DIR

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma"}


def get_audio_files(audio_dir: str = None) -> list[dict]:
    """
    Recursively collect all audio files from the specified directory.

    Args:
        audio_dir: Directory to search for audio files. Defaults to SEGMENT_AUDIO_DIR.

    Returns:
        List of dicts with relative path, display name, full path, size, and extension.
    """
    if audio_dir is None:
        audio_dir = str(SEGMENT_AUDIO_DIR)

    audio_files = []

    if not os.path.exists(audio_dir):
        logger.warning(f"Audio directory does not exist: {audio_dir}")
        return audio_files

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if Path(file).suffix.lower() in AUDIO_EXTENSIONS:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(audio_dir)
                unique_name = str(rel_path.with_suffix("")).replace(os.sep, " / ")

                audio_files.append(
                    {
                        "name": unique_name,
                        "path": str(rel_path),
                        "full_path": str(full_path),
                        "size": os.path.getsize(full_path),
                        "extension": full_path.suffix.lower(),
                    }
                )

    logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
    return sorted(audio_files, key=lambda x: x["name"])
