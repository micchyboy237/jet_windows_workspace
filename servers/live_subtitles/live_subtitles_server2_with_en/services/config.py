"""
Server configuration for live subtitles server.
Contains directory paths, file paths, and server-level constants.

For audio processing constants (SAMPLE_RATE, frame sizes, etc.),
see services.audio_config.
"""
import shutil
from pathlib import Path

# ──────────────────────────────────────────────
# Server Configuration
# ──────────────────────────────────────────────

# Base output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated"

# Number of recent segments to keep in rolling directory
N_SEGMENT_RESULTS = 50

# ──────────────────────────────────────────────
# Directory Paths (derived from OUTPUT_DIR)
# ──────────────────────────────────────────────

# Rolling segment directory (cleaned on startup, keeps last N segments)
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"

# Live audio buffer directory (cleaned on startup)
LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR / "live_audio_buffer"

# Permanent audio storage - PRESERVED across server restarts
# Stores WAV files by segment_id for playback in segment detail pages
SEGMENT_AUDIO_DIR = OUTPUT_DIR / "segment_audio"

# ──────────────────────────────────────────────
# File Paths
# ──────────────────────────────────────────────

# Segment index file (tracks next segment number)
SEGMENT_INDEX_PATH = LAST_N_SEGMENTS_DIR / "_segment_index.json"

# Speaker state persistence
SPEAKER_STATE_PATH = OUTPUT_DIR / "speaker_state.json"

# Audio index file that maps segment_id -> audio metadata
SEGMENT_AUDIO_INDEX = SEGMENT_AUDIO_DIR / "_audio_index.json"

# ──────────────────────────────────────────────
# Static & Template Directories (read-only)
# ──────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = _BASE_DIR / "static"
TEMPLATES_DIR = _BASE_DIR / "templates"

# ──────────────────────────────────────────────
# Temporary Directories (cleaned on startup)
# ──────────────────────────────────────────────

TEMP_DIRS = [
    LAST_N_SEGMENTS_DIR,
    LIVE_AUDIO_BUFFER_DIR,
]

# Clean temporary directories on startup
for temp_dir in TEMP_DIRS:
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create permanent directories (preserved across restarts)
SEGMENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Exports
# ──────────────────────────────────────────────

__all__ = [
    'OUTPUT_DIR',
    'N_SEGMENT_RESULTS',
    'LAST_N_SEGMENTS_DIR',
    'LIVE_AUDIO_BUFFER_DIR',
    'SEGMENT_AUDIO_DIR',
    'SEGMENT_INDEX_PATH',
    'SPEAKER_STATE_PATH',
    'SEGMENT_AUDIO_INDEX',
    'STATIC_DIR',
    'TEMPLATES_DIR',
]

# ──────────────────────────────────────────────
# Startup Summary (only in main process)
# ──────────────────────────────────────────────
if __name__ != "__main__":
    import sys
    if "pytest" not in sys.modules:
        print(f"[config] OUTPUT_DIR={OUTPUT_DIR}")
        print(f"[config] N_SEGMENT_RESULTS={N_SEGMENT_RESULTS}")
        print(f"[config] SEGMENT_AUDIO_DIR={SEGMENT_AUDIO_DIR} (preserved)")
