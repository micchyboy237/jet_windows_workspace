import json
import logging
import shutil
from pathlib import Path

# from audio_search import search_audio
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "success": "green bold",
            "warning": "yellow",
            "error": "red bold",
            "value": "white bold",
            "time": "magenta bold",
            "number": "bright_white",
            "uuid": "bright_blue",
            "speaker": "bright_green",
        }
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("live_subtitles_server_utils")

# Segment counter for sequential directory naming
_segment_counter: int = 0


def load_segment_counter(segment_index_path: Path) -> int:
    """Load the segment counter from disk, or start at 0."""
    global _segment_counter
    if segment_index_path.exists():
        try:
            with open(segment_index_path, "r") as f:
                data = json.load(f)
                _segment_counter = data.get("counter", 0)
                console.print(
                    f"[info]Loaded segment counter: {_segment_counter}[/info]"
                )
        except Exception as e:
            console.print(f"[warning]Could not load segment counter: {e}[/warning]")
            _segment_counter = 0
    return _segment_counter


def save_segment_counter(segment_index_path: Path) -> None:
    """Save the segment counter to disk."""
    try:
        with open(segment_index_path, "w") as f:
            json.dump({"counter": _segment_counter}, f, indent=2)
    except Exception as e:
        console.print(f"[warning]Could not save segment counter: {e}[/warning]")


def get_next_segment_number() -> int:
    """
    Get the next segment number using a circular approach.

    Numbers go from 1 to N_SEGMENT_RESULTS, then wrap around.
    Existing directories with the same number are removed before reuse.
    """
    global _segment_counter
    _segment_counter += 1

    # Wrap around if we exceed the max
    if _segment_counter > 999999:  # Safety limit
        _segment_counter = 1

    return _segment_counter


def prepare_segment_directory(segment_num: int, segments_dir: Path, segment_index_path: Path, n_results: int = 20) -> Path:
    """
    Prepare a segment directory, cleaning up old ones to maintain the limit.

    This uses a rolling window approach:
    - Segments are numbered 1, 2, 3, ... up to infinity
    - We only keep the last n_results segments
    - Old segments (num <= current - n_results) are deleted

    Parameters
    ----------
    segment_num : int
        The segment number for the new directory.

    Returns
    -------
    Path
        Path to the new (empty) segment directory.
    """
    # Create the new segment directory
    segment_dir = segments_dir / f"segment_{segment_num:03d}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Calculate the cutoff: keep only segments > (current - n_results)
    cutoff = segment_num - n_results

    # Find and remove old segment directories
    all_segment_dirs = sorted(
        [
            d
            for d in segments_dir.iterdir()
            if d.is_dir() and d.name.startswith("segment_")
        ],
        key=lambda d: d.name,
    )

    removed_count = 0
    for d in all_segment_dirs:
        try:
            # Extract the number from "segment_XXX"
            num_str = d.name.replace("segment_", "")
            dir_num = int(num_str)

            if dir_num <= cutoff:
                shutil.rmtree(d)
                removed_count += 1
                console.print(
                    f"[dim]Removed old segment directory: {d.name} "
                    f"(keeping segments {cutoff + 1} to {segment_num})[/dim]"
                )
        except (ValueError, Exception) as e:
            console.print(
                f"[warning]Could not process directory {d.name}: {e}[/warning]"
            )

    if removed_count > 0:
        console.print(
            f"[info]Segment cleanup: removed {removed_count} directories, "
            f"current segment: segment_{segment_num:03d}, "
            f"max kept: {n_results}[/info]"
        )

    save_segment_counter(segment_index_path)
    return segment_dir
