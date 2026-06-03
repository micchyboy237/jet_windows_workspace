# servers\live_subtitles\live_subtitles_server2_with_en\services\live_subtitles_server_utils.py

import json
import logging
import shutil
from pathlib import Path
from typing import List

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


def cleanup_global_files_for_segments(
    segments_dir: Path,
    removed_dirs: List[str]
) -> None:
    """
    Remove entries from global JSON files when segment directories are deleted.
    
    This function cleans up all_speakers.json and all_tag_events.json
    by removing entries that reference deleted segment directories.
    
    Args:
        segments_dir: The directory containing global JSON files
        removed_dirs: List of removed segment directory names (e.g., ["segment_001", "segment_002"])
    
    Debug logs trace:
        - Files being checked
        - Number of entries removed per file
        - Any errors encountered
    """
    if not removed_dirs:
        console.print("[dim]No directories to clean from global files[/dim]")
        return
    
    global_files = [
        segments_dir / "all_speakers.json",
        segments_dir / "all_tag_events.json",
    ]
    
    console.print(f"[dim]Cleaning global files for removed segments: {', '.join(removed_dirs)}[/dim]")
    
    for file_path in global_files:
        if not file_path.exists():
            console.print(f"[dim]Global file {file_path.name} does not exist, skipping[/dim]")
            continue
            
        try:
            # Read existing entries
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    entries = json.load(f)
                except json.JSONDecodeError:
                    console.print(f"[warning]Corrupted {file_path.name}, skipping cleanup[/warning]")
                    continue
            
            if not isinstance(entries, list):
                console.print(f"[warning]{file_path.name} is not a list, skipping cleanup[/warning]")
                continue
            
            original_count = len(entries)
            
            # Remove entries for deleted segment directories
            entries = [
                entry for entry in entries 
                if entry.get("segment_dir") not in removed_dirs
            ]
            
            removed_count = original_count - len(entries)
            
            if removed_count > 0:
                # Write back cleaned entries
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                
                console.print(
                    f"[dim]Cleaned {removed_count} entries from {file_path.name} "
                    f"({len(entries)} remaining)[/dim]"
                )
            else:
                console.print(f"[dim]No matching entries found in {file_path.name}[/dim]")
                
        except Exception as e:
            console.print(f"[warning]Failed to clean {file_path.name}: {e}[/warning]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def prepare_segment_directory(
    segment_num: int, 
    segments_dir: Path, 
    segment_index_path: Path, 
    n_results: int = 20
) -> Path:
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
    segments_dir : Path
        The base directory for segment folders.
    segment_index_path : Path
        Path to the segment index file.
    n_results : int
        Maximum number of segment directories to keep.

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
    removed_dir_names = []
    
    for d in all_segment_dirs:
        try:
            # Extract the number from "segment_XXX"
            num_str = d.name.replace("segment_", "")
            dir_num = int(num_str)

            if dir_num <= cutoff:
                shutil.rmtree(d)
                removed_count += 1
                removed_dir_names.append(d.name)
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
        
        # Clean up global JSON files for removed segments
        cleanup_global_files_for_segments(segments_dir, removed_dir_names)

    save_segment_counter(segment_index_path)
    return segment_dir
