import json
from typing import Any
from pathlib import Path


def save_file(data: Any, filepath: str | Path) -> None:
    """
    Saves arbitrary data as a JSON file in a generic, reusable way.
    
    - Creates parent directories if they don't exist
    - Uses UTF-8 encoding and ensures ASCII compatibility is off
    - Pretty-prints with consistent, readable formatting
    - Works with any JSON-serializable object (dict, list, dataclass instances via asdict, etc.)
    
    Args:
        data: The data to save (must be JSON-serializable)
        filepath: Target file path (str or pathlib.Path)
    
    Raises:
        TypeError: If data cannot be serialized to JSON
        OSError: If file cannot be written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,          # deterministic output for easier diffing/testing
            default=str              # fallback for objects that aren't natively serializable
        )

    print(f"Saved: {filepath}")