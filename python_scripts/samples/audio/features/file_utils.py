# python_scripts\samples\audio\features\file_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from rich.console import Console

# Create a shared console instance for consistent styling
console = Console()


def save_file(
    data: Any,
    file_path: Union[str, Path],
    *,
    indent: int = 2,
    log_success: bool = True,
) -> Path:
    """
    Save JSON-serializable data to a file with rich-formatted success logging.

    💡 Simple analogy: Like hitting "Save" in a text editor, but with a 
    pretty confirmation message that shows exactly where your file went.

    Args:
        data: Any data that can be converted to JSON (dict, list, str, etc.)
        file_path: Where to save the file (string or Path object)
        indent: How many spaces to use for JSON formatting (default: 2)
        log_success: Whether to print a success message (default: True)

    Returns:
        Path: The absolute path of the saved file (useful for chaining or logging)

    Raises:
        OSError: If the file cannot be written (permissions, disk full, etc.)
        TypeError: If data cannot be serialized to JSON

    Example:
        >>> save_file({"name": "test"}, "output/data.json")
        ✓ Saved: /full/path/to/output/data.json
    """
    # 🗂️ Step 1: Normalize the path
    path = Path(file_path)

    # 📁 Step 2: Make sure the folder exists (like mkdir -p)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 💾 Step 3: Write the data as nicely formatted JSON
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    # 🎯 Step 4: Get the full absolute path for the log message
    absolute_path = path.resolve()

    # 🎨 Step 5: Show a pretty success message with rich
    if log_success:
        # Rich lets us make file paths clickable!
        # Format: [link=file:///absolute/path]display text[/link]
        console.print(
            f"[bold green]✓[/bold green] Saved: "
            f"[link=file://{absolute_path}]{absolute_path}[/link]"
        )

    # 🔙 Step 6: Return the path so callers can use it if needed
    return absolute_path
