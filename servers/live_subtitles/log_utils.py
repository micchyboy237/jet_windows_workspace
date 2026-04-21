import sys
from pathlib import Path

# Paths to include and exclude in logs
INCLUDE_PATHS = ["Jet_Windows_Workspace/"]
EXCLUDE_PATHS = ["site-packages/"]

MAX_LOG_LENGTH = 100  # Max length for logged values


def get_entry_file_dir() -> str:
    """
    Returns the absolute directory path of the entry point script.
    Returns "server" if the entry point cannot be determined or is not a valid path.

    Returns:
        str: The absolute directory path of the entry point script, or "server" if invalid.
    """
    try:
        file_path = Path(sys.modules["__main__"].__file__).resolve()
        dir_path = file_path.parent
        if validate_filepath(str(file_path)):
            return str(dir_path)
        return ""
    except (KeyError, AttributeError):
        return ""


def get_entry_file_name(remove_extension: bool = False) -> str:
    """
    Returns the file name of the entry point script, optionally without the file extension.
    Args:
        remove_extension (bool): If True, returns the file name without the extension; otherwise, includes it.
    Returns:
        str: The file name of the entry point script, or "server" on error.
    """
    try:
        file_path = Path(sys.modules["__main__"].__file__)
        if remove_extension:
            return file_path.with_suffix("").name
        return file_path.name
    except (KeyError, AttributeError):
        return "server"


def validate_filepath(file_path: str) -> bool:
    # # Check if path should be included
    # if not any(path in file_path for path in INCLUDE_PATHS):
    #     return False  # Skip if not in allowed paths

    # Check if path should be excluded
    if any(path in file_path for path in EXCLUDE_PATHS):
        return False  # Skip if in excluded paths

    return True
