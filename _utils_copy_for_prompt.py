import os
import fnmatch
from pathlib import Path
from typing import List, Set, Optional, Iterable
from rich.console import Console

logger = Console()

# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)

def find_files(
    base_dir: str,
    include: List[str],
    exclude: List[str],
    include_content_patterns: List[str],
    exclude_content_patterns: List[str],
    case_sensitive: bool = False,
    extensions: List[str] = [],
    modified_after: Optional[float] = None,
) -> List[str]:
    """
    Optimized file finder with include/exclude filters, optional content matching,
    and support for double-wildcard absolute patterns.
    """

    normalized_extensions = {ext.lstrip(".").lstrip("*").lower() for ext in extensions}
    matched_files: Set[str] = set()
    base_path = Path(base_dir).resolve()

    if not base_path.exists():
        logger.print(f"Directory does not exist: {base_dir}")
        return []

    # Normalize includes/excludes
    def normalize_patterns(patterns: List[str], is_exclude=False) -> List[str]:
        out = []
        for pat in patterns:
            if os.path.isabs(pat):
                out.append(pat)
            else:
                if pat.endswith("/") or pat.endswith("/*"):
                    pat = pat.rstrip("/") + "/**/*"
                elif not is_exclude and (pat.startswith("*/") or pat.endswith("/*/")):
                    pat = pat.strip("*/").rstrip("/") + "/**/*"
                out.append(pat)
        return out

    adjusted_include = normalize_patterns(include)
    adjusted_exclude = normalize_patterns(exclude, is_exclude=True)

    # Default: search everything
    if not adjusted_include:
        adjusted_include = ["**/*"]

    # Pre-split excludes into absolute/relative
    abs_excludes = [p for p in adjusted_exclude if os.path.isabs(p)]
    rel_excludes = [p for p in adjusted_exclude if not os.path.isabs(p)]

    def is_excluded(file_path: Path) -> bool:
        """Check if file should be excluded early (absolute + relative)."""
        f_str = str(file_path)
        # Absolute patterns
        for pat in abs_excludes:
            if "**" in pat or "*" in pat or "?" in pat:
                if fnmatch.fnmatch(f_str, pat):
                    return True
            else:
                if Path(pat) in [file_path, *file_path.parents]:
                    return True
        # Relative patterns (match from base_path)
        rel = os.path.relpath(f_str, base_path)
        for pat in rel_excludes:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    # Collect candidates
    for pattern in adjusted_include:
        try:
            candidates: Iterable[Path]
            if os.path.isabs(pattern):
                abs_path = Path(pattern)
                if abs_path.is_file():
                    candidates = [abs_path]
                elif abs_path.is_dir():
                    candidates = abs_path.rglob("*")
                else:
                    # Handle absolute wildcard patterns
                    if any(x in pattern for x in ["*", "?", "**"]):
                        root = Path("/")
                        try:
                            candidates = root.glob(pattern.lstrip("/"))
                        except NotImplementedError:
                            # fallback: manual walk + fnmatch
                            candidates = [
                                p for p in root.rglob("*") if fnmatch.fnmatch(str(p), pattern)
                            ]
                    else:
                        continue
            else:
                candidates = base_path.rglob(pattern)

            for file_path in candidates:
                if not file_path.is_file():
                    continue

                # Exclude check (early)
                if is_excluded(file_path):
                    continue

                # Extension filter
                if normalized_extensions:
                    ext = file_path.suffix.lstrip(".").lower()
                    if ext not in normalized_extensions:
                        continue

                # Modified time filter
                if modified_after:
                    try:
                        if file_path.stat().st_mtime <= modified_after:
                            continue
                    except OSError as e:
                        logger.print_exception(f"Failed to get modified time for {file_path}: {e}")
                        continue

                norm_path = os.path.normpath(str(file_path)).replace("/private/var", "/var")
                matched_files.add(norm_path)

        except OSError as e:
            logger.print_exception(f"Error traversing {pattern}: {e}")

    # Final content filtering
    final_files = [
        f
        for f in matched_files
        if matches_content(f, include_content_patterns, exclude_content_patterns, case_sensitive)
    ]

    return sorted(final_files)


def matches_content(
    file_path: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    case_sensitive: bool = False,
) -> bool:
    if not include_patterns and not exclude_patterns:
        return True

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not case_sensitive:
            content = content.lower()
            include_patterns = [p.lower() for p in include_patterns]
            exclude_patterns = [p.lower() for p in exclude_patterns]

        if include_patterns and not any(
            fnmatch.fnmatch(content, p) if any(x in p for x in "*?") else p in content
            for p in include_patterns
        ):
            return False

        if exclude_patterns and any(
            fnmatch.fnmatch(content, p) if any(x in p for x in "*?") else p in content
            for p in exclude_patterns
        ):
            return False

        return True
    except (OSError, IOError) as e:
        logger.print_exception(f"Error reading {file_path}: {e}")
        return False

def get_file_length(file_path, shorten_funcs):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = clean_content(content, file_path, shorten_funcs)
        return len(content)
    except (OSError, IOError, UnicodeDecodeError):
        return 0

def format_file_structure(base_dir, include_files, exclude_files, include_content, exclude_content, case_sensitive=True, shorten_funcs=True, show_file_length=True):
    files: list[str] = find_files(base_dir, include_files, exclude_files,
                                  include_content, exclude_content, case_sensitive)
    # Create a new set for absolute file paths
    absolute_file_paths = set()

    # Iterate in reverse to avoid index shifting while popping
    for file in files:
        if not file.startswith("/"):
            file = os.path.join(file_dir, file)
        absolute_file_paths.add(os.path.relpath(file))

    files = list(absolute_file_paths)

    dir_structure = {}
    total_char_length = 0

    for file in files:
        # Convert to relative path
        file = os.path.relpath(file)

        dirs = file.split(os.sep)
        current_level = dir_structure

        if file.startswith("/"):
            dirs.pop(0)
        if ".." in dirs:
            dirs = [dir for dir in dirs if dir != ".."]

        for dir_name in dirs[:-1]:
            if dir_name not in current_level:
                current_level[dir_name] = {}
            current_level = current_level[dir_name]

        file_path = os.path.join(base_dir, file)
        file_length = get_file_length(file_path, shorten_funcs)
        total_char_length += file_length

        if show_file_length:
            current_level[f"{dirs[-1]} ({file_length})"] = None
        else:
            current_level[dirs[-1]] = None

    def print_structure(level, indent="", is_base_level=False):
        result = ""
        sorted_keys = sorted(level.items(), key=lambda x: (
            x[1] is not None, x[0].lower()))

        if is_base_level:
            for key, value in sorted_keys:
                if value is None:
                    result += key + "\n"
                else:
                    result += key + "/\n"
                    result += print_structure(value, indent + "    ", False)
        else:
            for key, value in sorted_keys:
                if value is None:
                    result += indent + "├── " + key + "\n"
                else:
                    result += indent + "├── " + key + "/\n"
                    result += print_structure(value, indent + "│   ", False)

        return result

    file_structure = print_structure(dir_structure, is_base_level=True)
    file_structure = file_structure.strip()
    # file_structure = f"Base dir: {file_dir}\n" + \
    #     f"\nFile structure:\n{file_structure}"
    print(
        f"\n----- FILES STRUCTURE -----\n{file_structure}\n----- END FILES STRUCTURE -----\n")
    print("\n")
    num_files = len(files)
    logger.log("Number of Files:", num_files)
    logger.log("Files Char Count:", total_char_length)
    return file_structure


import ast
import re
from textwrap import dedent


def get_signature(node, content, indent=0):
    source = ast.get_source_segment(content, node).splitlines()
    signature_lines = []
    for line in source:
        stripped = line.rstrip()
        signature_lines.append(stripped)
        if stripped.endswith(":"):
            break
    # Remove trailing colon for function definitions, but keep for classes
    if signature_lines and signature_lines[-1].endswith(":") and not isinstance(node, ast.ClassDef):
        signature_lines[-1] = signature_lines[-1][:-1]
    return "\n".join("    " * indent + line for line in signature_lines)


def shorten_functions(content: str, remove_class_vars: bool = False) -> str:
    """
    Shorten function and class definitions to their signatures, optionally removing class variables.
    Args:
        content: The source code to process.
        remove_class_vars: If True, exclude class-level variables in the output; if False, include them.
    Returns:
        A string containing only the signatures of functions and classes.
    """
    tree = ast.parse(content)
    definitions = []

    def process_node(node, indent=0):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not hasattr(node, 'parent') or not isinstance(node.parent, ast.ClassDef):
                definitions.append(get_signature(node, content, indent))
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                child.parent = node
            class_lines = [get_signature(node, content, indent)]
            if not remove_class_vars:
                for body_node in node.body:
                    if isinstance(body_node, ast.AnnAssign):
                        var_line = ast.get_source_segment(
                            content, body_node).rstrip()
                        class_lines.append("    " * (indent + 1) + var_line)
                    elif isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_lines.append(get_signature(
                            body_node, content, indent=indent + 1))
                    elif isinstance(body_node, ast.ClassDef):
                        for child in ast.iter_child_nodes(body_node):
                            child.parent = body_node
                        class_lines.append(process_node(
                            body_node, indent=indent + 1))
            else:
                for body_node in node.body:
                    if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_lines.append(get_signature(
                            body_node, content, indent=indent + 1))
                    elif isinstance(body_node, ast.ClassDef):
                        for child in ast.iter_child_nodes(body_node):
                            child.parent = body_node
                        class_lines.append(process_node(
                            body_node, indent=indent + 1))
            return "\n".join(class_lines)

    for node in tree.body:  # Only process top-level nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result = process_node(node)
            if result:
                definitions.append(result)

    return dedent("\n".join(definitions))

def strip_comments(content: str, remove_triple_quoted_definitions: bool = False) -> str:
    """
    Remove comments outside of triple-quoted strings and string literals.
    Preserves entire triple-quoted strings and inline '#' inside quotes.
    If remove_triple_quoted_definitions=True, removes all triple double quoted block definitions.
    """
    triple_quote_pattern = re.compile(r"('''|\"\"\")")
    lines = content.splitlines()
    result_lines = []
    in_triple_quote = False
    current_quote = ""

    for line in lines:
        if not in_triple_quote:
            match = triple_quote_pattern.search(line)
            if match:
                current_quote = match.group(1)
                if line.count(current_quote) == 2:
                    # Opening and closing on the same line
                    if not (remove_triple_quoted_definitions and current_quote == '"""'):
                        result_lines.append(line)
                    continue
                in_triple_quote = True
                if not (remove_triple_quoted_definitions and current_quote == '"""'):
                    result_lines.append(line)
            else:
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue  # remove full-line comment

                # walk through chars and detect # only if not inside quotes
                new_line = []
                in_single = in_double = False
                i = 0
                while i < len(line):
                    ch = line[i]
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        new_line.append(ch)
                    elif ch == '"' and not in_single:
                        in_double = not in_double
                        new_line.append(ch)
                    elif ch == '#' and not in_single and not in_double:
                        break  # start of comment outside quotes
                    else:
                        new_line.append(ch)
                    i += 1

                cleaned = "".join(new_line).rstrip()
                if cleaned:
                    result_lines.append(cleaned)
        else:
            # inside triple quotes
            if not (remove_triple_quoted_definitions and current_quote == '"""'):
                result_lines.append(line)

            if current_quote in line:
                if line.count(current_quote) % 2 == 1:
                    in_triple_quote = False

    cleaned = re.sub(r'\n\s*\n', '\n', '\n'.join(result_lines)).strip()
    return cleaned


def clean_newlines(content):
    """Removes consecutive newlines from the given content."""
    return re.sub(r'\n\s*\n+', '\n', content)

def clean_comments(content):
    """Removes comments from the given content."""
    return re.sub(r'#.*', '', content)


def clean_logging(content):
    """Removes logging statements from the given content, including multi-line ones."""
    logging_pattern = re.compile(
        r'logging\.(?:info|debug|error|warning|critical|exception|log|basicConfig|getLogger|disable|shutdown)\s*\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)',
        re.DOTALL
    )
    content = re.sub(logging_pattern, '', content)
    content = re.sub(r'\n\s*\n', '\n', content)
    return content


def clean_content(content: str, file_path: str, shorten_funcs: bool = True, remove_triple_quoted_definitions: bool = False):
    """Clean the content based on file type and apply various cleaning operations."""
    if file_path.endswith(".py"):
        content = strip_comments(content, remove_triple_quoted_definitions)
        if shorten_funcs:
            content = shorten_functions(content)
    if not file_path.endswith(".md"):
        content = clean_comments(content)
    content = clean_logging(content)
    # content = clean_print(content)
    return content


def remove_parent_paths(path: str) -> str:
    return os.path.join(
        *(part for part in os.path.normpath(path).split(os.sep) if part != ".."))

import pyperclip  # lazy import – only needed on Windows when used

def copy_to_clipboard(text: str) -> None:
    """
    Copy text to clipboard using pyperclip.
    Assumes pyperclip is installed (recommended and required for perfect Unicode support on Windows).
    """
    try:
        pyperclip.copy(text)
        logger.log("[bold green]Copied to clipboard[/] (via pyperclip)", len(text), "chars")
    except Exception as e:
        logger.print_exception()
        raise RuntimeError(f"Failed to copy to clipboard: {e}")
