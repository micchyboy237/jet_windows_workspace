import os
import fnmatch
import argparse
import subprocess
import json
from rich.console import Console
from tqdm import tqdm
from _utils_copy_for_prompt import (
    find_files,
    format_file_structure,
    clean_newlines,
    clean_content,
    remove_parent_paths,
    copy_to_clipboard,
)

logger = Console()

exclude_files = [
    "**/.git/",
    "**/.gitignore",
    "**/.DS_Store",
    "**/*.pyc",
    "**/_copy*.py",
    "**/__pycache__/",
    "**/.pytest_cache/",
    "**/node_modules/",
    "**/*lock.json",
    "**/*.lock",
    "**/public/",
    "**/mocks/",
    "**/.venv/",
    "**/dream/",
    "**/jupyter/",
    "**/*.png",
    # "**/_*",
    # "**/.cache/",
    "**/_git_stats.json",
    "**/stats_results/",
    # "**/generated/",
    # "**/.*",

    # Custom
    # "**/*.sh"
    # "**/__init__.py",
    # "*.md",
    "**/context.py",
]
include_files = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Examples\.vscode\launch.json",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\main.py",

    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\stream_vad.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\core\constants.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\core\stream_vad_postprocessor.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ReazonSpeech\pkg\k2-asr\src",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server3.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_client2.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_matcher_ja.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\diff_utils.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_utils.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\diff_words.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py",
    r"",
]

structure_include = [
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/libs/context_engineering/self_refinement_lab/practical_examples/",
    # "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/letta/alembic/",
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

DEFAULT_QUERY_MESSAGE = r"""
Analyze remaining issues carefully before fixing

pytest C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
================================================= test session starts =================================================
platform win32 -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0
benchmark: 5.1.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: C:\Users\druiv
plugins: anyio-4.12.0, hydra-core-1.3.2, langsmith-0.4.38, asyncio-1.3.0, benchmark-5.1.0, mock-3.15.1, snapshot-0.9.0, typeguard-4.5.1
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 18 items

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py ....FF.FF...F....F             [100%]

====================================================== FAILURES =======================================================
____________________ test_count_newly_appended_words[The quick brown-The quick blue fox jumps- -2] ____________________

a = 'The quick brown', b = 'The quick blue fox jumps', word_sep = ' ', expected = 2

    @pytest.mark.parametrize(
        "a, b, word_sep, expected",
        [
            # Basic append cases
            ("", "hello world", " ", 2),
            ("hello", "hello world", " ", 1),
            ("hello world", "hello world extra", " ", 1),
            ("a b c", "a b c d e f", " ", 3),

            # Changes in the middle + append at the end
            ("The quick brown", "The quick blue fox jumps", " ", 2),   # "fox", "jumps"
            ("Python is great", "Python is powerful and awesome", " ", 2),  # "and", "awesome"

            # No new words appended
            ("hello world", "hello world", " ", 0),
            ("hello world", "hello universe", " ", 0),
            ("hello world", "hello world!!!", " ", 0),

            # Edge cases
            ("", "", " ", 0),
            (" ", "word", " ", 1),
            ("word", " ", " ", 0),
            ("a b c d", "x y z a b c d e", " ", 1),   # only "e" is appended

            # Whitespace handling
            ("hello   world", "hello   world   extra", " ", 1),

            # Japanese example
            ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " ", 2),

            # Custom separators
            ("apple,banana,cherry", "apple,banana,cherry,date", ",", 1),
            ("1|2|3", "1|2|3|4|5", "|", 2),
        ],
    )
    def test_count_newly_appended_words(a: str, b: str, word_sep: str, expected: int):
>       assert count_newly_appended_words(a, b, word_sep) == expected
E       AssertionError: assert 3 == 2
E        +  where 3 = count_newly_appended_words('The quick brown', 'The quick blue fox jumps', ' ')

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py:43: AssertionError
_________________ test_count_newly_appended_words[Python is great-Python is powerful and awesome- -2] _________________

a = 'Python is great', b = 'Python is powerful and awesome', word_sep = ' ', expected = 2

    @pytest.mark.parametrize(
        "a, b, word_sep, expected",
        [
            # Basic append cases
            ("", "hello world", " ", 2),
            ("hello", "hello world", " ", 1),
            ("hello world", "hello world extra", " ", 1),
            ("a b c", "a b c d e f", " ", 3),

            # Changes in the middle + append at the end
            ("The quick brown", "The quick blue fox jumps", " ", 2),   # "fox", "jumps"
            ("Python is great", "Python is powerful and awesome", " ", 2),  # "and", "awesome"

            # No new words appended
            ("hello world", "hello world", " ", 0),
            ("hello world", "hello universe", " ", 0),
            ("hello world", "hello world!!!", " ", 0),

            # Edge cases
            ("", "", " ", 0),
            (" ", "word", " ", 1),
            ("word", " ", " ", 0),
            ("a b c d", "x y z a b c d e", " ", 1),   # only "e" is appended

            # Whitespace handling
            ("hello   world", "hello   world   extra", " ", 1),

            # Japanese example
            ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " ", 2),

            # Custom separators
            ("apple,banana,cherry", "apple,banana,cherry,date", ",", 1),
            ("1|2|3", "1|2|3|4|5", "|", 2),
        ],
    )
    def test_count_newly_appended_words(a: str, b: str, word_sep: str, expected: int):
>       assert count_newly_appended_words(a, b, word_sep) == expected
E       AssertionError: assert 3 == 2
E        +  where 3 = count_newly_appended_words('Python is great', 'Python is powerful and awesome', ' ')

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py:43: AssertionError
___________________________ test_count_newly_appended_words[hello world-hello universe- -0] ___________________________

a = 'hello world', b = 'hello universe', word_sep = ' ', expected = 0

    @pytest.mark.parametrize(
        "a, b, word_sep, expected",
        [
            # Basic append cases
            ("", "hello world", " ", 2),
            ("hello", "hello world", " ", 1),
            ("hello world", "hello world extra", " ", 1),
            ("a b c", "a b c d e f", " ", 3),

            # Changes in the middle + append at the end
            ("The quick brown", "The quick blue fox jumps", " ", 2),   # "fox", "jumps"
            ("Python is great", "Python is powerful and awesome", " ", 2),  # "and", "awesome"

            # No new words appended
            ("hello world", "hello world", " ", 0),
            ("hello world", "hello universe", " ", 0),
            ("hello world", "hello world!!!", " ", 0),

            # Edge cases
            ("", "", " ", 0),
            (" ", "word", " ", 1),
            ("word", " ", " ", 0),
            ("a b c d", "x y z a b c d e", " ", 1),   # only "e" is appended

            # Whitespace handling
            ("hello   world", "hello   world   extra", " ", 1),

            # Japanese example
            ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " ", 2),

            # Custom separators
            ("apple,banana,cherry", "apple,banana,cherry,date", ",", 1),
            ("1|2|3", "1|2|3|4|5", "|", 2),
        ],
    )
    def test_count_newly_appended_words(a: str, b: str, word_sep: str, expected: int):
>       assert count_newly_appended_words(a, b, word_sep) == expected
E       AssertionError: assert 1 == 0
E        +  where 1 = count_newly_appended_words('hello world', 'hello universe', ' ')

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py:43: AssertionError
___________________________ test_count_newly_appended_words[hello world-hello world!!!- -0] ___________________________

a = 'hello world', b = 'hello world!!!', word_sep = ' ', expected = 0

    @pytest.mark.parametrize(
        "a, b, word_sep, expected",
        [
            # Basic append cases
            ("", "hello world", " ", 2),
            ("hello", "hello world", " ", 1),
            ("hello world", "hello world extra", " ", 1),
            ("a b c", "a b c d e f", " ", 3),

            # Changes in the middle + append at the end
            ("The quick brown", "The quick blue fox jumps", " ", 2),   # "fox", "jumps"
            ("Python is great", "Python is powerful and awesome", " ", 2),  # "and", "awesome"

            # No new words appended
            ("hello world", "hello world", " ", 0),
            ("hello world", "hello universe", " ", 0),
            ("hello world", "hello world!!!", " ", 0),

            # Edge cases
            ("", "", " ", 0),
            (" ", "word", " ", 1),
            ("word", " ", " ", 0),
            ("a b c d", "x y z a b c d e", " ", 1),   # only "e" is appended

            # Whitespace handling
            ("hello   world", "hello   world   extra", " ", 1),

            # Japanese example
            ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " ", 2),

            # Custom separators
            ("apple,banana,cherry", "apple,banana,cherry,date", ",", 1),
            ("1|2|3", "1|2|3|4|5", "|", 2),
        ],
    )
    def test_count_newly_appended_words(a: str, b: str, word_sep: str, expected: int):
>       assert count_newly_appended_words(a, b, word_sep) == expected
E       AssertionError: assert 1 == 0
E        +  where 1 = count_newly_appended_words('hello world', 'hello world!!!', ' ')

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py:43: AssertionError
____________________________ test_count_newly_appended_words[a b c d-x y z a b c d e- -1] _____________________________

a = 'a b c d', b = 'x y z a b c d e', word_sep = ' ', expected = 1

    @pytest.mark.parametrize(
        "a, b, word_sep, expected",
        [
            # Basic append cases
            ("", "hello world", " ", 2),
            ("hello", "hello world", " ", 1),
            ("hello world", "hello world extra", " ", 1),
            ("a b c", "a b c d e f", " ", 3),

            # Changes in the middle + append at the end
            ("The quick brown", "The quick blue fox jumps", " ", 2),   # "fox", "jumps"
            ("Python is great", "Python is powerful and awesome", " ", 2),  # "and", "awesome"

            # No new words appended
            ("hello world", "hello world", " ", 0),
            ("hello world", "hello universe", " ", 0),
            ("hello world", "hello world!!!", " ", 0),

            # Edge cases
            ("", "", " ", 0),
            (" ", "word", " ", 1),
            ("word", " ", " ", 0),
            ("a b c d", "x y z a b c d e", " ", 1),   # only "e" is appended

            # Whitespace handling
            ("hello   world", "hello   world   extra", " ", 1),

            # Japanese example
            ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " ", 2),

            # Custom separators
            ("apple,banana,cherry", "apple,banana,cherry,date", ",", 1),
            ("1|2|3", "1|2|3|4|5", "|", 2),
        ],
    )
    def test_count_newly_appended_words(a: str, b: str, word_sep: str, expected: int):
>       assert count_newly_appended_words(a, b, word_sep) == expected
E       AssertionError: assert 8 == 1
E        +  where 8 = count_newly_appended_words('a b c d', 'x y z a b c d e', ' ')

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py:43: AssertionError
________________________________________________ test_complex_changes _________________________________________________

    def test_complex_changes():
        \"\"\"Heavy changes early, but new words appended at the end.\"\"\"
>       assert count_newly_appended_words(
            "The quick brown fox jumps over the lazy dog",
            "A completely different sentence that ends with new words here"
        ) == 3   # "new", "words", "here"
E       AssertionError: assert 10 == 3
E        +  where 10 = count_newly_appended_words('The quick brown fox jumps over the lazy dog', 'A completely different sentence that ends with new words here')

Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_diff_words.py:48: AssertionError
=============================================== short test summary info ===============================================
FAILED Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/test_diff_words.py::test_count_newly_appended_words[The quick brown-The quick blue fox jumps- -2] - AssertionError: assert 3 == 2
FAILED Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/test_diff_words.py::test_count_newly_appended_words[Python is great-Python is powerful and awesome- -2] - AssertionError: assert 3 == 2
FAILED Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/test_diff_words.py::test_count_newly_appended_words[hello world-hello universe- -0] - AssertionError: assert 1 == 0
FAILED Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/test_diff_words.py::test_count_newly_appended_words[hello world-hello world!!!- -0] - AssertionError: assert 1 == 0
FAILED Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/test_diff_words.py::test_count_newly_appended_words[a b c d-x y z a b c d e- -1] - AssertionError: assert 8 == 1
FAILED Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/test_diff_words.py::test_complex_changes - AssertionError: assert 10 == 3
============================================ 6 failed, 12 passed in 0.34s =============================================
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
Provide step by step analysis first.
Show unified diff for updated files, while show python code block for new files.
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
""".strip()

# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z

# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- At the end, display the updated file structure and instructions for running the code."
#     "\n- Provide complete working code for each file (should match file structure)"
# )

# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)

def get_language_from_extension(filename: str) -> str:
    """
    Simple file extension → markdown code fence language mapping
    Returns 'text' as safe fallback
    """
    ext = os.path.splitext(filename.lower())[1]

    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".md": "markdown",
        ".mdx": "mdx",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".sh": "bash",
        ".bash": "bash",
        ".sql": "sql",
        ".prisma": "prisma",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".php": "php",
        ".rb": "ruby",
    }

    return mapping.get(ext, "text")


def main():
    global exclude_files, include_files, include_content, exclude_content

    print("Running _copy_for_prompt.py")
    # Parse command-line options
    parser = argparse.ArgumentParser(
        description='Generate clipboard content from specified files.')
    parser.add_argument('-b', '--base-dir', default=file_dir,
                        help='Base directory to search files in (default: current directory)')
    parser.add_argument('-if', '--include-files', nargs='*', default=include_files,
                        help='Patterns of files to include (default: schema.prisma, episode)')
    parser.add_argument('-ef', '--exclude-files', nargs='*', default=exclude_files,
                        help='Directories or files to exclude (default: node_modules)')
    parser.add_argument('-ic', '--include-content', nargs='*', default=include_content,
                        help='Patterns of file content to include')
    parser.add_argument('-ec', '--exclude-content', nargs='*', default=exclude_content,
                        help='Patterns of file content to exclude')
    parser.add_argument('-cs', '--case-sensitive', action='store_true', default=False,
                        help='Make content pattern matching case-sensitive')
    parser.add_argument('-sf', '--shorten-funcs', action='store_true', default=SHORTEN_FUNCTS,
                        help='Shorten function and class definitions')
    parser.add_argument('-s', '--system', default=DEFAULT_SYSTEM_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-m', '--message', default=DEFAULT_QUERY_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-i', '--instructions', default=DEFAULT_INSTRUCTIONS_MESSAGE,
                        help='Instructions to include in the clipboard content')
    parser.add_argument('-fo', '--filenames-only', action='store_true',
                        help='Only copy the relative filenames, not their contents')
    parser.add_argument('-nl', '--no-length', action='store_true', default=INCLUDE_FILE_STRUCTURE,
                        help='Do not show file character length')

    args = parser.parse_args()
    base_dir = args.base_dir
    include = args.include_files
    exclude = args.exclude_files
    include_content = args.include_content
    exclude_content = args.exclude_content
    case_sensitive = args.case_sensitive
    shorten_funcs = args.shorten_funcs
    query_message = args.message
    system_message = args.system
    instructions_message = args.instructions
    filenames_only = args.filenames_only
    show_file_length = not args.no_length

    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(base_dir, include, exclude,
                               include_content, exclude_content, case_sensitive)

    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"\nFound files ({len(context_files)}):\n{
          json.dumps(context_files, indent=2)}")

    print("\n")

    # Initialize the clipboard content
    clipboard_content = ""

    if not context_files:
        print("No context files found matching the given patterns.")
    else:

        # Append relative filenames to the clipboard content
        for file in tqdm(
            context_files, desc=f"Processing {len(context_files)} files..."
        ):
            rel_path = os.path.relpath(path=file, start=file_dir)
            cleaned_rel_path = remove_parent_paths(rel_path)

            prefix = (
                f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n")
            if filenames_only:
                clipboard_content += f"{prefix}"
            else:
                file_path = os.path.relpath(os.path.join(base_dir, file))
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                            content = clean_content(content, file, shorten_funcs)
                            # ── NEW: Add fenced code block ───────────────────────────────
                            lang = get_language_from_extension(file)
                            fenced_content = f"```{lang}\n{content.rstrip()}\n```"
                            clipboard_content += f"{prefix}{fenced_content}\n\n"
                    except Exception:
                        # Continue to the next file
                        continue
                else:
                    clipboard_content += f"{prefix}\n"

        clipboard_content = clean_newlines(clipboard_content).strip()

    # Generate and format the file structure
    structure_include_files = structure_include
    if include:
        structure_include_files += include
    structure_exclude_files = structure_exclude
    if exclude:
        structure_exclude_files += exclude
    files_structure = format_file_structure(
        base_dir,
        include_files=structure_include_files,
        exclude_files=structure_exclude_files,
        include_content=include_content,
        exclude_content=exclude_content,
        case_sensitive=case_sensitive,
        shorten_funcs=shorten_funcs,
        show_file_length=show_file_length,
    )

    # Prepend system and query to the clipboard content then append instructions
    clipboard_content_parts = []

    if system_message:
        clipboard_content_parts.append(f"System\n{system_message}\n")
    # Query should come before instructions
    clipboard_content_parts.append(f"Query\n{query_message}\n")
    if instructions_message:
        clipboard_content_parts.append(f"Instructions\n{instructions_message}\n")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}\n")

    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}\n"
        )

    clipboard_content = "\n\n".join(clipboard_content_parts)

    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)

    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))

    # Newline
    print("\n")


if __name__ == "__main__":
    main()
