import os
import fnmatch
import argparse
import subprocess
import json
import tiktoken
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
from headroom import compress

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
    "**/*.svg",
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
]
include_files = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Examples\.vscode\launch.json",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\main.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_overlap_aware_diarization.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py",
    r"",
]

structure_include = [
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers",
]
structure_exclude = []

include_content = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\tagger",
]
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

COMPRESSION_MODEL = "gpt-4o"
TOKEN_BUDGET = 8000

DEFAULT_QUERY_MESSAGE = r"""
Browse how to resolve

python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py
13:17:22  INFO     Arguments parsed: strategy=resegment, condition=noisy, speakers=None, output=C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_overlap_aware_diarization
13:17:22  INFO     Output directory set to: C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_overlap_aware_diarization
13:17:22  INFO     ============================================================
13:17:22  INFO       ECAPA-TDNN Diarization  |  strategy=resegment  |  condition=noisy
13:17:22  INFO     ============================================================
13:17:22  INFO     Loading audio: C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.pretrained' was deprecated, redirecting to 'speechbrain.inference'. Please update your script. This is a change from SpeechBrain 1.0. See: https://github.com/speechbrain/speechbrain/releases/tag/v1.0.0
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.k2_integration' was deprecated, redirecting to 'speechbrain.integrations.k2_fsa'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.wordemb' was deprecated, redirecting to 'speechbrain.integrations.huggingface.wordemb'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.lobes.models.huggingface_transformers' was deprecated, redirecting to 'speechbrain.integrations.huggingface'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.lobes.models.spacy' was deprecated, redirecting to 'speechbrain.integrations.nlp'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.lobes.models.flair' was deprecated, redirecting to 'speechbrain.integrations.nlp'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
13:17:28  INFO     Numba verbose is deactivated. To enable it, set NUMBA_VERBOSE to 1.
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.nnet.loss.transducer_loss' was deprecated, redirecting to 'speechbrain.integrations.numba.transducer_loss'. Please update your script. This module depends on the optional 'numba' package. If you encounter an ImportError here, please install numba, for example with: pip install numba
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.wordemb' was deprecated, redirecting to 'speechbrain.integrations.huggingface.wordemb'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
13:17:31  INFO     Audio: 58.0s  |  16000 Hz  |  mono
13:17:31  INFO     Loading ECAPA-TDNN from speechbrain/spkrec-ecapa-voxceleb …
13:17:31  INFO     Fetch hyperparams.yaml: Fetching from HuggingFace Hub 'speechbrain/spkrec-ecapa-voxceleb' if not cached
13:17:32  INFO     HTTP Request: HEAD https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.yaml "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
13:17:32  WARNING  Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
13:17:32  INFO     HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/hyperparams.yaml "HTTP/1.1 200 OK"
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\fetching.py:153: UserWarning: Using SYMLINK strategy on Windows for fetching potentially requires elevated privileges and is not recommended. See `LocalStrategy` documentation.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py", line 1076, in <module>
    main()
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_overlap_aware_diarization.py", line 235, in main
    result = run_pipeline(
             ^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py", line 976, in run_pipeline
    encoder = load_ecapa()
              ^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py", line 206, in load_ecapa
    encoder = EncoderClassifier.from_hparams(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\inference\interfaces.py", line 489, in from_hparams
    return pretrained_from_hparams(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\inference\interfaces.py", line 183, in pretrained_from_hparams
    hparams_local_path = fetch(
                         ^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\fetching.py", line 429, in fetch
    download_file_hf(hf_kwargs, destination, local_strategy)
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\distributed.py", line 318, in main_proc_wrapped_func
    result = function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\fetching.py", line 277, in download_file_hf
    link_with_strategy(fetched_file, destination, local_strategy)
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\fetching.py", line 164, in link_with_strategy
    dst.symlink_to(src)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\pathlib.py", line 1386, in symlink_to
    os.symlink(target, self, target_is_directory)
OSError: [WinError 1314] A required privilege is not held by the client: 'C:\\Users\\druiv\\.cache\\huggingface\\hub\\models--speechbrain--spkrec-ecapa-voxceleb\\snapshots\\0f99f2d0ebe89ac095bcc5903c4dd8f72b367286\\hyperparams.yaml' -> 'C:\\Users\\druiv\\Desktop\\Jet_Files\\Jet_Windows_Workspace\\pretrained_ecapa\\hyperparams.yaml'
(jet_venv) PS C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace>
"""

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:
- Browse when beneficial or requested.
- Keep explanations simple and clear.
When coding:
- Provide step-by-step analysis and explain the flow.
- Use visuals, diagrams, or tables when helpful.
- Show full code for new files, then show full function code for new or updated functions.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Always add logs so we can trace and know if all features work correctly.
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
""".strip()
# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z
# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- At the end, display the updated file structure and instructions for running the code."
# "\n- Provide complete working code for each file (should match file structure)"
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
        description="Generate clipboard content from specified files."
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=file_dir,
        help="Base directory to search files in (default: current directory)",
    )
    parser.add_argument(
        "-if",
        "--include-files",
        nargs="*",
        default=include_files,
        help="Patterns of files to include (default: schema.prisma, episode)",
    )
    parser.add_argument(
        "-ef",
        "--exclude-files",
        nargs="*",
        default=exclude_files,
        help="Directories or files to exclude (default: node_modules)",
    )
    parser.add_argument(
        "-ic",
        "--include-content",
        nargs="*",
        default=include_content,
        help="Patterns of file content to include",
    )
    parser.add_argument(
        "-ec",
        "--exclude-content",
        nargs="*",
        default=exclude_content,
        help="Patterns of file content to exclude",
    )
    parser.add_argument(
        "-cs",
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Make content pattern matching case-sensitive",
    )
    parser.add_argument(
        "-sf",
        "--shorten-funcs",
        action="store_true",
        default=SHORTEN_FUNCTS,
        help="Shorten function and class definitions",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=DEFAULT_QUERY_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-i",
        "--instructions",
        default=DEFAULT_INSTRUCTIONS_MESSAGE,
        help="Instructions to include in the clipboard content",
    )
    parser.add_argument(
        "-fo",
        "--filenames-only",
        action="store_true",
        help="Only copy the relative filenames, not their contents",
    )
    parser.add_argument(
        "-nl",
        "--no-length",
        action="store_true",
        default=INCLUDE_FILE_STRUCTURE,
        help="Do not show file character length",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=False,
        help="Enable compression of the clipboard content before copying (default: False)",
    )
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
    compress_enabled = args.compress
    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(
        base_dir, include, exclude, include_content, exclude_content, case_sensitive
    )
    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"Compress enabled: {compress_enabled}")
    print(
        f"\nFound files ({len(context_files)}):\n{json.dumps(context_files, indent=2)}"
    )
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
            prefix = f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n"
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
    clipboard_content_parts.append(f"{query_message}\n\n")
    if instructions_message:
        clipboard_content_parts.append(f"Instructions\n{instructions_message}\n")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}\n")
    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}\n"
        )
    clipboard_content = "\n\n".join(clipboard_content_parts)
    # Compress to reduce tokens (optional)
    if compress_enabled:
        messages = [{"role": "user", "content": clipboard_content}]
        result = compress(
            messages,
            model=COMPRESSION_MODEL,  # headroom uses this for strategy selection only
            token_budget=TOKEN_BUDGET,  # enforce fit within llama-server context
            ccr_enabled=True,  # reversible compression (default)
            compress_user_messages=True,
            target_ratio=0.5,  # keep 50% — safe for mixed prose + code
            protect_recent=0,  # only 1 message, nothing to protect
            protect_analysis_context=False,  # do not protect code from compression
            # kompress_model="disabled",
        )
        # Log compression stats using logger.log for each result.*
        logger.log("Tokens before:", f"{result.tokens_before:,}")
        logger.log("Tokens after:", f"{result.tokens_after:,}")
        logger.log(
            "Tokens saved:",
            f"{result.tokens_saved:,} ({result.compression_ratio:.1%})",
        )
        logger.log(
            "Transforms applied:",
            str(result.transforms_applied),
        )
    else:
        logger.log("Compression skipped (use -c or --compress to enable)")
    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)
    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))
    logger.log("Tokens Count (gpt-4o):", count_tokens(clipboard_content))
    # Newline
    print("\n")


def count_tokens(
    text: str,
    model: str = "gpt-4o",  # Best default
    encoding_name: str | None = None,
) -> int:
    """
    Count the number of tokens in a string using tiktoken.
    Args:
        text: The input string to tokenize.
        model: OpenAI model name to determine the encoding
               (default: "gpt-4o" — recommended).
        encoding_name: Optional direct encoding name
                       (e.g., "o200k_base", "cl100k_base").
                       Takes precedence over model.
    Returns:
        Number of tokens.
    """
    if encoding_name:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


if __name__ == "__main__":
    main()
