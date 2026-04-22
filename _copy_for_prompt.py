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
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ten-vad\include\ten_vad.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ten-vad\examples\build-and-deploy-windows.bat",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ten-vad\examples\plot_pr_curves.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ten-vad\examples\test.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ten-vad\README.md",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\receiver_pipeline.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_firered.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_tenvad.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\audio_context_buffer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_matcher_ja.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\diff_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_funasr.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speech_waves.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\loader.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\file_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\energy.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_tenvad.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\temp4.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_firered2.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_firered.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\temp4.py",
    r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--separation-ami-1.0\snapshots\4d38e95cfd067c894b8b60b00761831fb01e4a8c\README.md",
    r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--speech-separation-ami-1.0\snapshots\9486b106945ae0cc0784041a08bfcdba5edadfb9\README.md",
    r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--speech-separation-ami-1.0\snapshots\9486b106945ae0cc0784041a08bfcdba5edadfb9\config.yaml",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\run_separation_pipeline.py",
    r"",
]

structure_include = [
    # r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--separation-ami-1.0\snapshots\4d38e95cfd067c894b8b60b00761831fb01e4a8c",
    # r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--speech-separation-ami-1.0\snapshots\9486b106945ae0cc0784041a08bfcdba5edadfb9",
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

DEFAULT_QUERY_MESSAGE = r"""
Browse how to resolve then fix run_separation_pipeline

Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\run_separation_pipeline.py", line 23, in <module>
    pipeline = Pipeline.from_pretrained(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\pipeline.py", line 245, in from_pretrained
    pipeline = Klass(**params)
               ^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\speech_separation.py", line 179, in __init__
    self._embedding = PretrainedSpeakerEmbedding(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\speaker_verification.py", line 762, in PretrainedSpeakerEmbedding
    return SpeechBrainPretrainedSpeakerEmbedding(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\speaker_verification.py", line 255, in __init__
    self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\inference\interfaces.py", line 487, in from_hparams
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
OSError: [WinError 1314] A required privilege is not held by the client: 'C:\\Users\\druiv\\.cache\\huggingface\\hub\\models--speechbrain--spkrec-ecapa-voxceleb\\snapshots\\0f99f2d0ebe89ac095bcc5903c4dd8f72b367286\\hyperparams.yaml' -> 'C:\\Users\\druiv\\None\\speechbrain\\hyperparams.yaml'
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
Provide step by step analysis and outline blueprint first.
Show unified diff for updated files, while show language code block for new files.
Use easy to understand terms when explaining.
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
