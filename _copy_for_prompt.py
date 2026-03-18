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
]
include_files = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Examples\.vscode\launch.json",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\main.py",

    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\README.md",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\requirements.txt",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\processing",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_handlers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\start_server.ps1",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_funasr.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\utils.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\audio_context_buffer.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_audio_context_buffer.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\utils.py",
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
Explain root cause of issue for context buffer and ja sentences based on below logs. Then show unified diff for the fix. Only update live_subtitles_server2 or audio_context_buffer for the fix.

[SERVER] Listening on ws://0.0.0.0:8765
connection open
[SERVER] Client connected — total 1
[empty context]
rtf_avg: 0.018: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.42it/s]
rtf_avg: 0.055: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.48it/s]
rtf_avg: 0.057, time_speech:  5.184, time_escape: 0.293: 100%|███████████████████████████| 1/1 [00:00<00:00,  3.41it/s]
 JA: 🎼世界 各国 が 水面下で 熾烈な 情報 戦 を 繰り広げる 時代。
 EN: 🎼The time when countries around the world engage in fierce information wars submerged beneath the surface.
[SERVER] Processed 2de12e05… → 🎼世界 各国 が 水面下で 熾烈な 情報 戦 を 繰り広げる 時代。…
rtf_avg: 0.009: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.14it/s]
  0%|                                                                                            | 0/1 [00:00<?, ?it/s]decoding, utt: tmpg37femar, empty speech
  0%|                                                                                            | 0/1 [00:00<?, ?it/s]
 FULL JA: [empty transcription]
 JA SENTS (0):[]
rtf_avg: 0.006: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 20.76it/s]
rtf_avg: 0.025: 100%|████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 11.47it/s]
rtf_avg: 0.023, time_speech:  7.776, time_escape: 0.177: 100%|███████████████████████████| 1/1 [00:00<00:00,  5.57it/s]
 JA: 🎼睨み 合う 2 つの 国、東の オスタニア、西 の ウェスタリス。戦争を企てるオスタ。
 EN: 🎼The eyes that meet in the land of Ostan and Vestris, East and West. The Ostan who plots war.
[SERVER] Processed 584dede7… → 🎼睨み 合う 2 つの 国、東の オスタニア、西 の ウェスタリス。戦争を企てるオスタ。…
rtf_avg: 0.006: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.72it/s]
rtf_avg: 0.134: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 11.10it/s]
rtf_avg: 0.004, time_speech:  26.572, time_escape: 0.093: 100%|██████████████████████████| 1/1 [00:00<00:00, 10.41it/s]
 FULL JA: 。
 JA SENTS (1):['\n1: 。']
rtf_avg: 0.007: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 19.31it/s]
rtf_avg: 0.031: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.13it/s]
rtf_avg: 0.029, time_speech:  6.912, time_escape: 0.198: 100%|███████████████████████████| 1/1 [00:00<00:00,  5.00it/s]
 JA: 🎼や 政府 要人 の 動向 を 探る べく、ウェスタリス は オペレーション ストリックス を 発動。
 EN: 🎼As Westalis, in order to investigate the trends of prominent government figures, she activates operation strix.
[SERVER] Processed 02b32c88… → 🎼や 政府 要人 の 動向 を 探る べく、ウェスタリス は オペレーション ストリックス を 発動…
rtf_avg: 0.006: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.09it/s]
rtf_avg: 0.087: 100%|████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 17.23it/s]
rtf_avg: 0.003, time_speech:  40.844, time_escape: 0.121: 100%|██████████████████████████| 1/1 [00:00<00:00,  8.12it/s]
 FULL JA: 。。
 JA SENTS (1):['\n1: 。。']
rtf_avg: 0.006: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 34.69it/s]
rtf_avg: 0.047: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.91it/s]
rtf_avg: 0.048, time_speech:  4.320, time_escape: 0.207: 100%|███████████████████████████| 1/1 [00:00<00:00,  4.79it/s]
 JA: 🎼線を担うスゴーデエージェント黄昏れ、百の顔を使いは。
 EN: A serpent born of the fading blue, uses 100 faces.
[SERVER] Processed cd7040b8… → 🎼線を担うスゴーデエージェント黄昏れ、百の顔を使いは。…
rtf_avg: 0.006: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.21it/s]
rtf_avg: 0.058: 100%|████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 25.04it/s]
rtf_avg: 0.002, time_speech:  49.980, time_escape: 0.124: 100%|██████████████████████████| 1/1 [00:00<00:00,  7.88it/s]
 FULL JA: 。。。
 JA SENTS (1):['\n1: 。。。']
rtf_avg: 0.006: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 36.47it/s]
rtf_avg: 0.033: 100%|████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 14.89it/s]
rtf_avg: 0.032, time_speech:  4.320, time_escape: 0.138: 100%|███████████████████████████| 1/1 [00:00<00:00,  7.15it/s]
 JA: 🎼てる彼の任務は？家族 を 作る こと。
 EN: His duty is to make me...
[SERVER] Processed a6beb7af… → 🎼てる彼の任務は？家族 を 作る こと。…
rtf_avg: 0.007: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.44it/s]
rtf_avg: 0.082: 100%|████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 17.94it/s]
rtf_avg: 0.004, time_speech:  59.000, time_escape: 0.227: 100%|██████████████████████████| 1/1 [00:00<00:00,  4.33it/s]
 FULL JA: 。。そ。。
 JA SENTS (2):['\n1: 。。', '\n2: そ。。']
rtf_avg: 0.008: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 26.86it/s]
rtf_avg: 0.039: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.34it/s]
rtf_avg: 0.037, time_speech:  4.320, time_escape: 0.162: 100%|███████████████████████████| 1/1 [00:00<00:00,  6.18it/s]
 JA: 🎼父ロイドフージ精神科医翔た。
 EN: Dr. Lloyd Fudge, psychiatrist flew in.
[SERVER] Processed 1e793869… → 🎼父ロイドフージ精神科医翔た。…
rtf_avg: 0.058: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.14it/s]
rtf_avg: 0.076: 100%|████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 18.99it/s]
rtf_avg: 0.004, time_speech:  68.020, time_escape: 0.267: 100%|██████████████████████████| 1/1 [00:00<00:00,  3.67it/s]
 FULL JA: 。う。。。。
 JA SENTS (2):['\n1: 。', '\n2: う。。。。']
rtf_avg: 0.007: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 18.10it/s]
rtf_avg: 0.037: 100%|████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 11.68it/s]
rtf_avg: 0.033, time_speech:  7.776, time_escape: 0.260: 100%|███████████████████████████| 1/1 [00:00<00:00,  3.82it/s]
 JA: 🎼スパイコードネームた昏れ。母、ヨルフォージャー市役所職員。しょう？
 EN: 🎼 Spy code name: Yoforge, mom, city council official. Shoyo?
[SERVER] Processed 8b0ef01d… → 🎼スパイコードネームた昏れ。母、ヨルフォージャー市役所職員。しょう？…
rtf_avg: 0.037: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.08it/s]
rtf_avg: 0.062: 100%|████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 23.23it/s]
rtf_avg: 0.003, time_speech:  72.932, time_escape: 0.218: 100%|██████████████████████████| 1/1 [00:00<00:00,  4.48it/s]
 FULL JA: あ。。。。。
 JA SENTS (1):['\n1: あ。。。。。']
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
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
