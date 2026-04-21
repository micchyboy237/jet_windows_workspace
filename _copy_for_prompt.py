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
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\audio_context_buffer.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_utils.py",
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
Fix
(jet_venv) PS C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace> python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
funasr version: 1.3.1.
Fetching 29 files: 100%|██████████████████████████████████████████████████████████████████| 29/29 [00:00<?, ?it/s]
WARNING:root:trust_remote_code: False
llama_context: n_ctx_per_seq (2048) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
Banned first-token IDs: [72803, 78191]
 'assistant' → tokens [78191]
 'Assistant' → tokens [72803]
Server listening on ws://0.0.0.0:8765
Client connected — total 1
VAD Reason: valley_detection
Context duration: 0.00s
Audio duration: 5.18s
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Processing 10cf41…
rtf_avg: 0.235: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.22s/it]
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023921_001:create_chat_completion called  stream=False
╭─────────────────────────── ⟶ LLM Request ───────────────────────────╮
│ Last user message: 世界各国が水面下で熾烈な情報戦を繰り広げる時代。 │
│                                                                     │
│ Params:                                                             │
│   temperature: 0.35                                                 │
│   top_p: 0.9                                                        │
│   top_k: 40                                                         │
│   min_p: 0.05                                                       │
│   typical_p: 0.95                                                   │
│   stream: False                                                     │
│   stop: ['\n\n', '<|eot_id|>', '<|end_of_text|>', '<|im_end|>']     │
│   seed: 3407                                                        │
│   max_tokens: 2000                                                  │
│   presence_penalty: 0.0                                             │
│   frequency_penalty: 0.0                                            │
│   repeat_penalty: 1.18                                              │
│   tfs_z: 1.0                                                        │
│   mirostat_mode: 0                                                  │
│   mirostat_tau: 5.0                                                 │
│   mirostat_eta: 0.1                                                 │
╰─────────────────────────────────────────────────────────────────────╯
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023921_001:request.json saved  messages=2
╭──────────────────────────────────── ⟵ LLM Response ────────────────────────────────────╮
│ These are dark times—worlds of nations engaging in fierce, secret information warfare. │
│                                                                                        │
│ Tokens — prompt: 489  completion: 17  total: 506                                       │
╰────────────────────────────────────────────────────────────────────────────────────────╯
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023921_001:Response saved  path=C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023921_001\response.md  uri=file:///C:/Users/druiv/Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_server2/llm_calls/20260422_023921_001/response.md
Response →
C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_
calls\20260422_023921_001\response.md
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023921_001:Non-stream done  elapsed=2.314s  tokens={'prompt_tokens': 489, 'completion_tokens': 17, 'total_tokens': 506}
New JA (24 chars):
世界各国が水面下で熾烈な情報戦を繰り広げる時代。
Full JA (1 sents):
世界各国が水面下で熾烈な情報戦を繰り広げる時代。
Full EN:
These are dark times—worlds of nations engaging in fierce, secret information warfare.

Audio Files:
  Long audio (A)   :      5.18s
  Short clip  (B)  :      5.18s

────────────────────────────────────────── Searching for partial matches ──────────────────────────────────────────

Match 1   Confidence: 1.0000
  Matched segment duration:   5.18 seconds
  • Covers 100.0% of long audio (A)
  • Covers 100.0% of short clip (B)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃    Signal    ┃ Start (s) ┃ End (s) ┃ Total (s) ┃ % of Total ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│   A (long)   │      0.00 │    5.18 │      5.18 │     100.0% │
├──────────────┼───────────┼─────────┼───────────┼────────────┤
│   B (clip)   │      0.00 │    5.18 │      5.18 │     100.0% │
└──────────────┴───────────┴─────────┴───────────┴────────────┘

1 partial match found (conf ≥ 0.75, matched ≥ 50% of short clip)
Processed successfully 10cf41…
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Processing b4f4d6…
VAD Reason: valley_detection
Context duration: 5.18s
Audio duration: 14.40s
rtf_avg: 0.028: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.84it/s]
✅ Accepted
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023933_002:create_chat_completion called  stream=False
╭───────────────────────────────────────────────── ⟶ LLM Request ─────────────────────────────────────────────────╮
│ Last user message:                                                                                              │
│ 睨み合う2つの国、東のオスタニア、西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべく、ウスタリスはオ │
│ ペレーションストリクスを発動。                                                                                  │
│                                                                                                                 │
│ Params:                                                                                                         │
│   temperature: 0.35                                                                                             │
│   top_p: 0.9                                                                                                    │
│   top_k: 40                                                                                                     │
│   min_p: 0.05                                                                                                   │
│   typical_p: 0.95                                                                                               │
│   stream: False                                                                                                 │
│   stop: ['\n\n', '<|eot_id|>', '<|end_of_text|>', '<|im_end|>']                                                 │
│   seed: 3407                                                                                                    │
│   max_tokens: 2000                                                                                              │
│   presence_penalty: 0.0                                                                                         │
│   frequency_penalty: 0.0                                                                                        │
│   repeat_penalty: 1.18                                                                                          │
│   tfs_z: 1.0                                                                                                    │
│   mirostat_mode: 0                                                                                              │
│   mirostat_tau: 5.0                                                                                             │
│   mirostat_eta: 0.1                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023933_002:request.json saved  messages=4
╭──────────────────────────────────────────────── ⟵ LLM Response ─────────────────────────────────────────────────╮
│ With Westaリス watching, Ostania—eying the movements of key officials planning a war between their nation and   │
│ the Eastern nation of Ostania—is activating Operation Striks.                                                   │
│                                                                                                                 │
│ Tokens — prompt: 574  completion: 35  total: 609                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023933_002:Response saved  path=C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023933_002\response.md  uri=file:///C:/Users/druiv/Desktop/Jet_Files/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_server2/llm_calls/20260422_023933_002/response.md
Response →
C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_
calls\20260422_023933_002\response.md
INFO:C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server2\llm_calls\20260422_023933_002:Non-stream done  elapsed=0.994s  tokens={'prompt_tokens': 574, 'completion_tokens': 35, 'total_tokens': 609}
History (2):
[{'role': 'user', 'content': '世界各国が水面下で熾烈な情報戦を繰り広げる時代。'}, {'role': 'assistant', 'content':
'These are dark times—worlds of nations engaging in fierce, secret information warfare.'}]
Client disconnected — total 0
ERROR:websockets.server:connection handler failed
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\websockets\asyncio\server.py", line 376, in conn_handler
    await self.handler(connection)
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py", line 543, in process_audio
    response = await future
               ^^^^^^^^^^^^
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py", line 305, in blocking_process_audio
    console.print(f"[success]Last Sentence (utt_id={last_utt_id[-6:]} | sent_idx={last_sent_idx}):[/success]")
                                                    ~~~~~~~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
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
