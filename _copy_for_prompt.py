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
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\processing",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_handlers.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_utils.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\start_server.ps1",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\requirements.txt",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\README.md",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\core",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\utils",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\bin/stream_vad.py",
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
Analyze the root cause of this issue

OSError: exception: access violation reading 0x0000000000000000
[03/10/26 04:59:22] INFO     [2258645f] Processing chunk 4 for utt 7e42ddb6-54b5-4509-9b47-c84df937ccf9 (partial)
[2258645f] Processing chunk 4 for utt 7e42ddb6-54b5-4509-9b47-c84df937ccf9 (partial)
rtf_avg: 0.003: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.72it/s]
rtf_avg: 0.023: 100%|████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.25it/s]
rtf_avg: 0.021, time_speech:  29.984, time_escape: 0.619: 100%|██████████████████████████| 1/1 [00:00<00:00,  1.61it/s]
[03/10/26 04:59:23] ERROR    [2258645f] Message handler error: exception: access violation reading 0x0000000000000000
                             ╭─────────────────────────── Traceback (most recent call last) ───────────────────────────╮
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server │
                             │ _subtitles_handlers.py:72 in handler                                                    │
                             │                                                                                         │
                             │    69 │   │   │   │   msg_type = data.get("type")                                       │
                             │    70 │   │   │   │                                                                     │
                             │    71 │   │   │   │   if msg_type in ("speech_chunk", "complete_utterance"):            │
                             │ ❱  72 │   │   │   │   │   await handle_speech_message(websocket, state, data)           │
                             │    73 │   │   │   │   elif msg_type == "speaker_diarization":                           │
                             │    74 │   │   │   │   │   await handle_speaker_diarization(websocket, state, data)      │
                             │    75 │   │   │   │   elif msg_type == "emotion_classification":                        │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server │
                             │ _subtitles_handlers.py:128 in handle_speech_message                                     │
                             │                                                                                         │
                             │   125 │   │   f"{'(final)' if is_final else '(partial)'}"                               │
                             │   126 │   )                                                                             │
                             │   127 │                                                                                 │
                             │ ❱ 128 │   await process_utterance(                                                      │
                             │   129 │   │   websocket,                                                                │
                             │   130 │   │   state,                                                                    │
                             │   131 │   │   sr,                                                                       │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server │
                             │ _subtitles_handlers.py:265 in process_utterance                                         │
                             │                                                                                         │
                             │   262 │   loop = asyncio.get_running_loop()                                             │
                             │   263 │                                                                                 │
                             │   264 │   # ── Fast path: transcription + translation ───────────────────────────────   │
                             │ ❱ 265 │   fast_result = await loop.run_in_executor(                                     │
                             │   266 │   │   executor_fast,                                                            │
                             │   267 │   │   process_fast_llm,                                                         │
                             │   268 │   │   audio_bytes,                                                              │
                             │                                                                                         │
                             │ C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\concurrent\futures\thread.py:59 in │
                             │ run                                                                                     │
                             │                                                                                         │
                             │    56 │   │   │   return                                                                │
                             │    57 │   │                                                                             │
                             │    58 │   │   try:                                                                      │
                             │ ❱  59 │   │   │   result = self.fn(*self.args, **self.kwargs)                           │
                             │    60 │   │   except BaseException as exc:                                              │
                             │    61 │   │   │   self.future.set_exception(exc)                                        │
                             │    62 │   │   │   # Break a reference cycle with the exception 'exc'                    │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\processin │
                             │ g\fast_processor.py:60 in process_fast_llm                                              │
                             │                                                                                         │
                             │    57 │   ja_text = trans_result.get("text_ja", "").strip()                             │
                             │    58 │                                                                                 │
                             │    59 │   # 2. Translation (llama.cpp)                                                  │
                             │ ❱  60 │   translation_result = translate_japanese_to_english(                           │
                             │    61 │   │   ja_text=ja_text,                                                          │
                             │    62 │   │   enable_scoring=False,  # set True when you want logprobs (slower)         │
                             │    63 │   │   history=None,          # can pass conversation history later              │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate │
                             │ _jp_en_llm.py:240 in translate_japanese_to_english                                      │
                             │                                                                                         │
                             │   237 │   │   completion_params["logprobs"] = True                                      │
                             │   238 │   │   completion_params["top_logprobs"] = 1                                     │
                             │   239 │                                                                                 │
                             │ ❱ 240 │   response = llm.create_chat_completion(                                        │
                             │   241 │   │   messages=messages,                                                        │
                             │   242 │   │   seed=3407,  # for reproducibility                                         │
                             │   243 │   │   logits_processor=LogitsProcessorList([no_assistant_first]),               │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py:2003   │
                             │ in create_chat_completion                                                               │
                             │                                                                                         │
                             │   2000 │   │   │   or self._chat_handlers.get(self.chat_format)                         │
                             │   2001 │   │   │   or llama_chat_format.get_chat_completion_handler(self.chat_format)   │
                             │   2002 │   │   )                                                                        │
                             │ ❱ 2003 │   │   return handler(                                                          │
                             │   2004 │   │   │   llama=self,                                                          │
                             │   2005 │   │   │   messages=messages,                                                   │
                             │   2006 │   │   │   functions=functions,                                                 │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama_chat_form │
                             │ at.py:669 in chat_completion_handler                                                    │
                             │                                                                                         │
                             │    666 │   │   │   │   │   llama_grammar.JSON_GBNF, verbose=llama.verbose               │
                             │    667 │   │   │   │   )                                                                │
                             │    668 │   │                                                                            │
                             │ ❱  669 │   │   completion_or_chunks = llama.create_completion(                          │
                             │    670 │   │   │   prompt=prompt,                                                       │
                             │    671 │   │   │   temperature=temperature,                                             │
                             │    672 │   │   │   top_p=top_p,                                                         │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py:1837   │
                             │ in create_completion                                                                    │
                             │                                                                                         │
                             │   1834 │   │   if stream:                                                               │
                             │   1835 │   │   │   chunks: Iterator[CreateCompletionStreamResponse] = completion_or_chu │
                             │   1836 │   │   │   return chunks                                                        │
                             │ ❱ 1837 │   │   completion: Completion = next(completion_or_chunks)  # type: ignore      │
                             │   1838 │   │   return completion                                                        │
                             │   1839 │                                                                                │
                             │   1840 │   def __call__(                                                                │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py:1322   │
                             │ in _create_completion                                                                   │
                             │                                                                                         │
                             │   1319 │   │                                                                            │
                             │   1320 │   │   finish_reason = "length"                                                 │
                             │   1321 │   │   multibyte_fix = 0                                                        │
                             │ ❱ 1322 │   │   for token in self.generate(                                              │
                             │   1323 │   │   │   prompt_tokens,                                                       │
                             │   1324 │   │   │   top_k=top_k,                                                         │
                             │   1325 │   │   │   top_p=top_p,                                                         │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py:914 in │
                             │ generate                                                                                │
                             │                                                                                         │
                             │    911 │   │                                                                            │
                             │    912 │   │   # Eval and sample                                                        │
                             │    913 │   │   while True:                                                              │
                             │ ❱  914 │   │   │   self.eval(tokens)                                                    │
                             │    915 │   │   │   while sample_idx < self.n_tokens:                                    │
                             │    916 │   │   │   │   token = self.sample(                                             │
                             │    917 │   │   │   │   │   top_k=top_k,                                                 │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py:648 in │
                             │ eval                                                                                    │
                             │                                                                                         │
                             │    645 │   │   │   self._batch.set_batch(                                               │
                             │    646 │   │   │   │   batch=batch, n_past=n_past, logits_all=self._logits_all          │
                             │    647 │   │   │   )                                                                    │
                             │ ❱  648 │   │   │   self._ctx.decode(self._batch)                                        │
                             │    649 │   │   │   # Save tokens                                                        │
                             │    650 │   │   │   self.input_ids[n_past : n_past + n_tokens] = batch                   │
                             │    651 │   │   │   # Save logits                                                        │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\_internals.py:3 │
                             │ 22 in decode                                                                            │
                             │                                                                                         │
                             │   319 │   # TODO: llama_save_session_file                                               │
                             │   320 │                                                                                 │
                             │   321 │   def decode(self, batch: LlamaBatch):                                          │
                             │ ❱ 322 │   │   return_code = llama_cpp.llama_decode(                                     │
                             │   323 │   │   │   self.ctx,                                                             │
                             │   324 │   │   │   batch.batch,                                                          │
                             │   325 │   │   )                                                                         │
                             ╰─────────────────────────────────────────────────────────────────────────────────────────╯
                             OSError: exception: access violation reading 0x0000000000000000
[2258645f] Message handler error: exception: access violation reading 0x0000000000000000
Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_handlers.py", line 72, in handler
    await handle_speech_message(websocket, state, data)
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_handlers.py", line 128, in handle_speech_message
    await process_utterance(
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\ws_server_subtitles_handlers.py", line 265, in process_utterance
    fast_result = await loop.run_in_executor(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\processing\fast_processor.py", line 60, in process_fast_llm
    translation_result = translate_japanese_to_english(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py", line 240, in translate_japanese_to_english
    response = llm.create_chat_completion(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py", line 2003, in create_chat_completion
    return handler(
           ^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama_chat_format.py", line 669, in chat_completion_handler
    completion_or_chunks = llama.create_completion(
                           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py", line 1837, in create_completion
    completion: Completion = next(completion_or_chunks)  # type: ignore
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py", line 1322, in _create_completion
    for token in self.generate(
                 ^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py", line 914, in generate
    self.eval(tokens)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\llama.py", line 648, in eval
    self._ctx.decode(self._batch)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\llama_cpp\_internals.py", line 322, in decode
    return_code = llama_cpp.llama_decode(
                  ^^^^^^^^^^^^^^^^^^^^^^^
OSError: exception: access violation reading 0x0000000000000000
[03/10/26 04:59:24] INFO     [2258645f] Processing chunk 5 for utt 7e42ddb6-54b5-4509-9b47-c84df937ccf9 (final)
[2258645f] Processing chunk 5 for utt 7e42ddb6-54b5-4509-9b47-c84df937ccf9 (final)
rtf_avg: 0.003: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.05it/s]
rtf_avg: 0.025: 100%|████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.73it/s]
rtf_avg: 0.022, time_speech:  36.000, time_escape: 0.808: 100%|██████████████████████████| 1/1 [00:00<00:00,  1.23it/s]
C:\Users\druiv\AppData\Local\Temp\pip-install-ptx8joy4\llama-cpp-python_ddecc1a71fa24729894a3f5b1d1a4bf6\vendor\llama.cpp\src\llama-context.cpp:1130: GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all) failed
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
