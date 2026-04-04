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
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_funasr.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\audio_context_buffer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\diff_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_utils.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speechbrain\speech_segments_extractor.py",
    r"",
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
Browse fix

python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speechbrain\speech_segments_extractor.py
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\torch_audio_backend.py:60: UserWarning: torchaudio._backend.list_audio_backends has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and video are being consolidated into TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release.
  available_backends = torchaudio.list_audio_backends()
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\torch_audio_backend.py:60: UserWarning:
torchaudio._backend.list_audio_backends has been deprecated. This deprecation is part of a large refactoring effort to
transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and
video are being consolidated into TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more
information. It will be removed from the 2.9 release.
  available_backends = torchaudio.list_audio_backends()
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.pretrained' was
deprecated, redirecting to 'speechbrain.inference'. Please update your script. This is a change from SpeechBrain 1.0.
See: https://github.com/speechbrain/speechbrain/releases/tag/v1.0.0
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.k2_integration'
was deprecated, redirecting to 'speechbrain.integrations.k2_fsa'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.wordemb' was
deprecated, redirecting to 'speechbrain.integrations.huggingface.wordemb'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module
'speechbrain.lobes.models.huggingface_transformers' was deprecated, redirecting to
'speechbrain.integrations.huggingface'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module
'speechbrain.lobes.models.spacy' was deprecated, redirecting to 'speechbrain.integrations.nlp'. Please update your
script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module
'speechbrain.lobes.models.flair' was deprecated, redirecting to 'speechbrain.integrations.nlp'. Please update your
script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module
'speechbrain.nnet.loss.transducer_loss' was deprecated, redirecting to 'speechbrain.integrations.numba.transducer_loss'.
Please update your script. This module depends on the optional 'numba' package. If you encounter an ImportError here,
please install numba, for example with: pip install numba
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.wordemb' was
deprecated, redirecting to 'speechbrain.integrations.huggingface.wordemb'. Please update your script.
  if ismodule(module) and hasattr(module, '__file__'):
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\parameter_transfer.py:234: UserWarning:
Requested Pretrainer collection using symlinks on Windows. This might not work; see `LocalStrategy` documentation.
Consider unsetting `collect_in` in Pretrainer to avoid symlinking altogether.
  warnings.warn(
✅ SpeechBrain VAD model ready
Processing: recording_spyx_1_speaker.wav
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchaudio\_backend\utils.py:337: UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.save_with_torchcodec` under the hood. Some parameters like format, encoding, bits_per_sample, buffer_size, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's encoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.encoders.AudioEncoder
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\dataio\audio_io.py", line 208, in info
    file_info = sf.info(path)
                ^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\soundfile.py", line 488, in info
    return _SoundFileInfo(file, verbose)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\soundfile.py", line 433, in __init__
    with SoundFile(file) as f:
         ^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\soundfile.py", line 690, in __init__
    self._file = self._open(file, mode_int, closefd)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\soundfile.py", line 1265, in _open
    raise LibsndfileError(err, prefix="Error opening {0!r}: ".format(self.name))
soundfile.LibsndfileError: Error opening 'C:\\Users\\druiv\\C:\\Users\\druiv\\AppData\\Local\\Temp\\tmpe0kqyrei.wav': System error.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speechbrain\speech_segments_extractor.py", line 217, in extract_speech_timestamps
    boundaries_sec = vad.get_speech_segments(
                     ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\inference\VAD.py", line 923, in get_speech_segments
    prob_chunks = self.get_speech_prob_file(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\inference\VAD.py", line 98, in get_speech_prob_file
    sample_rate, audio_len = self._get_audio_info(audio_file)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\inference\VAD.py", line 651, in _get_audio_info
    metadata = audio_io.info(str(audio_file))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\dataio\audio_io.py", line 217, in info
    raise RuntimeError(f"Failed to get info for {path}: {e}") from e
RuntimeError: Failed to get info for C:\Users\druiv\C:\Users\druiv\AppData\Local\Temp\tmpe0kqyrei.wav: Error opening 'C:\\Users\\druiv\\C:\\Users\\druiv\\AppData\\Local\\Temp\\tmpe0kqyrei.wav': System error.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speechbrain\speech_segments_extractor.py", line 549, in <module>
    segments, speech_probs = extract_speech_timestamps(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speechbrain\speech_segments_extractor.py", line 461, in extract_speech_timestamps
    os.remove(temp_path)
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'C:\\Users\\druiv\\AppData\\Local\\Temp\\tmpe0kqyrei.wav'
(jet_venv) PS C:\Users\druiv>

""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
Provide step by step analysis first.
Show unified diff for updated files unless specified otherwise, while show python code block for new files.
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
