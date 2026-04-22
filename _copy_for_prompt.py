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
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_firered2.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_firered.py",
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
How to correctly use is_same_speaker? Is it a problem with loader.load_audio? Can we apply the fix in load_audio?

(jet_venv) PS C:\Users\druiv> python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\run_verify_speakers.py `
>> C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\vad_firered2\segments\segment_001\sound.wav `
>> C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\vad_firered2\segments\segment_002\sound.wav
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\io.py:47: UserWarning:
torchcodec is not installed correctly so built-in audio decoding will fail. Solutions are:
* use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary;
* fix torchcodec installation. Error message was:

Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.8.0+cu128) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:

[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core8.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 7:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core7.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 6:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core6.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 5:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core5.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 4:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core4.dll' (or one of its dependencies). Try using the full path with constructor syntax.
[end of libtorchcodec loading traceback].
  warnings.warn(
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\torch_audio_backend.py:64: UserWarning: torchaudio._backend.list_audio_backends has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and video are being consolidated into TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release.
  available_backends = torchaudio.list_audio_backends()
[NeMo W 2026-04-22 09:11:34 megatron_init:62] Megatron num_microbatches_calculator not found, using Apex version.
W0422 09:11:34.492000 32112 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
[NeMo W 2026-04-22 09:11:37 nemo_logging:364] C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\lightning\pytorch\utilities\migration\migration.py:208: You have multiple `ModelCheckpoint` callback states in this checkpoint, but we found state keys that would end up colliding with each other after an upgrade, which means we can't differentiate which of your checkpoint callbacks needs which states. At least one of your `ModelCheckpoint` callbacks will not be able to reload the state.

Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.4.0. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint C:\Users\druiv\.cache\huggingface\hub\models--pyannote--embedding\snapshots\4db4899737a38b2d618bbd74350915aa10293cb2\pytorch_model.bin`
Same: False, Distance: 0.8576
(jet_venv) PS C:\Users\druiv>
(jet_venv) PS C:\Users\druiv> python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\run_is_same_speaker.py `
>> C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\vad_firered2\segments\segment_001\sound.wav `
>> C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\vad_firered2\segments\segment_002\sound.wav
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\io.py:47: UserWarning:
torchcodec is not installed correctly so built-in audio decoding will fail. Solutions are:
* use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary;
* fix torchcodec installation. Error message was:

Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.8.0+cu128) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:

[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core8.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 7:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core7.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 6:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core6.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 5:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core5.dll' (or one of its dependencies). Try using the full path with constructor syntax.

FFmpeg version 4:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1478, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core4.dll' (or one of its dependencies). Try using the full path with constructor syntax.
[end of libtorchcodec loading traceback].
  warnings.warn(
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\speechbrain\speechbrain\utils\torch_audio_backend.py:64: UserWarning: torchaudio._backend.list_audio_backends has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and video are being consolidated into TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release.
  available_backends = torchaudio.list_audio_backends()
[NeMo W 2026-04-22 09:12:00 megatron_init:62] Megatron num_microbatches_calculator not found, using Apex version.
W0422 09:12:00.947000 18844 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\run_is_same_speaker.py", line 25, in <module>
    same_speaker = is_same_speaker(emb1, emb2, args.threshold)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\speaker_verification_utils.py", line 337, in is_same_speaker
    return compute_cosine_distance(emb1, emb2) < threshold
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\speaker_verification_utils.py", line 269, in compute_cosine_distance
    return float(cdist(emb1, emb2, metric="cosine")[0, 0])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\scipy\spatial\distance.py", line 3104, in cdist
    XA = np.asarray(XA)
         ^^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
(jet_venv) PS C:\Users\druiv>
(jet_venv) PS C:\Users\druiv>
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
