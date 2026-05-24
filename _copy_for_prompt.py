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
    # r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--separation-ami-1.0\snapshots\4d38e95cfd067c894b8b60b00761831fb01e4a8c\README.md",
    # r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--speech-separation-ami-1.0\snapshots\9486b106945ae0cc0784041a08bfcdba5edadfb9\README.md",
    # r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--speech-separation-ami-1.0\snapshots\9486b106945ae0cc0784041a08bfcdba5edadfb9\config.yaml",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\run_separation_pipeline.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\transcribe_emotion_litagin_speech_improved.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\transcribe_emotion_litagin_speech.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\transcribe_emotion_litagin_window.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\transcribe_emotion_litagin.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_mac.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_funasr.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_funasr.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm_cached.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm_prefixed.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_matcher_ja.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\sentence_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\audio_context_buffer.py"
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\compare_speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_transcriber2.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_ja_en_pipeline.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_segments_buffer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_firered_hybrid.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\vad_extractors.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\live_vad_segments.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\config.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server3.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server3_jp_en.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\speech_segment_tracker.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\core\stream_vad_postprocessor.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\fireredvad_utils.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\word_utils_ja.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\test_word_utils_ja.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\speaker_verification.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\example_native_pyannote_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\example_onnx_wespeaker_embedding.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\example_native_pyannote_speaker_embedding.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\pyannote\examples\example_speechbrain_embedding.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\inference.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\clustering.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\cluster_speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\compare_speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\segment_speaker_labeler.py",
    # r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\segment_speaker_labeler_example.py",
    r"",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\pyannote\core\segment.py",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\pyannote\pipeline\pipeline.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\pipeline.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\utils\signal.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\__init__.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\multilabel.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\speech_separation.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\voice_activity_detection.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\_main_speaker_labeler.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\io.py",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\_metadata.py",
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
Its throwing error.
Can we use Inference -> crop or its not possible?
Also if its possible for the fix to be applied in MultiLabelSegmentation, then show full updated multilabel.



╭───────────────────────────────────────────────────────────────────╮
│ SpeakerLabeler Demo                                               │
│ Model: pyannote/segmentation-3.0                                  │
│ Audio: C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav │
╰───────────────────────────────────────────────────────────────────╯

Step 1: Initializing SpeakerLabeler
[19:42:49] Model classes: ['speaker#1', 'speaker#2', 'speaker#3']                                 speaker_labeler.py:110
           Overlap class indices: []                                                              speaker_labeler.py:113
           ✓ Model and pipelines loaded                                                           speaker_labeler.py:127
           ✓ SpeakerLabeler initialized successfully                                               speaker_labeler.py:98

Step 2: Processing audio file
File: C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav
Size: 3625.0 KB
╭───────────────────────────────────╮
│ Starting Speaker Labeling Process │
╰───────────────────────────────────╯
           Loading audio file: C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav         speaker_labeler.py:212
Audio Duration: 58.00s | Sample Rate: 16000Hz
           Processing chunk 1: 0.0s - 10.0s                                                       speaker_labeler.py:528
             → Running VAD...                                                                     speaker_labeler.py:288
⠋ Processing audio chunks...[NeMo W 2026-05-24 19:42:49 nemo_logging:364] C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\utils\reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
    It can be re-enabled by calling
       >>> import torch
       >>> torch.backends.cuda.matmul.allow_tf32 = True
       >>> torch.backends.cudnn.allow_tf32 = True
    See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.

      warnings.warn(

             → Running MultiLabel Segmentation (overlap detection)...                             speaker_labeler.py:293
⠦ Processing audio chunks...

Error: too many values to unpack (expected 2)
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\_main_speaker_labe │
│ ler.py:76 in main                                                                                │
│                                                                                                  │
│    73 │   │   console.print(f"[dim]File: {audio_path}[/dim]")                                    │
│    74 │   │   console.print(f"[dim]Size: {audio_path.stat().st_size / 1024:.1f} KB[/dim]")       │
│    75 │   │                                                                                      │
│ ❱  76 │   │   results = labeler.label_speakers(str(audio_path))                                  │
│    77 │   │                                                                                      │
│    78 │   │   # Display results                                                                  │
│    79 │   │   console.print(f"\n[bold]Step 3: Displaying Results[/bold]")                        │
│                                                                                                  │
│ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\speaker_labeler.py │
│ :533 in label_speakers                                                                           │
│                                                                                                  │
│   530 │   │   │   │   │   f"{chunk_start_time:.1f}s - {end_sample/sample_rate:.1f}s[/blue]"      │
│   531 │   │   │   │   )                                                                          │
│   532 │   │   │   │                                                                              │
│ ❱ 533 │   │   │   │   chunk_results = self.process_chunk(                                        │
│   534 │   │   │   │   │   chunk_waveform,                                                        │
│   535 │   │   │   │   │   sample_rate,                                                           │
│   536 │   │   │   │   │   chunk_start_time                                                       │
│                                                                                                  │
│ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\speaker_labeler.py │
│ :340 in process_chunk                                                                            │
│                                                                                                  │
│   337 │   │                                                                                      │
│   338 │   │   try:                                                                               │
│   339 │   │   │   # Run pipelines                                                                │
│ ❱ 340 │   │   │   results = self._process_single_chunk(temp_chunk_file, chunk_start_time)        │
│   341 │   │   │                                                                                  │
│   342 │   │   │   # Get raw segmentation for speaker labeling                                    │
│   343 │   │   │   console.log("[yellow]  → Running raw speaker segmentation...[/yellow]")        │
│                                                                                                  │
│ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\speaker_labeler.py │
│ :294 in _process_single_chunk                                                                    │
│                                                                                                  │
│   291 │   │                                                                                      │
│   292 │   │   # Run MultiLabel Segmentation                                                      │
│   293 │   │   console.log("[yellow]  → Running MultiLabel Segmentation (overlap detection)...[   │
│ ❱ 294 │   │   multilabel_result = self.multilabel_pipeline(chunk_file)                           │
│   295 │   │   results["multilabel"] = multilabel_result                                          │
│   296 │   │                                                                                      │
│   297 │   │   # Extract overlap regions                                                          │
│                                                                                                  │
│ C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\pipeline.py │
│ :478 in __call__                                                                                 │
│                                                                                                  │
│   475 │   │   # requested number of speakers in case of diarization                              │
│   476 │   │   track_pipeline_apply(self, file, **kwargs)                                         │
│   477 │   │                                                                                      │
│ ❱ 478 │   │   return self.apply(file, **kwargs)                                                  │
│   479 │                                                                                          │
│   480 │   def to(self, device: torch.device) -> Pipeline:                                        │
│   481 │   │   \"\"\"Send pipeline to `device`\"\"\"                                                    │
│                                                                                                  │
│ C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\multil │
│ abel.py:207 in apply                                                                             │
│                                                                                                  │
│   204 │   │   │   │   segmentations.data[:, i : i + 1], segmentations.sliding_window             │
│   205 │   │   │   )                                                                              │
│   206 │   │   │   # obtain hard segments                                                         │
│ ❱ 207 │   │   │   label_annotation: Annotation = self._binarize[label](label_segmentation)       │
│   208 │   │   │                                                                                  │
│   209 │   │   │   # add them to the pool of labels                                               │
│   210 │   │   │   detection.update(                                                              │
│                                                                                                  │
│ C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\utils\signal.py: │
│ 268 in __call__                                                                                  │
│                                                                                                  │
│   265 │   │   │   Binarized scores.                                                              │
│   266 │   │   \"\"\"                                                                                │
│   267 │   │                                                                                      │
│ ❱ 268 │   │   num_frames, num_classes = scores.data.shape                                        │
│   269 │   │   frames = scores.sliding_window                                                     │
│   270 │   │   timestamps = [frames[i].middle for i in range(num_frames)]                         │
│   271                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ValueError: too many values to unpack (expected 2)
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
Provide step-by-step analysis and explain the flow first.
Use visuals, diagrams, or tables when helpful.

Show full code for new files, then show full function code for updated functions.
Keep explanations simple and clear.

Write flexible, reusable, maintainable, and robust code.
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
