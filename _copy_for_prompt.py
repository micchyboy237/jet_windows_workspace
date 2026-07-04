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
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_config.py",
    r"",
   #  r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger.py",
    r"",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\generated\test_audio_tagger\chunk_summary.json",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\generated\test_audio_tagger\high_speech_segments.json",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\generated\test_audio_tagger\segments_result.json",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\generated\test_audio_tagger\speech_segments.json",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\test_audio_tagger.py",
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
Check


(jet_venv) PS C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace> python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\test_audio_tagger.py "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\segment_audio\segment_081.wav"
╭─── AudioTagger Configuration ────╮
│ AudioTagger Initialized          │
│ Model: model.onnx                │
│ Labels: class_labels_indices.csv │
│ Speech Threshold: 0.1            │
│ Speech Top N: 3                  │
│ Chunk Duration: 0.5s             │
│ Chunk Overlap: 0.25s             │
│ Min Chunk Duration: 0.5s         │
╰──────────────────────────────────╯

Analyzing audio: segment_081.wav

📊 Audio loaded: 2.86s, 16000Hz, 45760 samples
🔧 Chunk config: 0.5s chunks, 0.25s overlap, hop=4000 samples
📏 Calculated 10 chunk positions
🔍 Processing chunk 1/10: 0.00s - 0.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.243) | max_prob=0.2431 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.2431
🔍 Processing chunk 2/10: 0.25s - 0.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 3/10: 0.50s - 1.00s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.981) | max_prob=0.9813 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.9813
🔍 Processing chunk 4/10: 0.75s - 1.25s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 5/10: 1.00s - 1.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 6/10: 1.25s - 1.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.377) | max_prob=0.3772 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.3772
🔍 Processing chunk 7/10: 1.50s - 2.00s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 8/10: 1.75s - 2.25s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 9/10: 2.00s - 2.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 10/10: 2.25s - 2.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
📊 Avg speech probability: 0.5339 (from 3 speech chunks)
⏱ Total processing: 3.04s, RTF: 1.062x
🔧 Auto min_silence_duration_sec=0.500s (2× hop=0.250s, chunk=0.5s, overlap=0.25s)
╭─────────────────────────────────────── Segment-Based Audio Tagging ───────────────────────────────────────╮
│ tag_audio_segments                                                                                        │
│ speech_threshold=0.10 | min_silence=0.5s | min_speech=0.5s | resolution=10.0ms | include_non_speech=False │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
📊 Audio loaded: 2.86s, 16000Hz, 45760 samples
🔧 Chunk config: 0.5s chunks, 0.25s overlap, hop=4000 samples
📏 Calculated 10 chunk positions
🔍 Processing chunk 1/10: 0.00s - 0.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.243) | max_prob=0.2431 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.2431
🔍 Processing chunk 2/10: 0.25s - 0.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 3/10: 0.50s - 1.00s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.981) | max_prob=0.9813 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.9813
🔍 Processing chunk 4/10: 0.75s - 1.25s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 5/10: 1.00s - 1.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 6/10: 1.25s - 1.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.377) | max_prob=0.3772 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.3772
🔍 Processing chunk 7/10: 1.50s - 2.00s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 8/10: 1.75s - 2.25s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 9/10: 2.00s - 2.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 10/10: 2.25s - 2.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
📊 Avg speech probability: 0.5339 (from 3 speech chunks)
⏱ Total processing: 0.33s, RTF: 0.116x
🕑 Built prob timeline: 3/10 speech chunks → 150/275 cells with probability > 0 (54.5%) @ 10.0ms resolution,
total_end=2.750s
🎚 Using speech threshold: 0.1
📊 Timeline: 150/275 cells above threshold (54.5%)
🎤 Speech start at cell 0 (time=0.005s)
🔇 Speech end at cell 224 (time=2.245s) | segment: 0.005s-1.745s (silence=0.500s)
✅ 1 speech segment(s) detected
🔍 _build_segment_result: speech segment 1 0.005s–1.745s (dur=1.740s)
   Timeline cells: 175 | Overlapping chunks: 7 | Duration: 1.740s
   🟡 Medium Confidence: avg_prob=0.458≥0.4, density=85.7%≥50% (normal)
⏱ Segment detection complete: 0.34s, RTF: 0.118x
🔧 Auto min_silence_duration_sec=0.500s (2× hop=0.250s, chunk=0.5s, overlap=0.25s)
╭────────────────────────────── High Speech Segments Extraction ──────────────────────────────╮
│ extract_high_confidence_speech_segments                                                     │
│ min_duration=1.5s | min_silence=0.5s | filter: duration > 1.5s AND segment_type == 'speech' │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
🔍 Running tag_audio_segments...
╭─────────────────────────────────────── Segment-Based Audio Tagging ───────────────────────────────────────╮
│ tag_audio_segments                                                                                        │
│ speech_threshold=0.10 | min_silence=0.5s | min_speech=0.5s | resolution=10.0ms | include_non_speech=False │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
📊 Audio loaded: 2.86s, 16000Hz, 45760 samples
🔧 Chunk config: 0.5s chunks, 0.25s overlap, hop=4000 samples
📏 Calculated 10 chunk positions
🔍 Processing chunk 1/10: 0.00s - 0.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.243) | max_prob=0.2431 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.2431
🔍 Processing chunk 2/10: 0.25s - 0.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 3/10: 0.50s - 1.00s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.981) | max_prob=0.9813 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.9813
🔍 Processing chunk 4/10: 0.75s - 1.25s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 5/10: 1.00s - 1.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 6/10: 1.25s - 1.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   Speech classes found: Speech(0.377) | max_prob=0.3772 | threshold=0.1 | detected=True
   🎤 Speech detected! speech_prob=0.3772
🔍 Processing chunk 7/10: 1.50s - 2.00s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 8/10: 1.75s - 2.25s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 9/10: 2.00s - 2.50s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 10/10: 2.25s - 2.75s
   ✅ Tagged successfully: 5 predictions
🔍 _chunk_has_speech: checking top 3 predictions against threshold 0.1
   No speech classes in top 3 | detected=False
   🔇 No speech detected (speech_prob=0.0000)
📊 Avg speech probability: 0.5339 (from 3 speech chunks)
⏱ Total processing: 0.34s, RTF: 0.120x
🕑 Built prob timeline: 3/10 speech chunks → 150/275 cells with probability > 0 (54.5%) @ 10.0ms resolution,
total_end=2.750s
🎚 Using speech threshold: 0.1
📊 Timeline: 150/275 cells above threshold (54.5%)
🎤 Speech start at cell 0 (time=0.005s)
🔇 Speech end at cell 224 (time=2.245s) | segment: 0.005s-1.745s (silence=0.500s)
✅ 1 speech segment(s) detected
🔍 _build_segment_result: speech segment 1 0.005s–1.745s (dur=1.740s)
   Timeline cells: 175 | Overlapping chunks: 7 | Duration: 1.740s
   🟡 Medium Confidence: avg_prob=0.458≥0.4, density=85.7%≥50% (normal)
⏱ Segment detection complete: 0.35s, RTF: 0.122x
📊 Found 1 total speech segments
   ✅ Segment 0: 0.01s-1.75s (dur=1.74s, type=speech)
✅ Filtered 1 high speech segments (duration > 1.5s, type=speech)
📂 Loaded audio for extraction: 2.86s @ 16000Hz
   ✂ Extracted 1.74s audio (27840 samples)
⏱ Extraction complete: 0.35s | Total extracted: 1.74s
Chunks summary saved to: chunk_summary.json
Segment summary saved to: segments_result.json
Filtered speech segments saved to: high_speech_segments.json

Extracting individual segments...
[04:32:45] ✓ Saved audio:                                                                       test_audio_tagger.py:117
           C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_s
           ubtitles_server2_with_en\services\generated\test_audio_tagger\segments\segment_000\s
           ound.wav
           ✓ Saved metadata:                                                                    test_audio_tagger.py:123
           C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_s
           ubtitles_server2_with_en\services\generated\test_audio_tagger\segments\segment_000\s
           egment.json

✓ Extracted 1 segments to: segments
(jet_venv) PS C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace>
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:
- Browse when beneficial or requested.
- Keep explanations simple and clear.

When coding:
- Provide step-by-step analysis and explain the flow.
- Use visuals, diagrams, or tables when helpful.
- For new files, classes, methods, or functions: show the full code.
- For updates to existing files: show only the changed sections with context. Never output the full file unless it's small.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Always add logs for traceability and verification.
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
        clipboard_content_parts.append(f"<system>\n{system_message}\n</system>")
    # Query should come before instructions
    clipboard_content_parts.append(f"<query>\n{query_message}\n</query>")
    if instructions_message:
        clipboard_content_parts.append(f"<instructions>\n{instructions_message}\n</instructions>")
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
    # Disable special-token checks entirely — the input is arbitrary file
    # content/prompt text, not something where special tokens should be
    # interpreted as control tokens. This prevents ValueError crashes when
    # source files happen to contain strings like "<|endoftext|>".
    return len(encoding.encode(text, disallowed_special=()))


if __name__ == "__main__":
    main()
