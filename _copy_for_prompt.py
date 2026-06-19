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
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\state.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_segment_speaker_labeler.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\dashboard.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\speaker_metrics.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\single_plot.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\health_diagnostics.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\independent_analysis.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\similarity_network.js",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\pairwise_comparison.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\speaker_embedding_plot.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\dimension_diff_view.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\similarity_gauge.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\speaker_embedding_plot.html",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\pairwise_comparison.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\similarity_network.js",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\dimension_diff_view.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\dimension_diff_view.html",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_metrics_mixin.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\helpers\speaker_metrics.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler_health_mixin.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\test_segment_speaker_labeler_mixin_inheritance.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\vad_firered.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speech_waves.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_speech_waves.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\temp9.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\temp10.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_segment_speaker_labeler.py",
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
Evaluate

╭─── AudioTagger Configuration ────╮
│ AudioTagger Initialized          │
│ Model: model.onnx                │
│ Labels: class_labels_indices.csv │
│ Speech Threshold: 0.5            │
│ Speech Top N: 3                  │
│ Chunk Duration: 1.0s             │
│ Chunk Overlap: 0.5s              │
│ Min Chunk Duration: 0.5s         │
╰──────────────────────────────────╯
SegmentSpeakerLabeler initialized
  Segment duration: 1.0s - 5.0s
  Max speakers/segment: 3
  Split threshold: 0.35
  Audio tagger: enabled
  Speech filtering: enabled

Processing 23 audio segments...

⠋ Analyzing speakers...⚠️  Segment too short: 0.35s < 1.0s (min), skipping
Segment 1: skipped (too short) at t=0.00s
✅ Segment duration valid: 1.46s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 1.46s, 16000Hz, 23360 samples
📊 Audio loaded: 1.46s, 16000Hz, 23360 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 2 chunk positions
🔍 Processing chunk 1/2: 0.00s - 1.00s
⠏ Analyzing speakers...D:\a\sherpa-onnx\sherpa-onnx\sherpa-onnx\csrc\offline-zipformer-audio-tagging-model.cc:Init:69 version=1
model_type=zipformer2
model_author=k2-fsa
url=https://github.com/k2-fsa/icefall/tree/master/egs/audioset/AT/zipformer
comment=zipformer2 audio tagger


⠙ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0716)
🔍 Processing chunk 2/2: 0.50s - 1.46s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
📊 No speech chunks for avg calculation
⏱ Total processing: 2.57s, RTF: 1.761x
⚠ No speech segments found, returning empty array
AudioTagger: No speech detected, using original
Embedding shape: (1, 512), ndim: 2, filtered: False
Computed embedding for t=5.00s, got 0 top matches
Actual best score: 0.0000, should_create_new_speaker: True
Created new speaker: SPEAKER_01 (segment_count=1, total_speakers=1, next_id=2)
⚠️  New speaker: SPEAKER_01 (best sim: 0.000, mature: 0, young: 0, total: 1)
Segment 2: t=5.00s → [SPEAKER_01(1.000)] (primary: SPEAKER_01, speakers: 1, rejected: 0)
⚠️  Segment too long: 6.50s > 5.0s (max), splitting into 2 chunks of ~3.25s each
  Sub-segment 1/2: 10.00s - 13.25s (3.25s)
  Sub-segment 2/2: 12.44s - 15.69s (3.25s)
⠹ Analyzing speakers...Processing sub-segment 1/2 at t=10.00s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 3.25s, 16000Hz, 52000 samples
📊 Audio loaded: 3.25s, 16000Hz, 52000 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 6 chunk positions
🔍 Processing chunk 1/6: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0467)
🔍 Processing chunk 2/6: 0.50s - 1.50s
⠸ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.7813
🔍 Processing chunk 3/6: 1.00s - 2.00s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0501)
🔍 Processing chunk 4/6: 1.50s - 2.50s
⠴ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 5/6: 2.00s - 3.00s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.4844)
🔍 Processing chunk 6/6: 2.50s - 3.25s
⠦ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9265
📊 Avg speech probability: 0.8539 (from 2 speech chunks)
⏱ Total processing: 0.29s, RTF: 0.090x
🔍 Edges-only: first speech at 0.500s, last speech at 3.250s
🎤 Found 1 speech segment(s):
   Segment 1: 0.500s - 3.250s (duration: 2.750s)
✅ Speech extracted: 2.75s (removed 15.4% of audio)
🎯 AudioTagger speech extracted: 2.75s from 3.25s (84.6% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=10.00s, got 1 top matches
Actual best score: 0.1539, should_create_new_speaker: True
Created new speaker: SPEAKER_02 (segment_count=1, total_speakers=2, next_id=3)
⚠️  New speaker: SPEAKER_02 (best sim: 0.154, mature: 0, young: 0, total: 2)
Processing sub-segment 2/2 at t=12.44s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 3.25s, 16000Hz, 52000 samples
📊 Audio loaded: 3.25s, 16000Hz, 52000 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 6 chunk positions
🔍 Processing chunk 1/6: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9760
🔍 Processing chunk 2/6: 0.50s - 1.50s
⠇ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9679
🔍 Processing chunk 3/6: 1.00s - 2.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9850
🔍 Processing chunk 4/6: 1.50s - 2.50s
⠏ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.5910
🔍 Processing chunk 5/6: 2.00s - 3.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.6659
🔍 Processing chunk 6/6: 2.50s - 3.25s
⠋ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.6258
📊 Avg speech probability: 0.8019 (from 6 speech chunks)
⏱ Total processing: 0.30s, RTF: 0.092x
🔍 Edges-only: first speech at 0.000s, last speech at 3.250s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 3.250s (duration: 3.250s)
✅ Speech extracted: 3.25s (removed 0.0% of audio)
🎯 AudioTagger speech extracted: 3.25s from 3.25s (100.0% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=12.44s, got 1 top matches
Actual best score: 0.1573, should_create_new_speaker: True
Created new speaker: SPEAKER_03 (segment_count=1, total_speakers=3, next_id=4)
⚠️  New speaker: SPEAKER_03 (best sim: 0.157, mature: 0, young: 0, total: 3)
Segment 3: t=10.00s → [SPEAKER_02(1.000), SPEAKER_03(1.000), SPEAKER_01(0.154)] (primary: SPEAKER_02, speakers: 3,
rejected: 0, sub-segments: 2)
⚠️  Segment too short: 0.38s < 1.0s (min), skipping
Segment 4: skipped (too short) at t=15.00s
✅ Segment duration valid: 3.01s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 3.01s, 16000Hz, 48160 samples
📊 Audio loaded: 3.01s, 16000Hz, 48160 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 6 chunk positions
🔍 Processing chunk 1/6: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8897
🔍 Processing chunk 2/6: 0.50s - 1.50s
⠹ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9709
🔍 Processing chunk 3/6: 1.00s - 2.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9893
🔍 Processing chunk 4/6: 1.50s - 2.50s
⠸ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.5448
🔍 Processing chunk 5/6: 2.00s - 3.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8241
🔍 Processing chunk 6/6: 2.50s - 3.01s
⠼ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0295)
📊 Avg speech probability: 0.8438 (from 5 speech chunks)
⏱ Total processing: 0.29s, RTF: 0.095x
🔍 Edges-only: first speech at 0.000s, last speech at 3.000s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 3.000s (duration: 3.000s)
✅ Speech extracted: 3.00s (removed 0.3% of audio)
🎯 AudioTagger speech extracted: 3.00s from 3.01s (99.7% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=20.00s, got 1 top matches
Actual best score: 0.3363, should_create_new_speaker: False
✓ Updated SPEAKER_02 (sim=0.336, reason=passed)
Segment 5: t=20.00s → [SPEAKER_02(0.336)] (primary: SPEAKER_02, speakers: 3, rejected: 0)
✅ Segment duration valid: 1.60s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 1.60s, 16000Hz, 25600 samples
📊 Audio loaded: 1.60s, 16000Hz, 25600 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 3 chunk positions
🔍 Processing chunk 1/3: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9612
🔍 Processing chunk 2/3: 0.50s - 1.50s
⠦ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9106
🔍 Processing chunk 3/3: 1.00s - 1.60s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8238
📊 Avg speech probability: 0.8985 (from 3 speech chunks)
⏱ Total processing: 0.15s, RTF: 0.093x
🔍 Edges-only: first speech at 0.000s, last speech at 1.600s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 1.600s (duration: 1.600s)
✅ Speech extracted: 1.60s (removed 0.0% of audio)
🎯 AudioTagger speech extracted: 1.60s from 1.60s (100.0% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=25.00s, got 2 top matches
Actual best score: 0.3337, should_create_new_speaker: False
✓ Updated SPEAKER_03 (sim=0.334, reason=passed)
Segment 6: t=25.00s → [SPEAKER_03(0.334), SPEAKER_01(0.170)] (primary: SPEAKER_03, speakers: 3, rejected: 0)
⚠️  Segment too short: 0.88s < 1.0s (min), skipping
Segment 7: skipped (too short) at t=30.00s
✅ Segment duration valid: 2.88s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 2.88s, 16000Hz, 46080 samples
📊 Audio loaded: 2.88s, 16000Hz, 46080 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 5 chunk positions
🔍 Processing chunk 1/5: 0.00s - 1.00s
⠧ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8598
🔍 Processing chunk 2/5: 0.50s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9411
🔍 Processing chunk 3/5: 1.00s - 2.00s
⠏ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9678
🔍 Processing chunk 4/5: 1.50s - 2.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9941
🔍 Processing chunk 5/5: 2.00s - 2.88s
⠋ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9946
📊 Avg speech probability: 0.9515 (from 5 speech chunks)
⏱ Total processing: 0.25s, RTF: 0.086x
🔍 Edges-only: first speech at 0.000s, last speech at 2.880s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 2.880s (duration: 2.880s)
✅ Speech extracted: 2.88s (removed 0.0% of audio)
🎯 AudioTagger speech extracted: 2.88s from 2.88s (100.0% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=35.00s, got 1 top matches
Actual best score: 0.5040, should_create_new_speaker: False
✓ Updated SPEAKER_03 (sim=0.504, reason=passed)
Segment 8: t=35.00s → [SPEAKER_03(0.504)] (primary: SPEAKER_03, speakers: 3, rejected: 0)
⚠️  Segment too short: 0.70s < 1.0s (min), skipping
Segment 9: skipped (too short) at t=40.00s
⚠️  Segment too short: 0.96s < 1.0s (min), skipping
Segment 10: skipped (too short) at t=45.00s
⚠️  Segment too short: 0.27s < 1.0s (min), skipping
Segment 11: skipped (too short) at t=50.00s
⚠️  Segment too short: 0.42s < 1.0s (min), skipping
Segment 12: skipped (too short) at t=55.00s
⚠️  Segment too short: 0.96s < 1.0s (min), skipping
Segment 13: skipped (too short) at t=60.00s
✅ Segment duration valid: 4.22s (1.0s-5.0s)
🔀 Speaker change detected at 66.00s (sim=0.334 < 0.35)
🔀 Split segment into 2 sub-segments (1 change points detected)
Processing sub-segment 1/2 at t=65.00s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 1.00s, 16000Hz, 16000 samples
📊 Audio loaded: 1.00s, 16000Hz, 16000 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 1 chunk positions
🔍 Processing chunk 1/1: 0.00s - 1.00s
⠙ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
📊 No speech chunks for avg calculation
⏱ Total processing: 0.05s, RTF: 0.051x
⚠ No speech segments found, returning empty array
AudioTagger: No speech detected, using original
Embedding shape: (1, 512), ndim: 2, filtered: False
Computed embedding for t=65.00s, got 1 top matches
Actual best score: 0.2827, should_create_new_speaker: True
Created new speaker: SPEAKER_04 (segment_count=1, total_speakers=4, next_id=5)
⚠️  New speaker: SPEAKER_04 (best sim: 0.283, mature: 0, young: 2, total: 4)
Processing sub-segment 2/2 at t=66.00s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 3.22s, 16000Hz, 51520 samples
📊 Audio loaded: 3.22s, 16000Hz, 51520 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 6 chunk positions
🔍 Processing chunk 1/6: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8495
🔍 Processing chunk 2/6: 0.50s - 1.50s
⠸ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.7513
🔍 Processing chunk 3/6: 1.00s - 2.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9963
🔍 Processing chunk 4/6: 1.50s - 2.50s
⠼ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9969
🔍 Processing chunk 5/6: 2.00s - 3.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9757
🔍 Processing chunk 6/6: 2.50s - 3.22s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
⠴ Analyzing speakers...📊 Avg speech probability: 0.9139 (from 5 speech chunks)
⏱ Total processing: 0.29s, RTF: 0.090x
🔍 Edges-only: first speech at 0.000s, last speech at 3.000s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 3.000s (duration: 3.000s)
✅ Speech extracted: 3.00s (removed 6.8% of audio)
🎯 AudioTagger speech extracted: 3.00s from 3.22s (93.2% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=66.00s, got 2 top matches
Actual best score: 0.3410, should_create_new_speaker: False
✓ Updated SPEAKER_04 (sim=0.341, reason=passed)
🔧 Maintenance triggered: mature=0, young=2, orphan=2
Orphan remove: SPEAKER_01 (inactive 60.0s)
🔒 Protecting newborn SPEAKER_04 from cleanup
🔧 Maintenance: 4 → 3 speakers (removed 1 orphans, merged 0 young, merged 0 mature)
Segment 14: t=65.00s → [SPEAKER_04(1.000), SPEAKER_02(0.311)] (primary: SPEAKER_04, speakers: 3, rejected: 0,
sub-segments: 2)
✅ Segment duration valid: 1.13s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 1.13s, 16000Hz, 18080 samples
📊 Audio loaded: 1.13s, 16000Hz, 18080 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 2 chunk positions
🔍 Processing chunk 1/2: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9986
🔍 Processing chunk 2/2: 0.50s - 1.13s
⠧ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0088)
📊 Avg speech probability: 0.9986 (from 1 speech chunks)
⏱ Total processing: 0.10s, RTF: 0.087x
🔍 Edges-only: first speech at 0.000s, last speech at 1.000s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 1.000s (duration: 1.000s)
✅ Speech extracted: 1.00s (removed 11.5% of audio)
🎯 AudioTagger speech extracted: 1.00s from 1.13s (88.5% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=70.00s, got 1 top matches
Actual best score: 0.2765, should_create_new_speaker: True
Created new speaker: SPEAKER_05 (segment_count=1, total_speakers=4, next_id=6)
⚠️  New speaker: SPEAKER_05 (best sim: 0.277, mature: 0, young: 1, total: 4)
Segment 15: t=70.00s → [SPEAKER_05(1.000), SPEAKER_03(0.277)] (primary: SPEAKER_05, speakers: 4, rejected: 0)
⚠️  Segment too short: 0.43s < 1.0s (min), skipping
Segment 16: skipped (too short) at t=75.00s
✅ Segment duration valid: 1.05s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 1.05s, 16000Hz, 16800 samples
📊 Audio loaded: 1.05s, 16000Hz, 16800 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 2 chunk positions
🔍 Processing chunk 1/2: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9977
🔍 Processing chunk 2/2: 0.50s - 1.05s
⠇ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9999
📊 Avg speech probability: 0.9988 (from 2 speech chunks)
⏱ Total processing: 0.09s, RTF: 0.085x
🔍 Edges-only: first speech at 0.000s, last speech at 1.050s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 1.050s (duration: 1.050s)
✅ Speech extracted: 1.05s (removed 0.0% of audio)
🎯 AudioTagger speech extracted: 1.05s from 1.05s (100.0% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=80.00s, got 2 top matches
Actual best score: 0.4487, should_create_new_speaker: False
✓ Updated SPEAKER_03 (sim=0.449, reason=passed)
Segment 17: t=80.00s → [SPEAKER_03(0.449), SPEAKER_05(0.188)] (primary: SPEAKER_03, speakers: 4, rejected: 0)
⚠️  Segment too long: 7.08s > 5.0s (max), splitting into 2 chunks of ~3.54s each
  Sub-segment 1/2: 85.00s - 88.54s (3.54s)
  Sub-segment 2/2: 87.66s - 91.20s (3.54s)
🔀 Speaker change detected at 89.16s (sim=0.330 < 0.35)
🔀 Split segment into 2 sub-segments (1 change points detected)
Processing sub-segment 1/3 at t=85.00s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 3.54s, 16000Hz, 56640 samples
📊 Audio loaded: 3.54s, 16000Hz, 56640 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 7 chunk positions
🔍 Processing chunk 1/7: 0.00s - 1.00s
⠏ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9664
🔍 Processing chunk 2/7: 0.50s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9756
🔍 Processing chunk 3/7: 1.00s - 2.00s
⠙ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8351
🔍 Processing chunk 4/7: 1.50s - 2.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8283
🔍 Processing chunk 5/7: 2.00s - 3.00s
⠹ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9805
🔍 Processing chunk 6/7: 2.50s - 3.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9991
🔍 Processing chunk 7/7: 3.00s - 3.54s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
📊 Avg speech probability: 0.9308 (from 6 speech chunks)
⏱ Total processing: 0.34s, RTF: 0.095x
🔍 Edges-only: first speech at 0.000s, last speech at 3.500s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 3.500s (duration: 3.500s)
✅ Speech extracted: 3.50s (removed 1.1% of audio)
🎯 AudioTagger speech extracted: 3.50s from 3.54s (98.9% kept)
⠼ Analyzing speakers...Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=85.00s, got 2 top matches
Actual best score: 0.5615, should_create_new_speaker: False
✓ Updated SPEAKER_03 (sim=0.561, reason=passed)
Processing sub-segment 2/3 at t=87.66s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 1.50s, 16000Hz, 24000 samples
📊 Audio loaded: 1.50s, 16000Hz, 24000 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 3 chunk positions
🔍 Processing chunk 1/3: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9993
🔍 Processing chunk 2/3: 0.50s - 1.50s
⠴ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9962
🔍 Processing chunk 3/3: 1.00s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.3510)
📊 Avg speech probability: 0.9978 (from 2 speech chunks)
⏱ Total processing: 0.15s, RTF: 0.099x
🔍 Edges-only: first speech at 0.000s, last speech at 1.500s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 1.500s (duration: 1.500s)
✅ Speech extracted: 1.50s (removed 0.0% of audio)
🎯 AudioTagger speech extracted: 1.50s from 1.50s (100.0% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=87.66s, got 1 top matches
Actual best score: 0.2745, should_create_new_speaker: True
Created new speaker: SPEAKER_06 (segment_count=1, total_speakers=5, next_id=7)
⚠️  New speaker: SPEAKER_06 (best sim: 0.275, mature: 1, young: 3, total: 5)
Processing sub-segment 3/3 at t=89.16s
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 2.04s, 16000Hz, 32640 samples
📊 Audio loaded: 2.04s, 16000Hz, 32640 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 4 chunk positions
🔍 Processing chunk 1/4: 0.00s - 1.00s
⠦ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9872
🔍 Processing chunk 2/4: 0.50s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.3423)
🔍 Processing chunk 3/4: 1.00s - 2.00s
⠧ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 4/4: 1.50s - 2.04s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
📊 Avg speech probability: 0.9872 (from 1 speech chunks)
⏱ Total processing: 0.20s, RTF: 0.097x
🔍 Edges-only: first speech at 0.000s, last speech at 1.000s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 1.000s (duration: 1.000s)
✅ Speech extracted: 1.00s (removed 51.0% of audio)
🎯 AudioTagger speech extracted: 1.00s from 2.04s (49.0% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=89.16s, got 2 top matches
Actual best score: 0.2362, should_create_new_speaker: True
Created new speaker: SPEAKER_07 (segment_count=1, total_speakers=6, next_id=8)
⚠️  New speaker: SPEAKER_07 (best sim: 0.236, mature: 1, young: 3, total: 6)
🔧 Maintenance triggered: mature=1, young=3, orphan=3
🔒 Protecting newborn SPEAKER_06 from cleanup
🔒 Protecting newborn SPEAKER_07 from cleanup
🔒 Skipping newborn SPEAKER_06 (age=87.7s)
🔒 Skipping newborn SPEAKER_07 (age=89.2s)
🔍 Re-eval KEEP: SPEAKER_02 (2 segs) vs SPEAKER_03 sim=0.105 < 0.5
🔍 Re-eval KEEP: SPEAKER_04 (2 segs) vs SPEAKER_03 sim=0.046 < 0.5
🔍 Re-eval KEEP: SPEAKER_05 (1 segs) vs SPEAKER_03 sim=0.254 < 0.5
Segment 18: t=85.00s → [SPEAKER_06(1.000), SPEAKER_07(1.000), SPEAKER_03(0.561)] (primary: SPEAKER_06, speakers: 6,
rejected: 0, sub-segments: 3)
⚠️  Segment too short: 0.98s < 1.0s (min), skipping
Segment 19: skipped (too short) at t=90.00s
✅ Segment duration valid: 2.44s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 2.44s, 16000Hz, 39040 samples
📊 Audio loaded: 2.44s, 16000Hz, 39040 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 4 chunk positions
🔍 Processing chunk 1/4: 0.00s - 1.00s
⠏ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.4812)
🔍 Processing chunk 2/4: 0.50s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.7408
🔍 Processing chunk 3/4: 1.00s - 2.00s
⠋ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.6667
🔍 Processing chunk 4/4: 1.50s - 2.44s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9995
📊 Avg speech probability: 0.8023 (from 3 speech chunks)
⏱ Total processing: 0.24s, RTF: 0.097x
🔍 Edges-only: first speech at 0.500s, last speech at 2.440s
🎤 Found 1 speech segment(s):
   Segment 1: 0.500s - 2.440s (duration: 1.940s)
✅ Speech extracted: 1.94s (removed 20.5% of audio)
🎯 AudioTagger speech extracted: 1.94s from 2.44s (79.5% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=95.00s, got 3 top matches
Actual best score: 0.4346, should_create_new_speaker: False
✓ Updated SPEAKER_03 (sim=0.435, reason=passed)
🔧 Maintenance triggered: mature=1, young=3, orphan=3
🔒 Protecting newborn SPEAKER_06 from cleanup
⠹ Analyzing speakers...🔒 Protecting newborn SPEAKER_07 from cleanup
🔒 Skipping newborn SPEAKER_06 (age=87.7s)
🔒 Skipping newborn SPEAKER_07 (age=89.2s)
🔍 Re-eval KEEP: SPEAKER_02 (2 segs) vs SPEAKER_03 sim=0.104 < 0.5
🔍 Re-eval KEEP: SPEAKER_04 (2 segs) vs SPEAKER_03 sim=0.057 < 0.5
🔍 Re-eval KEEP: SPEAKER_05 (1 segs) vs SPEAKER_03 sim=0.291 < 0.5
Segment 20: t=95.00s → [SPEAKER_03(0.435), SPEAKER_05(0.318), SPEAKER_07(0.202)] (primary: SPEAKER_03, speakers: 6,
rejected: 0)
✅ Segment duration valid: 4.28s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 4.28s, 16000Hz, 68480 samples
📊 Audio loaded: 4.28s, 16000Hz, 68480 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 8 chunk positions
🔍 Processing chunk 1/8: 0.00s - 1.00s
   ✅ Tagged successfully: 5 predictions
⠸ Analyzing speakers...   🎤 Speech detected! speech_prob=0.9914
🔍 Processing chunk 2/8: 0.50s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8935
🔍 Processing chunk 3/8: 1.00s - 2.00s
⠼ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9982
🔍 Processing chunk 4/8: 1.50s - 2.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9971
🔍 Processing chunk 5/8: 2.00s - 3.00s
⠦ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.2059)
🔍 Processing chunk 6/8: 2.50s - 3.50s
   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.8297
🔍 Processing chunk 7/8: 3.00s - 4.00s
⠧ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🎤 Speech detected! speech_prob=0.9919
🔍 Processing chunk 8/8: 3.50s - 4.28s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
📊 Avg speech probability: 0.9503 (from 6 speech chunks)
⏱ Total processing: 0.44s, RTF: 0.103x
🔍 Edges-only: first speech at 0.000s, last speech at 4.000s
🎤 Found 1 speech segment(s):
   Segment 1: 0.000s - 4.000s (duration: 4.000s)
✅ Speech extracted: 4.00s (removed 6.5% of audio)
🎯 AudioTagger speech extracted: 4.00s from 4.28s (93.5% kept)
Embedding shape: (1, 512), ndim: 2, filtered: True
Computed embedding for t=100.00s, got 3 top matches
Actual best score: 0.6799, should_create_new_speaker: False
✓ Updated SPEAKER_03 (sim=0.680, reason=passed)
🔧 Maintenance triggered: mature=1, young=3, orphan=3
🔒 Protecting newborn SPEAKER_06 from cleanup
🔒 Protecting newborn SPEAKER_07 from cleanup
🔒 Skipping newborn SPEAKER_06 (age=87.7s)
🔒 Skipping newborn SPEAKER_07 (age=89.2s)
🔍 Re-eval KEEP: SPEAKER_02 (2 segs) vs SPEAKER_03 sim=0.101 < 0.5
🔍 Re-eval KEEP: SPEAKER_04 (2 segs) vs SPEAKER_03 sim=0.054 < 0.5
🔍 Re-eval KEEP: SPEAKER_05 (1 segs) vs SPEAKER_03 sim=0.316 < 0.5
Segment 21: t=100.00s → [SPEAKER_03(0.680), SPEAKER_05(0.317), SPEAKER_07(0.261)] (primary: SPEAKER_03, speakers: 6,
rejected: 0)
⚠️  Segment too short: 0.31s < 1.0s (min), skipping
Segment 22: skipped (too short) at t=105.00s
✅ Segment duration valid: 2.78s (1.0s-5.0s)
╭─ Speech Extraction ─╮
│ extract_speech_only │
│ edges_only=True     │
│ prob_threshold=0.5  │
╰─────────────────────╯
📊 Audio loaded: 2.78s, 16000Hz, 44480 samples
📊 Audio loaded: 2.78s, 16000Hz, 44480 samples
🔧 Chunk config: 1.0s chunks, 0.5s overlap, hop=8000 samples
📏 Calculated 5 chunk positions
🔍 Processing chunk 1/5: 0.00s - 1.00s
⠇ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 2/5: 0.50s - 1.50s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.1951)
🔍 Processing chunk 3/5: 1.00s - 2.00s
⠋ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 4/5: 1.50s - 2.50s
   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
🔍 Processing chunk 5/5: 2.00s - 2.78s
⠙ Analyzing speakers...   ✅ Tagged successfully: 5 predictions
   🔇 No speech detected (speech_prob=0.0000)
📊 No speech chunks for avg calculation
⏱ Total processing: 0.24s, RTF: 0.088x
⚠ No speech segments found, returning empty array
AudioTagger: No speech detected, using original
Embedding shape: (1, 512), ndim: 2, filtered: False
Computed embedding for t=110.00s, got 1 top matches
Actual best score: 0.1559, should_create_new_speaker: True
Created new speaker: SPEAKER_08 (segment_count=1, total_speakers=7, next_id=9)
⚠️  New speaker: SPEAKER_08 (best sim: 0.156, mature: 1, young: 5, total: 7)
🔧 Maintenance triggered: mature=1, young=5, orphan=5
Orphan remove: SPEAKER_05 (inactive 40.0s)
🔒 Protecting newborn SPEAKER_08 from cleanup
🔒 Skipping newborn SPEAKER_08 (age=110.0s)
🔍 Re-eval KEEP: SPEAKER_02 (2 segs) vs SPEAKER_03 sim=0.101 < 0.5
🔍 Re-eval KEEP: SPEAKER_04 (2 segs) vs SPEAKER_03 sim=0.054 < 0.5
🔍 Re-eval KEEP: SPEAKER_06 (1 segs) vs SPEAKER_03 sim=0.276 < 0.5
🔍 Re-eval KEEP: SPEAKER_07 (1 segs) vs SPEAKER_03 sim=0.272 < 0.5
🔧 Maintenance: 7 → 6 speakers (removed 1 orphans, merged 0 young, merged 0 mature)
Segment 23: t=110.00s → [SPEAKER_08(1.000), SPEAKER_07(0.156)] (primary: SPEAKER_08, speakers: 6, rejected: 0)
  Analyzing speakers...
🎤 Speaker Analysis Results
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃  # ┃ Dir                  ┃ Duration ┃ Rank ┃  Speaker   ┃ Confidence ┃    Match Type    ┃ Primary ┃ ▶️ Play ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│  2 │ segment_001_wave_002 │    1.46s │  —   │ SPEAKER_01 │      1.000 │  first_speaker   │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  3 │ segment_001_wave_003 │    6.50s │  —   │ SPEAKER_02 │      1.000 │   new_speaker    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  —   │ SPEAKER_03 │      1.000 │   new_speaker    │   ⭐    │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #3  │ SPEAKER_01 │      0.154 │ weak_alternative │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  5 │ segment_001_wave_005 │    3.01s │  —   │ SPEAKER_02 │      0.336 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  6 │ segment_001_wave_006 │    1.60s │  —   │ SPEAKER_03 │      0.334 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_01 │      0.170 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  8 │ segment_001_wave_008 │    2.88s │  —   │ SPEAKER_03 │      0.504 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 14 │ segment_001_wave_014 │    4.22s │  —   │ SPEAKER_04 │      1.000 │   new_speaker    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_02 │      0.311 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 15 │ segment_001_wave_015 │    1.13s │  —   │ SPEAKER_05 │      1.000 │   new_speaker    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_03 │      0.277 │ weak_alternative │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 17 │ segment_001_wave_017 │    1.05s │  —   │ SPEAKER_03 │      0.449 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_05 │      0.188 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 18 │ segment_001_wave_018 │    7.08s │  —   │ SPEAKER_06 │      1.000 │   new_speaker    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  —   │ SPEAKER_07 │      1.000 │   new_speaker    │   ⭐    │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  —   │ SPEAKER_03 │      0.561 │  possible_match  │   ⭐    │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #4  │ SPEAKER_05 │      0.199 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 20 │ segment_001_wave_020 │    2.44s │  —   │ SPEAKER_03 │      0.435 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_05 │      0.318 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #3  │ SPEAKER_07 │      0.202 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 21 │ segment_001_wave_021 │    4.28s │  —   │ SPEAKER_03 │      0.680 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_05 │      0.317 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #3  │ SPEAKER_07 │      0.261 │    weak_match    │         │        │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 23 │ segment_001_wave_023 │    2.78s │  —   │ SPEAKER_08 │      1.000 │   new_speaker    │   ⭐    │ ▶️ Play │
├────┼──────────────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │                      │          │  #2  │ SPEAKER_07 │      0.156 │ weak_alternative │         │        │
└────┴──────────────────────┴──────────┴──────┴────────────┴────────────┴──────────────────┴─────────┴────────┘
╭────────────────────────────────────────────────────── Summary ───────────────────────────────────────────────────────╮
│                                                                                                                      │
│  Total segments: 23                                                                                                  │
│  Total results (incl. alternatives): 26                                                                              │
│  Total duration: 59.1s                                                                                               │
│  Unique speakers: 8                                                                                                  │
│  Average matches per segment: 1.1                                                                                    │
│                                                                                                                      │
╰────────────────────────────────
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
