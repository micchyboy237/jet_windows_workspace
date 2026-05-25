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
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\multilabel.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\speech_separation.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\voice_activity_detection.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\_main_speaker_labeler.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\io.py",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\_metadata.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_segment_speaker.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\_main_segment_speaker_labeler.py",
    r"",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr_onnx\sensevoice_bin.py",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr_onnx\utils\utils.py",
    # r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunASR\runtime\python\onnxruntime\funasr_onnx\sensevoice_bin.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunASR\runtime\python\onnxruntime\funasr_onnx\utils\utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunAudioLLM_SenseVoice\demo_onnx.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunAudioLLM_SenseVoice\jet_funasr_onnx_model_fixer.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\funasr_onnx_result_analyzer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_funasr_onnx_with_analysis.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_funasr_onnx.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_funasr.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_segment_speaker.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunASR\funasr\utils\postprocess_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_funasr.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_segment_speaker.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\segment_speaker_labeler.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\_main_segment_speaker_labeler.py",
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
Evaluate

Processing 20 audio segments...

⠋ Analyzing speakers...Computed embedding for t=0.00s, got 0 top matches
Actual best score: 0.0000, should_create_new_speaker: True
📥 Adding embedding to SPEAKER_01: quality=1.000, current count=0, stability=0.000
💾 Storage OK: 1/50 embeddings for SPEAKER_01
📐 Updating centroid for SPEAKER_01 with 1 embeddings
📌 Single embedding: centroid = direct copy, stability=0.1
📊 Centroid updated for SPEAKER_01: stability 0.000 ↑ 0.100 (Δ=+0.100)
🕐 First seen timestamp set: 0.00s
✨ Created new speaker: SPEAKER_01 (quality=1.000, max_embeddings=50)
📥 Adding embedding to SPEAKER_01: quality=1.000, current count=1, stability=0.100
💾 Storage OK: 2/50 embeddings for SPEAKER_01
📐 Updating centroid for SPEAKER_01 with 2 embeddings
📌 Two embeddings: centroid = mean, stability=0.3, inter-embedding similarity=1.000
📊 Centroid updated for SPEAKER_01: stability 0.100 ↑ 0.300 (Δ=+0.200)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_01: base=0.30, stability=0.300, final=0.195
📊 Storage stats for SPEAKER_01:
  Utilization: 4.0% (2/50)
  Stability: 0.300, Quality: 0.195
  Diversity: 0.500 (higher is better)
⚠️  New speaker: SPEAKER_01 (best sim: 0.000, mature: 0, young: 1, total: 1)
Segment 1: t=0.00s → [SPEAKER_01(1.000)] (primary: SPEAKER_01, speakers: 1)
🏷️  Centroid quality for SPEAKER_01: base=0.30, stability=0.300, final=0.195
Computed embedding for t=0.00s, got 1 top matches
Actual best score: 0.1087, should_create_new_speaker: True
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=0, stability=0.000
💾 Storage OK: 1/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 1 embeddings
📌 Single embedding: centroid = direct copy, stability=0.1
📊 Centroid updated for SPEAKER_02: stability 0.000 ↑ 0.100 (Δ=+0.100)
🕐 First seen timestamp set: 0.00s
✨ Created new speaker: SPEAKER_02 (quality=1.000, max_embeddings=50)
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=1, stability=0.100
💾 Storage OK: 2/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 2 embeddings
📌 Two embeddings: centroid = mean, stability=0.3, inter-embedding similarity=1.000
📊 Centroid updated for SPEAKER_02: stability 0.100 ↑ 0.300 (Δ=+0.200)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.30, stability=0.300, final=0.195
📊 Storage stats for SPEAKER_02:
  Utilization: 4.0% (2/50)
  Stability: 0.300, Quality: 0.195
  Diversity: 0.500 (higher is better)
⚠️  New speaker: SPEAKER_02 (best sim: 0.109, mature: 0, young: 2, total: 2)
Segment 2: t=0.00s → [SPEAKER_02(1.000), SPEAKER_01(0.109)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.30, stability=0.300, final=0.195
🏷️  Centroid quality for SPEAKER_02: base=0.30, stability=0.300, final=0.195
Computed embedding for t=0.00s, got 1 top matches
Actual best score: 0.2448, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=2, stability=0.300
💾 Storage OK: 3/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 3 embeddings
🎯 Calculating weighted centroid with 3 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.60653066 1.        ]...
  Similarity weights: [1.         1.         0.24478611]... (mean=0.748)
  Combined weights: [0.32864683 0.35768266 0.31367051]... (sum=1.000)
📈 Stability estimate for SPEAKER_02: 0.600 (based on count only, < 5 samples)
📊 Centroid updated for SPEAKER_02: stability 0.300 ↑ 0.600 (Δ=+0.300)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.60, stability=0.600, final=0.480
📊 Storage stats for SPEAKER_02:
  Utilization: 6.0% (3/50)
  Stability: 0.600, Quality: 0.480
  Diversity: 0.669 (higher is better)
Segment 3: t=0.00s → [SPEAKER_02(0.245)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.30, stability=0.300, final=0.195
🏷️  Centroid quality for SPEAKER_02: base=0.60, stability=0.600, final=0.480
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.3011, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=3, stability=0.600
💾 Storage OK: 4/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 4 embeddings
🎯 Calculating weighted centroid with 4 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.51341712 0.71653131]...
  Similarity weights: [0.9322841  0.9322841  0.44479637]... (mean=0.698)
  Combined weights: [0.24562175 0.2591958  0.23267277]... (sum=1.000)
📈 Stability estimate for SPEAKER_02: 0.800 (based on count only, < 5 samples)
📊 Centroid updated for SPEAKER_02: stability 0.600 ↑ 0.800 (Δ=+0.200)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.60, stability=0.800, final=0.540
📊 Storage stats for SPEAKER_02:
  Utilization: 8.0% (4/50)
  Stability: 0.800, Quality: 0.540
  Diversity: 0.722 (higher is better)
Segment 4: t=0.00s → [SPEAKER_02(0.301), SPEAKER_01(0.215)] (primary: SPEAKER_02, speakers: 2)
⠙ Analyzing speakers...🏷️  Centroid quality for SPEAKER_01: base=0.30, stability=0.300, final=0.195
🏷️  Centroid quality for SPEAKER_02: base=0.60, stability=0.800, final=0.540
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.2811, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=4, stability=0.800
💾 Storage OK: 5/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 5 embeddings
🎯 Calculating weighted centroid with 5 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.47236655 0.60653066]...
  Similarity weights: [0.83170255 0.83170255 0.4548687 ]... (mean=0.606)
  Combined weights: [0.19602694 0.20411339 0.18533274]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.666
  Variance of similarities: 0.0165
  Stability: 0.800 → 0.655 (Δ=-0.145)
📊 Centroid updated for SPEAKER_02: stability 0.800 ↓ 0.655 (Δ=-0.145)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.655, final=0.662
📊 Storage stats for SPEAKER_02:
  Utilization: 10.0% (5/50)
  Stability: 0.655, Quality: 0.662
  Diversity: 0.756 (higher is better)
Segment 5: t=0.00s → [SPEAKER_02(0.281), SPEAKER_01(0.210)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.30, stability=0.300, final=0.195
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.655, final=0.662
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.4514, should_create_new_speaker: False
Temporal smoothing: keeping 'SPEAKER_02' over 'SPEAKER_01' (sim=0.451)
📥 Adding embedding to SPEAKER_01: quality=1.000, current count=2, stability=0.300
💾 Storage OK: 3/50 embeddings for SPEAKER_01
📐 Updating centroid for SPEAKER_01 with 3 embeddings
🎯 Calculating weighted centroid with 3 samples
📊 Weight components for SPEAKER_01:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.60653066 1.        ]...
  Similarity weights: [1.         1.         0.45137402]... (mean=0.817)
  Combined weights: [0.32058889 0.34891281 0.3304983 ]... (sum=1.000)
📈 Stability estimate for SPEAKER_01: 0.600 (based on count only, < 5 samples)
📊 Centroid updated for SPEAKER_01: stability 0.300 ↑ 0.600 (Δ=+0.300)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
📊 Storage stats for SPEAKER_01:
  Utilization: 6.0% (3/50)
  Stability: 0.600, Quality: 0.480
  Diversity: 0.577 (higher is better)
Segment 6: t=0.00s → [SPEAKER_01(0.451), SPEAKER_02(0.239)] (primary: SPEAKER_01, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.655, final=0.662
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.4563, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=5, stability=0.655
💾 Storage OK: 6/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 6 embeddings
🎯 Calculating weighted centroid with 6 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.44932896 0.54881164]...
  Similarity weights: [0.8230876  0.8230876  0.48619182]... (mean=0.616)
  Combined weights: [0.16229512 0.16753176 0.15226774]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.648
  Variance of similarities: 0.0139
  Stability: 0.655 → 0.639 (Δ=-0.016)
📊 Centroid updated for SPEAKER_02: stability 0.655 ↓ 0.639 (Δ=-0.016)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.639, final=0.655
📊 Storage stats for SPEAKER_02:
  Utilization: 12.0% (6/50)
  Stability: 0.639, Quality: 0.655
  Diversity: 0.746 (higher is better)
Segment 7: t=0.00s → [SPEAKER_02(0.456), SPEAKER_01(0.291)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.639, final=0.655
Computed embedding for t=0.00s, got 1 top matches
Actual best score: 0.2038, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=6, stability=0.639
💾 Storage OK: 7/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 7 embeddings
🎯 Calculating weighted centroid with 7 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.43459821 0.51341712]...
  Similarity weights: [0.80518492 0.80518492 0.48939551]... (mean=0.556)
  Combined weights: [0.14151893 0.14528608 0.13190599]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.599
  Variance of similarities: 0.0194
  Stability: 0.639 → 0.588 (Δ=-0.051)
📊 Centroid updated for SPEAKER_02: stability 0.639 ↓ 0.588 (Δ=-0.051)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.588, final=0.635
📊 Storage stats for SPEAKER_02:
  Utilization: 14.0% (7/50)
  Stability: 0.588, Quality: 0.635
  Diversity: 0.782 (higher is better)
Segment 8: t=0.00s → [SPEAKER_02(0.204)] (primary: SPEAKER_02, speakers: 2)
⠹ Analyzing speakers...🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.588, final=0.635
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.4570, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=7, stability=0.588
💾 Storage OK: 8/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 8 embeddings
🎯 Calculating weighted centroid with 8 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.42437285 0.48954166]...
  Similarity weights: [0.76046272 0.76046272 0.48450397]... (mean=0.564)
  Combined weights: [0.12129932 0.12408304 0.11369635]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.595
  Variance of similarities: 0.0135
  Stability: 0.588 → 0.587 (Δ=-0.000)
📊 Centroid updated for SPEAKER_02: stability 0.588 → 0.587 (Δ=-0.000)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.587, final=0.635
📊 Storage stats for SPEAKER_02:
  Utilization: 16.0% (8/50)
  Stability: 0.587, Quality: 0.635
  Diversity: 0.770 (higher is better)
Segment 9: t=0.00s → [SPEAKER_02(0.457), SPEAKER_01(0.251)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.587, final=0.635
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.2386, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=8, stability=0.587
💾 Storage OK: 9/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 9 embeddings
🎯 Calculating weighted centroid with 9 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.41686202 0.47236655]...
  Similarity weights: [0.69450118 0.69450118 0.43519752]... (mean=0.523)
  Combined weights: [0.10671079 0.10889259 0.09981489]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.562
  Variance of similarities: 0.0141
  Stability: 0.587 → 0.554 (Δ=-0.033)
📊 Centroid updated for SPEAKER_02: stability 0.587 ↓ 0.554 (Δ=-0.033)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.554, final=0.622
📊 Storage stats for SPEAKER_02:
  Utilization: 18.0% (9/50)
  Stability: 0.554, Quality: 0.622
  Diversity: 0.789 (higher is better)
Segment 10: t=0.00s → [SPEAKER_02(0.239), SPEAKER_01(0.198)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=0.80, stability=0.554, final=0.622
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.4275, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=9, stability=0.554
💾 Storage OK: 10/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 10 embeddings
🎯 Calculating weighted centroid with 10 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.41111229 0.45942582]...
  Similarity weights: [0.67643949 0.67643949 0.42819878]... (mean=0.528)
  Combined weights: [0.09513261 0.0968624  0.08886308]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.553
  Variance of similarities: 0.0148
  Stability: 0.554 → 0.545 (Δ=-0.009)
📊 Centroid updated for SPEAKER_02: stability 0.554 → 0.545 (Δ=-0.009)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.545, final=0.773
📊 Storage stats for SPEAKER_02:
  Utilization: 20.0% (10/50)
  Stability: 0.545, Quality: 0.773
  Diversity: 0.789 (higher is better)
Segment 11: t=0.00s → [SPEAKER_02(0.427), SPEAKER_01(0.215)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.545, final=0.773
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.5184, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=10, stability=0.545
💾 Storage OK: 11/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 11 embeddings
🎯 Calculating weighted centroid with 11 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.40656966 0.44932896]...
  Similarity weights: [0.62786007 0.62786007 0.39124989]... (mean=0.519)
  Combined weights: [0.08503138 0.08644391 0.07936667]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.546
  Variance of similarities: 0.0168
  Stability: 0.545 → 0.537 (Δ=-0.008)
📊 Centroid updated for SPEAKER_02: stability 0.545 → 0.537 (Δ=-0.008)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.537, final=0.768
📊 Storage stats for SPEAKER_02:
  Utilization: 22.0% (11/50)
  Stability: 0.537, Quality: 0.768
  Diversity: 0.781 (higher is better)
Segment 12: t=0.00s → [SPEAKER_02(0.518), SPEAKER_01(0.294)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.537, final=0.768
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.4919, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=11, stability=0.537
⠼ Analyzing speakers...💾 Storage OK: 12/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 12 embeddings
🎯 Calculating weighted centroid with 12 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.40289032 0.44123317]...
  Similarity weights: [0.60150067 0.60150067 0.37040156]... (mean=0.524)
  Combined weights: [0.07693378 0.07810349 0.0716635 ]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.543
  Variance of similarities: 0.0213
  Stability: 0.537 → 0.532 (Δ=-0.005)
📊 Centroid updated for SPEAKER_02: stability 0.537 → 0.532 (Δ=-0.005)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.532, final=0.766
📊 Storage stats for SPEAKER_02:
  Utilization: 24.0% (12/50)
  Stability: 0.532, Quality: 0.766
  Diversity: 0.779 (higher is better)
Segment 13: t=0.00s → [SPEAKER_02(0.492), SPEAKER_01(0.225)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.532, final=0.766
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.5778, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=12, stability=0.532
💾 Storage OK: 13/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 13 embeddings
🎯 Calculating weighted centroid with 13 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.39984965 0.43459821]...
  Similarity weights: [0.55283196 0.55283196 0.33361154]... (mean=0.523)
  Combined weights: [0.06954365 0.07053002 0.06483854]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.576
  Variance of similarities: 0.0252
  Stability: 0.532 → 0.562 (Δ=+0.030)
📊 Centroid updated for SPEAKER_02: stability 0.532 ↑ 0.562 (Δ=+0.030)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.562, final=0.781
📊 Storage stats for SPEAKER_02:
  Utilization: 26.0% (13/50)
  Stability: 0.562, Quality: 0.781
  Diversity: 0.770 (higher is better)
Segment 14: t=0.00s → [SPEAKER_02(0.578), SPEAKER_01(0.217)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.562, final=0.781
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.5496, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=13, stability=0.562
💾 Storage OK: 14/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 14 embeddings
🎯 Calculating weighted centroid with 14 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.39729471 0.429062  ]...
  Similarity weights: [0.52543473 0.52543473 0.31335024]... (mean=0.530)
  Combined weights: [0.06364026 0.06448098 0.0593273 ]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.600
  Variance of similarities: 0.0237
  Stability: 0.562 → 0.586 (Δ=+0.024)
📊 Centroid updated for SPEAKER_02: stability 0.562 ↑ 0.586 (Δ=+0.024)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.586, final=0.793
📊 Storage stats for SPEAKER_02:
  Utilization: 28.0% (14/50)
  Stability: 0.586, Quality: 0.793
  Diversity: 0.765 (higher is better)
Segment 15: t=0.00s → [SPEAKER_02(0.550), SPEAKER_01(0.225)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.586, final=0.793
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.6197, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=14, stability=0.586
💾 Storage OK: 15/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 15 embeddings
🎯 Calculating weighted centroid with 15 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.39511776 0.42437285]...
  Similarity weights: [0.48206727 0.48206727 0.2918346 ]... (mean=0.529)
  Combined weights: [0.05827071 0.05899768 0.05470127]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.613
  Variance of similarities: 0.0308
  Stability: 0.586 → 0.595 (Δ=+0.008)
📊 Centroid updated for SPEAKER_02: stability 0.586 → 0.595 (Δ=+0.008)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.595, final=0.797
📊 Storage stats for SPEAKER_02:
  Utilization: 30.0% (15/50)
  Stability: 0.595, Quality: 0.797
  Diversity: 0.757 (higher is better)
Segment 16: t=0.00s → [SPEAKER_02(0.620), SPEAKER_01(0.166)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.595, final=0.797
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.2577, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=15, stability=0.595
💾 Storage OK: 16/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 16 embeddings
🎯 Calculating weighted centroid with 16 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.39324072 0.42035038]...
  Similarity weights: [0.47282189 0.47282189 0.28699254]... (mean=0.519)
  Combined weights: [0.05460918 0.05524622 0.05125937]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.602
  Variance of similarities: 0.0362
  Stability: 0.595 → 0.580 (Δ=-0.015)
📊 Centroid updated for SPEAKER_02: stability 0.595 ↓ 0.580 (Δ=-0.015)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.580, final=0.790
⠴ Analyzing speakers...📊 Storage stats for SPEAKER_02:
  Utilization: 32.0% (16/50)
  Stability: 0.580, Quality: 0.790
  Diversity: 0.770 (higher is better)
Segment 17: t=0.00s → [SPEAKER_02(0.258), SPEAKER_01(0.220)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.580, final=0.790
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.5274, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=16, stability=0.580
💾 Storage OK: 17/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 17 embeddings
🎯 Calculating weighted centroid with 17 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.39160563 0.41686202]...
  Similarity weights: [0.46033156 0.46033156 0.27731867]... (mean=0.520)
  Combined weights: [0.05109283 0.05165365 0.04792473]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.645
  Variance of similarities: 0.0123
  Stability: 0.580 → 0.637 (Δ=+0.057)
📊 Centroid updated for SPEAKER_02: stability 0.580 ↑ 0.637 (Δ=+0.057)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.637, final=0.819
📊 Storage stats for SPEAKER_02:
  Utilization: 34.0% (17/50)
  Stability: 0.637, Quality: 0.819
  Diversity: 0.764 (higher is better)
Segment 18: t=0.00s → [SPEAKER_02(0.527), SPEAKER_01(0.199)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.637, final=0.819
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.5729, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=17, stability=0.637
💾 Storage OK: 18/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 18 embeddings
🎯 Calculating weighted centroid with 18 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.39016854 0.4138081 ]...
  Similarity weights: [0.4410028  0.4410028  0.26946107]... (mean=0.526)
  Combined weights: [0.04770116 0.04819748 0.04490411]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.645
  Variance of similarities: 0.0125
  Stability: 0.637 → 0.637 (Δ=-0.000)
📊 Centroid updated for SPEAKER_02: stability 0.637 → 0.637 (Δ=-0.000)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.637, final=0.818
📊 Storage stats for SPEAKER_02:
  Utilization: 36.0% (18/50)
  Stability: 0.637, Quality: 0.818
  Diversity: 0.758 (higher is better)
Segment 19: t=0.00s → [SPEAKER_02(0.573), SPEAKER_01(0.227)] (primary: SPEAKER_02, speakers: 2)
🏷️  Centroid quality for SPEAKER_01: base=0.60, stability=0.600, final=0.480
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.637, final=0.818
Computed embedding for t=0.00s, got 2 top matches
Actual best score: 0.4976, should_create_new_speaker: False
📥 Adding embedding to SPEAKER_02: quality=1.000, current count=18, stability=0.637
💾 Storage OK: 19/50 embeddings for SPEAKER_02
📐 Updating centroid for SPEAKER_02 with 19 embeddings
🎯 Calculating weighted centroid with 19 samples
📊 Weight components for SPEAKER_02:
  Quality weights: [1. 1. 1.]... (mean=1.000)
  Recency weights: [0.36787944 0.38889556 0.41111229]...
  Similarity weights: [0.44326399 0.44326399 0.280096  ]... (mean=0.525)
  Combined weights: [0.04527086 0.04571452 0.04273897]... (sum=1.000)
📊 Stability updated for SPEAKER_02:
  Mean similarity to centroid: 0.623
  Variance of similarities: 0.0112
  Stability: 0.637 → 0.616 (Δ=-0.021)
📊 Centroid updated for SPEAKER_02: stability 0.637 ↓ 0.616 (Δ=-0.021)
🕐 First seen timestamp set: 0.00s
🏷️  Centroid quality for SPEAKER_02: base=1.00, stability=0.616, final=0.808
📊 Storage stats for SPEAKER_02:
  Utilization: 38.0% (19/50)
  Stability: 0.616, Quality: 0.808
  Diversity: 0.755 (higher is better)
Segment 20: t=0.00s → [SPEAKER_02(0.498), SPEAKER_01(0.356)] (primary: SPEAKER_02, speakers: 2)
  Analyzing speakers...
🎤 Speaker Analysis Results
┏━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃  # ┃ Dir         ┃ Duration ┃ Rank ┃  Speaker   ┃ Confidence ┃    Match Type    ┃ Primary ┃ ▶️ Play ┃
┡━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│  1 │ segment_120 │    1.29s │  —   │ SPEAKER_01 │      1.000 │  first_speaker   │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  2 │ segment_121 │    3.70s │  —   │ SPEAKER_02 │      1.000 │   new_speaker    │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.109 │ weak_alternative │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  3 │ segment_122 │    1.91s │  —   │ SPEAKER_02 │      0.245 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  4 │ segment_123 │    2.67s │  —   │ SPEAKER_02 │      0.301 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.215 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  5 │ segment_124 │    3.34s │  —   │ SPEAKER_02 │      0.281 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.210 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  6 │ segment_125 │    2.97s │  —   │ SPEAKER_01 │      0.451 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_02 │      0.239 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  7 │ segment_126 │    6.88s │  —   │ SPEAKER_02 │      0.456 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.291 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  8 │ segment_127 │    3.55s │  —   │ SPEAKER_02 │      0.204 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│  9 │ segment_128 │    3.40s │  —   │ SPEAKER_02 │      0.457 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.251 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 10 │ segment_129 │    1.33s │  —   │ SPEAKER_02 │      0.239 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.198 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 11 │ segment_130 │    1.15s │  —   │ SPEAKER_02 │      0.427 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.215 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 12 │ segment_131 │    1.13s │  —   │ SPEAKER_02 │      0.518 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.294 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 13 │ segment_132 │    1.70s │  —   │ SPEAKER_02 │      0.492 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.225 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 14 │ segment_133 │    1.48s │  —   │ SPEAKER_02 │      0.578 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.217 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 15 │ segment_134 │    3.80s │  —   │ SPEAKER_02 │      0.550 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.225 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 16 │ segment_135 │    1.24s │  —   │ SPEAKER_02 │      0.620 │   strong_match   │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.166 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 17 │ segment_136 │    1.33s │  —   │ SPEAKER_02 │      0.258 │    weak_match    │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.220 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 18 │ segment_137 │    4.47s │  —   │ SPEAKER_02 │      0.527 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.199 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 19 │ segment_138 │    2.08s │  —   │ SPEAKER_02 │      0.573 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.227 │    weak_match    │         │        │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│ 20 │ segment_139 │    3.10s │  —   │ SPEAKER_02 │      0.498 │  possible_match  │   ⭐    │ ▶️ Play │
├────┼─────────────┼──────────┼──────┼────────────┼────────────┼──────────────────┼─────────┼────────┤
│    │             │          │  #2  │ SPEAKER_01 │      0.356 │  possible_match  │         │        │
└────┴─────────────┴──────────┴──────┴────────────┴────────────┴──────────────────┴─────────┴────────┘
╭────────────────────────────────────────────────────── Summary ───────────────────────────────────────────────────────╮
│                                                                                                                      │
│  Total segments: 20                                                                                                  │
│  Total results (incl. alternatives): 37                                                                              │
│  Total duration: 52.5s                                                                                               │
│  Unique speakers: 2                                                                                                  │
│  Average matches per segment: 1.9                                                                                    │
│                                                                                                                      │
╰─────
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
Provide step-by-step analysis and explain the flow first.
Use visuals, diagrams, or tables when helpful.

Show full code for new files, then show full function code for new or updated functions.
Keep explanations simple and clear.

Write smart, flexible, reusable, maintainable, and robust code.
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
