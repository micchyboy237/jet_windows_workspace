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
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\evaluate_speaker_embeddings.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_evaluate_speaker_embeddings.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\evaluate_speaker_cluster.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_evaluate_speaker_cluster.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_overlap_aware_diarization.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\overlap_aware_diarization.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\embedding_model_factory.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\main.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\state.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing\speaker_labeling.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing\transcription.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\save_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\config.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\multi_speaker_labelling\main\_main_nemo_titanet.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\multi_speaker_labelling\nemo_titanet.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\embedding_model_factory.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.embedders.ini",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.llm.ini",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.rerankers.ini",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Start-Llama-Server-Reranker.ps1",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Start-Llama-Server-Llm.ps1",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Start-Llama-Server-Embedder.ps1",
    r"",
]

structure_include = [
    r"",
    # r"C:\Users\druiv\.cache\files\audio\speakers",
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

. C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Start-Llama-Server-Llm.ps1
0.00.096.046 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.098.943 I srv   load_models: Loaded 0 cached model presets
0.00.102.339 I srv   load_models: Loaded 38 custom model presets from C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.llm.ini
0.00.116.112 I srv    operator(): Available models (38) (*: custom preset)
0.00.116.120 I srv    operator():   * alma-ja-7b (aliases: alma-ja:7b)
0.00.116.123 I srv    operator():   * deepseek-coder-v2-lite (aliases: deepseek-coder-v2-lite:16b-ish)
0.00.116.123 I srv    operator():   * deepseek-r1-1.5b-q5kl (aliases: deepseek-r1:1.5b-q5kl)
0.00.116.124 I srv    operator():   * deepseek-r1-1.5b-q5km (aliases: deepseek-r1:1.5b-q5km)
0.00.116.125 I srv    operator():   * deepseek-r1-7b (aliases: deepseek-r1:7b)
0.00.116.126 I srv    operator():   * default
0.00.116.127 I srv    operator():   * dolphin-2.6-phi-2b (aliases: dolphin-2.6-phi:2b)
0.00.116.128 I srv    operator():   * elyza-jp-8b-iq2 (aliases: elyza-jp:8b-iq2)
0.00.116.128 I srv    operator():   * fiendish-llama-3b (aliases: fiendish-llama:3b)
0.00.116.129 I srv    operator():   * gemma-2-jpn-translate (aliases: gemma-2-jpn-translate:2b)
0.00.116.130 I srv    operator():   * gemma-3-4b (aliases: gemma-3:4b)
0.00.116.131 I srv    operator():   * gemma-3-vision-4b (aliases: gemma-3-vision:4b)
0.00.116.131 I srv    operator():   * gemma3-uncensored-1b (aliases: gemma3-uncensored:1b)
0.00.116.132 I srv    operator():   * impish-llama-4b (aliases: impish-llama:4b)
0.00.116.133 I srv    operator():   * lfm2-enjp-350m (aliases: lfm2-enjp:350m)
0.00.116.133 I srv    operator():   * llama-3.1-8b (aliases: llama-3.1:8b)
0.00.116.134 I srv    operator():   * llama-3.2-3b (aliases: llama-3.2:3b)
0.00.116.135 I srv    operator():   * llama-3.2-uncensored-3b (aliases: llama-3.2-uncensored:3b)
0.00.116.136 I srv    operator():   * ministral-3b (aliases: ministral:3b)
0.00.116.136 I srv    operator():   * mistral-nemo (aliases: mistral-nemo:12b-ish)
0.00.116.137 I srv    operator():   * nano-imp-1b (aliases: nano-imp:1b-q8)
0.00.116.138 I srv    operator():   * qwen2.5-7b (aliases: qwen2.5:7b)
0.00.116.138 I srv    operator():   * qwen2.5-vl-7b (aliases: qwen2.5-vl:7b)
0.00.116.139 I srv    operator():   * qwen3-4b (aliases: qwen3:4b)
0.00.116.139 I srv    operator():   * qwen3.5-0.8b (aliases: qwen3.5:0.8b)
0.00.116.140 I srv    operator():   * qwen3.5-0.8b-vision (aliases: qwen3.5:0.8b-vision)
0.00.116.141 I srv    operator():   * qwen3.5-2b (aliases: qwen3.5:2b)
0.00.116.141 I srv    operator():   * qwen3.5-2b-uncensored (aliases: qwen3.5-uncensored:2b)
0.00.116.142 I srv    operator():   * qwen3.5-2b-vision (aliases: qwen3.5:2b-vision)
0.00.116.142 I srv    operator():   * qwen3.5-4b (aliases: qwen3.5:4b)
0.00.116.143 I srv    operator():   * qwen3.5-4b-uncensored (aliases: qwen3.5-uncensored:4b)
0.00.116.144 I srv    operator():   * qwen3.5-4b-vision (aliases: qwen3.5:4b-vision)
0.00.116.144 I srv    operator():   * sarashina-3b (aliases: sarashina:3b)
0.00.116.145 I srv    operator():   * shisa-lfm2-1.2b (aliases: shisa-lfm2:1.2b)
0.00.116.146 I srv    operator():   * shisa-llama3.2-3b-iq4 (aliases: shisa-llama3.2:3b-iq4)
0.00.116.146 I srv    operator():   * shisa-llama3.2-3b-q4 (aliases: shisa-llama3.2:3b-q4)
0.00.116.147 I srv    operator():   * smollm3-3b (aliases: smollm3:3b)
0.00.116.147 I srv    operator():   * wizardlm-uncensored-7b (aliases: wizardlm-uncensored:7b)
0.00.116.158 I srv   load_models: (startup) loading model qwen3.5-2b-uncensored
0.00.117.059 I srv          load: spawning server instance with name=qwen3.5-2b-uncensored on port 58026
0.00.117.088 I srv          load: spawning server instance with args:
0.00.117.090 I srv          load:   C:\Users\druiv\.cache\llama-cpp-bin\llama-server.exe
0.00.117.090 I srv          load:   --chat-template-file
0.00.117.091 I srv          load:   C:\Users\druiv\.cache\llama.cpp\jinja-templates\Qwen3.5_2B.jinja
0.00.117.091 I srv          load:   --host
0.00.117.091 I srv          load:   127.0.0.1
0.00.117.092 I srv          load:   --min-p
0.00.117.092 I srv          load:   0.05
0.00.117.092 I srv          load:   --mlock
0.00.117.093 I srv          load:   --no-mmap
0.00.117.093 I srv          load:   --port
0.00.117.093 I srv          load:   58026
0.00.117.094 I srv          load:   --temperature
0.00.117.094 I srv          load:   0.75
0.00.117.094 I srv          load:   --top-k
0.00.117.095 I srv          load:   40
0.00.117.095 I srv          load:   --top-p
0.00.117.095 I srv          load:   0.92
0.00.117.096 I srv          load:   --alias
0.00.117.096 I srv          load:   qwen3.5-2b-uncensored
0.00.117.096 I srv          load:   --ctx-size
0.00.117.096 I srv          load:   4096
0.00.117.097 I srv          load:   --cont-batching
0.00.117.097 I srv          load:   --cache-type-k
0.00.117.097 I srv          load:   q8_0
0.00.117.098 I srv          load:   --cache-type-v
0.00.117.098 I srv          load:   q8_0
0.00.117.098 I srv          load:   --flash-attn
0.00.117.099 I srv          load:   off
0.00.117.099 I srv          load:   --model
0.00.117.099 I srv          load:   C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
0.00.117.100 I srv          load:   --n-gpu-layers
0.00.117.100 I srv          load:   999
0.00.117.100 I srv          load:   --parallel
0.00.117.101 I srv          load:   1
0.00.117.101 I srv          load:   --threads
0.00.117.101 I srv          load:   6
0.00.117.101 I srv          load:   --threads-batch
0.00.117.102 I srv          load:   6
0.00.122.566 W srv  llama_server: -----------------
0.00.122.566 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.122.567 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.122.568 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.122.568 W srv  llama_server: -----------------
0.00.122.576 W srv  llama_server: -----------------
0.00.122.576 W srv  llama_server: the following feature(s) are enabled:
0.00.122.576 W srv  llama_server:     router mode
0.00.122.577 W srv  llama_server: do not expose the server to untrusted environments
0.00.122.577 W srv  llama_server: -----------------
0.00.122.583 I srv  llama_server: starting server in router mode. models will be automatically loaded on-demand
0.00.127.063 I srv  llama_server: listening on http://0.0.0.0:8080
[58026] 0.00.169.651 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
[58026] 0.00.256.595 W srv  llama_server: -----------------
[58026] 0.00.256.604 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
[58026] 0.00.256.605 W srv  llama_server: this can be a security risk (cross-origin attacks)
[58026] 0.00.256.605 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
[58026] 0.00.256.606 W srv  llama_server: -----------------
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.0}}
[58026] 0.00.263.484 I srv    load_model: loading model 'C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
[58026] 0.00.800.120 E llama_init_from_model: V cache quantization requires flash_attn
[58026] 0.00.856.706 E common_fit_params: encountered an error while trying to fit params to free device memory: failed to create llama_context from model
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.0}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.2610487937927246}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.29925107955932617}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.339667409658432}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.37788230180740356}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.41636425256729126}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.4649154543876648}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.505331814289093}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.5467804670333862}}
[58026] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":1.0}}
[58026] 0.03.559.647 E llama_init_from_model: V cache quantization requires flash_attn
[58026] 0.03.559.675 E cmn  common_init_: failed to create context with model 'C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
[58026] 0.03.559.680 E cmn  common_init_: failed to create context with model 'C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
[58026] 0.03.559.719 E srv    load_model: failed to create_context with model 'C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
[58026] 0.03.559.731 I srv    operator(): operator(): cleaning up before exit...
[58026] 0.03.561.165 E srv  llama_server: exiting due to model loading error
0.03.976.464 I srv    operator(): instance name=qwen3.5-2b-uncensored exited with status 1



. C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Start-Llama-Server-Embedder.ps1
0.00.107.563 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.110.584 I srv   load_models: Loaded 0 cached model presets
0.00.111.823 I srv   load_models: Loaded 7 custom model presets from C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.embedders.ini
0.00.112.148 I srv    operator(): Available models (7) (*: custom preset)
0.00.112.154 I srv    operator():   * all-minilm-l12 (aliases: all-minilm:l12-q4)
0.00.112.155 I srv    operator():   * default
0.00.112.156 I srv    operator():   * embedding-gemma-300m (aliases: embedding-gemma:300m)
0.00.112.158 I srv    operator():   * nomic-embed-1.5 (aliases: nomic-embed:1.5)
0.00.112.159 I srv    operator():   * nomic-embed-2-moe (aliases: nomic-embed:2-moe)
0.00.112.159 I srv    operator():   * qwen3-embed-0.6b (aliases: qwen3-embed:0.6b)
0.00.112.160 I srv    operator():   * qwen3-embed-4b (aliases: qwen3-embed:4b-q5_0)
0.00.112.163 I srv   load_models: (startup) loading model nomic-embed-2-moe
0.00.112.966 I srv          load: spawning server instance with name=nomic-embed-2-moe on port 51164
0.00.112.997 I srv          load: spawning server instance with args:
0.00.112.999 I srv          load:   C:\Users\druiv\.cache\llama-cpp-bin\llama-server.exe
0.00.113.000 I srv          load:   --embeddings
0.00.113.000 I srv          load:   --host
0.00.113.001 I srv          load:   127.0.0.1
0.00.113.001 I srv          load:   --no-jinja
0.00.113.002 I srv          load:   --mlock
0.00.113.002 I srv          load:   --no-mmap
0.00.113.002 I srv          load:   --pooling
0.00.113.003 I srv          load:   mean
0.00.113.003 I srv          load:   --port
0.00.113.003 I srv          load:   51164
0.00.113.004 I srv          load:   --alias
0.00.113.004 I srv          load:   nomic-embed-2-moe
0.00.113.004 I srv          load:   --batch-size
0.00.113.005 I srv          load:   256
0.00.113.005 I srv          load:   --ctx-size
0.00.113.005 I srv          load:   2048
0.00.113.006 I srv          load:   --cont-batching
0.00.113.006 I srv          load:   --cache-type-k
0.00.113.006 I srv          load:   q8_0
0.00.113.007 I srv          load:   --cache-type-v
0.00.113.007 I srv          load:   q8_0
0.00.113.007 I srv          load:   --flash-attn
0.00.113.008 I srv          load:   off
0.00.113.008 I srv          load:   --model
0.00.113.008 I srv          load:   C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf
0.00.113.008 I srv          load:   --n-gpu-layers
0.00.113.009 I srv          load:   999
0.00.113.009 I srv          load:   --parallel
0.00.113.009 I srv          load:   1
0.00.113.010 I srv          load:   --threads
0.00.113.010 I srv          load:   6
0.00.113.010 I srv          load:   --threads-batch
0.00.113.011 I srv          load:   6
0.00.113.011 I srv          load:   --ubatch-size
0.00.113.011 I srv          load:   256
0.00.120.069 W srv  llama_server: -----------------
0.00.120.074 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.120.074 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.120.074 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.120.075 W srv  llama_server: -----------------
0.00.120.085 W srv  llama_server: -----------------
0.00.120.085 W srv  llama_server: the following feature(s) are enabled:
0.00.120.086 W srv  llama_server:     router mode
0.00.120.086 W srv  llama_server: do not expose the server to untrusted environments
0.00.120.086 W srv  llama_server: -----------------
0.00.120.093 I srv  llama_server: starting server in router mode. models will be automatically loaded on-demand
0.00.127.232 I srv  llama_server: listening on http://0.0.0.0:8081
[51164] 0.00.102.227 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
[51164] 0.00.205.320 W srv  llama_server: -----------------
[51164] 0.00.205.325 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
[51164] 0.00.205.326 W srv  llama_server: this can be a security risk (cross-origin attacks)
[51164] 0.00.205.326 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
[51164] 0.00.205.327 W srv  llama_server: -----------------
[51164] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.0}}
[51164] 0.00.221.564 I srv    load_model: loading model 'C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf'
[51164] 0.01.075.200 E llama_init_from_model: V cache quantization requires flash_attn
[51164] 0.01.255.907 E common_fit_params: encountered an error while trying to fit params to free device memory: failed to create llama_context from model
[51164] 0.01.966.274 W load: model vocab missing newline token, using special_pad_id instead
[51164] 0.02.096.388 W load: Mask token is missing in vocab, please reconvert model!
[51164] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.0}}
[51164] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.2582736909389496}}
[51164] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.5328506231307983}}
[51164] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.9999908804893494}}
[51164] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":1.0}}
[51164] 0.03.006.080 E llama_init_from_model: V cache quantization requires flash_attn
[51164] 0.03.006.087 E cmn  common_init_: failed to create context with model 'C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf'
[51164] 0.03.006.092 E cmn  common_init_: failed to create context with model 'C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf'
[51164] 0.03.006.103 E srv    load_model: failed to create_context with model 'C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf'
[51164] 0.03.006.115 I srv    operator(): operator(): cleaning up before exit...
[51164] 0.03.007.028 E srv  llama_server: exiting due to model loading error
0.03.525.146 I srv    operator(): instance name=nomic-embed-2-moe exited with status 1




. C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Start-Llama-Server-Reranker.ps1
0.00.095.288 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.098.231 I srv   load_models: Loaded 0 cached model presets
0.00.098.981 I srv   load_models: Loaded 5 custom model presets from C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.rerankers.ini
0.00.099.196 I srv    operator(): Available models (5) (*: custom preset)
0.00.099.202 I srv    operator():   * bge-rerank-large (aliases: bge-rerank:large)
0.00.099.202 I srv    operator():   * bge-rerank-v2-m3 (aliases: bge-rerank:v2-m3)
0.00.099.203 I srv    operator():   * default
0.00.099.204 I srv    operator():   * qwen3-rerank-0.6b (aliases: qwen3-rerank:0.6b)
0.00.099.205 I srv    operator():   * qwen3-rerank-4b (aliases: qwen3-rerank:4b)
0.00.099.208 I srv   load_models: (startup) loading model bge-rerank-v2-m3
0.00.099.885 I srv          load: spawning server instance with name=bge-rerank-v2-m3 on port 51168
0.00.099.906 I srv          load: spawning server instance with args:
0.00.099.907 I srv          load:   C:\Users\druiv\.cache\llama-cpp-bin\llama-server.exe
0.00.099.908 I srv          load:   --host
0.00.099.908 I srv          load:   127.0.0.1
0.00.099.908 I srv          load:   --no-jinja
0.00.099.908 I srv          load:   --mlock
0.00.099.909 I srv          load:   --no-mmap
0.00.099.909 I srv          load:   --pooling
0.00.099.909 I srv          load:   rank
0.00.099.910 I srv          load:   --port
0.00.099.910 I srv          load:   51168
0.00.099.910 I srv          load:   --reranking
0.00.099.911 I srv          load:   --alias
0.00.099.911 I srv          load:   bge-rerank-v2-m3
0.00.099.911 I srv          load:   --batch-size
0.00.099.911 I srv          load:   64
0.00.099.912 I srv          load:   --ctx-size
0.00.099.912 I srv          load:   1024
0.00.099.912 I srv          load:   --cont-batching
0.00.099.913 I srv          load:   --cache-type-k
0.00.099.913 I srv          load:   q8_0
0.00.099.913 I srv          load:   --cache-type-v
0.00.099.913 I srv          load:   q8_0
0.00.099.914 I srv          load:   --flash-attn
0.00.099.914 I srv          load:   off
0.00.099.914 I srv          load:   --model
0.00.099.915 I srv          load:   C:\Users\druiv\.cache\llama.cpp\rerankers\bge-reranker-v2-m3-Q4_K_M.gguf
0.00.099.915 I srv          load:   --n-gpu-layers
0.00.099.915 I srv          load:   999
0.00.099.915 I srv          load:   --parallel
0.00.099.915 I srv          load:   1
0.00.099.916 I srv          load:   --threads
0.00.099.916 I srv          load:   6
0.00.099.916 I srv          load:   --threads-batch
0.00.099.917 I srv          load:   6
0.00.099.917 I srv          load:   --ubatch-size
0.00.099.917 I srv          load:   64
0.00.104.278 W srv  llama_server: -----------------
0.00.104.281 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.104.282 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.104.282 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.104.282 W srv  llama_server: -----------------
0.00.104.292 W srv  llama_server: -----------------
0.00.104.293 W srv  llama_server: the following feature(s) are enabled:
0.00.104.293 W srv  llama_server:     router mode
0.00.104.294 W srv  llama_server: do not expose the server to untrusted environments
0.00.104.294 W srv  llama_server: -----------------
0.00.104.299 I srv  llama_server: starting server in router mode. models will be automatically loaded on-demand
0.00.115.780 I srv  llama_server: listening on http://0.0.0.0:8082
[51168] 0.00.095.761 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
[51168] 0.00.185.475 W srv  llama_server: -----------------
[51168] 0.00.185.481 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
[51168] 0.00.185.481 W srv  llama_server: this can be a security risk (cross-origin attacks)
[51168] 0.00.185.481 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
[51168] 0.00.185.482 W srv  llama_server: -----------------
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.0}}
[51168] 0.00.196.341 I srv    load_model: loading model 'C:\Users\druiv\.cache\llama.cpp\rerankers\bge-reranker-v2-m3-Q4_K_M.gguf'
[51168] 0.01.065.877 E llama_init_from_model: V cache quantization requires flash_attn
[51168] 0.01.251.325 E common_fit_params: encountered an error while trying to fit params to free device memory: failed to create llama_context from model
[51168] 0.01.992.614 W load: model vocab missing newline token, using special_pad_id instead
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.0}}
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.13433636724948883}}
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.2774430811405182}}
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.4300605058670044}}
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":0.922234296798706}}
[51168] cmd_child_to_router:state:{"state":"loading","payload":{"stages":["text_model"],"current":"text_model","value":1.0}}
[51168] 0.03.450.734 E llama_init_from_model: V cache quantization requires flash_attn
[51168] 0.03.450.748 E cmn  common_init_: failed to create context with model 'C:\Users\druiv\.cache\llama.cpp\rerankers\bge-reranker-v2-m3-Q4_K_M.gguf'
[51168] 0.03.450.753 E cmn  common_init_: failed to create context with model 'C:\Users\druiv\.cache\llama.cpp\rerankers\bge-reranker-v2-m3-Q4_K_M.gguf'
[51168] 0.03.450.761 E srv    load_model: failed to create_context with model 'C:\Users\druiv\.cache\llama.cpp\rerankers\bge-reranker-v2-m3-Q4_K_M.gguf'
[51168] 0.03.450.775 I srv    operator(): operator(): cleaning up before exit...
[51168] 0.03.451.712 E srv  llama_server: exiting due to model loading error
0.03.908.457 I srv    operator(): instance name=bge-rerank-v2-m3 exited with status 1

""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:

- Browse when beneficial or requested.
- Always use easy to understand terms.
- Dont use memory from previous artifacts.

My device:

- Mac M1 for coding work
- Windows 11 for local servers with below specs:
  - CPU: AMD Ryzen 5 3600
  - GPU: GTX 1660
  - RAM: 16GB dual sticks

When coding:

- Provide step-by-step analysis and trace the flows first.
- Use visuals, diagrams, or tables when helpful.
- For new files, classes, methods, or functions: show the full code.
- For updates to existing files: show only the changed sections with context. Never output the full file unless it's small.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Ask for clarifications before giving detailed answers if needed.
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
    parser.add_argument(
        "-q",
        "--query-only",
        action="store_true",
        default=False,
        help="Include only the query message and files, omitting system and instructions",
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
    query_only = args.query_only

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
    # Build the clipboard content parts
    clipboard_content_parts = []
    if not query_only:
        if system_message:
            clipboard_content_parts.append(f"<system>\n{system_message}\n</system>")
    # Query should come before instructions
    clipboard_content_parts.append(f"<query>\n{query_message}\n</query>")
    if not query_only:
        if instructions_message:
            clipboard_content_parts.append(
                f"<instructions>\n{instructions_message}\n</instructions>"
            )
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
