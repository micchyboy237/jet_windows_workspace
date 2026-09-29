import argparse
import json
import os

import tiktoken
from rich.console import Console
from tqdm import tqdm

from _utils_copy_for_prompt import (
    clean_content,
    clean_newlines,
    copy_to_clipboard,
    find_files,
    format_file_structure,
    remove_parent_paths,
)

# Initialize Rich Console
console = Console()

exclude_files = [
    "**/.git/",
    "**/.gitignore",
    "**/.DS_Store",
    "**/_copy*.py",
    "**/__pycache__/",
    "**/.pytest_cache/",
    "**/node_modules/",
    "**/*lock.json",
    "**/*.lock",
    "**/public/",
    "**/mocks/",
    "**/dream/",
    "**/jupyter/",
    "**/*.png",
    "**/*.wav",
    "**/*.svg",
    "**/*.pyc",
    "**/_git_stats.json",
    "**/stats_results/",
    # "**/_*",
    # "**/.cache/",
    # "**/.venv/",
    "**/generated/",
    "**/.env",
    # "**/.*",
    # Custom
    # "**/*.sh"
    # "**/__init__.py",
    # "**/*.md",
    # "**/tests/",
    # "**/pretrained_models/",
    "**/hls.min.js",
]
include_files = [
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\jet_apps\sparse_vectors",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\jet_apps\sparse_vectors\demo_opensearch_neural_sparse.py",
    r"C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\sparse_encoder\model.py",
    r"",
]

structure_include = [
    # "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/unstructured/example-docs",
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False
INCLUDE_FILE_STRUCTURE = False

COMPRESSION_MODEL = "gpt-4o"
TOKEN_BUDGET = 8000

DEFAULT_QUERY_MESSAGE = r"""
Search the web for code exaxmples for "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte".
Analyze the issues below then search the web for forums on how to resolve.

python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\jet_apps\sparse_vectors\demo_opensearch_neural_sparse.py
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████| 140/140 [00:01<00:00, 138.48it/s]
[transformers] NewForMaskedLM LOAD REPORT from: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
Key                    | Status  |
-----------------------+---------+-
lm_head.decoder.weight | MISSING |

Notes:
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [0,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [1,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [2,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [3,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [4,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [5,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [6,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [7,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [8,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [9,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [10,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [11,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [12,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [13,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [14,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [15,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [16,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [17,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [18,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [19,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [20,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [21,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [22,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [23,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [24,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [25,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [26,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [27,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [28,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [29,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [30,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\IndexKernelUtils.cu:16: block: [0,0,0], thread: [31,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\jet_apps\sparse_vectors\demo_opensearch_neural_sparse.py", line 51, in <module>
    main()
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\jet_apps\sparse_vectors\demo_opensearch_neural_sparse.py", line 25, in main
    doc_tensor = model.encode_document(doc_text)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\util\decorators.py", line 46, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\sparse_encoder\model.py", line 486, in encode_document
    return self.encode(
           ^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\util\decorators.py", line 46, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\sparse_encoder\model.py", line 763, in encode
    embeddings = self(features, **forward_kwargs)["sentence_embedding"]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\base\model.py", line 569, in forward
    input = module(input, **module_kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\base\modules\router.py", line 504, in forward
    features = module(features, **module_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\sentence_transformers\base\modules\transformer.py", line 1650, in forward
    model_output = model_method(**filtered_kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\huggingface\modules\transformers_modules\Alibaba_hyphen_NLP\new_hyphen_impl\40ced75c3017eb27626c9d4ea981bde21a2662f4\modeling.py", line 1039, in forward
    outputs = self.new(
              ^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\huggingface\modules\transformers_modules\Alibaba_hyphen_NLP\new_hyphen_impl\40ced75c3017eb27626c9d4ea981bde21a2662f4\modeling.py", line 901, in forward
    (embedding_output, attention_mask, rope_embeds, length) = self.embeddings(
                                                              ^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\huggingface\modules\transformers_modules\Alibaba_hyphen_NLP\new_hyphen_impl\40ced75c3017eb27626c9d4ea981bde21a2662f4\modeling.py", line 415, in forward
    embeddings = self.LayerNorm(embeddings)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\normalization.py", line 229, in forward
    return F.layer_norm(
           ^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\functional.py", line 2935, in layer_norm
    return torch.layer_norm(
           ^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:

- Always search the web first when requested or if it's beneficial.
- Use easy to understand terms.

My device:

- Mac M1 for coding work
- Windows 11 for local servers with below specs:
  - CPU: AMD Ryzen 5 3600
  - GPU: GTX 1660
  - RAM: 16GB

When coding:

- First provide step-by-step analysis with summarized flows as separate lists and flow diagrams.
- For new files, classes, methods, or functions: show the full code.
- For updates to existing files: show only the changed sections with context. Never output the full file unless it's small.
- Write smart, reusable, maintainable, optimal, robust, and minimal code.
- If possible, avoid business specific logic to keep code generic.
- Ask for clarifications before giving detailed answers if something is confusing or need confirmation.
- Include logs that cover all steps.
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
    console.print("[bold green]Running _copy_for_prompt.py[/bold green]")
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
    console.print()
    context_files = find_files(
        base_dir, include, exclude, include_content, exclude_content, case_sensitive
    )
    console.print()
    console.print(f"[bold]Include patterns:[/bold] {include}")
    console.print(f"[bold]Exclude patterns:[/bold] {exclude}")
    console.print(f"[bold]Include content patterns:[/bold] {include_content}")
    console.print(f"[bold]Exclude content patterns:[/bold] {exclude_content}")
    console.print(f"[bold]Case sensitive:[/bold] {case_sensitive}")
    console.print(f"[bold]Filenames only:[/bold] {filenames_only}")
    console.print(f"[bold]Compress enabled:[/bold] {compress_enabled}")
    console.print(
        f"\n[bold]Found files ({len(context_files)}):[/bold]\n{json.dumps(context_files, indent=2)}"
    )
    console.print()

    # Initialize the clipboard content
    clipboard_content = ""
    if not context_files:
        console.print(
            "[yellow]No context files found matching the given patterns.[/yellow]"
        )
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
        from headroom import compress

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
        # Log compression stats using rich console
        console.print(f"[bold cyan]Tokens before:[/bold cyan] {result.tokens_before:,}")
        console.print(f"[bold cyan]Tokens after:[/bold cyan] {result.tokens_after:,}")
        console.print(
            f"[bold green]Tokens saved:[/bold green] {result.tokens_saved:,} ({result.compression_ratio:.1%})"
        )
        console.print(
            f"[bold magenta]Transforms applied:[/bold magenta] {result.transforms_applied}"
        )
    else:
        console.print("[dim]Compression skipped (use -c or --compress to enable)[/dim]")

    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)

    # Print the copied content character count
    console.print(f"[bold blue]Prompt Char Count:[/bold blue] {len(clipboard_content)}")
    console.print(
        f"[bold blue]Tokens Count (gpt-4o):[/bold blue] {count_tokens(clipboard_content)}"
    )
    # Newline
    console.print()


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
