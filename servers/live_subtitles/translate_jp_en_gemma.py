from typing import Literal, Dict, Any, Union
import time
import uuid
from threading import Lock

from rich.console import Console
from rich.markdown import Markdown
from llama_cpp import Llama
from llama_cpp.llama_types import CreateCompletionResponse, CompletionChoice, CompletionUsage
import numpy as np

console = Console()

# ────────────────────────────────────────────────
# Config – adjust paths & settings to match your setup
# ────────────────────────────────────────────────

MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators\gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"
)

CTX_SIZE        = 2048
GPU_LAYERS      = -1
CACHE_TYPE_K    = "q8_0"
CACHE_TYPE_V    = "q8_0"
TOK_ADD_BOS     = False

def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert numpy scalar types (float32, float64, int64, etc.) 
    to native Python float/int.
    Leaves other types unchanged.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()           # converts to Python float or int
    elif isinstance(obj, np.ndarray):
        return obj.tolist()         # converts array → list (with recursion)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_to_python(item) for item in obj)
    else:
        return obj

# ────────────────────────────────────────────────
# Lazy LLM initialization (thread-safe)
# ────────────────────────────────────────────────

_llm_instance: Llama | None = None
_llm_lock = Lock()

def get_llm() -> Llama:
    """
    Lazily initialize and return the global Llama instance.
    Thread-safe – safe to call concurrently from multiple threads.
    """
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:  # double-checked locking
                console.print("[bold yellow]Initializing Gemma translator model...[/bold yellow]")
                _llm_instance = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=CTX_SIZE,
                    n_gpu_layers=GPU_LAYERS,
                    flash_attn=True,
                    logits_all=True,           # required for logprobs
                    cache_type_k=CACHE_TYPE_K,
                    cache_type_v=CACHE_TYPE_V,
                    verbose=False,
                    tokenizer_kwargs={"add_bos_token": TOK_ADD_BOS},
                )
                console.print("[bold green]Gemma model loaded and ready[/bold green]")
    return _llm_instance

# ────────────────────────────────────────────────
# No eager global initialization! Use get_llm().

def translate_text(
    text: str,
    max_tokens: int = 512,
    logprobs: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    stop: Union[str, list[str], None] = ["\n\n", "English:", "</s>"],
    echo: bool = False,
) -> CreateCompletionResponse:
    """
    Translate Japanese text to natural English using Gemma-2-2b-jpn-it.

    Returns a complete CreateCompletionResponse compatible with OpenAI-style output.
    All numpy float types are converted to Python float recursively.
    """
    prompt = f"""Translate the following Japanese text to natural English.
Japanese:
{text}
English:""".strip()

    generation_params: Dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "logprobs": logprobs,
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "echo": echo,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "mirostat_mode": 0,
        "seed": None,
        "stream": False,
    }

    raw_response = get_llm()(**generation_params)

    # ────────────────────────────────────────────────
    # Final assembly + numpy cleanup
    # ────────────────────────────────────────────────
    response: CreateCompletionResponse = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": get_llm().model_path.rsplit("/", 1)[-1] if get_llm().model_path else "gemma-2-2b-jpn-it",
        "choices": raw_response["choices"],
        "usage": raw_response.get("usage") or CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        ),
    }

    # Convert all numpy scalars → Python float/int recursively
    response = convert_numpy_to_python(response)

    return response


if __name__ == "__main__":
    result = translate_text(
        "こんにちは、お元気ですか？今日はとても良い天気ですね。",
        max_tokens=64,
        logprobs=3,               # ← ask for top-3 logprobs per token
        temperature=0.0,          # deterministic → easier to inspect
    )
    console.print(Markdown(result["choices"][0]["text"].strip()))

    from rich.pretty import pprint

    print("\n[bold cyan]Translation Result:[/bold cyan]")
    pprint(result, expand_all=True)

    if (logprobs := result["choices"][0].get("logprobs")):
        print("\nFirst few tokens + top logprobs:")
        for token, lp, top_lp in zip(
            logprobs["tokens"][:8],
            logprobs["token_logprobs"][:8],
            logprobs["top_logprobs"][:8]
        ):
            print(f"{token:12} {lp:8.3f}   | top: {top_lp}")
    else:
        print("Still no logprobs → check logits_all=True in Llama()")