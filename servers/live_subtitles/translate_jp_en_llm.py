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

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf"

SYSTEM_MESSAGE = (
    "You are a professional, natural-sounding Japanese-to-English translator. "
    "Translate accurately while making the English sound fluent and idiomatic "
    "as if written by a native English speaker."
)

MODEL_SETTINGS = {
    "n_ctx": 1024,
    "n_gpu_layers": -1,
    "flash_attn": True,
    "logits_all": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "tokenizer_kwargs": {"add_bos_token": False},
    "n_batch": 128,
    "n_threads": 6,
    "n_threads_batch": 6,
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
}

# Recommended defaults for Japanese → English translation
TRANSLATION_DEFAULTS = {
    "temperature": 0.5,
    "top_p": 1.0,
    "min_p": 0.5,
    "repeat_penalty": 1.05,
    "max_tokens": 512,
    # For confidence scores
    "logprobs": 3,
}

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
                _llm_instance = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)
                console.print("[bold green]Gemma model loaded and ready[/bold green]")
    return _llm_instance

# ────────────────────────────────────────────────
# No eager global initialization! Use get_llm().

def translate_text(
    text: str,
    system_message: str = SYSTEM_MESSAGE,
    **generation_params,
) -> CreateCompletionResponse:
    """
    Translate Japanese text to natural English.

    Returns a complete CreateCompletionResponse compatible with OpenAI-style output.
    All numpy float types are converted to Python float recursively.
    """
    prompt = f"""{system_message}
Japanese: {text}
English:""".strip()

    generation_params: Dict[str, Any] = {
        "prompt": prompt,
        **TRANSLATION_DEFAULTS,
        **generation_params
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
        "本商品は30日経過後の返品・交換はお受けできませんのでご了承ください。",
        logprobs=3,               # ← ask for top-3 logprobs per token
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