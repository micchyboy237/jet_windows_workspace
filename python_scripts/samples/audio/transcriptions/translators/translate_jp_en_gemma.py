from typing import Literal, Dict, Any, Union
import time
import uuid

from rich.console import Console
from rich.markdown import Markdown
from llama_cpp import Llama
from llama_cpp.llama_types import CreateCompletionResponse, CompletionChoice, CompletionUsage

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

# ────────────────────────────────────────────────
# Initialize model
# ────────────────────────────────────────────────

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_gpu_layers=GPU_LAYERS,
    flash_attn=True,
    logits_all=True,           # ← required for logprobs to be populated
    cache_type_k=CACHE_TYPE_K,
    cache_type_v=CACHE_TYPE_V,
    verbose=False,
    tokenizer_kwargs={"add_bos_token": TOK_ADD_BOS},
)

def translate_text(
    text: str,
    max_tokens: int = 512,
    logprobs: int | None = None,           # ← new optional param
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
    """
    prompt = f"""Translate the following Japanese text to natural English.
Japanese:
{text}
English:""".strip()

    # Optional: you can customize these further per call
    generation_params: Dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "logprobs": logprobs,               # ← pass through
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "echo": echo,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "mirostat_mode": 0,
        "seed": None,
        "stream": False,
    }

    raw_response = llm(**generation_params)

    response: CreateCompletionResponse = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": llm.model_path.rsplit("/", 1)[-1] if llm.model_path else "gemma-2-2b-jpn-it",
        "choices": raw_response["choices"],
        "usage": raw_response.get("usage") or CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        ),
    }

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