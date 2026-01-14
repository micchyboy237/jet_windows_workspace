import time
import uuid
import json
from typing import Dict, Any, Union, Iterator, List, Tuple, Optional
from threading import Lock

from rich import print
from rich.console import Console
from rich.live import Live
from rich.text import Text

from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)

console = Console()

# ────────────────────────────────────────────────
# Configuration – easy to tweak
# ────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf"

MODEL_SETTINGS = {
    "n_ctx": 1024,
    "n_gpu_layers": -1,
    "flash_attn": True,
    "logits_all": True,
    "type_k": 8,
    "type_v": 8,
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
}

# ────────────────────────────────────────────────
# Lazy + thread-safe model loading
# ────────────────────────────────────────────────
_llm: Llama | None = None
_llm_lock = Lock()


def get_llm() -> Llama:
    """Lazily initialize and return the Llama instance (thread-safe)"""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                console.print("[bold yellow]Loading LFM2-350M translator...[/bold yellow]")
                _llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)
                console.print("[bold green]Translation LLM ready ✓[/bold green]")
    return _llm


# ────────────────────────────────────────────────
# Main translation interface
# ────────────────────────────────────────────────
def translate_japanese_to_english(
    text: str,
    stream: bool = False,
    **generation_params,
) -> Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]]:
    """
    High-level Japanese → natural English translation using chat format.

    Uses llama_cpp's create_chat_completion under the hood.

    Args:
        text: Japanese text to translate
        temperature: Controls randomness (0.0 = deterministic, ~0.6-0.8 natural)
        top_p: Nucleus sampling
        min_p: Minimum probability sampling (very helpful on smaller models)
        repeat_penalty: Penalizes repetitions
        max_tokens: Maximum output length
        stream: Whether to return a streaming iterator
        stop: Optional stop strings

    Returns:
        Full response object or stream iterator depending on `stream` flag

    Recommended usage:
        # Blocking
        result = translate_japanese_to_english("こんにちは！", stream=False)
        print(result["choices"][0]["message"]["content"])

        # Streaming (beautiful console)
        for chunk in translate_japanese_to_english("こんにちは！", stream=True):
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
    """
    messages: List[ChatCompletionRequestMessage] = [
        {
            "role": "system",
            "content": (
                "You are a professional, natural-sounding Japanese-to-English translator. "
                "Translate accurately while making the English sound fluent and idiomatic "
                "as if written by a native English speaker."
            ),
        },
        {"role": "user", "content": text.strip()},
    ]

    params: Dict[str, Any] = {
        "messages": messages,
        "stream": stream,
        **TRANSLATION_DEFAULTS,
        **generation_params
    }

    llm = get_llm()
    return llm.create_chat_completion(**params)


def translate_text(text: str, logprobs: Optional[int] = None, **generation_params) -> dict:
    """Translate with beautiful real-time streaming display using rich"""
    full_text = ""

    _generation_params: Dict[str, Any] = {
        **TRANSLATION_DEFAULTS,
        **generation_params
    }

    if logprobs:
        _generation_params["logprobs"] = True
        _generation_params["top_logprobs"] = logprobs

    stream = translate_japanese_to_english(
        text=text,
        stream=True,
        **_generation_params,
    )

    llm = get_llm()
    all_logprobs: List[Tuple[str, float, List[dict]]] = []
    try:
        role: str = None
        finish_reason: str = None
    
        with Live(auto_refresh=False) as live:
            for chunk in stream:
                if "choices" not in chunk or not chunk["choices"]:
                    continue

                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                logprobs = choice["logprobs"] or {}
                logprobs_content = logprobs.get("content", [])
                logprobs_tokens = [(l["token"], l["logprob"], l["top_logprobs"]) for l in logprobs_content]

                if "role" in delta:
                    role = delta["role"]

                content = delta.get("content", "")

                if content:
                    full_text += content
                    live.update(Text(full_text))
                    live.refresh()

                    all_logprobs.extend(logprobs_tokens)

                if choice["finish_reason"]:
                    finish_reason = choice["finish_reason"]
    finally:
        # IMPORTANT:
        # Clear KV cache + decoding state after a streaming completion.
        # Without this, subsequent calls can hit llama_decode returned -1.
        llm.reset()

    return {
        "role": role,
        "text": full_text,
        "finish_reason": finish_reason,
        "logprobs": all_logprobs,
    }


# ────────────────────────────────────────────────
# Quick demo
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # logprobs = None
    logprobs = 5
    examples = [
        "本商品は30日経過後の返品・交換はお受けできませんのでご了承ください。",
    ]

    for i, jp_text in enumerate(examples, 1):
        console.rule(f"Example {i}")
        console.print("[dim]Japanese:[/dim]")
        console.print(jp_text, style="italic cyan")
        console.print()

        console.print("[bold green]English (streaming):[/bold green]")
        result = translate_text(jp_text, logprobs=logprobs)
        full_text = result.pop("text")
        all_logprobs = result.pop("logprobs")

        from rich.pretty import pprint

        print(f"\n[bold cyan]Logprobs {i}:[/bold cyan]")
        pprint(all_logprobs)

        print(f"\n[bold cyan]Meta {i}:[/bold cyan]")
        pprint(result, expand_all=True)

        print(f"\n[bold cyan]Translation {i}:[/bold cyan]")
        pprint(full_text, expand_all=True)

    print()
