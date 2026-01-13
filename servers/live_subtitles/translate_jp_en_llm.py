from typing import Dict, Any, Union, Iterator, List
import time
import uuid
from threading import Lock

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
    "n_ctx": 4096,
    "n_gpu_layers": 20,
    "flash_attn": False,
    "logits_all": False,
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
    "temperature": 0.6,
    "top_p": 0.95,
    "min_p": 0.05,
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
                console.print("[bold green]Model ready ✓[/bold green]")
    return _llm


# ────────────────────────────────────────────────
# Main translation interface
# ────────────────────────────────────────────────
def translate_japanese_to_english(
    text: str,
    temperature: float = TRANSLATION_DEFAULTS["temperature"],
    top_p: float = TRANSLATION_DEFAULTS["top_p"],
    min_p: float = TRANSLATION_DEFAULTS["min_p"],
    repeat_penalty: float = TRANSLATION_DEFAULTS["repeat_penalty"],
    max_tokens: int = TRANSLATION_DEFAULTS["max_tokens"],
    stream: bool = False,
    stop: Union[str, List[str], None] = None,
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
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if stop is not None:
        params["stop"] = stop

    llm = get_llm()
    return llm.create_chat_completion(**params)


def translate_and_print_streaming(text: str) -> str:
    """Translate with beautiful real-time streaming display using rich"""
    full_text = ""

    stream = translate_japanese_to_english(
        text=text,
        temperature=TRANSLATION_DEFAULTS["temperature"],
        stream=True,
    )

    llm = get_llm()

    try:
        with Live(auto_refresh=False) as live:
            for chunk in stream:
                if "choices" not in chunk or not chunk["choices"]:
                    continue

                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    full_text += content
                    live.update(Text(full_text))
                    live.refresh()
    finally:
        # IMPORTANT:
        # Clear KV cache + decoding state after a streaming completion.
        # Without this, subsequent calls can hit llama_decode returned -1.
        llm.reset()

    return full_text.strip()


# ────────────────────────────────────────────────
# Quick demo
# ────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        "本商品は30日経過後の返品・交換はお受けできませんのでご了承ください。",
        "この度はご注文いただき誠にありがとうございます。明日発送予定でございます。",
        "大変申し訳ございませんが、現在在庫切れとなっております。",
    ]

    for i, jp_text in enumerate(examples, 1):
        console.rule(f"Example {i}")
        console.print("[dim]Japanese:[/dim]")
        console.print(jp_text, style="italic cyan")
        console.print()

        console.print("[bold green]English (streaming):[/bold green]")
        english = translate_and_print_streaming(jp_text)

        console.print("\n" + "─" * 70)
        console.print(english)
        console.print()