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

SYSTEM_PROMPT = """
You are a professional, natural-sounding Japanese-to-English translator. Translate accurately while making the English sound fluent and idiomatic as if written by a native English speaker.
""".strip()

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
    ja_text: str,
    enable_scoring: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    stream: bool = False,
    **generation_params,
) -> Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]]:
    """
    High-level Japanese → natural English translation using chat format.

    Uses llama_cpp's create_chat_completion under the hood.

    Args:
        ja_text: Japanese text to translate
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
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": ja_text.strip()},
    ]

    params: Dict[str, Any] = {
        "messages": messages,
        "stream": stream,
        **TRANSLATION_DEFAULTS,
        **generation_params
    }

    llm = get_llm()
    response = llm.create_chat_completion(**params)
    en_text = response["choices"][0]["message"]["content"].strip()

    return {
        "text": en_text,
        "log_prob": None,
        "confidence": None,
        "quality": None,
    }


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


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path
    from rich import box
    from rich.console import Console
    from rich.panel import Panel

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console = Console()

    parser = argparse.ArgumentParser(
        description="Japanese → English subtitle translator using llama.cpp"
    )
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        default="""
恥ずかしい…見ないでください…
んっ…そこ、弱いんです…
はぁ…はぁ…気持ちいい…
もう…ダメかも…頭おかしくなりそう…
お願い…もっと激しくして…壊して…！
あぁんっ！すごい…奥まで届いてる…♡
出さないで…まだ中にいてて…
        """.strip(),
        help="Japanese text to translate (multi-line ok)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRANSLATION_DEFAULTS["max_tokens"],
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TRANSLATION_DEFAULTS["temperature"],
        help="Sampling temperature",
    )
    parser.add_argument(
        "--scoring",
        "--enable-scoring",
        action="store_true",
        default=False,
        help="Enable logprobs and confidence scoring (slower)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        default=False,
        help="Disable rich console formatting (plain output)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # You can override MODEL_PATH here if you want to support --model
    # MODEL_PATH = args.model   # ← uncomment if you want to use CLI model path
    # llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)  # reload if needed
    # -------------------------------------------------------------------------

    if args.no_rich:
        # Very simple plain output mode
        result = translate_japanese_to_english(
            ja_text=args.text,
            max_tokens=args.max_tokens,
            enable_scoring=args.scoring,
            temperature=args.temperature,
            history=None,
        )
        print("Japanese:")
        print(args.text)
        print("\nEnglish:")
        print(result["text"])
        if args.scoring:
            print(f"\nlogprob   : {result['log_prob']}")
            print(f"confidence : {result['confidence']}")
            print(f"quality    : {result['quality']}")
    else:
        console.rule("Japanese → English Translation", style="bold cyan")

        console.print(
            Panel(
                args.text,
                title="[bold magenta]Japanese Input[/]",
                border_style="magenta",
                padding=(1, 2),
                expand=False,
                box=box.ROUNDED,
            )
        )

        console.print(
            f"[dim]Translating with temperature={args.temperature}  "
            f"max_tokens={args.max_tokens}  scoring={args.scoring}[/]"
        )

        result = translate_japanese_to_english(
            ja_text=args.text,
            max_tokens=args.max_tokens,
            enable_scoring=args.scoring,
            temperature=args.temperature,
            history=None,
        )

        metrics = ""
        if args.scoring:
            metrics = (
                f"\n\n[dim]Metrics:[/] logprob = {result['log_prob']}   "
                f"confidence = {result['confidence']}   quality = {result['quality']}"
            )

        console.print(
            Panel(
                result["text"] + metrics,
                title="[bold green]English Translation[/]",
                border_style="green",
                padding=(1, 2),
                expand=False,
                box=box.DOUBLE,
            )
        )

    result_json_path = OUTPUT_DIR / "translation_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]Saved JSON result to:[/bold green] {result_json_path}")
