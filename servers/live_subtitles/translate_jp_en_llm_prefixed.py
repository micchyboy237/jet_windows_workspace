"""
translate_jp_en_llm_prefixed.py
──────────────────────
Japanese → English translator using llama-cpp-python and the
shisa-v2.1-llama3.2-3b Q4_K_M GGUF model.
The Llama class is loaded once at module level (singleton) so the KV
cache persists across calls within a single process — ideal for the
progressive-subtitle demo.
"""
from __future__ import annotations
import json
import time
import os
import sys
from functools import lru_cache
from typing import Any
from llama_cpp import Llama
from sentence_matcher_ja import fuzzy_match_prefix_texts

MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators"
    r"\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
)

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.05
N_CTX = 8192

SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator. "
    "Translate the user's Japanese text into natural, fluent English. "
    "Output ONLY the English translation — no explanations, no romaji, "
    "no Japanese text, no extra commentary."
)


@lru_cache(maxsize=1)
def _get_llm() -> Llama:
    """
    Load the GGUF model once and cache it for the lifetime of the process.
    llama-cpp-python auto-detects the Llama-3.2 chat template from GGUF
    metadata, so we don't need to pass chat_format explicitly.
    """
    print(f"[translate_llm] Loading translator model from {MODEL_PATH} ...", flush=True)
    t0 = time.perf_counter()

    # Suppress stderr temporarily to hide the llama.cpp context warning
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=-1,       # offload all layers to GPU if available
            offload_kqv=True,      # keep KV cache on GPU for faster re-use
            verbose=False,         # suppress llama.cpp startup noise
        )
    finally:
        sys.stderr = old_stderr

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[translate_llm] Model loaded in {elapsed_ms:.0f} ms", flush=True)
    return llm


llm = _get_llm()


def translate_japanese_to_english(
    text: str,
    history: list[dict[str, str]] | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
) -> dict[str, Any]:
    """
    Translate a Japanese string to English.

    Parameters
    ----------
    text        : Japanese text to translate.
    history     : Optional list of prior {role, content} turns.
                  Pass the accumulated conversation for KV-cache reuse.
    max_tokens  : Maximum tokens to generate.
    temperature : Sampling temperature (Shisa recommends 0.6).
    top_p       : Nucleus sampling threshold.
    top_k       : Top-K sampling.
    repeat_penalty : Repetition penalty.

    Returns
    -------
    dict with keys:
        text              – English translation (str)
        tokens_evaluated  – prompt tokens processed this call (int)
        tokens_cached     – prompt tokens served from KV cache (int)
        tokens_generated  – new tokens generated (int)
        latency_ms        – wall-clock time in ms (float)
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": text})

    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        stream=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    translation: str = response["choices"][0]["message"]["content"].strip()
    usage = response.get("usage", {})
    tokens_evaluated: int = usage.get("prompt_tokens", 0)
    tokens_generated: int = usage.get("completion_tokens", 0)
    tokens_cached: int = getattr(llm, "_n_past_cached", 0)

    return {
        "text": translation,
        "tokens_evaluated": tokens_evaluated,
        "tokens_cached": tokens_cached,
        "tokens_generated": tokens_generated,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    from rich.rule import Rule

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]llama-server KV Cache Demo[/bold cyan]\n[dim]Progressive / Growing Subtitles[/dim]",
        border_style="cyan"
    ))

    history = []
    progressive_subtitles = [
        "今日は",  #  "Today," (incomplete)
        "今日はとても",  #  "is very" (still incomplete)
        "今日はとても疲れる。",  #  "tiring."
        # New sentence — no overlap → triggers reset
        "あなたのことが好きだよ。",  #  "I like you."
        # Another new sentence
        "早く行かないと電車に乗り遅れる！",  #  "Hurry up or we'll miss the train!"
    ]

    console.print(Rule("[bold]Progressive Growing Input Test[/bold]"))
    console.print()

    results = []  # store full data
    prev_ja = None
    prev_en = None

    for i, growing_text in enumerate(progressive_subtitles, start=1):
        ja_text = growing_text

        console.print(f"[bold yellow]\[Step {i}][/bold yellow] [white]{ja_text}[/white]")
        
        with Progress(SpinnerColumn(), TextColumn("[dim]Translating...[/dim]"), console=console, transient=True) as progress:
            progress.add_task("", total=None)
            
            result = translate_japanese_to_english(
                ja_text, 
                history=history, 
            )

            en_text = result["text"]

            fuzzy_texts_result = fuzzy_match_prefix_texts({
                "full_ja": ja_text,
                "full_en": result["text"],
                "prev_ja": prev_ja,
                "prev_en": prev_en,
            })

            new_ja = fuzzy_texts_result["new_ja"]
            new_en = fuzzy_texts_result["new_en"]
        
        en_text = fuzzy_texts_result["full_en"]
        is_continuation = fuzzy_texts_result["is_continuation"]

        console.print(f" [green]↳ EN:[/green] [italic]{en_text}[/italic]")
        if new_en:
            console.print(f" [bright_green]   Δ New:[/bright_green] [bold]{new_en}[/bold]")
        console.print()

        results.append((i, ja_text, en_text, new_en, is_continuation))

        # Update history and previous state
        history.append({"role": "user", "content": ja_text})
        history.append({"role": "assistant", "content": en_text})

        console.print(f"[bold magenta]History ({len(history)}):[/bold magenta]")
        console.print(
            json.dumps(history, indent=1, ensure_ascii=False),
            style="bright_blue on grey11",
        )
        console.print("\n")

        prev_ja = ja_text
        prev_en = en_text

    # === Final Results Table ===
    console.print(Rule("[bold]Final Results[/bold]"))
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        expand=True,
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Japanese Input", style="cyan", ratio=2)
    table.add_column("English Translation", style="green", ratio=3)
    table.add_column("New EN (incremental)", style="bright_green", ratio=3)
    table.add_column("Is Continuation", style="yellow", width=16, justify="center")

    for step, jp, en, new_en, is_continuation in results:
        table.add_row(
            str(step),
            jp,
            en,
            new_en if new_en else "[dim]— (first sentence)[/dim]",
            "✅" if is_continuation else "❌",
        )

    console.print(table)
