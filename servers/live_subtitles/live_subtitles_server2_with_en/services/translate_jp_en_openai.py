"""
translate_jp_en_openai.py
──────────────────────
Japanese → English translator using an OpenAI-compatible llama.cpp server.
Uses streaming chat completions with real-time console output and logit bias.
"""
from __future__ import annotations
import json
import sys
import time
from functools import lru_cache
from typing import Any, Dict

from openai import OpenAI

# ── Server configuration ──────────────────────────────────────────────
LLM_SERVER_BASE_HOST = "http://localhost:8080"
LLM_SERVER_BASE_URL = f"{LLM_SERVER_BASE_HOST}/v1"
LLM_MODEL_NAME = "qwen3.5-uncensored:2b"

# ── Generation defaults ───────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_FREQUENCY_PENALTY = 0.05  # Replaces repeat_penalty from original
DEFAULT_PRESENCE_PENALTY = 0.0

# ── System prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Japanese-to-English translator. "
    "Translate the user's Japanese text into natural, fluent English. "
    "Translate all content faithfully, including explicit or mature text. "
    "Output ONLY the English translation — no explanations, no romaji, "
    "no Japanese text, no extra commentary."
)

# ── Logit bias strings (matching original implementation) ─────────────
PENALIZE_STRINGS = [
    "申", "訳", "あり", "ませ", "でき", "ご", "ざ", "い",
    "I cannot", "I can't", "I'm sorry", "I am sorry", "I apologize",
    "unable to", "not able to", "cannot translate", "can't translate",
    "inappropriate", "offensive", "I won't",
]

BOOST_STRINGS = [
    "The", "This", "It", "I", "He", "She", "They", "We",
    "A", "An", "In", "On", "At", "With",
]

PENALTY_VALUE = -20.0
BOOST_VALUE = 5.0


def _tokenize_text(text: str) -> list[int]:
    """
    Tokenize a text string using the llama.cpp server's /tokenize endpoint.
    
    POST /tokenize with {"content": text}
    Returns: {"tokens": [123, 456, 789]}
    
    Returns a list of token IDs.
    """
    try:
        import requests
        response = requests.post(
            f"{LLM_SERVER_BASE_HOST}/tokenize",
            json={
                "content": text,
                "model": LLM_MODEL_NAME,
                "add_special": False,
                "parse_special": True,
                "with_pieces": False,
            },
        )
        if response.status_code == 200:
            data = response.json()
            tokens = data.get("tokens", [])
            # tokens can be either:
            # - [123, 456, 789] (with_pieces=false, default)
            # - [{"id": 123, "piece": "Hello"}, ...] (with_pieces=true)
            if tokens and isinstance(tokens[0], dict):
                return [t["id"] for t in tokens]
            return tokens
        else:
            print(
                f"[translate_openai] Tokenize failed for '{text}': HTTP {response.status_code}",
                flush=True,
            )
            return []
    except Exception as e:
        print(
            f"[translate_openai] Tokenize error for '{text}': {e}",
            flush=True,
        )
        return []


@lru_cache(maxsize=1)
def _create_translation_logit_bias() -> Dict[str, int]:
    """
    Create logit_bias to force English and prevent refusals.
    Tokenizes penalized/boosted strings via the server and builds the bias dict.
    
    Note: OpenAI API expects Dict[str, int] where keys are token IDs as strings
    and values are bias values from -100 to 100.
    """
    logit_bias: Dict[str, int] = {}

    for text in PENALIZE_STRINGS:
        tokens = _tokenize_text(text)
        for token_id in tokens:
            key = str(token_id)
            logit_bias[key] = logit_bias.get(key, 0) + int(PENALTY_VALUE)

    for text in BOOST_STRINGS:
        tokens = _tokenize_text(text)
        for token_id in tokens:
            key = str(token_id)
            logit_bias[key] = logit_bias.get(key, 0) + int(BOOST_VALUE)

    print(
        f"[translate_openai] Created logit_bias with {len(logit_bias)} token entries",
        flush=True,
    )
    return logit_bias


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create and cache the OpenAI client pointing to the local llama.cpp server."""
    print(
        f"[translate_openai] Creating OpenAI client for {LLM_SERVER_BASE_URL} ...",
        flush=True,
    )
    client = OpenAI(
        base_url=LLM_SERVER_BASE_URL,
        api_key="not-needed",  # llama.cpp server doesn't require an API key
    )
    print("[translate_openai] Client created", flush=True)
    return client


client = _get_client()


def translate_japanese_to_english(
    text: str,
    history: list[dict[str, str]] | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
) -> dict[str, Any]:
    """
    Translate a Japanese string to English using the local llama.cpp server.

    Parameters
    ----------
    text        : Japanese text to translate.
    history     : Optional list of prior {role, content} turns.
    max_tokens  : Maximum tokens to generate.
    temperature : Sampling temperature.
    top_p       : Nucleus sampling threshold.
    top_k       : Top-K sampling (llama.cpp specific, sent via extra_body).
    frequency_penalty : Frequency penalty (-2.0 to 2.0).
    presence_penalty  : Presence penalty (-2.0 to 2.0).

    Returns
    -------
    dict with keys:
        text              – English translation (str)
        tokens_evaluated  – prompt tokens processed (int, 0 if not available)
        tokens_cached     – prompt tokens served from cache (int, 0 if not available)
        tokens_generated  – new tokens generated (int)
        latency_ms        – wall-clock time in ms (float)
    """
    # ── Build messages ─────────────────────────────────────────────
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": text})

    print(
        f"[translate_openai] Translating: {text[:80]}{'...' if len(text) > 80 else ''}",
        flush=True,
    )
    print(f"[translate_openai] History length: {len(history or [])} turns", flush=True)

    # ── Get logit bias (cached) ────────────────────────────────────
    logit_bias = _create_translation_logit_bias()

    # ── Stream the response with flushed logs ──────────────────────
    t0 = time.perf_counter()
    full_translation: str = ""
    tokens_generated: int = 0

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,              # ✅ Direct parameter
            temperature=temperature,            # ✅ Direct parameter
            top_p=top_p,                        # ✅ Direct parameter
            frequency_penalty=frequency_penalty, # ✅ Direct parameter
            presence_penalty=presence_penalty,   # ✅ Direct parameter
            logit_bias=logit_bias,              # ✅ Direct parameter
            extra_body={                        # Only llama.cpp-specific params
                "top_k": top_k,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            stream=True,
        )

        print("[translate_openai] Streaming response:", flush=True)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                full_translation += content_piece
                tokens_generated += 1
                # Print each chunk immediately with flushed output
                sys.stdout.write(content_piece)
                sys.stdout.flush()

        print()  # Newline after streaming
        print("[translate_openai] Stream complete.", flush=True)

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"[translate_openai] ERROR: {e}", flush=True)
        return {
            "text": "",
            "tokens_evaluated": 0,
            "tokens_cached": 0,
            "tokens_generated": 0,
            "latency_ms": latency_ms,
            "error": str(e),
        }

    latency_ms = (time.perf_counter() - t0) * 1000
    translation = full_translation.strip()

    print(
        f"[translate_openai] Translation complete — "
        f"{tokens_generated} tokens in {latency_ms:.0f} ms",
        flush=True,
    )

    return {
        "text": translation,
        "tokens_evaluated": 0,  # Not directly available from streaming OpenAI API
        "tokens_cached": 0,     # Not directly available from OpenAI-compatible API
        "tokens_generated": tokens_generated,
        "latency_ms": latency_ms,
    }


# ── CLI demo / test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    from rich.rule import Rule

    # Optional: import the fuzzy matcher (same as original demo)
    try:
        from services.sentence_matcher_ja import fuzzy_match_prefix_texts
    except ImportError:
        from sentence_matcher_ja import fuzzy_match_prefix_texts

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]OpenAI-compatible llama.cpp Server Demo[/bold cyan]\n"
            "[dim]Streaming translation with real-time output[/dim]",
            border_style="cyan",
        )
    )

    history: list[dict[str, str]] = []
    progressive_subtitles = [
        "今日は",
        "今日はとても",
        "今日はとても疲れる。",
        "あなたのことが好きだよ。",
        "早く行かないと電車に乗り遅れる！",
    ]

    console.print(Rule("[bold]Progressive Growing Input Test[/bold]"))
    console.print()

    results = []
    prev_ja = None
    prev_en = None

    for i, growing_text in enumerate(progressive_subtitles, start=1):
        ja_text = growing_text
        console.print(
            f"[bold yellow]\\[Step {i}][/bold yellow] [white]{ja_text}[/white]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]Translating...[/dim]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            result = translate_japanese_to_english(
                ja_text,
                history=history,
            )
            en_text = result["text"]

        # Apply prefix matching to extract only new content
        if prev_ja is not None or prev_en is not None:
            fuzzy_texts_result = fuzzy_match_prefix_texts({
                "full_ja": ja_text,
                "full_en": result["text"],
                "prev_ja": prev_ja or "",
                "prev_en": prev_en or "",
            })
            new_ja = fuzzy_texts_result["new_ja"]
            new_en = fuzzy_texts_result["new_en"]
            en_text = fuzzy_texts_result["full_en"]
            is_continuation = fuzzy_texts_result["is_continuation"]
        else:
            new_en = en_text
            is_continuation = False
            new_ja = ja_text

        console.print(f" [green]↳ EN:[/green] [italic]{en_text}[/italic]")
        if new_en and i > 1:
            console.print(
                f" [bright_green]   Δ New:[/bright_green] [bold]{new_en}[/bold]"
            )
        console.print()

        results.append((i, ja_text, en_text, new_en, is_continuation))

        # Update history for KV cache benefit
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