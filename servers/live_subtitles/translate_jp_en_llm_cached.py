"""
translate_jp_en_llm_cached.py
──────────────────────
Japanese → English translator using llama-cpp-python and the
shisa-v2.1-llama3.2-3b Q4_K_M GGUF model.
The Llama class is loaded once at module level (singleton) so the KV
cache persists across calls within a single process — ideal for the
progressive-subtitle demo.
"""
from __future__ import annotations
import time
import os
import sys
from functools import lru_cache
from typing import Any
from llama_cpp import Llama

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
