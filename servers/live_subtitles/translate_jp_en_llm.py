import time
import uuid
import json
import math
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
MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf"
)

MODEL_SETTINGS = {
    "n_ctx": 4096,  # Increased to reduce context warning and allow longer context
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

TRANSLATION_DEFAULTS = {
    "temperature": 0.7,   # Slightly lower for less hallucination on small model
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.12,  # Tighter to reduce repetitions
    "max_tokens": 512,
}

SYSTEM_MESSAGE = """You are a professional subtitle translator from Japanese to English for anime, drama, and general media.
Correct any transcription errors using the provided context for consistency.
Preserve the exact number of lines as the input Japanese text.
Use natural, fluent, and engaging English while maintaining the original tone.
Preserve symbols such as 🎼.
Output ONLY the translated English text. No explanations, no Japanese text, no additional content, and no repetition of instructions."""

llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)

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
            "content": SYSTEM_MESSAGE,
        },
        {"role": "user", "content": text.strip()},
    ]

    params: Dict[str, Any] = {
        "messages": messages,
        "stream": stream,
        **TRANSLATION_DEFAULTS,
        **generation_params,
    }

    return llm.create_chat_completion(**params)


def translation_quality_label(avg_logprob: float) -> str:
    if not math.isfinite(avg_logprob):
        return "N/A"
    if avg_logprob > -0.3:
        return "Very High"
    if avg_logprob > -0.7:
        return "High"
    if avg_logprob > -1.2:
        return "Medium"
    if avg_logprob > -2.0:
        return "Low"
    return "Very Low"


def translate_japanese_to_english_structured(
    text: str,
    *,
    beam_size: int = 1,  # kept for compatibility with server
    max_decoding_length: int = 512,
    min_tokens_for_confidence: int = 3,
    enable_scoring: bool = True,
    context_prompt: str | None = None,
    **generation_params,
) -> Tuple[str, Optional[float], Optional[float], str]:
    """
    Server-safe translation wrapper.
    Returns:
        (
            english_text,
            avg_logprob,
            confidence,
            quality_label
        )
    """

    params = {
        "max_tokens": max_decoding_length,
        "logprobs": True if enable_scoring else None,
        "top_logprobs": 1 if enable_scoring else None,
        **generation_params,
    }

    user_content = text.strip()
    if context_prompt:
        user_content = (
            f"Previous context (do NOT translate again, just use for reference):\n"
            f"{context_prompt}\n\n"
            f"Current line to translate:\n{user_content}"
        )

    try:
        response = translate_japanese_to_english(
            text=text,
            stream=False,
            **params,
        )

        choice = response["choices"][0]
        message = choice["message"]["content"]
        logprobs_data = choice.get("logprobs")

        avg_logprob = None
        confidence = None
        quality = "N/A"

        if enable_scoring and logprobs_data and logprobs_data.get("content"):
            token_logprobs = [
                tok["logprob"]
                for tok in logprobs_data["content"]
                if tok.get("logprob") is not None
            ]

            if len(token_logprobs) >= min_tokens_for_confidence:
                avg_logprob = sum(token_logprobs) / len(token_logprobs)
                confidence = float(math.exp(avg_logprob))
                quality = translation_quality_label(avg_logprob)

        return message.strip(), avg_logprob, confidence, quality

    finally:
        # critical for llama_cpp stability
        llm.reset()


def translate_text(
    text: str, logprobs: Optional[int] = None, **generation_params
) -> dict:
    """Translate with beautiful real-time streaming display using rich"""
    full_text = ""

    _generation_params: Dict[str, Any] = {**TRANSLATION_DEFAULTS, **generation_params}

    if logprobs:
        _generation_params["logprobs"] = True
        _generation_params["top_logprobs"] = logprobs

    stream = translate_japanese_to_english(
        text=text,
        stream=True,
        **_generation_params,
    )

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
                logprobs_tokens = [
                    (l["token"], l["logprob"], l["top_logprobs"])
                    for l in logprobs_content
                ]

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
# ────────────────────────────────────────────────
# Quick demo
# ────────────────────────────────────────────────
if __name__ == "__main__":
    console.rule("LFM2-350M-ENJP-MT Translator Demo", style="bold magenta")

    eng, avg_logprob, conf, qual = translate_japanese_to_english_structured(
        text='🎼世界 核府 が 水面下 で 熾烈 な 情報 線 を 繰り 広げる 時代、ニラ 繰り 広げる 時代、睨み 合う 二 つの 国、東の オスタニア、西 の ウェスタリス。',
        context_prompt='🎼世界 核府 が 水面下 で 熾烈 な 情報戦 を 繰り 広げる 時代。',
        # temperature=0.75,
        max_decoding_length=512,
        min_tokens_for_confidence=3,
        enable_scoring=False,
    )
    console.print(f"   → [green]{eng}[/green]")
    if avg_logprob is not None:
        console.print(
            f"   [dim]logprob: {avg_logprob:.3f}  |  conf: {conf:.3%}  |  {qual}[/dim]"
        )
    console.print()
