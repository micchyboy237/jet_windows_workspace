import argparse
import os

from performance_tracker import (
    PerformanceMetrics,
    PerformanceTracker,
)
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from llama_cpp import ChatCompletionRequestMessage, Llama, LogitsProcessorList
from rich.console import Console

console = Console()

TRANSLATION_DEFAULTS = {
    "max_tokens": 2000,
    "temperature": 0.2,
    "top_p": 0.9,
    # "presence_penalty": 2.0,
    # "top_k": 20,
    # "chat_template_kwargs": {
    #     "enable_thinking": False,
    # },
    # "logprobs": True,
    # "top_logprobs": 3
    "stop": ["\n\n","*","("],
}

SYSTEM_PROMPT = """You are an expert real-time Japanese-to-English subtitle translator for live-streamed audio.

Your ONLY job is to produce accurate, natural, subtitle-ready English translations.

Strict rules you must follow every time:
- Translate with 100% fidelity to the original meaning. Never add, omit, embellish, or moralize.
- Infer the most likely intended meaning from Whisper transcription errors, but stay extremely close to the provided text.
- Use natural, spoken English suitable for live subtitles: concise, flowing, and easy to read on screen.
- Translate EVERYTHING exactly as it is — including profanity, slang, politics, or adult content. Keep the original tone and intensity.
- If the Japanese switches speakers, separate them with newlines. Do not add speaker names unless they appear in the Japanese.
- Output **ONLY** the clean English translation. 

ABSOLUTELY FORBIDDEN:
- Any labels like "JA:", "EN:", "Translation:", "Alternative:", "OR"
- Multiple versions or "alternate subtitle" suggestions
- Numbered lists, bullets, quotes around the whole text, or markdown
- Any explanations, reasoning, or meta text

Your response must contain nothing but the English subtitle text itself.

Correct example:
Fierce information battles are raging silently across the world right now.

Wrong example (never output this):
JA: 世界各国が水面下で...
EN: The world carries out...
OR (alternate...): Fierce info wars...
"""

USER_PROMPT = "{japanese_text}"


def log_metrics(metrics: PerformanceMetrics) -> None:
    console.print("\n\n[bold yellow]=== Completion Details (llama.cpp aligned) ===[/bold yellow]")
    console.print(f"[cyan]Prompt tokens     : {metrics.prompt_tokens}[/cyan]")
    console.print(f"[cyan]Completion tokens : {metrics.completion_tokens}[/cyan]")
    console.print(f"[cyan]Total tokens      : {metrics.total_tokens}[/cyan]")

    if metrics.ttft is not None:
        console.print(f"[magenta]TTFT              : {metrics.ttft:.3f}s[/magenta]")

    if metrics.prompt_eval_speed is not None:
        console.print(
            f"[green]Prompt eval speed : {metrics.prompt_eval_speed:.2f} tokens/s (approx)[/green]"
        )

    if metrics.decode_speed is not None:
        console.print(f"[green]Decode speed      : {metrics.decode_speed:.2f} tokens/s (eval)[/green]")

    console.print(f"[yellow]Total latency     : {metrics.total_latency:.3f}s[/yellow]")

    # Optional: keep but clearly marked as non-standard
    if metrics.end_to_end_throughput is not None:
        console.print(
            f"[bold]End-to-end throughput : {metrics.end_to_end_throughput:.2f} tokens/s[/bold]"
        )


class TranslationResult(TypedDict):
    text: str
    log_prob: Optional[float]
    confidence: Optional[float]
    quality: Optional[str]


def translate_japanese_to_english(
    ja_text: str,
    enable_scoring: bool = False,
    history: Optional[List[ChatCompletionRequestMessage]] = None,
    temperature: float = TRANSLATION_DEFAULTS["temperature"],
    max_tokens: int = TRANSLATION_DEFAULTS["max_tokens"],
    **kwargs,
) -> TranslationResult:
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-1234",
    )

    messages = [
        # {
        #     "role": "system",
        #     "content": SYSTEM_PROMPT,
        # },
        {
            "role": "user",
            "content": USER_PROMPT.format(japanese_text=ja_text),
        }
    ]

    console.print(f"[dim]User prompt:\n{USER_PROMPT.format(japanese_text=ja_text)}[/dim]")

    tracker = PerformanceTracker()

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model="Qwen/Qwen3.5-2B",
        messages=messages,
        max_tokens=32768,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=2.0,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
        stream=True,
    )

    en_text = ""
    stream_started = False
    for part in stream:
        if not stream_started:
            stream_started = True
            console.print("Response:")

        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Check for reasoning_content first
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                tracker.mark_token()
                console.print(f"[orange1]{delta.reasoning_content}[/orange1]", end="")
            # Then check for regular content
            elif hasattr(delta, "content") and delta.content:
                tracker.mark_token()
                console.print(f"[bright_cyan]{delta.content}[/bright_cyan]", end="")
                en_text += delta.content

        usage = getattr(part, "usage", None)
        if usage is not None:
            metrics = tracker.finalize(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            log_metrics(metrics)

    # Extra safety: ensure only clean English is returned
    en_text = en_text.strip()
    
    # Remove common unwanted prefixes that the model might still add
    unwanted_prefixes = [
        "JA:", "EN:", "Translation:", "English:", "Subtitle:", 
        "OR", "Alternate:", "Alternative:", "1.", "2."
    ]
    for prefix in unwanted_prefixes:
        if en_text.upper().startswith(prefix.upper()):
            en_text = en_text.split(":", 1)[-1].strip()
            en_text = en_text.split("\n", 1)[-1].strip() if "\n" in en_text else en_text
    
    # Remove any surrounding quotes if the whole output is quoted
    if (en_text.startswith('"') and en_text.endswith('"')) or \
       (en_text.startswith("'") and en_text.endswith("'")):
        en_text = en_text[1:-1].strip()

    return {
        "text": en_text,
        "log_prob": None,
        "confidence": None,
        "quality": None,
    }



if __name__ == "__main__":
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
    args = parser.parse_args()

    result = translate_japanese_to_english(args.text)
