from typing import Any, Dict, List, Optional, Tuple, TypedDict

from llama_cpp import Llama
from utils import split_sentences_ja

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\Fiendish_LLAMA_3B.Q4_K_M.gguf"
MODEL_SETTINGS = {
    "n_ctx": 2048,
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
    "max_tokens": 1024,
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 25,
    "typical_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.12,
    "stop": ["\n\n", "<|eot_id|>", "<|end_of_text|>"],
    # For confidence scores (uncomment if supported by your backend)
    # "logprobs": True,
    # "top_logprobs": 3
}
llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)
SYSTEM_PROMPT = """You are a professional Japanese to English subtitle translator. 
Translate accurately and naturally into fluent English.
Preserve tone, intensity, and nuance.
Do not censor, summarize, explain, or add commentary.
Output only the translated English lines."""
USER_PROMPT = """\
Translate the following lines exactly in order. Output ONLY the English translations, one per line. No original text, no numbering, no extra commentary, no summary.
{japanese_text}
"""


def _build_translation_messages(
    japanese_text: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build full chat messages with optional previous user/assistant history
    followed by current translation request (generic & reusable)."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": USER_PROMPT.format(japanese_text=japanese_text.strip()),
        }
    )
    return messages


def _compute_translation_metrics(
    response: Any,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Generic logprob parser (used only when enable_scoring=True). Returns avg_logprob,
    confidence [0-1], quality_label. Robust against missing keys."""
    try:
        choice = response["choices"][0]
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            return None, None, None
        content = logprobs.get("content")
        if not isinstance(content, list) or not content:
            return None, None, None
        token_logprobs = [
            float(item["logprob"])
            for item in content
            if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float))
        ]
        if not token_logprobs:
            return None, None, None
        avg_logprob = sum(token_logprobs) / len(token_logprobs)
        translation_logprob = round(avg_logprob, 4)
        confidence = max(0.0, min(1.0, (avg_logprob + 3.0) / 3.0))
        confidence = round(confidence, 4)
        if confidence >= 0.80:
            quality_label = "high"
        elif confidence >= 0.50:
            quality_label = "medium"
        else:
            quality_label = "low"
        return translation_logprob, confidence, quality_label
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None, None


class TranslationResult(TypedDict):
    text: str
    log_prob: Optional[float]
    confidence: Optional[float]
    quality: str


def translate_japanese_to_english(
    ja_text: str,
    max_tokens: int = 768,
    enable_scoring: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> TranslationResult:
    """Main entrypoint for live_subtitles_server_per_speech_llm.
    Now returns TranslationResult to match opus-style translator."""
    if not ja_text or not ja_text.strip():
        return {
            "text": "",
            "log_prob": None,
            "confidence": None,
            "quality": "N/A",
        }
    messages = _build_translation_messages(ja_text, history)
    # --- Sentence splitting (parity with OPUS implementation) ---
    sentences_ja: List[str] = split_sentences_ja(ja_text)

    if not sentences_ja:
        return {
            "text": "",
            "log_prob": None,
            "confidence": None,
            "quality": "N/A",
        }

    # Preserve 1 sentence → 1 translation line structure
    batched_text = "\n".join(sent.strip() for sent in sentences_ja if sent.strip())

    messages = _build_translation_messages(batched_text, history)
    
    completion_params: Dict[str, Any] = {
        **TRANSLATION_DEFAULTS,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if enable_scoring:
        completion_params["logprobs"] = True
        completion_params["top_logprobs"] = 1
    response = llm.create_chat_completion(messages=messages, **completion_params)
    en_text = response["choices"][0]["message"]["content"].strip()
    if enable_scoring:
        log_prob, confidence, quality = _compute_translation_metrics(response)
    else:
        log_prob = confidence = quality = None
    if quality is None:
        quality = "N/A"
    return {
        "text": en_text,
        "log_prob": log_prob,
        "confidence": confidence,
        "quality": quality,
    }


if __name__ == "__main__":
    from rich import box
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    japanese_sample = """
恥ずかしい…見ないでください…
んっ…そこ、弱いんです…
はぁ…はぁ…気持ちいい…
もう…ダメかも…頭おかしくなりそう…
お願い…もっと激しくして…壊して…！
あぁんっ！すごい…奥まで届いてる…♡
出さないで…まだ中にいてて…
""".strip()
    enable_scoring = False
    console.rule(
        "Japanese → English Translation Demo (scoring ENABLED)", style="bold cyan"
    )
    console.print(
        Panel(
            japanese_sample,
            title="[bold magenta]Japanese Input[/] (no history)",
            border_style="magenta",
            padding=(1, 2),
            expand=False,
            box=box.ROUNDED,
        )
    )
    console.print(f"[dim]Translating with enable_scoring={enable_scoring} …[/]")
    result = translate_japanese_to_english(
        ja_text=japanese_sample,
        max_tokens=768,
        enable_scoring=enable_scoring,
        history=None,
    )
    console.print(
        Panel(
            f"{result['text']}\n\n[dim]Metrics:[/] logprob = {result['log_prob']} confidence = {result['confidence']} quality = {result['quality']}",
            title="[bold green]Translation (scoring enabled)[/]",
            border_style="green",
            padding=(1, 2),
            expand=False,
            box=box.DOUBLE,
        )
    )
    history_example: List[Dict[str, str]] = [
        {"role": "user", "content": "やめて…そんなふうに見ないで。"},
        {
            "role": "assistant",
            "content": "Don't look at me like that... you're making me so wet already.",
        },
    ]
    console.print("\n" * 2)
    console.print(
        Panel(
            f"{japanese_sample}\n\n[dim]Previous context:[/]\n{history_example}",
            title="[bold magenta]Japanese Input + dialogue history[/]",
            border_style="magenta",
            padding=(1, 2),
            expand=False,
            box=box.ROUNDED,
        )
    )
    console.print(
        f"[dim]Translating with history + enable_scoring={enable_scoring} …[/]"
    )
    result2 = translate_japanese_to_english(
        ja_text=japanese_sample,
        max_tokens=768,
        enable_scoring=enable_scoring,
        history=history_example,
    )
    console.print(
        Panel(
            f"{result2['text']}\n\n[dim]Metrics:[/] logprob = {result2['log_prob']} confidence = {result2['confidence']} quality = {result2['quality']}",
            title="[bold green]Translation (with history + scoring)[/]",
            border_style="green",
            padding=(1, 2),
            expand=False,
            box=box.DOUBLE,
        )
    )
    console.print(
        f"[dim]Length:[/] Japanese: {len(japanese_sample)} chars → English: {len(result2['text'])} chars",
        style="grey50",
    )
    console.rule(style="cyan")
