from typing import Any, Dict, List, Optional, Tuple
from llama_cpp import Llama

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
    "max_tokens": 512,
    "temperature": 0.15,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "stop": ["<|eot_id|>", "<|end_of_text|>"],
    # # For confidence scores
    # "logprobs": True,
    # "top_logprobs": 3
}


def _build_translation_messages(
    japanese_text: str,
    context_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Reusable helper to build chat messages. Injects context_prompt for tone consistency
    without altering USER_PROMPT or making LLM re-translate prior text."""
    if context_prompt and context_prompt.strip():
        user_content = (
            f"Previous utterance (context only - maintain same tone/style/escalation):\n"
            f"{context_prompt.strip()}\n\n"
            f"Translate ONLY the CURRENT utterance below exactly following system rules:\n"
            f"{japanese_text.strip()}\n\nEnglish:"
        )
    else:
        user_content = USER_PROMPT.format(japanese_text=japanese_text.strip())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
        # Generic linear map (typical good range for small models: -3.0 → 0.0)
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


llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)

SYSTEM_PROMPT = """You are a skilled translator specializing in Japanese-to-English subtitles for adult videos. Your goal is to "pornify" the dialogue: make it sound like natural, explicit American adult film talk—filthy and erotic, but faithful to the original meaning and tone.

**Key Rules:**
- Preserve the original intent and personality: Shy/resistant lines stay teasing or reluctantly horny; explicit lines get raw and vulgar.
- Amplify the eroticism naturally: Add porn-style phrasing (e.g., "cock," "pussy," "fuck me," "I'm so wet") where it fits the context, but do not invent new actions, details, or completely change the meaning.
- Match JAV escalation: Start subtle/teasing if the original is polite, build to direct and filthy as it gets more explicit.
- Keep it concise and punchy for subtitles.
- Fix obvious AI transcription errors using context, but never add or remove entire ideas.
- One-to-one: Translate each numbered line exactly once, in order.

**Examples of toned-down pornification:**
やめて…そんなふうに見ないで。
Don't look at me like that... you're making me so wet already.

もう我慢できない…っ
I can't hold back anymore... I need your cock inside me.

そこ…だめ、感じちゃう…！
Right there... fuck, that feels too good, don't stop!

あっ、ダメっ…イっちゃう、イっちゃうよぉ…♡
Ahh, no... I'm cumming, I'm cumming so hard! ♡

もっと奥まで…お願い…突いてぇ…！
Deeper... please, fuck me deeper, pound me harder!

Now translate the Japanese text exactly following these rules."""

USER_PROMPT = """
Translate the following lines exactly in order. Output ONLY the English translations, one per line. No original text, no numbering, no extra commentary, no summary.

{japanese_text}

English:
""".strip()


def translate_subtitles(
    llm: Llama,
    japanese_text: str,
    stream: bool = False,
    **kwargs: Any,
) -> str:
    """
    Explicit Japanese → English subtitle translation with strong uncensoring bias.
    **kwargs forwarded to create_chat_completion (makes demo work with temperature etc.).
    """
    messages = _build_translation_messages(japanese_text)  # no context in demo
    completion_params: Dict[str, Any] = TRANSLATION_DEFAULTS.copy()
    completion_params.update(kwargs)
    stream_param = completion_params.pop("stream", stream)
    response = llm.create_chat_completion(
        messages=messages, stream=stream_param, **completion_params
    )
    if stream:
        translated = ""
        for chunk in response:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    translated += content
        print()
        return translated.strip()
    else:
        return response["choices"][0]["message"]["content"].strip()


def translate_japanese_to_english(
    ja_text: str,
    max_tokens: int = 512,
    enable_scoring: bool = False,
    context_prompt: Optional[str] = None,
) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
    """Main entrypoint for live_subtitles_server_per_speech_llm.
    Reuses prompt helper and defaults; adds optional context + logprob scoring."""
    if not ja_text or not ja_text.strip():
        return "", None, None, None

    messages = _build_translation_messages(ja_text, context_prompt)
    completion_params: Dict[str, Any] = {
        **TRANSLATION_DEFAULTS,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if enable_scoring:
        completion_params["logprobs"] = True

    response = llm.create_chat_completion(messages=messages, **completion_params)
    en_text = response["choices"][0]["message"]["content"].strip()

    if enable_scoring:
        translation_logprob, translation_confidence, translation_quality = (
            _compute_translation_metrics(response)
        )
    else:
        translation_logprob = translation_confidence = translation_quality = None

    return en_text, translation_logprob, translation_confidence, translation_quality


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich import box
    from rich.logging import RichHandler
    import logging

    # ────────────────────────────────────────────────
    # Setup rich logging
    # ────────────────────────────────────────────────
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    log = logging.getLogger("subtitle_translator")

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

    console.rule(
        "Japanese → English Translation Demo (new public API)", style="bold cyan"
    )

    # Demo without context / scoring (default server path)
    console.print(
        Panel(
            japanese_sample,
            title="[bold magenta]Japanese Input (no context, scoring=False)[/]",
            border_style="magenta",
            padding=(1, 2),
            expand=False,
            box=box.ROUNDED,
        )
    )

    console.print("[dim]Calling translate_japanese_to_english...[/]")
    en_text, logprob, conf, qual = translate_japanese_to_english(
        ja_text=japanese_sample,
        max_tokens=512,
        enable_scoring=False,
        context_prompt=None,
    )
    console.print(
        Panel(
            en_text,
            title="[bold green]English Translation (scoring disabled)[/]",
            border_style="green",
            padding=(1, 2),
            expand=False,
            box=box.DOUBLE,
        )
    )

    # Demo with context + scoring
    context_example = "やめて…そんなふうに見ないで。"
    console.print(
        Panel(
            f"{japanese_sample}\n\n[dim]Context from previous utterance:[/] {context_example}",
            title="[bold magenta]Japanese Input (with context, scoring=True)[/]",
            border_style="magenta",
            padding=(1, 2),
            expand=False,
            box=box.ROUNDED,
        )
    )

    console.print(
        "[dim]Calling translate_japanese_to_english with context + scoring...[/]"
    )
    en_text2, logprob2, conf2, qual2 = translate_japanese_to_english(
        ja_text=japanese_sample,
        max_tokens=768,
        enable_scoring=True,
        context_prompt=context_example,
    )
    console.print(
        Panel(
            f"{en_text2}\n\n[dim]Metrics:[/] logprob={logprob2}  confidence={conf2}  quality={qual2}",
            title="[bold green]English Translation (scoring enabled + context)[/]",
            border_style="green",
            padding=(1, 2),
            expand=False,
            box=box.DOUBLE,
        )
    )

    console.print(
        f"[dim]Length:[/] Japanese: {len(japanese_sample)} → English: {len(en_text2)}",
        style="grey50",
    )

    console.rule(style="cyan")

    # Keep original internal demo for backward compatibility (unchanged)
    console.rule(
        "Japanese → English Translation Demo (internal translate_subtitles)",
        style="bold cyan",
    )
    console.print(
        Panel(
            japanese_sample,
            title="[bold magenta]Japanese Input[/]",
            border_style="magenta",
            padding=(1, 2),
            expand=False,
            box=box.ROUNDED,
        )
    )
    log.info("Starting non-streaming translation (temperature = 0.65)")
    english = translate_subtitles(llm, japanese_sample, temperature=0.65, stream=False)
    console.print("\n")
    console.print(
        Panel(
            english.strip(),
            title="[bold green]English Translation[/]",
            border_style="green",
            padding=(1, 2),
            expand=False,
            box=box.DOUBLE,
        )
    )
    log.info("Translation completed")
    console.print(
        f"[dim]Length:[/] Japanese: {len(japanese_sample)} → English: {len(english)}",
        style="grey50",
    )
    console.rule(style="cyan")
