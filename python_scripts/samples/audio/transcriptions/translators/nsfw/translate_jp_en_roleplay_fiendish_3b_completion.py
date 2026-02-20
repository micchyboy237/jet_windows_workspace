from __future__ import annotations

import time
from typing import List

from llama_cpp import Llama
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown


console = Console()


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
    "stop": ["### Japanese", "### English"]
    # # For confidence scores
    # "logprobs": True,
    # "top_logprobs": 3
}


class FiendishLlama:
    """Singleton-style model loader to avoid reloading on every call."""

    _instance: Llama | None = None

    @classmethod
    def get(cls) -> Llama:
        if cls._instance is None:
            with console.status("[bold green]Loading Fiendish_LLAMA_3B ...[/bold green]"):
                start = time.perf_counter()

                cls._instance = Llama(
                    model_path=MODEL_PATH,
                    **MODEL_SETTINGS
                )

                took = time.perf_counter() - start
            console.print(f"[green]Model loaded in {took:.1f} seconds[/green]\n")

        return cls._instance


def build_translation_prompt(japanese_lines: List[str]) -> str:
    """
    Builds a completion-style prompt inspired by pornify.txt, but toned down for more
    natural, faithful pornified translations. Output is simplified to only the English
    lines, one per line under an 'English:' header—no Original lines, no prefixes,
    no summary/scene tags.
    """
    # Filter empty lines and create numbered list for clear one-to-one mapping
    numbered_lines = []
    for i, line in enumerate(japanese_lines, 1):
        stripped = line.strip()
        if stripped:
            numbered_lines.append(f"{i}. {stripped}")

    lines_part = "\n".join(numbered_lines)

    prompt = f"""You are a skilled translator specializing in Japanese-to-English subtitles for adult videos. Your goal is to "pornify" the dialogue: make it sound like natural, explicit American adult film talk—filthy and erotic, but faithful to the original meaning and tone.

**Key Rules:**
- Preserve the original intent and personality: Shy/resistant lines stay teasing or reluctantly horny; explicit lines get raw and vulgar.
- Amplify the eroticism naturally: Add porn-style phrasing (e.g., "cock," "pussy," "fuck me," "I'm so wet") where it fits the context, but do not invent new actions, details, or completely change the meaning.
- Match JAV escalation: Start subtle/teasing if the original is polite, build to direct and filthy as it gets more explicit.
- Keep it concise and punchy for subtitles.
- Fix obvious AI transcription errors using context, but never add or remove entire ideas.
- One-to-one: Translate each numbered line exactly once, in order.

**Examples of toned-down pornification:**
1. やめて…そんなふうに見ないで。
   → Don't look at me like that... you're making me so wet already.

2. もう我慢できない…っ
   → I can't hold back anymore... I need your cock inside me.

3. そこ…だめ、感じちゃう…！
   → Right there... fuck, that feels too good, don't stop!

4. あっ、ダメっ…イっちゃう、イっちゃうよぉ…♡
   → Ahh, no... I'm cumming, I'm cumming so hard! ♡

5. もっと奥まで…お願い…突いてぇ…！
   → Deeper... please, fuck me deeper, pound me harder!

Translate the following lines exactly in order. Output ONLY the English translations, one per line, starting with "English:" on the first line. No original text, no numbering, no extra commentary, no summary.

{lines_part}

English:
"""

    return prompt.strip()


def translate_jp_to_en(japanese_lines: List[str]) -> str:
    """
    Translate Japanese adult dialogue lines into natural, uncensored English.
    """
    llm = FiendishLlama.get()
    prompt = build_translation_prompt(japanese_lines)

    completion = llm.create_completion(
        prompt=prompt,
        **TRANSLATION_DEFAULTS,
    )

    return completion["choices"][0]["text"].strip()


def main():
    console.rule("JAV Dialogue Translation Example", style="cyan")

    japanese_dialogue = [
        "恥ずかしい…見ないでください…",
        "んっ…そこ、弱いんです…",
        "はぁ…はぁ…気持ちいい…",
        "もう…ダメかも…頭おかしくなりそう…",
        "お願い…もっと激しくして…壊して…！",
        "あぁんっ！すごい…奥まで届いてる…♡",
        "出さないで…まだ中にいてて…",
    ]

    console.print("[bold magenta]Japanese:[/bold magenta]")
    for line in japanese_dialogue:
        console.print(f"  {line}")

    console.print("\n[bold cyan]Translating …[/bold cyan]")

    english = translate_jp_to_en(japanese_dialogue)

    console.print("\n[bold cyan]English:[/bold cyan]")
    console.print(english)


if __name__ == "__main__":
    main()