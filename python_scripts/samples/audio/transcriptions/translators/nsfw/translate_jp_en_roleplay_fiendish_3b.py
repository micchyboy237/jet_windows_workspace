from __future__ import annotations
import time
from typing import List, Union
from llama_cpp import Llama
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\Fiendish_LLAMA_3B.Q4_K_M.gguf"

MODEL_SETTINGS = {
    "n_ctx": 2048,
    "n_threads": 8,
    "n_batch": 256,
    "verbose": False,
    "n_gpu_layers": -1,
}

TRANSLATION_DEFAULTS = {
    "max_tokens": 768,
    "temperature": 0.20,
    "top_p": 0.92,
    "repeat_penalty": 1.12,
    "stop": ["</s>", "<|end|>", "\n\n\n", "Original>", "English:"],
}

# ────────────────────────────────────────────────────────────────
#  Inspired by standard.txt + tuned for more natural & lewd output
# ────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are an expert English translator specializing in Japanese → English subtitle translation for Japanese Adult Videos (JAV).

Rules you MUST follow:
- Preserve the exact adult/erotic/lewd/teasing/pornographic tone and intent
- Polite & suggestive → gradually more direct/explicit/vulgar as arousal increases
- Use natural-sounding, sexy English that matches the character's personality
- During sex: very explicit, dirty, vulgar language is expected and desired
- Fix obvious Whisper transcription errors when context makes it clear
- Keep the SAME NUMBER of lines as input — translate line-by-line
- Do NOT merge, split, or skip lines
- Output format: ONLY numbered English lines — nothing else

Examples of desired tone:
1. 気持ちいい… → "It feels so good…" / "Fuck… it feels amazing…"
2. だめっ、だめぇ！ → "Nooo! Stop— ahh!" / "I can't— I'm gonna cum!"
3. もっと奥まで入れて… → "Deeper… fuck me deeper…" 
4. いやらしい声出ちゃう… → "I can't help making such dirty sounds…"

Now translate the following Japanese subtitle lines:

{japanese_text}

English:
"""

class FiendishLlama:
    _instance: Llama | None = None

    @classmethod
    def get(cls) -> Llama:
        if cls._instance is None:
            with console.status("[bold green]Loading Fiendish_LLAMA_3B ...[/bold green]"):
                start = time.perf_counter()
                cls._instance = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)
                took = time.perf_counter() - start
            console.print(f"[green]Model loaded in {took:.1f} seconds[/green]\n")
        return cls._instance


def build_translation_prompt(japanese_lines: Union[str, List[str]]) -> str:
    if isinstance(japanese_lines, str):
        japanese_lines = [japanese_lines]

    # Filter empty lines but keep original count/structure
    cleaned_lines = []
    for line in japanese_lines:
        stripped = line.strip()
        if stripped:  # only add non-empty
            cleaned_lines.append(stripped)
        else:
            cleaned_lines.append("")  # preserve blank lines if they existed

    numbered = []
    for i, line in enumerate(cleaned_lines, 1):
        if line.strip():
            numbered.append(f"{i}. {line}")
        else:
            numbered.append(f"{i}.")

    japanese_text = "\n".join(numbered)

    return PROMPT_TEMPLATE.format(japanese_text=japanese_text).strip()


def translate_jp_to_en(japanese_input: Union[str, List[str]]) -> str:
    """
    Translate Japanese adult dialogue (str or list[str]) → natural uncensored English
    """
    llm = FiendishLlama.get()
    prompt = build_translation_prompt(japanese_input)

    completion = llm(prompt=prompt, **TRANSLATION_DEFAULTS)
    text = completion["choices"][0]["text"].strip()

    # Minimal post-processing — remove leading numbers if model adds them again
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            # Remove possible "1. " "2. " etc that some models love to add
            if line[0].isdigit() and line[1:3] in (". ", ") "):
                line = line.split(" ", 1)[1].strip()
            lines.append(line)

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
#               Diverse & realistic JAV-style samples
# ────────────────────────────────────────────────────────────────
DIVERSE_SAMPLES = [
    # Polite teasing → escalating
    [
        "ふふっ…そんなに見つめられたら恥ずかしいよ…",
        "もう…触らないでって言ってるのに…",
        "あっ…そこ…気持ちいい…",
        "だめっ、イっちゃう…！",
    ],

    # Whisper-style error + horny girlfriend
    [
        "はぁはぁ…もう我慢できないよぉ…",
        "おちんぽ…早く入れてぇ…",
        "奥まで…ずんずんって…してぇ♡",
        "あっ、だめ、そこぉ！イく、イっちゃうよぉ！",
    ],

    # Hotel affair style — nervous at first
    [
        "こんなところで…本当にいいの…？",
        "声…出ちゃうから…手で口塞いで…",
        "んっ…！すごい…奥まで入ってる…",
        "もっと…激しくして…お願い…！",
    ],

    # Very vulgar / intense scene
    [
        "マンコぐちゃぐちゃだよ…見て…",
        "お前のチンポでめちゃくちゃにされてる…♡",
        "中に出して！奥にいっぱい出してぇ！",
        "あぁっ！イク！一緒にイこぉ！！",
    ],

    # Single long line (common in some transcriptions)
    "んっ…あっ…だめぇ…そんなに激しくされたら…おかしくなっちゃうよぉ…！もう…イきたい…イかせてぇ…！",

    # Teasing / denial style
    [
        "まだイっちゃダメだよ？♡",
        "我慢して…もっと気持ちよくしてあげるから…",
        "ほら…おちんぽビクビクしてる…かわいい…",
        "いいよ…もう我慢しなくていい…出して？",
    ],
]


def main():
    console.rule("JAV Dialogue Translation — Fiendish 3B", style="magenta")

    for i, sample in enumerate(DIVERSE_SAMPLES, 1):
        console.print(f"\n[bold underline]Sample {i}[/bold underline]")

        console.print("[bold magenta]Japanese:[/bold magenta]")
        for line in sample:
            console.print(f"  {line}")

        console.print("\n[bold cyan]Translating…[/bold cyan]")
        time.sleep(0.4)  # just for visual pacing

        english = translate_jp_to_en(sample)

        console.print("[bold green]English:[/bold green]")
        console.print(english)
        console.print("─" * 60)


if __name__ == "__main__":
    main()