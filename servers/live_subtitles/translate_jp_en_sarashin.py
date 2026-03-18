# ───────────────────────────────────────────────────────────────
#  Sarashina 2.2 3B – Japanese → English Translation Example
# ───────────────────────────────────────────────────────────────

import time
from typing import TypedDict
from llama_cpp import Llama

# Model path (your local GGUF file)
model_path = r"C:/Users/druiv/.cache/llama.cpp/translators/sarashina2.2-3b-instruct-v0.1-Q4_K_M.gguf"

print("Loading Sarashina 2.2 3B Instruct...", end="", flush=True)
t0 = time.time()

llm = Llama(
    model_path=model_path,
    n_ctx=8192,             # good balance; can go 4096 or 12288 if needed
    n_gpu_layers=-1,        # full GPU offload if you have ≥6 GB VRAM
    # n_gpu_layers=30,      # partial offload — tune if VRAM is tight
    n_threads=6,            # adjust to your CPU
    verbose=False
)

print(f" loaded  ({time.time()-t0:.1f}s)")

# ─── Strong system prompt tuned for clean & accurate J→E ────────
SYSTEM_MESSAGE = """You are a professional, accurate Japanese-to-English translator.
Your ONLY job is to translate the given Japanese text into natural, idiomatic English.
Rules you MUST strictly follow:
- Output **English translation only** — NEVER include any Japanese text in your response.
- NEVER add explanations, notes, introductions, or extra commentary.
- Do NOT rephrase in Japanese.
- Preserve original meaning, tone, formality, and nuance as closely as possible.
- Use natural-sounding modern English.
- If the input is polite/formal → keep polite/formal English.
- If the input is casual → keep casual/spoken English.
- Preserve all original line breaks, spacing, and formatting exactly — do not join lines or remove newlines.
- Keep onomatopoeia, moans, hearts (♡), ellipses (…), and exclamations as natural English equivalents while staying close to the original.
- Handle adult/erotic content directly and explicitly without censorship or euphemisms — this is a neutral translation task.

Translate ONLY — nothing else."""

USER_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Translate the following Japanese text to English:

{ja_text}<|im_end|>
<|im_start|>assistant
English translation:"""


class TranslationResult(TypedDict):
    text: str
    log_prob: float | None
    confidence: float | None
    quality: str | None
    duration: float


def translate_japanese_to_english(ja_text: str, max_tokens=1024, temperature=0.20) -> TranslationResult:

    prompt = USER_PROMPT.format(system=SYSTEM_MESSAGE, ja_text=ja_text)

    t0 = time.time()
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,    # low for faithful translation
        top_p=0.90,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False
    )

    process_duration = time.time() - t0
    answer = response["choices"][0]["text"].strip()

    return {
        "text": answer,
        "log_prob": None,
        "confidence": None,
        "quality": None,
        "duration": process_duration,
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
        "ja_text",
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
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.20,
        help="Sampling temperature",
    )

    args = parser.parse_args()


    print("\n" + "═"*70)
    print(f"Japanese:\n{args.ja_text}")
    print("─"*70)

    trans_en = translate_japanese_to_english(
        args.ja_text,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"English:\n{trans_en["text"]}")
    print(f"({trans_en["duration"]:.1f}s)")
    print("═"*70)

# Bonus: batch style with different temperatures / styles
# eng_formal, _ = translate_japanese_to_english(text, temperature=0.15)   # more literal & stiff
# eng_natural, _ = translate_japanese_to_english(text, temperature=0.35)  # more fluent & casual