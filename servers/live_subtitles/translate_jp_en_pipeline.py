# servers\live_subtitles\translate_jp_en_pipeline.py
import argparse
from typing import Dict, List, Optional, TypedDict

from transformers import Pipeline, pipeline

# =============================================================================
# Global translator (GPU by default)
# =============================================================================
translator: Pipeline = pipeline(
    "translation",
    model="Mitsua/elan-mt-bt-ja-en",
    device=0,  # 0 = GPU; change to -1 for CPU if needed
    # torch_dtype="float16",      # optional: faster + lower VRAM on GPU
)


# Same result structure as LLM version
class TranslationResult(TypedDict):
    text: str
    log_prob: Optional[float]
    confidence: Optional[float]
    quality: str


DEFAULT_JA_TEXT = """
恥ずかしい…見ないでください…
んっ…そこ、弱いんです…
はぁ…はぁ…気持ちいい…
もう…ダメかも…頭おかしくなりそう…
お願い…もっと激しくして…壊して…！
あぁんっ！すごい…奥まで届いてる…♡
出さないで…まだ中にいてて…
""".strip()

# Centralized defaults for consistency
DEFAULT_MAX_TOKENS = 768
DEFAULT_NUM_BEAMS = 4  # Matches model card BLEU evaluation
DEFAULT_TEMPERATURE = 1.0  # Neutral value (ignored in beam search)
DEFAULT_DO_SAMPLE = False  # Deterministic by default


def translate_japanese_to_english(
    ja_text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_scoring: bool = False,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    top_k: int = 50,
    top_p: float = 0.95,
    num_beams: int = DEFAULT_NUM_BEAMS,
    history: Optional[List[Dict[str, str]]] = None,  # unused here
    **kwargs,
) -> TranslationResult:
    if not ja_text or not ja_text.strip():
        return {"text": "", "log_prob": None, "confidence": None, "quality": "N/A"}

    gen_kwargs = {
        "max_length": max_tokens,
        "early_stopping": True,
    }

    if do_sample:
        # Sampling mode (more creative/variable — opt-in only)
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": max(0.1, temperature),
                "top_k": top_k,
                "top_p": top_p,
                "num_return_sequences": 1,
            }
        )
    else:
        # Default: deterministic beam search (best for subtitles/quality)
        gen_kwargs["num_beams"] = num_beams

    result = translator(
        ja_text,
        **gen_kwargs,
    )

    if not result or "translation_text" not in result[0]:
        return {
            "text": "[Translation error]",
            "log_prob": None,
            "confidence": None,
            "quality": "low",
        }

    en_text = result[0]["translation_text"].strip()

    # Metrics placeholders (no real logprobs in standard pipeline)
    if enable_scoring:
        log_prob = None
        confidence = None
        quality = "high" if not do_sample else "variable (sampling)"
    else:
        log_prob = confidence = None
        quality = "N/A"

    return {
        "text": en_text,
        "log_prob": log_prob,
        "confidence": confidence,
        "quality": quality,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Japanese → English subtitle translator using Mitsua/elan-mt-bt-ja-en (deterministic by default)"
    )
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        default=DEFAULT_JA_TEXT,
        help="Japanese text to translate (multi-line ok)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output length (tokens)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (only used when --do-sample is enabled)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=DEFAULT_DO_SAMPLE,
        help="Enable probabilistic sampling (less deterministic, more varied output)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help="Number of beams for deterministic beam search (default: 4)",
    )
    parser.add_argument(
        "--scoring",
        "--enable-scoring",
        action="store_true",
        default=False,
        help="Show placeholder metrics (no real logprobs available)",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        default=False,
        help="Disable rich console formatting",
    )

    args = parser.parse_args()

    # Rich formatting (optional)
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        use_rich = not args.no_rich
    except ImportError:
        use_rich = False

    if use_rich:
        console.rule(
            "Japanese → English Translation (Deterministic Pipeline)", style="bold cyan"
        )
        console.print(
            Panel(
                args.text,
                title="[bold magenta]Japanese Input[/]",
                border_style="magenta",
                padding=(1, 2),
                expand=False,
                box=box.ROUNDED,
            )
        )
        mode = "sampling" if args.do_sample else f"beam search (beams={args.num_beams})"
        console.print(
            f"[dim]Translating with max_length={args.max_tokens}, mode={mode}, "
            f"temperature={args.temperature} (ignored in beam search), scoring={args.scoring}[/]"
        )

        result = translate_japanese_to_english(
            ja_text=args.text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_scoring=args.scoring,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            history=None,
        )

        metrics = ""
        if args.scoring:
            metrics = (
                f"\n\n[dim]Metrics:[/] logprob = {result['log_prob']} "
                f"confidence = {result['confidence']} quality = {result['quality']}"
            )

        console.print(
            Panel(
                result["text"] + metrics,
                title="[bold green]English Translation[/]",
                border_style="green",
                padding=(1, 2),
                expand=False,
                box=box.DOUBLE,
            )
        )
    else:
        result = translate_japanese_to_english(
            ja_text=args.text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_scoring=args.scoring,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            history=None,
        )
        print("Japanese:")
        print(args.text)
        print("\nEnglish:")
        print(result["text"])
        if args.scoring:
            print(f"\nlogprob    : {result['log_prob']}")
            print(f"confidence : {result['confidence']}")
            print(f"quality    : {result['quality']}")
