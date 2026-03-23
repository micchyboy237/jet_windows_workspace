from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LogitsProcessorList

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"

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
    "temperature": 0.35,
    "top_p": 0.90,
    "top_k": 40,
    "typical_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.18,
    "stop": ["\n\n", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
    # "logprobs": True,
    # "top_logprobs": 3
}


SYSTEM_PROMPT = """You are an expert Japanese-to-English translator for anime-style erotic dialogue and subtitles.

Translate faithfully and naturally. Preserve tone, emotion, pauses (...), moans/gasp sounds (ahh, nngh, hah, etc.), symbols (♡) and original speaker perspective.
Output ONLY the pure English translation — nothing else."""

USER_PROMPT = """Translate the following Japanese text to natural English:

{japanese_text}"""


class ReferenceExample(TypedDict):
    ja: str
    en: str


DEFAULT_REFERENCE_EXAMPLES: List[ReferenceExample] = [
    # Erotic / sensual (1st-person) — NO number
    {"ja": "恥ずかしい…見ないでください…", "en": "It's so embarrassing... Please don't look..."},
    {"ja": "お願い…もっと激しくして…壊して…！", "en": "Please... do it harder... Break me...!"},

    # Erotic with count (numeric)
    {"ja": "んっ…もう3回イッちゃった…", "en": "Nngh... I've already come 3 times..."},

    # Cute / playful with age (numeric)
    {"ja": "きゃっ！もう18歳なのに、まだ子供扱い～♡", "en": "Kyaa! I'm already 18 but you're still treating me like a kid~♡"},

    # Angry / tsundere with repetition (numeric)
    {"ja": "ばか！3回も言ったでしょ！", "en": "Idiot! I already said it 3 times!"},

    # Sad / emotional with duration (numeric)
    {"ja": "…ごめんね。あと1年で会えなくなるかも…", "en": "...I'm sorry. In just 1 year we might not be able to meet anymore..."},

    # Neutral conversation with date (numeric)
    {"ja": "2024年3月15日に会いましょう。", "en": "Let's meet on March 15, 2024."},

    # Neutral narrative / third-person with measurement (numeric)
    {"ja": "彼女は5メートル先に静かに立っていた。", "en": "She was quietly standing 5 meters ahead."},

    # Playful / teasing with count (numeric)
    {"ja": "ふふっ、10回目だよ？もう忘れちゃったの？", "en": "Hehe, this is the 10th time, you know? Did you already forget?"},
]

llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)


class BanFirstTokenProcessor:
    """
    Bans specified strings from being generated as the first token(s).
    Uses llm.tokenizer.encode() to convert strings → token IDs.
    """
    def __init__(
        self,
        banned_strings: List[str],
        tokenizer,                    # pass llm.tokenizer (the object)
        ban_all_tokens: bool = False, # if True, bans every token in the strings
    ):
        self.tokenizer = tokenizer    # LlamaTokenizer instance
        self.first_step = True
        
        banned_token_ids: Set[int] = set()
        
        for text in banned_strings:
            if not text.strip():
                continue
                
            # Correct call: use .encode()
            tokens = self.tokenizer.encode(
                text,
                add_bos=False,       # no BOS needed for prefix banning
                special=True
            )
            
            if tokens:
                if ban_all_tokens:
                    banned_token_ids.update(tokens)
                else:
                    # Safer & usually sufficient: only ban possible *starting* tokens
                    banned_token_ids.add(tokens[0])
        
        self.banned_token_ids = banned_token_ids
        
        # Debug print – very useful to see what you're actually banning
        print("Banned first-token IDs:", sorted(self.banned_token_ids))
        for s in banned_strings:
            tks = self.tokenizer.encode(s, add_bos=False, special=True)
            print(f"  '{s}' → tokens {tks}")

    def __call__(
        self,
        input_ids: npt.NDArray[np.intc],
        scores: npt.NDArray[np.single],
    ) -> npt.NDArray[np.single]:
        if self.first_step and self.banned_token_ids:
            for tid in self.banned_token_ids:
                if 0 <= tid < scores.shape[-1]:
                    scores[tid] = -np.inf
            self.first_step = False
        return scores

# The strings you most likely want to block
banned_starts = [
    "assistant",
    # " assistant",  # ← very common (leading space)
    "Assistant",
    # " assistant:",
    # "assistant:",
]

no_assistant_first = BanFirstTokenProcessor(
    banned_strings=banned_starts,
    tokenizer=llm.tokenizer(),  # ← pass your tokenizer here
    ban_all_tokens=False,  # usually best — only block starting tokens
)


def _build_translation_messages(
    japanese_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    reference_examples: List[ReferenceExample] = DEFAULT_REFERENCE_EXAMPLES,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if reference_examples:
        for example in reference_examples:
            messages.append(
                {
                    "role": "user",
                    "content": USER_PROMPT.format(japanese_text=example["ja"].strip()),
                }
            )
            messages.append({"role": "assistant", "content": example["en"].strip()})

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
    except Exception:
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
    temperature: float = 0.5,
    history: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> TranslationResult:
    if not ja_text or not ja_text.strip():
        return {"text": "", "log_prob": None, "confidence": None, "quality": "N/A"}

    messages = _build_translation_messages(ja_text, history)

    completion_params: Dict[str, Any] = {
        **TRANSLATION_DEFAULTS,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    if enable_scoring:
        completion_params["logprobs"] = True
        completion_params["top_logprobs"] = 1

    response = llm.create_chat_completion(
        messages=messages,
        seed=3407,  # for reproducibility
        logits_processor=LogitsProcessorList([no_assistant_first]),
        **completion_params,
    )
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRANSLATION_DEFAULTS["max_tokens"],
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TRANSLATION_DEFAULTS["temperature"],
        help="Sampling temperature",
    )
    parser.add_argument(
        "--scoring",
        "--enable-scoring",
        action="store_true",
        default=False,
        help="Enable logprobs and confidence scoring (slower)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        default=False,
        help="Disable rich console formatting (plain output)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # You can override MODEL_PATH here if you want to support --model
    # MODEL_PATH = args.model   # ← uncomment if you want to use CLI model path
    # llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)  # reload if needed
    # -------------------------------------------------------------------------

    if args.no_rich:
        # Very simple plain output mode
        result = translate_japanese_to_english(
            ja_text=args.text,
            max_tokens=args.max_tokens,
            enable_scoring=args.scoring,
            temperature=args.temperature,
            history=None,
        )
        print("Japanese:")
        print(args.text)
        print("\nEnglish:")
        print(result["text"])
        if args.scoring:
            print(f"\nlogprob   : {result['log_prob']}")
            print(f"confidence : {result['confidence']}")
            print(f"quality    : {result['quality']}")
    else:
        console.rule("Japanese → English Translation", style="bold cyan")

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

        console.print(
            f"[dim]Translating with temperature={args.temperature}  "
            f"max_tokens={args.max_tokens}  scoring={args.scoring}[/]"
        )

        result = translate_japanese_to_english(
            ja_text=args.text,
            max_tokens=args.max_tokens,
            enable_scoring=args.scoring,
            temperature=args.temperature,
            history=None,
        )

        metrics = ""
        if args.scoring:
            metrics = (
                f"\n\n[dim]Metrics:[/] logprob = {result['log_prob']}   "
                f"confidence = {result['confidence']}   quality = {result['quality']}"
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

    result_json_path = OUTPUT_DIR / "translation_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]Saved JSON result to:[/bold green] {result_json_path}")
