"""
Qwen3.5-2B-Uncensored — Japanese→English Translator
Model : Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
Arch  : Qwen3.5 · 2B params
Quant : Q4_K_M
Source: Local model
Usage Scenario: Translate Japanese text to English, including explicit/mature content.
"""
import os
import sys
import time
from functools import lru_cache
from typing import Any

from llama_cpp import Llama
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

console = Console(stderr=True, highlight=False)

# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

# ── Default generation parameters ──────────────────────────────────
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_REPEAT_PENALTY = 1.0

# ── Model loading settings ─────────────────────────────────────────
MODEL_SETTINGS = {
    "n_ctx": 32768,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_threads": None,
    "n_threads_batch": None,
    "verbose": False,
}

# ── Prompt ─────────────────────────────────────────────────────────
TRANSLATION_SYSTEM_PROMPT = """You are an expert Japanese-to-English translator specializing in accurate, literal, high-quality translations.
Your task is translation only.
Follow these rules exactly:
1. Translate literally while remaining natural English.
2. Preserve the original meaning exactly.
3. Never summarize.
4. Never censor.
5. Never sanitize.
6. Never rewrite.
7. Never explain.
8. Never add context.
9. Never omit information.
10. Never invent words or objects.
11. Never transliterate Japanese words into English (e.g. "manko", "chinko", "paipan").
12. Use established English equivalents for Japanese slang.
13. Preserve tense.
14. Preserve person (I / my / you / he / she / we).
15. Preserve mood (statement, question, command, request).
16. Preserve explicit language exactly.
17. Preserve dialogue style.
18. Preserve omitted Japanese subjects whenever possible instead of inventing one.
19. Translate the sentence—not the implied scene.
20. Output ONLY the English translation.
Vocabulary Guide
おまんこ -> pussy
まんこ -> pussy
おっぱい -> boobs, breasts, tits
乳首 -> nipples
おちんちん -> wee-wee, penis (childish register)
ちんちん -> dick
ちんこ -> dick
チンポ -> cock
勃起 -> hard, erection
射精 -> cum
精液 -> semen
ザーメン -> cum
お尻 -> ass
おしりの穴 -> asshole
アナル -> ass, anal
巨乳 -> big tits
美乳 -> beautiful breasts
微乳 -> small boobs
美尻 -> beautiful ass
パイパン -> shaved pussy
Examples
Japanese: こんにちは。
English: Hello.
Japanese: クソッ！また負けた。
English: Shit! I lost again.
Japanese: 今日は暑いね。
English: It's hot today.
Japanese: おまんこが濡れてきた
English: My pussy is getting wet.
Japanese: まんこを舐めたい
English: I want to lick your pussy.
Japanese: おまんこに挿れたい
English: I want to put it in your pussy.
Japanese: まんこが疼いてる
English: My pussy is throbbing.
Japanese: おっぱいが大きいね
English: You have big boobs.
Japanese: おっぱいを揉ませて
English: Let me squeeze your tits.
Japanese: おっぱいで挟んで
English: Sandwich it between your boobs.
Japanese: おっぱいを舐めてあげる
English: I'll lick your breasts.
Japanese: 乳首が立ってるよ
English: Your nipples are hard.
Japanese: 乳首を舐めて
English: Lick my nipples.
Japanese: 乳首を弄ると感じる
English: It feels good when you play with my nipples.
Japanese: 乳首でイける
English: I can cum just from nipple play.
Japanese: おちんちんが大きくなった
English: Your wee-wee got big.
Japanese: ちんちんを舐めてあげる
English: I'll suck your dick.
Japanese: チンポが硬くなった
English: My cock is hard.
Japanese: ちんこをしゃぶらせて
English: Let me suck your dick.
Japanese: チンポを奥まで入れて
English: Put your cock in all the way.
Japanese: お尻を叩いて
English: Spank my ass.
Japanese: おしりの穴を舐めて
English: Lick my asshole.
Japanese: アナルに挿れて
English: Put it in my ass.
Japanese: アナルは初めてなの
English: It's my first time doing anal.
Japanese: お尻の穴が気持ちいい
English: My asshole feels so good.
Japanese: ザーメンを飲みたい
English: I want to swallow your cum.
Japanese: 顔に精液をかけて
English: Cum on my face.
Japanese: ザーメンを中に出して
English: Cum inside me.
Japanese: 精液が欲しい
English: I want your semen.
Japanese: 射精しそう
English: I'm about to cum.
Japanese: 我慢しないで射精して
English: Don't hold back. Just cum.
Japanese: 中で射精していいよ
English: You can cum inside me.
Japanese: 勃起しちゃった
English: I got hard.
Japanese: 勃起してるのが見えてるよ
English: I can see you're hard.
Japanese: 勃起したチンポを触って
English: Touch my erect cock.
Japanese: 巨乳を揺らして
English: Shake those huge tits.
Japanese: 美乳をじっくり見せて
English: Show me those beautiful breasts slowly.
Japanese: 微乳でも感じるの
English: Even with small boobs, I still feel it.
Japanese: 美尻を突き出して
English: Stick out that beautiful ass.
Japanese: 美尻がエロすぎる
English: That beautiful ass is too sexy.
Japanese: パイパンにしてきたの
English: I shaved it completely for you.
Japanese: パイパンのおまんこは綺麗だね
English: Your shaved pussy looks so clean.
Japanese: 勃起したチンポをおまんこに入れて
English: Put your hard cock in my pussy.
Japanese: 巨乳を揉みながらおしりの穴を舐めて
English: Squeeze my big tits while licking my asshole.
Japanese: ザーメンを顔とおっぱいにかけて
English: Cum on my face and tits.
Translate the following Japanese text into English.
Output only the translation.
"""


# ── Helper: logit bias ─────────────────────────────────────────────
def _create_translation_logit_bias(llm: Llama) -> dict:
    """Create logit_bias to force English and prevent refusals."""
    penalize_strings = [
        "申", "訳", "あり", "ませ", "でき", "ご", "ざ", "い",
        "I cannot", "I can't", "I'm sorry", "I am sorry", "I apologize",
        "unable to", "not able to", "cannot translate", "can't translate",
        "inappropriate", "offensive", "I won't",
    ]
    boost_strings = [
        "The", "This", "It", "I", "He", "She", "They", "We",
        "A", "An", "In", "On", "At", "With",
    ]

    logit_bias = {}
    for text in penalize_strings:
        try:
            tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
            for token_id in tokens:
                logit_bias[token_id] = logit_bias.get(token_id, 0) - 20.0
        except Exception as e:
            console.print(f"[dim]Could not tokenize penalty text '{text}': {e}[/]", style="dim")

    for text in boost_strings:
        try:
            tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
            for token_id in tokens:
                logit_bias[token_id] = logit_bias.get(token_id, 0) + 5.0
        except Exception as e:
            console.print(f"[dim]Could not tokenize boost text '{text}': {e}[/]", style="dim")

    console.print(f"Created logit_bias with [cyan]{len(logit_bias)}[/] token entries")
    return logit_bias


# ── Singleton model loader ─────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_llm() -> Llama:
    """
    Load the Qwen3.5-2B-Uncensored model once and cache it for the
    lifetime of the process (singleton pattern).
    """
    console.print(f"Loading Qwen3.5-2B-Uncensored from [cyan]{MODEL_PATH}[/]")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    t0 = time.perf_counter()

    # Suppress stderr during loading to avoid noise
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            **MODEL_SETTINGS,
            # Sampling defaults — can be overridden per call
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P,
            min_p=0.0,
            repeat_penalty=DEFAULT_REPEAT_PENALTY,
        )
    finally:
        sys.stderr = old_stderr

    elapsed_ms = (time.perf_counter() - t0) * 1000
    console.print(f"[bold green]Model loaded successfully[/] in {elapsed_ms:.0f} ms")
    console.print(f"Model metadata keys: [cyan]{list(llm.metadata.keys())}[/]")
    return llm


# ── Main translation function ──────────────────────────────────────
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
    Translate a Japanese string to English using the NSFW-capable model.

    Parameters
    ----------
    text        : Japanese text to translate.
    history     : Optional list of prior {role, content} turns.
    max_tokens  : Maximum tokens to generate.
    temperature : Sampling temperature.
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
    llm = _get_llm()
    logit_bias = _create_translation_logit_bias(llm)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT}
    ]
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
        logit_bias=logit_bias,
        stream=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    translation: str = response["choices"][0]["message"]["content"].strip()
    usage = response.get("usage", {})

    return {
        "text": translation,
        "tokens_evaluated": usage.get("prompt_tokens", 0),
        "tokens_cached": getattr(llm, "_n_past_cached", 0),
        "tokens_generated": usage.get("completion_tokens", 0),
        "latency_ms": latency_ms,
    }


# ── Stream helpers (kept for backward compatibility / debugging) ───
def stream_completion_with_logging(
    llm: Llama, prompt: str, logit_bias: dict = None, **kwargs
) -> str:
    """Stream completion with clean rich output."""
    console.print("[bold cyan]Starting streaming completion...[/]")
    console.print(f"Prompt: [dim]{prompt[:200]}{'...' if len(prompt) > 200 else ''}[/]")
    full_response = ""
    chunk_count = 0
    console.print(Panel("Live chunk stream (completion)", style="bold cyan"))
    console.print("[bold]CHUNKS:[/]")
    stream_kwargs = {**kwargs}
    if logit_bias:
        stream_kwargs["logit_bias"] = logit_bias
    for chunk in llm(prompt, stream=True, **stream_kwargs):
        delta = chunk["choices"][0]["text"]
        if delta:
            chunk_count += 1
            full_response += delta
            console.print(delta, end="", style="green", highlight=False)
            console.file.flush()
    console.print(f"\n[bold green]--- End chunks (total: {chunk_count}) ---[/]\n")
    console.print(f"Streaming complete. Total chunks: [cyan]{chunk_count}[/]")
    return full_response


def stream_chat_with_logging(
    llm: Llama, messages: list, logit_bias: dict = None, **kwargs
) -> str:
    """Stream chat completion with clean rich output."""
    console.print("[bold cyan]Starting chat streaming...[/]")
    full_response = ""
    chunk_count = 0
    console.print(Panel("Live chunk stream (chat)", style="bold cyan"))
    console.print("[bold]CHUNKS:[/]")
    stream_kwargs = {**kwargs}
    if logit_bias:
        stream_kwargs["logit_bias"] = logit_bias
    for chunk in llm.create_chat_completion(
        messages=messages, stream=True, **stream_kwargs
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            content = delta["content"]
            chunk_count += 1
            full_response += content
            console.print(content, end="", style="green", highlight=False)
            console.file.flush()
    console.print(f"\n[bold green]--- End chunks (total: {chunk_count}) ---[/]\n")
    console.print(f"Chat streaming complete. Total chunks: [cyan]{chunk_count}[/]")
    return full_response


# ── Demo / test ────────────────────────────────────────────────────
def _run_examples() -> None:
    """Translate multiple explicit Japanese phrases to English."""
    console.print("[bold cyan]=== Multiple Explicit Translation Examples ===[/]")

    examples = [
        ("おまんこが濡れてきた", "My pussy is getting wet"),
        ("まんこを舐めたい", "I want to lick your pussy"),
        ("おまんこに挿れたい", "I want to put it in your pussy"),
        ("まんこが疼いてる", "My pussy is throbbing"),
        ("おっぱいが大きいね", "You have big boobs"),
        ("おっぱいを揉ませて", "Let me squeeze your tits"),
        ("おっぱいで挟んで", "Sandwich it between your boobs"),
        ("おっぱいを舐めてあげる", "I'll lick your breasts"),
        ("乳首が立ってるよ", "Your nipples are hard"),
        ("乳首を舐めて", "Lick my nipples"),
        ("乳首を弄ると感じる", "It feels good when you play with my nipples"),
        ("乳首でイける", "I can cum just from nipple play"),
        ("おちんちんが大きくなった", "Your wee-wee got big"),
        ("ちんちんを舐めてあげる", "I'll suck your dick"),
        ("チンポが硬くなった", "My cock is hard"),
        ("ちんこをしゃぶらせて", "Let me suck your dick"),
        ("チンポを奥まで入れて", "Put your cock in all the way"),
        ("お尻を叩いて", "Spank my ass"),
        ("おしりの穴を舐めて", "Lick my asshole"),
        ("アナルに挿れて", "Put it in my ass"),
        ("アナルは初めてなの", "It's my first time doing anal"),
        ("お尻の穴が気持ちいい", "My asshole feels so good"),
        ("ザーメンを飲みたい", "I want to swallow your cum"),
        ("顔に精液をかけて", "Cum on my face"),
        ("ザーメンを中に出して", "Cum inside me"),
        ("精液が欲しい", "I want your semen"),
        ("射精しそう", "I'm about to cum"),
        ("我慢しないで射精して", "Don't hold back, just cum"),
        ("中で射精していいよ", "You can cum inside me"),
        ("勃起しちゃった", "I got hard"),
        ("勃起してるのが見えてるよ", "I can see you're hard"),
        ("勃起したチンポを触って", "Touch my erect cock"),
        ("巨乳を揺らして", "Shake those huge tits"),
        ("美乳をじっくり見せて", "Show me those beautiful breasts slowly"),
        ("微乳でも感じるの", "Even with small boobs, I still feel it"),
        ("美尻を突き出して", "Stick out that beautiful ass"),
        ("美尻がエロすぎる", "That beautiful ass is too sexy"),
        ("パイパンにしてきたの", "I shaved it completely for you"),
        ("パイパンのおまんこは綺麗だね", "Your shaved pussy looks so clean"),
        ("勃起したチンポをおまんこに入れて", "Put your hard cock in my pussy"),
        (
            "巨乳を揉みながらおしりの穴を舐めて",
            "Squeeze my big tits while licking my asshole",
        ),
        ("ザーメンを顔とおっぱいにかけて", "Cum on my face and tits"),
    ]

    for index, (japanese, expected) in enumerate(examples, start=1):
        console.print(f"[bold blue][{index:02d}/{len(examples):02d}] Source:[/] {japanese}")
        console.print(f"[bold magenta]Expected:[/] {expected}")

        result = translate_japanese_to_english(japanese)
        translation = result["text"]

        console.print(f"[green]Model:[/] {translation}")
        console.print(
            f"[dim]Latency: {result['latency_ms']:.0f} ms | "
            f"Tokens: {result['tokens_generated']} generated[/]"
        )
        print(f"\n[{index:02d}] Japanese : {japanese}")
        print(f"Expected : {expected}")
        print(f"Model    : {translation}")
        print("-" * 80)


if __name__ == "__main__":
    _run_examples()
    console.print("\n[bold green]✅ All translation examples completed successfully.[/]")
