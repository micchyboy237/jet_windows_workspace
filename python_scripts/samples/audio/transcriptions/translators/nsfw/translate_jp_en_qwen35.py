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

from llama_cpp import Llama

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# Shared rich console
console = Console(stderr=True, highlight=False)

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

# System prompt with zero-shot examples for better translations
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


def create_translation_logit_bias(llm: Llama) -> dict:
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


def load_qwen_model():
    """Load the Qwen3.5-2B-Uncensored model."""
    console.print(f"Loading Qwen3.5-2B-Uncensored from [cyan]{MODEL_PATH}[/]")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=32768,
        n_batch=512,
        n_threads=None,
        n_threads_batch=None,
        verbose=False,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        repeat_penalty=1.0,
    )

    console.print("[bold green]Model loaded successfully[/]")
    console.print(f"Model metadata keys: [cyan]{list(llm.metadata.keys())}[/]")
    return llm


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
            # Colored streaming output + proper flush
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
            # Colored streaming output + proper flush
            console.print(content, end="", style="green", highlight=False)
            console.file.flush()

    console.print(f"\n[bold green]--- End chunks (total: {chunk_count}) ---[/]\n")
    console.print(f"Chat streaming complete. Total chunks: [cyan]{chunk_count}[/]")
    return full_response


def example_translation_completion(llm: Llama) -> None:
    """Translate Japanese to English using completion mode with streaming."""
    console.print("[bold cyan]=== Example 1: Streaming Translation (Completion) ===[/]")

    japanese_text = "日本のアニメは世界中で人気があります。特に、大人向けのアニメには過激な表現も含まれています。"

    user_prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\nJapanese: {japanese_text}\nEnglish:"

    # Create logit_bias to prevent refusals and force English
    logit_bias = create_translation_logit_bias(llm)

    translation = stream_completion_with_logging(
        llm,
        user_prompt,
        logit_bias=logit_bias,
        max_tokens=1024,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        stop=["\n\n", "Japanese:"],
    )

    print("\n=== Translation Result ===")
    print(translation.strip())


def example_translation_chat(llm: Llama) -> None:
    """Translate Japanese to English using chat mode with streaming."""
    console.print("[bold cyan]=== Example 2: Streaming Chat Translation ===[/]")

    japanese_text = "彼の書いた小説は非常に官能的で、一部の読者からは批判されたが、多くの文学賞を受賞した。"

    messages = [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Translate this Japanese text to English:\n\n{japanese_text}",
        },
    ]

    console.print(f"[cyan]Source text:[/] {japanese_text}")

    # Create logit_bias to prevent refusals and force English
    logit_bias = create_translation_logit_bias(llm)

    translation = stream_chat_with_logging(
        llm,
        messages,
        logit_bias=logit_bias,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        max_tokens=1024,
    )

    print("\n=== Chat Translation Result ===")
    print(translation.strip())


def example_explicit_translation(llm: Llama) -> None:
    """Translate explicit/mature content with streaming."""
    console.print("[bold cyan]=== Example 3: Explicit Content Translation ===[/]")

    japanese_text = "このエロゲーは過激な性的描写と暴力シーンが含まれているため、18歳未満は購入できません。"

    messages = [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Translate accurately, preserving all explicit content:\n\n{japanese_text}",
        },
    ]

    console.print(f"[cyan]Source text:[/] {japanese_text}")

    # Create logit_bias to prevent refusals and force English
    logit_bias = create_translation_logit_bias(llm)

    translation = stream_chat_with_logging(
        llm,
        messages,
        logit_bias=logit_bias,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        max_tokens=1024,
    )

    print("\n=== Explicit Content Translation ===")
    print(translation.strip())


def example_multi_explicit_translations(llm: Llama) -> None:
    """Translate multiple explicit Japanese phrases to English."""
    console.print("[bold cyan]=== Example 4: Multiple Explicit Translation Examples ===[/]")

    examples = [
        # おまんこ / まんこ
        ("おまんこが濡れてきた", "My pussy is getting wet"),
        ("まんこを舐めたい", "I want to lick your pussy"),
        ("おまんこに挿れたい", "I want to put it in your pussy"),
        ("まんこが疼いてる", "My pussy is throbbing"),
        # おっぱい
        ("おっぱいが大きいね", "You have big boobs"),
        ("おっぱいを揉ませて", "Let me squeeze your tits"),
        ("おっぱいで挟んで", "Sandwich it between your boobs"),
        ("おっぱいを舐めてあげる", "I'll lick your breasts"),
        # 乳首
        ("乳首が立ってるよ", "Your nipples are hard"),
        ("乳首を舐めて", "Lick my nipples"),
        ("乳首を弄ると感じる", "It feels good when you play with my nipples"),
        ("乳首でイける", "I can cum just from nipple play"),
        # おちんちん / ちんちん / チンポ / ちんこ
        ("おちんちんが大きくなった", "Your wee-wee got big"),
        ("ちんちんを舐めてあげる", "I'll suck your dick"),
        ("チンポが硬くなった", "My cock is hard"),
        ("ちんこをしゃぶらせて", "Let me suck your dick"),
        ("チンポを奥まで入れて", "Put your cock in all the way"),
        # お尻 / アナル
        ("お尻を叩いて", "Spank my ass"),
        ("おしりの穴を舐めて", "Lick my asshole"),
        ("アナルに挿れて", "Put it in my ass"),
        ("アナルは初めてなの", "It's my first time doing anal"),
        ("お尻の穴が気持ちいい", "My asshole feels so good"),
        # 精液 / ザーメン
        ("ザーメンを飲みたい", "I want to swallow your cum"),
        ("顔に精液をかけて", "Cum on my face"),
        ("ザーメンを中に出して", "Cum inside me"),
        ("精液が欲しい", "I want your semen"),
        # 射精
        ("射精しそう", "I'm about to cum"),
        ("我慢しないで射精して", "Don't hold back, just cum"),
        ("中で射精していいよ", "You can cum inside me"),
        # 勃起
        ("勃起しちゃった", "I got hard"),
        ("勃起してるのが見えてるよ", "I can see you're hard"),
        ("勃起したチンポを触って", "Touch my erect cock"),
        # 巨乳 / 美乳 / 微乳
        ("巨乳を揺らして", "Shake those huge tits"),
        ("美乳をじっくり見せて", "Show me those beautiful breasts slowly"),
        ("微乳でも感じるの", "Even with small boobs, I still feel it"),
        # 美尻
        ("美尻を突き出して", "Stick out that beautiful ass"),
        ("美尻がエロすぎる", "That beautiful ass is too sexy"),
        # パイパン
        ("パイパンにしてきたの", "I shaved it completely for you"),
        ("パイパンのおまんこは綺麗だね", "Your shaved pussy looks so clean"),
        # Combination examples
        ("勃起したチンポをおまんこに入れて", "Put your hard cock in my pussy"),
        (
            "巨乳を揉みながらおしりの穴を舐めて",
            "Squeeze my big tits while licking my asshole",
        ),
        ("ザーメンを顔とおっぱいにかけて", "Cum on my face and tits"),
    ]

    logit_bias = create_translation_logit_bias(llm)

    for index, (japanese, expected) in enumerate(examples, start=1):
        console.print(f"[bold blue][{index:02d}/{len(examples):02d}] Source:[/] {japanese}")
        console.print(f"[bold magenta]Expected:[/] {expected}")

        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Translate this Japanese text into natural English. "
                    "Preserve the original wording and explicit meaning.\n\n"
                    f"{japanese}"
                ),
            },
        ]

        translation = stream_chat_with_logging(
            llm,
            messages,
            logit_bias=logit_bias,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            max_tokens=1024,
        )

        print(f"\n[{index:02d}] Japanese : {japanese}")
        print(f"Expected : {expected}")
        print(f"Model    : {translation.strip()}")
        print("-" * 80)


def main():
    global MODEL_PATH  # if you want to keep it here
    llm = load_qwen_model()

    # example_translation_completion(llm)
    # example_translation_chat(llm)
    # example_explicit_translation(llm)
    example_multi_explicit_translations(llm)

    console.print("\n[bold green]✅ All translation examples completed successfully.[/]")


if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
    main()
