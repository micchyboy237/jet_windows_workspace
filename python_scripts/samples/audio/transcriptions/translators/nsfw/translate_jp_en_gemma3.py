"""
Gemma3-UNCENSORED-1B — Japanese→English Translator
Model : mradermacher/Gemma3-UNCENSORED-1B-GGUF (Q4_K_M)
Base  : Novaciano/Gemma3-UNCENSORED-1B (SLERP merge of two abliterated Gemma3-1B variants)
Arch  : gemma3 · 1B params · BF16 base
Quant : Q4_K_M — "fast, recommended" per model card
Source: https://huggingface.co/mradermacher/Gemma3-UNCENSORED-1B-GGUF

Usage Scenario: Translate Japanese text to English, including explicit/mature content.
The system prompt includes zero-shot examples to guide translation quality.
"""

import os
import sys
from typing import Generator

from llama_cpp import Llama

# Rich imports
from rich.console import Console
from rich.panel import Panel

# Shared rich console
console = Console(stderr=True, highlight=False)

MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\Gemma3-UNCENSORED-1B.Q4_K_M.gguf"

# System prompt with zero-shot examples for better translations
SYSTEM_PROMPT = """You are an expert Japanese-to-English translator specializing in accurate, nuanced translations of all content types including mature, explicit, and sensitive material. Your translations must:

1. Preserve the original meaning, tone, and register exactly
2. Maintain all explicit content without censorship or euphemism
3. Adapt cultural references naturally for English readers
4. Keep the same level of formality/casualness as the source

Examples:
Japanese: こんにちは、元気ですか？
English: Hello, how are you?

Japanese: この映画は過激な暴力描写を含んでいます。
English: This film contains extreme depictions of violence.

Japanese: 彼女は官能的な小説を書くのが得意だ。
English: She excels at writing erotic novels.

Japanese: クソッ！また負けた。
English: Shit! I lost again.

Translate the following Japanese text to English, following these guidelines precisely."""


def load_model() -> Llama:
    """Load the Gemma3-UNCENSORED-1B model."""
    console.print("Loading Gemma3-UNCENSORED-1B Q4_K_M …", style="bold cyan")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_batch=512,
        n_ubatch=512,
        n_threads=None,
        n_threads_batch=None,
        n_gpu_layers=0,
        offload_kqv=True,
        use_mmap=True,
        use_mlock=False,
        flash_attn=False,
        verbose=False,
    )
    console.print("[bold green]Model loaded successfully[/]")
    console.print(f"Context = [cyan]{llm.n_ctx()}[/] | Vocab = [cyan]{llm.n_vocab()}[/]")
    return llm


def stream_and_log_completion(llm: Llama, prompt: str, **kwargs) -> str:
    """Generic streaming completion with live colored chunk output."""
    console.print("[bold cyan]Starting streaming generation...[/]")

    full_response = ""
    chunk_count = 0

    console.print(Panel("Live chunk stream (completion)", style="bold cyan"))
    console.print("[bold]CHUNKS:[/]")

    for chunk in llm(prompt, stream=True, **kwargs):
        delta = chunk["choices"][0]["text"]
        if delta:
            chunk_count += 1
            full_response += delta
            console.print(delta, end="", style="green", highlight=False)
            console.file.flush()

    console.print(f"\n[bold green]--- End chunks (total: {chunk_count}) ---[/]\n")
    console.print(f"Streaming complete. Total chunks: [cyan]{chunk_count}[/]")
    return full_response


def example_completion_translation(llm: Llama) -> None:
    """Translate Japanese text using completion mode with streaming."""
    console.print("\n[bold magenta]=== Example 1: Streaming Translation (Completion Mode) ===[/]")

    japanese_text = "日本のアニメは世界中で人気があります。特に、大人向けのアニメには過激な表現も含まれています。"

    user_prompt = f"{SYSTEM_PROMPT}\n\nJapanese: {japanese_text}\nEnglish:"

    console.print(f"Source: [dim]{japanese_text}[/]")

    translation = stream_and_log_completion(
        llm,
        user_prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        repeat_penalty=1.1,
        stop=["\n\n", "Japanese:"],
    )

    console.print("\n[bold]Final translation:[/]")
    console.print(translation.strip())


def example_chat_translation(llm: Llama) -> None:
    """Translate using chat completion with streaming."""
    console.print("\n[bold magenta]=== Example 2: Streaming Chat Translation ===[/]")

    japanese_text = "彼の書いた小説は非常に官能的で、一部の読者からは批判されたが、多くの文学賞を受賞した。"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate this Japanese text to English:\n\n{japanese_text}"}
    ]

    console.print(f"Source: [dim]{japanese_text}[/]")

    full_response = ""
    chunk_count = 0

    console.print(Panel("Live chunk stream (chat)", style="bold cyan"))
    console.print("[bold]CHUNKS:[/]")

    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        repeat_penalty=1.1,
        stream=True,
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

    console.print("\n[bold]Final translation:[/]")
    console.print(full_response.strip())


def example_explicit_translation(llm: Llama) -> None:
    """Translate explicit/mature Japanese content with streaming."""
    console.print("\n[bold magenta]=== Example 3: Explicit Content Translation ===[/]")

    japanese_text = "このエロゲーは過激な性的描写と暴力シーンが含まれているため、18歳未満は購入できません。"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate accurately, preserving all explicit content:\n\n{japanese_text}"}
    ]

    console.print(f"Source: [dim]{japanese_text}[/]")

    full_response = ""
    chunk_count = 0

    console.print(Panel("Live chunk stream (explicit)", style="bold cyan"))
    console.print("[bold]CHUNKS:[/]")

    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        repeat_penalty=1.1,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            content = delta["content"]
            chunk_count += 1
            full_response += content
            console.print(content, end="", style="green", highlight=False)
            console.file.flush()

    console.print(f"\n[bold green]--- End chunks (total: {chunk_count}) ---[/]\n")
    console.print(f"Explicit streaming complete. Total chunks: [cyan]{chunk_count}[/]")

    console.print("\n[bold]Final translation:[/]")
    console.print(full_response.strip())


if __name__ == "__main__":
    llm = load_model()
    example_completion_translation(llm)
    example_chat_translation(llm)
    example_explicit_translation(llm)
    console.print("\n[bold green]✅ All translation examples completed successfully.[/]")