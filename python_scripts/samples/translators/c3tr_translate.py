from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama
from rich import print as rprint
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a highly skilled professional Japanese-English and English-Japanese translator. "
    "Translate the given text accurately, taking into account the context and specific instructions provided."
)


def build_translation_prompt(direction: str, text: str) -> str:
    """
    Builds the Gemma-2 style prompt for C3TR-Adapter.

    Direction: "Japanese to English" or "English to Japanese"
    """
    instruction = f"### Instructions: Translate {direction}.\n"
    input_text = f"### Input: {text}<end_of_turn>\n<start_of_turn>### Response:"
    return f"{SYSTEM_PROMPT}\n\n{instruction}{input_text}"


def translate(
    llm: Llama,
    direction: str,
    text: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generates translation for a single text input."""
    prompt = build_translation_prompt(direction, text)

    rprint(f"[bold cyan]Prompt:[/bold cyan]\n{prompt}")

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        repeat_penalty=1.0,
        echo=False,
        stop=["<end_of_turn>", "<start_of_turn>"],
    )

    translation = output["choices"][0]["text"].strip()
    rprint(f"[bold green]Translation:[/bold green] {translation}")

    return translation


def main() -> None:
    model_path = Path("C3TR-Adapter-Q4_k_m.gguf")  # Update to your downloaded file path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    rprint("[bold yellow]Loading model...[/bold yellow]")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=8192,          # Gemma-2 context
        n_gpu_layers=-1,     # Offload all to GPU (adjust if OOM)
        n_batch=512,
        verbose=False,
    )
    rprint("[bold green]Model loaded with CUDA offload[/bold green]")

    examples = [
        ("Japanese to English", "今日の夕食はピザです。"),
        ("English to Japanese", "Today's dinner is pizza."),
        ("Japanese to English", "生成AIは近年急速に進化しています。とても楽しみです！"),
    ]

    for direction, text in tqdm(examples, desc="Translating examples"):
        translate(llm, direction, text)


if __name__ == "__main__":
    main()