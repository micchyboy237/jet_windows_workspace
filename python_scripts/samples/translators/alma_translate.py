from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama
from rich import print as rprint
from tqdm import tqdm


def build_translation_prompt(source_lang: str, target_lang: str, text: str) -> str:
    """
    Builds the prompt format used by ALMA-7B-Ja-V2 for bidirectional translation.

    Examples from model usage:
    - Ja → En: "Translate this from Japanese to English:\nJapanese: <text>\nEnglish:"
    - En → Ja: "Translate this from English to Japanese:\nEnglish: <text>\nJapanese:"
    """
    return (
        f"Translate this from {source_lang} to {target_lang}:\n"
        f"{source_lang}: {text}\n"
        f"{target_lang}:"
    )


def translate(
    llm: Llama,
    source_lang: str,
    target_lang: str,
    text: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generates translation for a single text input."""
    prompt = build_translation_prompt(source_lang, target_lang, text)

    rprint(f"[bold cyan]Prompt:[/bold cyan]\n{prompt}")

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=False,
        stop=["\n"],  # Stop at newline to avoid extra output
    )

    translation = output["choices"][0]["text"].strip()
    rprint(f"[bold green]Translation:[/bold green] {translation}")

    return translation


def main() -> None:
    model_path = r"~\.cache\llama.cpp\translators\ALMA-7B-Ja-V2.Q4_K_M.gguf"
    model_path = Path(model_path).expanduser().resolve()  # Update to your downloaded file path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load model with GPU offload (adjust n_gpu_layers based on VRAM fit; -1 for all)
    rprint("[bold yellow]Loading model...[/bold yellow]")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=35,  # Offload ~35 layers to GPU (GTX 1660 6GB fits Q4_K_M 7B with ~5.2GB VRAM usage + KV cache headroom)
        n_batch=512,
        verbose=False,
        use_mlock=True,  # Lock model in RAM to prevent swapping (leverages 16GB system RAM)
        flash_attn=True,  # Enable Flash Attention for faster inference on supported models
    )
    rprint("[bold green]Model loaded with CUDA offload[/bold green]")

    # Example translations
    examples = [
        ("Japanese", "English", "今日の夕食はピザです。"),
        ("English", "Japanese", "Today's dinner is pizza."),
        ("Japanese", "English", "生成AIは近年急速に進化しています。"),
        ("Japanese", "English", "世界各国が水面架で知列な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、戦"),
        ("Japanese", "English", "争を加わだてるオスタニア政府要順の動向をさせ、"),
    ]

    for source, target, text in tqdm(examples, desc="Translating examples"):
        translate(llm, source, target, text)


if __name__ == "__main__":
    main()