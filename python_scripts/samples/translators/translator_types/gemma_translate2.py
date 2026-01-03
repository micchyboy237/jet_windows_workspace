# translate_ja_en_gemma.py
# Requirements:
#   pip install llama-cpp-python rich
#   (preferably the CUDA version if you built llama.cpp with GPU support)

from typing import Literal
from rich.console import Console
from rich.markdown import Markdown
from llama_cpp import Llama

console = Console()

# ────────────────────────────────────────────────
# Config – adjust paths & settings to match your setup
# ────────────────────────────────────────────────

MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators\gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"
)

# From your Start-LlamaServer-Llm.ps1 for option 9
CTX_SIZE        = 2048
GPU_LAYERS      = -1          # = 999 → full GPU offload (GTX 1660 6 GB should handle 2B Q4 fine)
CACHE_TYPE_K    = "q8_0"
CACHE_TYPE_V    = "q8_0"
TOK_ADD_BOS     = False       # --override-kv tokenizer.ggml.add_bos_token=bool:false

# ────────────────────────────────────────────────
# Initialize model
# ────────────────────────────────────────────────

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_gpu_layers=GPU_LAYERS,
    flash_attn=True,               # usually good on modern GPUs
    logits_all=False,
    cache_type_k=CACHE_TYPE_K,     # helps with quality / speed on small context
    cache_type_v=CACHE_TYPE_V,
    verbose=False,                 # set True when debugging
    # Important for your custom model
    tokenizer_kwargs={"add_bos_token": TOK_ADD_BOS},
    # chat_format=None               # ← do NOT force any built-in format
)


def translate_ja_to_en(
    text: str,
    temperature: float = 0.3,
    top_p: float = 0.90,
    max_tokens: int = 768,
) -> str:
    """Send Japanese text → model → clean English output"""
    # Very simple instruction that most ja→en fine-tunes understand
    # Feel free to experiment with more/less verbose variants
    prompt = f"""Translate the following Japanese text to natural English.

Japanese:
{text}

English:""".strip()

    # ── Generation ───────────────────────────────────────
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        repeat_penalty=1.05,
        stop=["</s>", "<|eot_id|>", "<eos>", "\n\n\n"],  # common stop tokens
        echo=False,
    )

    # Extract generated text (most custom translators put answer right after prompt)
    generated = response["choices"][0]["text"].strip()

    # Minimal post-processing – remove leftover prompt leakage if any
    if generated.startswith("English:"):
        generated = generated.split("English:", 1)[-1].strip()

    return generated


# ────────────────────────────────────────────────
# Interactive loop
# ────────────────────────────────────────────────

def main():
    console.print("[bold cyan]Japanese → English translator (gemma-2-2b-jpn-it-translate)[/]", justify="center")
    console.print("[dim]Ctrl+C to quit | empty line = quit\n[/]")

    while True:
        try:
            ja_text = console.input("[bold bright_white]Japanese:[/] ").strip()
            if not ja_text:
                console.print("[yellow]Goodbye.[/]")
                break

            with console.status("[progress]Translating...", spinner="dots"):
                english = translate_ja_to_en(
                    ja_text,
                    temperature=0.25,     # lower = more deterministic
                    top_p=0.92,
                    max_tokens=1024,
                )

            console.print()
            console.print(Markdown(f"**English:**\n{english}"))
            console.print("─" * 70, style="dim")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            break
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}", style="bold red")


if __name__ == "__main__":
    main()