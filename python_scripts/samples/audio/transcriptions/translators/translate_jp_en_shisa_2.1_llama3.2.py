# Recommended installation (run once)
# For Mac M1/Metal:
# pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir \
#     -C cmake.args="-DLLAMA_METAL=ON"

# For Windows NVIDIA GTX 1660:
# pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir \
#     -C cmake.args="-DLLAMA_CUBLAS=ON"

from typing import Optional
from llama_cpp import Llama
from rich.console import Console

console = Console()

# ── Model configuration ──────────────────────────────────────────────
MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"

# You can also try Q5_K_M / Q6_K if you have enough VRAM/RAM (~4–5GB needed)
# Q4_K_M → good speed/quality balance on your GTX 1660 / M1

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,           # Full GPU offload (Metal on M1, cuBLAS on NVIDIA)
    n_ctx=8192,                # Good for medium-long Japanese texts
    n_batch=1024,
    verbose=False,
)


def translate_ja_to_en(
    text: str,
    temperature: float = 0.65,
    max_tokens: int = 1024,
    top_p: float = 0.92,
    repeat_penalty: float = 1.05,
) -> str:
    """
    Translate Japanese text to natural, high-quality English using Shisa v2.1 3B
    
    Args:
        text: Japanese input text (can be paragraph, dialogue, novel snippet, etc.)
        temperature: Lower = more literal/accurate, Higher = more creative
        max_tokens: Safety limit for very long outputs
    
    Returns:
        Clean English translation
    """
    prompt = f"""<|im_start|>system
You are an excellent, professional Japanese-to-English translator.
Translate the following Japanese text into natural, idiomatic English.
Preserve nuance, tone, politeness level, and intent as accurately as possible.
Do not add explanations or notes unless explicitly requested.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=0.05,
        repeat_penalty=repeat_penalty,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )

    translated = response["choices"][0]["text"].strip()
    return translated


# ── Example usage ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test sentences (feel free to replace with your real text)
    japanese_samples = [
        "雨のマニラの夜に、ネオンの光が濡れたアスファルトに映り込んでいた。",
        "お疲れ様です。明日の会議資料はもうご確認いただけましたでしょうか？",
        "彼女は静かに目を閉じ、遠い夏の記憶に身を委ねた。蝉の声が、どこか懐かしく響く。",
    ]

    console.print("[bold magenta]Japanese → English Translation Demo[/bold magenta]\n")

    for i, ja_text in enumerate(japanese_samples, 1):
        console.print(f"[bold white]Example {i}:[/bold white]")
        console.print(f"[dim]{ja_text}[/dim]\n")

        with console.status("[bold green]Translating...[/bold green]"):
            english = translate_ja_to_en(ja_text, temperature=0.65)

        console.print("[bold cyan]→ English:[/bold cyan]")
        console.print(f"{english}\n")
        console.rule(style="dim")