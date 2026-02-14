# Recommended installation (run once)
# For Mac M1/Metal:
# pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir \
#     -C cmake.args="-DLLAMA_METAL=ON"

# For Windows NVIDIA GTX 1660:
# pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir \
#     -C cmake.args="-DLLAMA_CUBLAS=ON"

from typing import Optional, Dict, Any
from llama_cpp import Llama
from rich.console import Console

console = Console()

# ── Model configuration ──────────────────────────────────────────────
MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\translators\gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"

MODEL_SETTINGS = {
    "n_ctx": 1024,
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

# Recommended defaults for Japanese → English translation
TRANSLATION_DEFAULTS = {
    "temperature": 0.65,
    "top_p": 0.92,
    "min_p": 0.05,
    # "repeat_penalty": 1.05,
    "max_tokens": 512,
    # "stop": ["<|im_end|>", "<|im_start|>"],
    "echo": False,
    # For confidence scores
    # "logprobs": 3,
}

# You can also try Q5_K_M / Q6_K if you have enough VRAM/RAM (~4–5GB needed)
# Q4_K_M → good speed/quality balance on your GTX 1660 / M1

llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)


def translate_text(
    text: str,
    # temperature: float = 0.65,
    # max_tokens: int = 1024,
    # top_p: float = 0.92,
    # repeat_penalty: float = 1.05,
    **generation_params,
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
    prompt = f"""\
Translate the following Japanese text into natural English.

JA: {text}
EN: """

    params: Dict[str, Any] = {
        "prompt": prompt,
        **TRANSLATION_DEFAULTS,
        **generation_params
    }

    response = llm(
        # prompt,
        # max_tokens=max_tokens,
        # temperature=temperature,
        # top_p=top_p,
        # min_p=0.05,
        # repeat_penalty=repeat_penalty,
        # stop=["<|im_end|>", "<|im_start|>"],
        # echo=False,
        **params
    )

    translated = response["choices"][0]["text"].strip()
    return translated


# ── Example usage ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test sentences (feel free to replace with your real text)
    japanese_samples = [
        """
世界各国が水面下で熾烈な情報戦を繰り広げる時代 にらみ合う2つの国 東のオスタニア、西のウェスタリス
戦争を企てるオスタニア政府要人の動向 戦争を企てるオスタニア政府要人の動向を探るべく
ウェスタリスはオペレーションストリクスを発動 作戦を担うスゴーデエージェントたそがれ 100の顔を使い分ける彼の任務は
家族の顔を使い分ける彼の任務は 家族を作ること 父ロイドフォージャー 精神科医 正体、コードネームたそがれ
母ヨルフォージャー 母、夜フォージャー、市役所職員、正体、殺し屋、コードネーム、イバラ姫、娘。
娘、アーニャフォージャー、正体、心を読むことができるエスパー。
正体、心を読むことができるエスパー、犬、ボンドフォージャー、正体、未来を予知できる超能力権、物狩りのため、疑似家族を作り
、互いに正体を隠した。 二次家族を作り互いに正体を隠した彼らのミッションは続く
""",
    ]

    console.print("[bold magenta]Japanese → English Translation Demo[/bold magenta]\n")

    for i, ja_text in enumerate(japanese_samples, 1):
        console.print(f"[bold white]Example {i}:[/bold white]")
        console.print(f"[dim]{ja_text}[/dim]\n")

        with console.status("[bold green]Translating...[/bold green]"):
            english = translate_text(ja_text)

        console.print("[bold cyan]→ English:[/bold cyan]")
        console.print(f"{english}\n")
        console.rule(style="dim")