from typing import Literal, Dict, Any, Union
import time
import uuid

from rich.console import Console
from rich.markdown import Markdown
from llama_cpp import Llama
from llama_cpp.llama_types import CreateCompletionResponse, CompletionChoice, CompletionUsage
from temp_utils2 import split_sentences_ja

console = Console()

# ────────────────────────────────────────────────
# Config – adjust paths & settings to match your setup
# ────────────────────────────────────────────────

MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
)

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
    "repeat_penalty": 1.05,
    "max_tokens": 512,
    "stop": ["<|im_end|>", "<|im_start|>"],
    "echo": False,
    # For confidence scores
    # "logprobs": 3,
}

CTX_SIZE        = 2048
GPU_LAYERS      = -1
CACHE_TYPE_K    = 8
CACHE_TYPE_V    = 8
TOK_ADD_BOS     = False

# ────────────────────────────────────────────────
# Initialize model
# ────────────────────────────────────────────────

llm = Llama(model_path=MODEL_PATH, **MODEL_SETTINGS)

def translate_text(
    text: str,
    # max_tokens: int = 512,
    # logprobs: int | None = None,           # ← new optional param
    # temperature: float = 0.7,
    # top_p: float = 0.9,
    # top_k: int = 40,
    # repeat_penalty: float = 1.1,
    # stop: Union[str, list[str], None] =  ["<|im_end|>", "<|im_start|>"],
    # echo: bool = False,
    **generation_params,
) -> CreateCompletionResponse:
    """
    Translate Japanese text to natural English using Gemma-2-2b-jpn-it.

    Returns a complete CreateCompletionResponse compatible with OpenAI-style output.
    """
    prompt = f"""Translate the following Japanese text to natural English.
Japanese:
{text}
English:""".strip()

    # Optional: you can customize these further per call
    generation_params: Dict[str, Any] = {
        "prompt": prompt,
        **TRANSLATION_DEFAULTS,
        **generation_params
    }

    raw_response = llm(**generation_params)

    response: CreateCompletionResponse = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": llm.model_path.rsplit("/", 1)[-1] if llm.model_path else "gemma-2-2b-jpn-it",
        "choices": raw_response["choices"],
        "usage": raw_response.get("usage") or CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        ),
    }

    return response


if __name__ == "__main__":
    # ja_text = "本商品は30日経過後の返品・交換はお受けできませんのでご了承ください。"

    ja_text = """
世界各国が水面下で熾烈な情報戦を繰り広げる時代にらみ合う2つの国東のオスタニア西の西のウェスタリス戦争を企てるオスタニア
政の動向を探るべくウェスタリスはオペレーションを担うディエンとたそがれ100の顔を使い分正体ロイドフォージャー
コードネームたそがれ母ヨルフォージャー市役所職員正体殺し屋コードネーム茨原姫 母ヨルフォージャー正体
仕事職員正体、コロシアコードネームイバラヒメ娘。妻に正体、正体、心を読むことができるエスパー犬、女ボンドフォージャー、正
体、未来を予知できる超能力家族を作り物狩りのため疑似家族を作り互いに正体を隠した彼らのミッションは続く
"""
    ja_sentences = split_sentences_ja(ja_text)
    ja_text = ja_sentences[1]

    max_tokens = 512
    temperature = 0.0
    logprobs = 1

    result = translate_text(
        ja_text,
        max_tokens=max_tokens,
        temperature=temperature,          # deterministic → easier to inspect
        logprobs=logprobs,               # ← ask for top-3 logprobs per token
    )
    choice = result["choices"][0].copy()
    full_text = choice.pop("text")
    all_logprobs = choice.pop("logprobs", [])

    from rich.pretty import pprint

    print(f"\n[bold cyan]Logprobs:[/bold cyan]")
    pprint(all_logprobs)

    print(f"\n[bold cyan]Meta:[/bold cyan]")
    pprint(choice, expand_all=True)

    print(f"\n[bold cyan]Translation:[/bold cyan]")
    pprint(full_text, expand_all=True)