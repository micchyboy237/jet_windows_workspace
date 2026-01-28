from typing import Literal, Dict, Any, Union, List
import time
import uuid

from rich.console import Console
from rich.markdown import Markdown
from llama_cpp import Llama, LlamaState
from llama_cpp.llama_types import CreateCompletionResponse, CompletionChoice, CompletionUsage
from split_sentences_ja import split_sentences_ja

console = Console()

# ────────────────────────────────────────────────
# Config – adjust paths & settings to match your setup
# ────────────────────────────────────────────────

MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators\gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"
)

CTX_SIZE        = 4096              # ← increased for growing context
GPU_LAYERS      = -1
CACHE_TYPE_K    = "q8_0"
CACHE_TYPE_V    = "q8_0"
TOK_ADD_BOS     = False

# ────────────────────────────────────────────────
# Initialize model (once!)
# ────────────────────────────────────────────────

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_gpu_layers=GPU_LAYERS,
    flash_attn=True,
    logits_all=True,           # required for logprobs
    cache_type_k=CACHE_TYPE_K,
    cache_type_v=CACHE_TYPE_V,
    tokenizer_kwargs={"add_bos_token": TOK_ADD_BOS},
    verbose=True,
)

# We'll keep track of the current KV cache state and the tokens we've already processed
current_state: LlamaState | None = None
processed_tokens: List[int] = []   # growing list of all tokens seen so far


def translate_text_incremental(
    new_ja_text: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    stop: Union[str, list[str], None] = None,
    logprobs: int | None = None,
    echo: bool = False,
) -> CreateCompletionResponse:
    """
    Incrementally translate Japanese text to English using a chat-formatted prompt for Gemma-2.
    Applies various recommended fixes for improved output (see docstring).
    """
    global current_state, processed_tokens

    # (5) Clean up Japanese text before translation. Can tweak as needed for your OCR data.
    clean_ja_text = new_ja_text.replace(" ", "").replace("、", "、 ").strip()

    # 2. (Keep track of previous Japanese chunks if needed)
    previous_text = (
        llm.detokenize(processed_tokens).decode("utf-8", errors="ignore").strip()
        if processed_tokens
        else ""
    )
    # For chat template, just concat as context
    ja_block = (previous_text + "\n" + clean_ja_text).strip() if previous_text else clean_ja_text

    # (1) Use chat template for Gemma-2
    messages = [
        {
            "role": "user",
            "content": f"""Translate the following Japanese text to natural, fluent English.

Japanese:
{ja_block}

Provide only the English translation, nothing else:"""
        }
    ]

    # (3) & (2) Max tokens, stop instructions
    if stop is None:
        stop = ["\n\n", "</s>"]

    # Streaming is not set up for logprobs; can add if needed
    generation_params: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "logprobs": logprobs,
        # "echo": echo,  # (4) echo=True if logprobs wanted on prompt
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "mirostat_mode": 0,
        "seed": None,
        "stream": False,
    }

    if current_state is not None:
        llm.load_state(current_state)

    raw_response = llm.create_chat_completion(**generation_params)

    current_state = llm.save_state()
    # Optionally update processed_tokens: No easy auto-growth for chat, so may be left as is.

    # Format response
    response: CreateCompletionResponse = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": llm.model_path.rsplit("/", 1)[-1] if llm.model_path else "gemma-2-2b-jpn-it",
        "choices": raw_response["choices"],
        "usage": raw_response.get("usage") or CompletionUsage(
            prompt_tokens=0, # Not tracked in chat (unless you want to tokenize yourself)
            completion_tokens=0,
            total_tokens=0
        ),
    }

    return response


# ────────────────────────────────────────────────
# Example: Incrementally translate growing text
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Reset state for this example
    current_state = None
    processed_tokens = []

    # Example growing Japanese text (split into chunks)
    full_ja_text = """
世界各国が水面下で熾烈な情報戦を繰り広げる時代にらみ合う2つの国東のオスタニア西の西のウェスタリス戦争を企てるオスタニア
政の動向を探るべくウェスタリスはオペレーションを担うディエンとたそがれ100の顔を使い分正体ロイドフォージャー
コードネームたそがれ母ヨルフォージャー市役所職員正体殺し屋コードネーム茨原姫 母ヨルフォージャー正体
仕事職員正体、コロシアコードネームイバラヒメ娘。妻に正体、正体、心を読むことができるエスパー犬、女ボンドフォージャー、正
体、未来を予知できる超能力家族を作り物狩りのため疑似家族を作り互いに正体を隠した彼らのミッションは続く
"""

    ja_sentences = split_sentences_ja(full_ja_text)
    # For demo: process in chunks
    chunks = [
        "\n".join(ja_sentences[:2]),
        "\n".join(ja_sentences[2:5]),
        "\n".join(ja_sentences[5:]),
    ]

    max_tokens = 1024
    temperature = 0.0
    logprobs = 1  # or try None if debugging

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[bold green]Chunk {i}/{len(chunks)}:[/bold green]")
        print(Markdown(f"**New Japanese:**\n{chunk}"))

        result = translate_text_incremental(
            new_ja_text=chunk,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            # echo=True, # logprobs on prompt if needed
        )

        choice = result["choices"][0].copy()
        full_text = choice.pop("message", {}).get("content", "") if "message" in choice else choice.pop("text", "")
        all_logprobs = choice.pop("logprobs", [])

        print(f"\n[bold cyan]Logprobs (first few):[/bold cyan]")
        from rich.pretty import pprint
        
        logprobs_data = choice.get("logprobs")

        if logprobs_data is None:
            print("[yellow]No logprobs returned by the model[/yellow]")
        elif not isinstance(logprobs_data, list):
            print(f"[yellow]Unexpected logprobs type: {type(logprobs_data).__name__}[/yellow]")
            pprint(logprobs_data)
        elif len(logprobs_data) == 0:
            print("[dim]Logprobs list is empty[/dim]")
        else:
            pprint(logprobs_data[:3])

        print(f"\n[bold cyan]Meta:[/bold cyan]")
        pprint(choice, expand_all=True)

        print(f"\n[bold cyan]Translation so far:[/bold cyan]")
        print(full_text.strip())
        print("-" * 80)