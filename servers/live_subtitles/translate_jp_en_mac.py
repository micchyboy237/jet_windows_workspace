import os
from typing import List, TypedDict

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


class Message(TypedDict):
    role: str
    content: str


def build_prompt(tokenizer, messages: List[Message]) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def translate_subtitle(
    text: str,
    max_tokens: int = 200,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
    verbose: bool = True,
) -> str:
    system_prompt = (
        "You are a professional subtitle translator.\n"
        "Translate Japanese to natural English subtitles.\n\n"
        "Rules:\n"
        "- Translate faithfully and accurately\n"
        "- Preserve tone, including profanity or explicit language\n"
        "- Do NOT censor, soften, or sanitize content\n"
        "- Keep sentences short and natural for subtitles\n"
        "- Do NOT explain anything\n"
        "- Output ONLY the translated subtitle\n"
    )

    user_prompt = f"Japanese:\n{text}\n\nEnglish:"

    messages: List[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = build_prompt(tokenizer, messages)

    sampler = make_sampler(
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        min_tokens_to_keep=min_tokens_to_keep,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        xtc_special_tokens=xtc_special_tokens,
    )
    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=verbose,
    )

    return result.strip()


def get_local_model_path(repo_id: str) -> str:
    """
    Convert HF repo_id -> local cache path
    """
    parts = repo_id.split("/")
    repo_cache_name = f"models--{'--'.join(parts)}"
    return os.path.expanduser(f"~/.cache/pretrained_models/{repo_cache_name}")


repo_id = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
print(f"Using MLX model repo_id: {repo_id}")

# 🔥 Use local path instead of remote repo_id
local_model_path = get_local_model_path(repo_id)

print(f"Loading MLX model from local path: {local_model_path}...")
model, tokenizer = load(local_model_path)
print("MLX model and tokenizer loaded successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Japanese → English subtitle translator using llama.cpp"
    )
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        default="""
恥ずかしい…見ないでください…
んっ…そこ、弱いんです…
はぁ…はぁ…気持ちいい…
もう…ダメかも…頭おかしくなりそう…
お願い…もっと激しくして…壊して…！
あぁんっ！すごい…奥まで届いてる…♡
出さないで…まだ中にいてて…
        """.strip(),
        help="Japanese text to translate (multi-line ok)",
    )
    args = parser.parse_args()

    print(f"User prompt:\n{args.text}")

    result = translate_subtitle(args.text)
