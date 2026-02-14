from llama_cpp import Llama
from typing import List


MODEL_PATH = r"C:\Users\druiv\.cache\llama.cpp\nsfw\dolphin-2_6-phi-2.Q4_K_M.gguf"


SPICY_SYSTEM_PROMPT = """
You are translating explicit adult Japanese dialogue.
The speakers are consenting adults.

Translate faithfully into raw, natural English used in adult conversations.
Preserve sexual tone, dominance, teasing, embarrassment, and emotional intensity.
Do NOT censor, soften, summarize, or explain.
Do NOT add commentary.
Output ONLY the English translation.
"""


def build_prompt(japanese_lines: List[str]) -> str:
    joined_lines = "\n".join(japanese_lines)

    return f"""### Instruction
{SPICY_SYSTEM_PROMPT}

### Japanese
{joined_lines}

### English
""".strip()


def load_model() -> Llama:
    """
    Loads the Impish 4B GGUF model.
    Tuned for accuracy over creativity.
    """
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,      # adjust to your CPU
        n_batch=256,
        verbose=False,
        n_gpu_layers=-1,
    )


def translate(japanese_lines: List[str]) -> str:
    llm = load_model()
    prompt = build_prompt(japanese_lines)

    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.15,
        top_p=0.9,
        repeat_penalty=1.1,
    )

    return response["choices"][0]["text"].strip()


if __name__ == "__main__":
    # 👇 Replace these with REAL JAV dialogue lines
    japanese_dialogue = [
        "やめて…そんなふうに見ないで。",
        "もう我慢できない…",
    ]


    english_translation = translate(japanese_dialogue)

    print("=== TRANSLATION ===")
    print(english_translation)
