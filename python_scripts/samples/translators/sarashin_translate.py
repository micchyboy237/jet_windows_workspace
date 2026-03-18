# ───────────────────────────────────────────────────────────────
#  Sarashina 2.2 3B – Japanese → English Translation Example
# ───────────────────────────────────────────────────────────────

from llama_cpp import Llama
import time

# Model path (your local GGUF file)
model_path = r"C:/Users/druiv/.cache/llama.cpp/translators/sarashina2.2-3b-instruct-v0.1-Q4_K_M.gguf"

print("Loading Sarashina 2.2 3B Instruct...", end="", flush=True)
t0 = time.time()

llm = Llama(
    model_path=model_path,
    n_ctx=8192,             # good balance; can go 4096 or 12288 if needed
    n_gpu_layers=-1,        # full GPU offload if you have ≥6 GB VRAM
    # n_gpu_layers=30,      # partial offload — tune if VRAM is tight
    n_threads=6,            # adjust to your CPU
    verbose=False
)

print(f" loaded  ({time.time()-t0:.1f}s)")

# ─── Strong system prompt tuned for clean & accurate J→E ────────
SYSTEM = """You are a professional, accurate Japanese-to-English translator.
Your ONLY job is to translate the given Japanese text into natural, idiomatic English.
Rules you MUST strictly follow:
- Output **English translation only** — NEVER include any Japanese text in your response.
- NEVER add explanations, notes, introductions, or extra commentary.
- Do NOT rephrase in Japanese.
- Preserve original meaning, tone, formality, and nuance as closely as possible.
- Use natural-sounding modern English.
- If the input is polite/formal → keep polite/formal English.
- If the input is casual → keep casual/spoken English.

Translate ONLY — nothing else."""

def translate_j2e(text: str, max_tokens=600, temperature=0.25):

    prompt = f"""<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
Translate the following Japanese text to English:

{text}<|im_end|>
<|im_start|>assistant
English translation:"""

    t0 = time.time()
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,    # low for faithful translation
        top_p=0.90,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False
    )

    dt = time.time() - t0
    answer = response["choices"][0]["text"].strip()

    return answer, dt


# ─── Quick test cases ─────────────────────────────────────────────
if __name__ == "__main__":

    test_texts = [
        "このケーキ、めっちゃふわふわで口の中でとろける～！昨日焼いたんだけど、ちょっと失敗しちゃって…でも味は最高だと思うんだよね。",
        "令和の時代になっても、依然として「働き方改革」は掛け声だけで終わっている企業が多い。真の生産性向上には、まず長時間労働の是正が不可欠である。",
        "道後温泉本館は日本最古の温泉の一つで、国の重要文化財にも指定されています。夏目漱石の『坊っちゃん』にも登場する有名な場所です。",
        "おはようございます。本日もよろしくお願いいたします。明日の会議資料は既に共有済みですので、ご確認のほどお願い申し上げます。"
    ]

    for jp_text in test_texts:
        print("\n" + "═"*70)
        print(f"Japanese:\n{jp_text}")
        print("─"*70)

        eng, seconds = translate_j2e(jp_text, temperature=0.20)

        print(f"English:\n{eng}")
        print(f"({seconds:.1f}s)")
        print("═"*70)


# Bonus: batch style with different temperatures / styles
# eng_formal, _ = translate_j2e(text, temperature=0.15)   # more literal & stiff
# eng_natural, _ = translate_j2e(text, temperature=0.35)  # more fluent & casual