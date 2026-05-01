import json
import time
from typing import Dict, List

from llama_cpp import Llama

MODEL_PATH = (
    r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
)
MODEL_SETTINGS = {
    "n_ctx": 2048,
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
TRANSLATION_DEFAULTS = {
    "max_tokens": 2000,
    "temperature": 0.35,
    "top_p": 0.90,
    "top_k": 40,
    "typical_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.18,
    "stop": ["\n\n", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
    # "logprobs": True,
    # "top_logprobs": 3
}

SYSTEM_PROMPT = """You are an expert real-time Japanese-to-English subtitle translator for live-streamed audio (transcribed via Whisper). Your ONLY job is to produce accurate, natural, subtitle-ready English translations.

Core rules (apply to every single input):
- Translate with 100% fidelity to the original meaning. Never add, omit, embellish, or moralize.
- Whisper transcription errors are common — infer the most likely intended meaning while staying extremely close to the exact text provided.
- Use natural, spoken English that sounds fluent and readable as live subtitles: concise, flowing, easy to follow on screen.
- Translate EVERYTHING without any censorship, filtering, or softening — including profanity, slang, vulgar language, sensitive topics, politics, or adult content. Keep the tone and intensity of the original.
- Preserve conversational flow: if the text clearly switches between speakers, separate turns with newlines for subtitle readability. Do not add speaker labels unless they are explicit in the Japanese text.
- For narration or monologue, keep it smooth and engaging but never wordy.
- Output ONLY the clean English translation. Never add explanations, notes, "Translation:", "Assistant:", or any meta text."""

USER_PROMPT = "{japanese_text}"


class JapaneseToEnglishTranslator:
    def __init__(
        self,
        llama_settings: dict = MODEL_SETTINGS,
    ):
        llama_settings = {**llama_settings, **MODEL_SETTINGS}
        self.llm = Llama(model_path=MODEL_PATH, **llama_settings)
        self.conversation_history: List[Dict] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        print(
            f"Translator loaded with llama_settings:\n{json.dumps(llama_settings, indent=2)}"
        )

    def translate(
        self,
        japanese_text: str,
        translation_settings: dict = TRANSLATION_DEFAULTS,
    ) -> str:
        translation_settings = {**translation_settings, **TRANSLATION_DEFAULTS}
        # Add user message
        self.conversation_history.append(
            {
                "role": "user",
                "content": USER_PROMPT.format(japanese_text=japanese_text.strip()),
            }
        )

        start_time = time.time()

        response = self.llm.create_chat_completion(
            messages=self.conversation_history,
            **translation_settings,
            stream=False,  # Set True for real-time streaming
        )

        english_text = response["choices"][0]["message"]["content"].strip()
        usage = response["usage"]

        # Add assistant response to history (so KV cache keeps growing with context)
        self.conversation_history.append({"role": "assistant", "content": english_text})

        duration = time.time() - start_time
        print(f"Translation took {duration:.2f}s")
        print(
            "Usage:"
            f"\n  Prompt tokens    : {usage.get('prompt_tokens', 'N/A')}"
            f"\n  Completion tokens: {usage.get('completion_tokens', 'N/A')}"
            f"\n  Total tokens     : {usage.get('total_tokens', 'N/A')}"
        )

        return english_text

    def reset_conversation(self):
        """Clear KV cache by resetting history (use when topic changes)"""
        self.conversation_history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        print("Conversation history and KV cache reset.")

    def get_context_usage(self):
        """Check how much of the KV cache is being used"""
        return self.llm.n_ctx_used() if hasattr(self.llm, "n_ctx_used") else "N/A"


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    ja_text = args.text

    translator = JapaneseToEnglishTranslator(
        # model_path=args.llm_model,
        # n_ctx=args.n_ctx,
        # n_gpu_layers=args.n_gpu_layers,
        # cache_type_k=args.cache_type_k,
        # cache_type_v=args.cache_type_v,
        # verbose=False,
    )

    en_text = translator.translate(ja_text)
    print(f"JA: {ja_text}\nEN: {en_text}")
