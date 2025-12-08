from typing import List

import ctranslate2
from transformers import AutoTokenizer

from translator_types import (
    Device,
    BatchType,
    TranslationOptions
)

# ── Constants ─────────────────────────────────────────────────────────────────
QUANTIZED_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
DEFAULT_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"


# ── Single Translation ───────────────────────────────────────────────────────
def translate_ja_to_en(
    text: str,
    *,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: Device = "cpu",
    **options: TranslationOptions,
) -> str:
    """
    Translate a single Japanese sentence to English using a quantized Opus-MT model.

    All translation options are fully type-checked via :class:`TranslationOptions`.

    Args:
        text: Japanese input text.
        model_path: Path to the CTranslate2-converted model.
        tokenizer_name: Hugging Face tokenizer (must match the model).
        beam_size: Beam size (1 = greedy decoding).
        max_decoding_length: Maximum number of generated tokens.
        device: Device to run on ("cpu", "cuda", or "auto").
        **options: Any additional valid keys from :class:`TranslationOptions`
                   (e.g. ``return_scores=True``, ``sampling_temperature=0.8``).

    Returns:
        Translated English string (stripped whitespace).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # ctranslate2.Translator is the class itself
    translator = ctranslate2.Translator(model_path, device=device)

    # Tokenize → convert to subword tokens expected by CTranslate2
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    source_batch = [source_tokens]

    results = translator.translate_batch(
        source_batch,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        **options,
    )

    best_hypothesis: list[str] = results[0].hypotheses[0]
    translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(best_hypothesis))
    return translated.strip()


# ── Batch Translation ────────────────────────────────────────────────────────
def batch_translate_ja_to_en(
    texts: List[str],
    *,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: Device = "cpu",
    max_batch_size: int = 32,
    batch_type: BatchType = "examples",
    **options: TranslationOptions,
) -> List[str]:
    """
    Efficient batch translation of Japanese to English.

    Fully typed — IDEs and static checkers will catch invalid options instantly.

    Args:
        texts: List of Japanese sentences.
        model_path: Path to the CTranslate2 model.
        tokenizer_name: Matching tokenizer.
        beam_size: Beam size for decoding.
        max_decoding_length: Max tokens to generate.
        device: Runtime device.
        max_batch_size: Split large inputs to reduce padding (0 = auto).
        batch_type: "examples" or "tokens".
        **options: Any additional :class:`TranslationOptions`.

    Returns:
        List of English translations in the same order.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # ctranslate2.Translator is the class itself; avoid instantiating a second time
    translator = ctranslate2.Translator(model_path, device=device)

    source_batch: list[list[str]] = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts
    ]

    results = translator.translate_batch(
        source_batch,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
        **options,
    )

    translations = [
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(result.hypotheses[0])
        ).strip()
        for result in results
    ]
    return translations


# ── Example Usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ja_example = "おい、そんな一気に冷たいものを食べると腹を壊すぞ！"

    print("=== Single Translation ===")
    print(f"JA:  {ja_example}")
    print(f"EN:  {translate_ja_to_en(ja_example, beam_size=5)}")

    ja_batch = [
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！",
        "今日の天気はとても良いです。",
    ]

    print("\n=== Batch Translation ===")
    en_batch = batch_translate_ja_to_en(
        ja_batch,
        beam_size=4,
        max_batch_size=16,
        return_scores=True,           # type-checked!
        sampling_temperature=0.9,     # type-checked!
        replace_unknowns=True,        # type-checked!
    )

    for ja, en in zip(ja_batch, en_batch):
        print(f"JA → {ja}")
        print(f"EN → {en}\n")