from typing import List

from transformers import AutoTokenizer

from translator_types import (
    Device,
    BatchType,
    TranslationOptions,
    Translator,
)

# ── Device auto-detection with memory safety ──────────────────────────────────
import torch

MIN_FREE_VRAM_GB = 2.0  # Safe threshold for quantized Opus-MT models

# ── Constants ─────────────────────────────────────────────────────────────────
QUANTIZED_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
DEFAULT_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"


def detect_device() -> Device:
    """
    Automatically choose the best device:
      - "cuda"  → if GPU available AND ≥ 2 GB free VRAM
      - "cpu"   → otherwise

    Uses PyTorch (already a dependency via transformers).
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Get current free memory in bytes
        free_memory_bytes = torch.cuda.mem_get_info()[0]  # (free, total)
        free_memory_gb = free_memory_bytes / (1024 ** 3)
        print(f"Detected free GPU VRAM: {free_memory_gb:.2f} GB")  # Print free memory

        if free_memory_gb >= MIN_FREE_VRAM_GB:
            return "cuda"
        else:
            print(
                f"Warning: GPU has only {free_memory_gb:.2f} GB free VRAM "
                f"(< {MIN_FREE_VRAM_GB} GB), falling back to CPU"
            )
            return "cpu"
    except Exception as e:
        print(f"Warning: Could not query GPU memory ({e}), using CPU")
        return "cpu"


# ── Single Translation ───────────────────────────────────────────────────────
def translate_ja_to_en(
    text: str,
    *,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: Device | None = None,  # ← now optional!
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
        device: Device to run on ("cpu", "cuda", "auto", or ``None`` for smart auto-detect).
        **options: Any additional valid keys from :class:`TranslationOptions`
                   (e.g. ``return_scores=True``, ``sampling_temperature=0.8``).

    Returns:
        Translated English string (stripped whitespace).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Resolve device: None or "auto" → smart detection, otherwise use as-is
    if device is None or device == "auto":
        device = detect_device()
    # Explicit "cpu" or "cuda" → respect user choice (no memory check)

    translator = Translator(model_path, device=device)  # our thin wrapper

    # Tokenize → convert to subword tokens expected by CTranslate2
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    source_batch = [source_tokens]

    results = translator.translate_batch(
        source_batch,
        options=TranslationOptions(
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            **options,  # user overrides
        ),
    )

    # ctranslate2 returns a list of TranslationResult objects
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
    device: Device | None = None,
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
        device: Runtime device ("cpu", "cuda", "auto", or ``None`` → auto-detect with memory check).
        max_batch_size: Split large inputs to reduce padding (0 = auto).
        batch_type: "examples" or "tokens".
        **options: Any additional :class:`TranslationOptions`.

    Returns:
        List of English translations in the same order.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if device is None or device == "auto":
        device = detect_device()

    # ctranslate2.Translator is the class itself; avoid instantiating a second time
    translator = Translator(model_path, device=device)  # our thin wrapper

    source_batch: list[list[str]] = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts
    ]

    results = translator.translate_batch(
        source_batch,
        options=TranslationOptions(
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            max_batch_size=max_batch_size,
            batch_type=batch_type,
            **options,
        ),
    )

    translations = [
        # result is now a real ctranslate2.TranslationResult → use attribute access
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
    # No device argument → automatically uses GPU if available
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
        return_scores=True,
        sampling_temperature=0.9,
        replace_unknowns=True,
        # device=None → auto-detect (will use GPU on your Windows machine)
    )

    for ja, en in zip(ja_batch, en_batch):
        print(f"JA → {ja}")
        print(f"EN → {en}\n")