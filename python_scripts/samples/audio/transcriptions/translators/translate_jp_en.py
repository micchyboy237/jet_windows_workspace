from typing import List, Union, Tuple
from typing import cast

from transformers import AutoTokenizer

from translator_types import (
    Device,
    BatchType,
    TranslationOptions,
    Translator,
)

# Near top of file, add this small helper
def quality_label(log_prob: float | None) -> str:
    if log_prob is None:
        return "N/A"
    if log_prob > -0.40:
        return "High"
    elif log_prob > -0.80:
        return "Medium"
    else:
        return "Low"

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
) -> Union[str, Tuple[str, float]]:
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
        Translated English string (stripped whitespace) or (string, score) if return_scores is True.
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
            **{
                "beam_size": beam_size,
                "max_decoding_length": max_decoding_length,
                **options
            },  # user overrides
        ),
    )

    # ctranslate2 returns a list of TranslationResult objects
    best_hypothesis: list[str] = results[0].hypotheses[0]
    translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(best_hypothesis))
    translation = translated.strip()  # type: str

    # Use direct attribute access — TranslationResult has public .scores when requested
    if options.get("return_scores", False) and hasattr(results[0], "scores") and results[0].scores:
        score = results[0].scores[0]  # log-prob of the best hypothesis
        return translation, score

    return translation

    # Returns: str if return_scores=False, Tuple[str, float] otherwise (when scores available)


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
) -> Union[List[str], List[Tuple[str, float]]]:
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
        List of English translations or (translation, score) pairs if return_scores is True.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if device is None or device == "auto":
        device = detect_device()

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
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(result.hypotheses[0])
        ).strip()
        for result in results  # type: ignore
    ]
    # If return_scores was requested, pair each translation with its score
    if options.get("return_scores", False):
        out = []
        for trans, result in zip(translations, results):
            score = result.scores[0] if hasattr(result, "scores") and result.scores else None
            out.append((trans, score))
        return out

    return translations

    # Returns: Union[List[str], List[Tuple[str, float | None]]]


# ────────────────────────────────────────────────
# Then update the three printing sections in __main__:

if __name__ == "__main__":
    common_opts = dict(return_scores=True, beam_size=5)

    ja_example = "おい、そんな一気に冷たいものを食べると腹を壊すぞ！"

    print("=== Single Translation ===")
    print(f"JA: {ja_example}")
    result_single = translate_ja_to_en(ja_example, **common_opts)
    # Single (default beam)
    if isinstance(result_single, tuple):
        en, score = result_single
        print(f"EN: {en}")
        print(f"   (log-prob: {score:.4f} | quality: {quality_label(score)})")
    else:
        print(f"EN: {result_single}")

    print("\n=== Single Translation (longer beam) ===")
    result_longer = translate_ja_to_en(
        ja_example,
        **{
            "beam_size": 8,
            **common_opts
        }  # type: ignore  # mypy knows it's tuple due to return_scores=True
    )
    # Single (longer beam)
    en_long, score_long = cast(Tuple[str, float], result_longer)
    print(f"JA: {ja_example}")
    print(f"EN: {en_long}")
    print(f"   (log-prob: {score_long:.4f} | quality: {quality_label(score_long)})")

    ja_batch = [
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！",
        "今日の天気はとても良いです。",
    ]
    print("\n=== Batch Translation ===")
    results_batch = batch_translate_ja_to_en(
        ja_batch,
        max_batch_size=16,
        sampling_temperature=0.9,
        replace_unknowns=True,
        **{
            "beam_size": 4,
            **common_opts
        }
    )

    # Batch
    for ja, item in zip(ja_batch, results_batch):
        en, score = item
        print(f"JA → {ja}")
        print(f"EN → {en}")
        print(f"   (log-prob: {score:.4f} | quality: {quality_label(score)})")
        print()