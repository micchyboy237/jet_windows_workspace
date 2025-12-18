from typing import Any, Dict, List, Sequence, Tuple
from transformers import AutoTokenizer
import threading

# Import helper functions and translation options defaults
from translator_types.translator import translation_options_as_dict, _TRANSLATION_OPTIONS_DEFAULTS

from translator_types import (
    Device,
    BatchType,
    TranslationOptions,
    Translator,
)

# Global thread-safe cache for tokenizer and translator instances
_model_cache: Dict[Tuple[str, Device], Tuple[AutoTokenizer, Translator]] = {}
_cache_lock = threading.RLock()

# ── Device auto-detection with memory safety ──────────────────────────────────
import torch
from analyzer import (
    analyze_logits,
    analyze_translation_results,
    print_analysis_table,
    print_logits_insights,
)
from utils.logger import get_logger

log = get_logger("ct2_cache")

MIN_FREE_VRAM_GB = 2.0  # Safe threshold for quantized Opus-MT models

# ── Constants ─────────────────────────────────────────────────────────────────
QUANTIZED_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
DEFAULT_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"


def detect_device() -> Device:
    """Auto-detect GPU if enough VRAM, else CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free_memory_bytes = torch.cuda.mem_get_info()[0]
        free_memory_gb = free_memory_bytes / (1024**3)
        print(f"Detected free GPU VRAM: {free_memory_gb:.2f} GB")
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


# ── Shared Core Translation Logic ─────────────────────────────────────────────
def _translate_core(
    source_texts: Sequence[str],
    *,
    model_path: str,
    tokenizer_name: str,
    beam_size: int,
    max_decoding_length: int,
    device: Device,
    max_batch_size: int = 0,
    batch_type: BatchType = "examples",
    **options: TranslationOptions,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any] | None]:
    # Thread-safe global cache for tokenizer + translator pair
    cache_key = (model_path, device)
    with _cache_lock:
        cached = _model_cache.get(cache_key)
        if cached is None:
            log.info(
                f"[bold yellow]Loading translation model[/] "
                f"[dim]→[/] [cyan]{model_path}[/] | [green]{device}[/] | [blue]{device}[/]"
            )

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            translator = Translator(model_path, device=device)
            _model_cache[cache_key] = (tokenizer, translator)

            log.info(
                f"[bold green]Translation model ready & cached[/] "
                f"[dim]→[/] [bright_white]{cache_key}[/]"
            )
        else:
            log.info(f"[dim]Translation model cache hit[/] → {cached}")

            tokenizer, translator = cached

    source_batches: List[List[str]] = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        for text in source_texts
    ]

    # Build full options dict with defaults filled in (missing keys → default)
    full_opts: TranslationOptions = {
        **_TRANSLATION_OPTIONS_DEFAULTS,  # from translator.py
        "beam_size": beam_size,
        "max_decoding_length": max_decoding_length,
        "max_batch_size": max_batch_size,
        "batch_type": batch_type,
        **options,  # user-provided overrides (including return_logits_vocab etc.)
    }

    # Clean to only send non-default values to CTranslate2 (keeps signature clean)
    cleaned_opts = translation_options_as_dict(full_opts)

    results = translator.translate_batch(
        source_batches,
        **cleaned_opts,  # Note: no `options=` kwarg – pass directly as kwargs
    )

    translations = [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0])).strip()
        for result in results
    ]

    translation_analysis = analyze_translation_results(results)

    # Robust logits extraction with improved check
    logits_analysis = None
    if options.get("return_logits_vocab", False):
        logits_for_analysis = []
        for r in results:
            if not hasattr(r, "logits") or not r.logits or not r.logits[0]:
                continue
            hyp_logits = r.logits[0]  # list of StorageView (one per generated token)
            try:
                converted = [token_logits.to_numpy() for token_logits in hyp_logits]
                logits_for_analysis.append(converted)
            except Exception as e:
                print(f"Warning: Failed to extract logits: {e}")
                continue

        logits_analysis = analyze_logits(logits_for_analysis)

    return translations, translation_analysis, logits_analysis


def select_model_tokenizer(language: str):
    pass



# ── Public APIs ─────────────────────────────────────────────────────────────
def translate_text(
    text: str,
    *,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: Device | None = None,
    language: str | None = None,
    **options: TranslationOptions,
) -> str:
    if device is None or device == "auto":
        device = detect_device()

    translations, analysis, logits_analysis = _translate_core(
        source_texts=[text],
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        device=device,
        max_batch_size=0,
        batch_type="examples",
        **options,
    )

    print_analysis_table(analysis)
    if logits_analysis:
        print_logits_insights(logits_analysis)

    return translations[0]


def batch_translate_text(
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
    if device is None or device == "auto":
        device = detect_device()

    translations, analysis, logits_analysis = _translate_core(
        source_texts=texts,
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        device=device,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
        **options,
    )

    print_analysis_table(analysis)
    if logits_analysis:
        print_logits_insights(logits_analysis)

    return translations


# ── Example Usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ja_example = "おい、そんな一気に冷たいものを食べると腹を壊すぞ！"
    print("=== Single Translation ===")
    print(f"JA: {ja_example}")

    en_single = translate_text(
        ja_example,
        beam_size=5,
        return_scores=True,
        return_attention=True,
        return_logits_vocab=True,
    )
    print(f"EN: {en_single}")

    ja_batch = [
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！",
        "今日の天気はとても良いです。",
    ]

    print("\n=== Batch Translation ===")
    en_batch = batch_translate_text(
        ja_batch,
        beam_size=4,
        max_batch_size=16,
        sampling_temperature=0.9,
        replace_unknowns=True,
        return_scores=True,
        return_attention=True,
        return_logits_vocab=True,
    )

    for ja, en in zip(ja_batch, en_batch):
        print(f"JA → {ja}")
        print(f"EN → {en}\n")