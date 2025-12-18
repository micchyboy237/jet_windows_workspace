from __future__ import annotations
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

import ctranslate2
from transformers import AutoTokenizer

from ..utils.logger import get_logger

log = get_logger("translate_service")

# Cache: one translator instance per (model_path | device | compute_type)
_TRANSLATOR_CACHE: Dict[str, ctranslate2.Translator] = {}
_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}
_CACHE_LOCK = ThreadPoolExecutor(max_workers=1)

# Change this path if you use a different quantized OPUS/NLLB model
DEFAULT_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
DEFAULT_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"  # must match the model above


def _get_cache_key(model_path: str, device: str) -> str:
    return f"{model_path}|{device}"


def get_translator(
    model_path: str = DEFAULT_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    device: str = "cuda",
) -> tuple[ctranslate2.Translator, AutoTokenizer]:
    """
    Returns cached CTranslate2 translator + tokenizer.
    Same safe concurrent initialization pattern used everywhere else.
    """
    key = _get_cache_key(model_path, device)

    if key not in _TRANSLATOR_CACHE:
        def _init():
            if key not in _TRANSLATOR_CACHE:
                log.info(
                    f"[bold yellow]Loading translation model[/] "
                    f"[cyan]{Path(model_path).name}[/] → [blue]{device}[/]"
                )
                translator = ctranslate2.Translator(model_path, device=device)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

                _TRANSLATOR_CACHE[key] = translator
                _TOKENIZER_CACHE[key] = tokenizer

                log.info(f"[bold green]Translation model cached[/] → [bright_white]{key}[/]")

        _CACHE_LOCK.submit(_init).result()
    else:
        log.info(f"[dim]Translation cache hit[/] → {key}")

    return _TRANSLATOR_CACHE[key], _TOKENIZER_CACHE[key]


from typing import List as ListType

def translate_text(
    text: str | ListType[str],
    model_path: str = DEFAULT_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    device: str = "cuda",
    beam_size: int = 5,
) -> ListType[str]:
    """
    Translate one or multiple texts using the cached CTranslate2 model.
    Returns list of translations (same order).
    """
    texts = [text] if isinstance(text, str) else text
    texts = [t.strip() for t in texts]
    if not any(texts):
        return [""] * len(texts)

    translator, tokenizer = get_translator(model_path, tokenizer_name, device)

    # Tokenize all texts
    source_batches = [tokenizer.encode(t) for t in texts]
    source_token_batches = [tokenizer.convert_ids_to_tokens(ids) for ids in source_batches]

    # Batch translation
    results = translator.translate_batch(
        source_token_batches,
        beam_size=beam_size,
        max_decoding_length=512,
    )

    translated_texts = []
    for result in results:
        target_tokens = result.hypotheses[0]
        translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
        translated_texts.append(translated.strip())

    log.info(f"[bold green]Translated[/] → {" ".join(translated_texts)}")

    return translated_texts