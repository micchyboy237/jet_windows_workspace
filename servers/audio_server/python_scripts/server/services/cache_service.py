# servers\audio_server\python_scripts\server\services\cache_service.py (updated sections only)

import os
from typing import Optional

from faster_whisper import WhisperModel
from transformers import AutoTokenizer

from python_scripts.server.services.translator_types import Translator
from python_scripts.server.utils.logger import get_logger

log = get_logger("batch_transcription")

# Constants for transcriber defaults
WHISPER_MODEL_NAME = "kotoba-tech/kotoba-whisper-v2.0-faster"

# Constants for translator defaults
TRANSLATOR_MODEL_PATH: str = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER: str = "Helsinki-NLP/opus-mt-ja-en"

transcriber_model: Optional[WhisperModel] = None
translator: Optional[Translator] = None
tokenizer: Optional[AutoTokenizer] = None


# Existing function (unchanged)
def load_transcriber_model(
    model_name: str = WHISPER_MODEL_NAME,
    device: str = "cuda",
    compute_type: str = "float32"
) -> WhisperModel:
    global transcriber_model
    if transcriber_model:
        log.info(f"Reusing transcriber model cache hit [bold magenta]{model_name}[/bold magenta] ({device} {compute_type})")
        return transcriber_model
    log.info(f"Loading transcriber model [bold magenta]{model_name}[/bold magenta] ({device} {compute_type})")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    transcriber_model = model
    log.info("[bold green]Transcriber model loaded and cached successfully[/bold green]")
    return model


def load_translator_model() -> Translator:
    """Load and cache the CTranslate2 Japanese-to-English translation model."""
    global translator
    if translator is not None:
        log.info("Reusing cached translator model")
        return translator

    if not os.path.exists(TRANSLATOR_MODEL_PATH):
        raise FileNotFoundError(f"Translator model path not found: {TRANSLATOR_MODEL_PATH}")

    log.info(f"Loading translator model from [bold magenta]{TRANSLATOR_MODEL_PATH}[/bold magenta]")
    translator_instance = Translator(
        TRANSLATOR_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=4,
    )
    translator = translator_instance
    log.info("[bold green]Translator model loaded and cached successfully[/bold green]")
    return translator_instance


def load_translator_tokenizer() -> AutoTokenizer:
    """Load and cache the tokenizer for the Japanese-to-English model."""
    global tokenizer
    if tokenizer is not None:
        log.info("Reusing cached translator tokenizer")
        return tokenizer

    log.info(f"Loading tokenizer [bold magenta]{TRANSLATOR_TOKENIZER}[/bold magenta]")
    tokenizer_instance = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    tokenizer = tokenizer_instance
    log.info("[bold green]Translator tokenizer loaded and cached successfully[/bold green]")
    return tokenizer_instance