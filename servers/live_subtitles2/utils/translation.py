"""Japanese to English translation utility."""

from __future__ import annotations

from typing import Optional

import ctranslate2
import torch
from transformers import AutoTokenizer


class JapaneseToEnglishTranslator:
    """Reusable translator class."""

    def __init__(
        self,
        model_path: str = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2",
        tokenizer_name: str = "Helsinki-NLP/opus-mt-ja-en",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        max_input_length: int = 512,
        max_output_length: int = 384,
    ):
        """Load translation pipeline."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.translator = ctranslate2.Translator(
            model_path,
            device=device,
            compute_type=compute_type,
        )
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def translate_japanese_to_english(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        # Fixed version
        input_ids = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_length,
        )
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        results = self.translator.translate_batch(
            [input_tokens],
            beam_size=4,
            max_decoding_length=self.max_output_length
        )

        output_tokens = results[0].hypotheses[0]
        translated = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )
        return translated.strip()
