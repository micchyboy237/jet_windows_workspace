"""Japanese to English translation utility."""

from __future__ import annotations

from typing import Any

import torch
from transformers import pipeline


class JapaneseToEnglishTranslator:
    """Reusable translator class."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-ja-en"):
        """Load translation pipeline."""
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "translation", model=model_name, device=device, max_length=400
        )

    def translate_japanese_to_english(self, text: str) -> str:
        """Translate JP to EN. Core utility function."""
        if not text or not text.strip():
            return ""
        result: list[dict[str, Any]] = self.pipeline(text)
        return result[0]["translation_text"].strip()
