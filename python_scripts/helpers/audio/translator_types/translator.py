from __future__ import annotations

from typing import Any, Callable, Literal, Sequence
from typing_extensions import NotRequired, TypedDict

import ctranslate2

Device = Literal["cpu", "cuda", "auto"]
BatchType = Literal["examples", "tokens"]

# ----------------------------------------------------------------------
# Result containers – keep as TypedDict (never instantiated)
# ----------------------------------------------------------------------
class ExecutionStats(TypedDict):
    num_tokens: int
    num_examples: int
    total_time_in_ms: float

class TranslationResult(TypedDict):
    hypotheses: list[list[str]]
    scores: NotRequired[list[float]]
    attention: NotRequired[list[list[list[float]]]]

class ScoringResult(TypedDict):
    tokens: list[str]
    tokens_score: list[float]

# ----------------------------------------------------------------------
# Options → TypedDict with externalized defaults and helper
# ----------------------------------------------------------------------
class TranslationOptions(TypedDict, total=False):
    target_prefix: Sequence[Sequence[str]] | None
    beam_size: int
    patience: float
    num_hypotheses: int
    length_penalty: float
    coverage_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    prefix_bias_beta: float
    max_input_length: int
    max_decoding_length: int
    min_decoding_length: int
    sampling_topk: int
    sampling_topp: float
    sampling_temperature: float
    return_scores: bool
    return_attention: bool
    return_alternatives: bool
    min_alternative_expansion_prob: float
    return_logits_vocab: bool
    disable_unk: bool
    suppress_sequences: Sequence[Sequence[str]] | None
    end_token: str | Sequence[str] | None
    return_end_token: bool
    use_vmap: bool
    replace_unknowns: bool
    callback: Callable[[str], bool] | None
    max_batch_size: int
    batch_type: BatchType
    asynchronous: bool

_TRANSLATION_OPTIONS_DEFAULTS: dict[str, Any] = {
    "target_prefix": None,
    "beam_size": 2,
    "patience": 1.0,
    "num_hypotheses": 1,
    "length_penalty": 1.0,
    "coverage_penalty": 0.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "prefix_bias_beta": 0.0,
    "max_input_length": 1024,
    "max_decoding_length": 256,
    "min_decoding_length": 1,
    "sampling_topk": 1,
    "sampling_topp": 1.0,
    "sampling_temperature": 1.0,
    "return_scores": False,
    "return_attention": False,
    "return_alternatives": False,
    "min_alternative_expansion_prob": 0.0,
    "return_logits_vocab": False,
    "disable_unk": False,
    "suppress_sequences": None,
    "end_token": None,
    "return_end_token": False,
    "use_vmap": False,
    "replace_unknowns": False,
    "callback": None,
    "max_batch_size": 0,
    "batch_type": "examples",
    "asynchronous": False,
}

def translation_options_as_dict(options: TranslationOptions) -> dict[str, Any]:
    """Return only non-default values for passing to ctranslate2."""
    return {k: v for k, v in options.items() if v != _TRANSLATION_OPTIONS_DEFAULTS.get(k)}

# ----------------------------------------------------------------------
# Thin, future-proof wrapper around the real CTranslate2 translator
# ----------------------------------------------------------------------
class Translator(ctranslate2.Translator):
    """100% compatible with ctranslate2.Translator + room for helpers."""

    def translate_batch(  # type: ignore[override]
        self,
        source: Sequence[Sequence[str]],
        **opts: Any,
    ) -> list[TranslationResult]:
        """
        Wrapper around ctranslate2.Translator.translate_batch that automatically
        applies default values for any TranslationOptions that are not provided
        in ``opts``. This ensures non-default values are kept while missing keys
        fall back to the library's intended defaults.
        """
        # Merge user-provided opts with defaults (user values take precedence)
        full_opts: TranslationOptions = {
            **_TRANSLATION_OPTIONS_DEFAULTS,
            **opts,
        }
        # Remove keys that match the default value – ctranslate2 ignores unknown keys,
        # but sending only changed values is cleaner and avoids future surprises.
        cleaned_opts = translation_options_as_dict(full_opts)

        return super().translate_batch(
            source,
            **cleaned_opts,
        )
