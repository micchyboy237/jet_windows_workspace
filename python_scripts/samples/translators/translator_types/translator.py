# jet/translators/types.py
from __future__ import annotations

from typing import (
    Literal,
    TypedDict,
    Sequence,
    Callable,
    overload,
    Any,
)
from typing_extensions import NotRequired
import concurrent.futures

# ── Core Literals ─────────────────────────────────────────────────────────────
Device = Literal["cpu", "cuda", "auto"]
BatchType = Literal["examples", "tokens"]

# ── Execution Statistics ──────────────────────────────────────────────────────
class ExecutionStats(TypedDict):
    num_tokens: int
    num_examples: int
    total_time_in_ms: float

# ── Result Structures (exposed in Python bindings) ─────────────────────────────
class TranslationResult(TypedDict):
    hypotheses: list[list[str]]
    scores: NotRequired[list[float]]
    attention: NotRequired[list[list[list[float]]]]
    # ... other optional fields added by return_* flags

class ScoringResult(TypedDict):
    tokens: list[str]
    tokens_score: list[float]
    # normalized_score() is a method in C++, exposed as property in Python

# ── Options ───────────────────────────────────────────────────────────────────
class TranslationOptions(TypedDict, total=False):
    # Core decoding
    beam_size: int
    patience: float
    length_penalty: float
    coverage_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    prefix_bias_beta: float
    num_hypotheses: int

    # Length control
    max_input_length: int
    max_decoding_length: int
    min_decoding_length: int

    # Sampling
    sampling_topk: int
    sampling_topp: float
    sampling_temperature: float

    # Output control
    return_scores: bool
    return_attention: bool
    return_alternatives: bool
    min_alternative_expansion_prob: float
    return_logits_vocab: bool

    # Miscellaneous
    disable_unk: bool
    suppress_sequences: Sequence[Sequence[str]] | None
    end_token: str | Sequence[str] | None
    return_end_token: bool
    use_vmap: bool
    replace_unknowns: bool
    callback: Callable[[str], bool] | None  # only used when beam_size == 1


class ScoringOptions(TypedDict, total=False):
    max_input_length: int
    offset: int
    asynchronous: bool


# ── Translator Typed Interface ────────────────────────────────────────────────
SourceBatch = Sequence[Sequence[str]]                     # List[List[str]]
TargetBatch = Sequence[Sequence[str]]                     # List[List[str]]

Tokenizer = Callable[[str], list[str]]
Detokenizer = Callable[[list[str]], str]

class Translator:
    # ── Constructor (from pybind11) ───────────────────────────────────────────
    def __init__(
        self,
        model_path: str,
        device: Device = "cpu",
        *,
        device_index: int | Sequence[int] = 0,
        compute_type: str | dict[Device, str] = "default",
        inter_threads: int = 1,
        intra_threads: int = 0,
        max_queued_batches: int = 0,
        flash_attention: bool = False,
        tensor_parallel: bool = False,
        files: dict[str, bytes | Any] | None = None,
    ) -> None: ...

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def device(self) -> Device: ...
    @property
    def device_index(self) -> list[int]: ...
    @property
    def compute_type(self) -> str: ...
    @property
    def num_translators(self) -> int: ...
    @property
    def num_queued_batches(self) -> int: ...
    @property
    def num_active_batches(self) -> int: ...
    @property
    def model_is_loaded(self) -> bool: ...
    @property
    def tensor_parallel(self) -> bool: ...

    # ── Async translation ─────────────────────────────────────────────────────
    @overload
    def translate_batch_async(
        self,
        source: SourceBatch,
        *,
        target_prefix: None = None,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        **options: TranslationOptions,
    ) -> list[concurrent.futures.Future[TranslationResult]]: ...

    @overload
    def translate_batch_async(
        self,
        source: SourceBatch,
        target_prefix: TargetBatch,
        *,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        **options: TranslationOptions,
    ) -> list[concurrent.futures.Future[TranslationResult]]: ...

    def translate_batch_async(
        self,
        source: SourceBatch,
        target_prefix: TargetBatch | None = None,
        *,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        **options: TranslationOptions,
    ) -> list[concurrent.futures.Future[TranslationResult]]: ...

    # ── Sync translation ──────────────────────────────────────────────────────
    @overload
    def translate_batch(
        self,
        source: SourceBatch,
        *,
        target_prefix: None = None,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        asynchronous: bool = False,
        **options: TranslationOptions,
    ) -> list[TranslationResult]: ...

    @overload
    def translate_batch(
        self,
        source: SourceBatch,
        target_prefix: TargetBatch,
        *,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        asynchronous: bool = False,
        **options: TranslationOptions,
    ) -> list[TranslationResult]: ...

    def translate_batch(
        self,
        source: SourceBatch,
        target_prefix: TargetBatch | None = None,
        *,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        asynchronous: bool = False,
        **options: TranslationOptions,
    ) -> list[TranslationResult]: ...

    # ── Scoring ───────────────────────────────────────────────────────────────
    def score_batch_async(
        self,
        source: SourceBatch,
        target: TargetBatch,
        *,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        **options: ScoringOptions,
    ) -> list[concurrent.futures.Future[ScoringResult]]: ...

    def score_batch(
        self,
        source: SourceBatch,
        target: TargetBatch,
        *,
        max_batch_size: int = 0,
        batch_type: BatchType = "examples",
        **options: ScoringOptions,
    ) -> list[ScoringResult]: ...

    # ── File-based translation (high-level) ───────────────────────────────────
    def translate_file(
        self,
        source_path: str,
        output_path: str,
        target_path: str | None = None,
        *,
        max_batch_size: int = 32,
        read_batch_size: int = 0,
        batch_type: BatchType = "examples",
        with_scores: bool = False,
        **options: TranslationOptions,
    ) -> ExecutionStats: ...

    def score_file(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        *,
        max_batch_size: int = 32,
        read_batch_size: int = 0,
        batch_type: BatchType = "examples",
        with_tokens_score: bool = False,
        **options: ScoringOptions,
    ) -> ExecutionStats: ...

    # ── Raw stream versions (with custom tokenizers) ──────────────────────────
    def translate_raw_text_file(
        self,
        source: str | object,  # str or istream-like
        output: str | object,  # str or ostream-like
        source_tokenizer: Tokenizer,
        target_detokenizer: Detokenizer,
        target_tokenizer: Tokenizer | None = None,
        target_prefix_path: str | None = None,
        *,
        max_batch_size: int = 32,
        read_batch_size: int = 0,
        batch_type: BatchType = "examples",
        with_scores: bool = False,
        **options: TranslationOptions,
    ) -> ExecutionStats: ...

    # ── Model management ──────────────────────────────────────────────────────
    def unload_model(self, to_cpu: bool = False) -> None: ...
    def load_model(self, keep_cache: bool = False) -> None: ...