from __future__ import annotations

from typing import Literal, TypedDict, Sequence, Callable, Any
from typing_extensions import TypeAlias  # for Python <3.12 compatibility

# Common types
Device: TypeAlias = Literal["cpu", "cuda", "auto"]
BatchType: TypeAlias = Literal["examples", "tokens"]

ComputeTypeValue: TypeAlias = Literal[
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]

# compute_type can be a string or a dict mapping device → compute_type
ComputeType: TypeAlias = ComputeTypeValue | dict[Device, ComputeTypeValue]

# StringOrMap in C++ is either a string or a dict[str, str]
StringOrMap: TypeAlias = str | dict[str, str]

# files argument: dict mapping filename → bytes or file-like object
FilesDict: TypeAlias = dict[str, bytes | Any]  # Any = file-like with .read()


class TranslatorInitOptions(TypedDict, total=False):
    """Options for Translator.__init__"""

    model_path: str
    device: Device
    device_index: int | Sequence[int]
    compute_type: ComputeType
    inter_threads: int
    intra_threads: int
    max_queued_batches: int
    flash_attention: bool
    tensor_parallel: bool
    files: FilesDict | None


class TranslationOptions(TypedDict, total=False):
    """Options for translate_batch() and translate_file()"""

    max_batch_size: int
    batch_type: BatchType
    asynchronous: bool

    # Decoding strategy
    beam_size: int
    patience: float
    num_hypotheses: int
    length_penalty: float
    coverage_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    disable_unk: bool

    # Token / sequence control
    suppress_sequences: Sequence[Sequence[str]] | None
    end_token: str | Sequence[str] | None
    return_end_token: bool

    # Prefix / constrained decoding
    prefix_bias_beta: float

    # Length control
    max_input_length: int
    max_decoding_length: int
    min_decoding_length: int

    # Vocabulary / post-processing
    use_vmap: bool
    replace_unknowns: bool

    # Output control
    return_scores: bool
    return_logits_vocab: bool
    return_attention: bool
    return_alternatives: bool
    min_alternative_expansion_prob: float

    # Sampling
    sampling_topk: int
    sampling_topp: float
    sampling_temperature: float

    # Streaming callback (only when beam_size == 1)
    callback: Callable[[str], bool] | None


class TranslateFileOptions(TranslationOptions, total=False):
    """Additional options specific to translate_file()"""

    read_batch_size: int
    with_scores: bool

    # Tokenization hooks
    source_tokenize_fn: Callable[[str], list[str]] | None
    target_tokenize_fn: Callable[[str], list[str]] | None
    target_detokenize_fn: Callable[[list[str]], str] | None


class ScoringOptions(TypedDict, total=False):
    """Options for score_batch() and score_file()"""

    max_batch_size: int
    batch_type: BatchType
    max_input_length: int
    offset: int
    asynchronous: bool


class ScoreFileOptions(ScoringOptions, total=False):
    """Additional options specific to score_file()"""

    read_batch_size: int
    with_tokens_score: bool

    source_tokenize_fn: Callable[[str], list[str]] | None
    target_tokenize_fn: Callable[[str], list[str]] | None
    target_detokenize_fn: Callable[[list[str]], str] | None


class UnloadModelOptions(TypedDict, total=False):
    to_cpu: bool


class LoadModelOptions(TypedDict, total=False):
    keep_cache: bool