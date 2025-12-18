from __future__ import annotations

import os
from pathlib import Path
import ctranslate2
import numpy as np
import librosa
import torch
import transformers
from typing import Any, Literal, Tuple
from typing import Union
import numpy.typing as npt
from utils.logger import get_logger

log = get_logger("ct2_cache")

QuantizedModelSizes = Literal[
    "tiny", "base", "small", "medium", "large-v2", "large-v3"
]

# Global single cache for loaded model and processor
# Key: model_size (str), Value: Tuple[model, processor]
_MODEL_CACHE: dict[str, Tuple[ctranslate2.models.Whisper, transformers.WhisperProcessor]] = {}


def load_whisper_ct2_model(
    model_size: QuantizedModelSizes,
    model_dir: str,
    device: str = "cpu",
    compute_type: str = "int8",  # or "int8" on CPU
) -> Tuple[ctranslate2.models.Whisper, transformers.WhisperProcessor]:
    """Load quantized CTranslate2 Whisper model + processor with global caching."""
    # Use model_size as cache key (ensures one instance per size regardless of model_dir calls)
    key = f"{model_size}|{model_dir}|{compute_type}|{device}"
    if key in _MODEL_CACHE:
        log.info(f"[dim]CTranslate2 cache hit[/] → {key}")
        return _MODEL_CACHE[key]

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found: {model_dir}\n"
            f"Convert with:\n"
            f" ct2-transformers-converter --model openai/whisper-{model_size} "
            f"--output_dir {model_dir} --quantization int8_float16"
        )

    log.info(
        f"[bold yellow]Loading CTranslate2 model[/] "
        f"[dim]→[/] [cyan]{model_size}[/] | [green]{compute_type}[/] | [blue]{device}[/]"
    )
    processor = transformers.WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
    model = ctranslate2.models.Whisper(model_dir)

    # Cache the loaded pair
    _MODEL_CACHE[key] = (model, processor)

    log.info(
        f"[bold green]CTranslate2 model ready & cached[/] "
        f"[dim]→[/] [bright_white]{key}[/]"
    )

    return model, processor

AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> np.ndarray:
    """
    Robust audio loader for ASR pipelines with correct datatype, normalization, layout, and resampling.
    
    Handles:
      - File paths
      - In-memory WAV bytes
      - NumPy arrays (any shape/layout/dtype/sr)
      - Torch tensors
      - Automatically normalizes to [-1.0, 1.0] float32
      - Always resamples to target_sr
      - Correctly converts stereo → mono regardless of channel position
    Returns
    -------
    np.ndarray
        Shape (samples,), float32, [-1.0, 1.0], exactly `sr` Hz
    """
    # ─────── FIX 1: In-memory arrays/tensors have unknown original sr ───────
    import io
    current_sr: int | None
    if isinstance(audio, (str, os.PathLike)):
        y, current_sr = librosa.load(audio, sr=None, mono=False)
    elif isinstance(audio, bytes):
        y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
    elif isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = None
    elif isinstance(audio, torch.Tensor):
        y = audio.float().cpu().numpy()
        current_sr = None
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # ─────── FIX 2: Correct normalization (NumPy, not torch) ───────
    if np.issubdtype(y.dtype, np.integer):
        y = y / (2 ** (np.iinfo(y.dtype).bits - 1))
    elif np.abs(y).max() > 1.0 + 1e-6:
        y = y / np.abs(y).max()

    # ─────── FIX 3: Always make (channels, time) layout ───────
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    # Mono conversion
    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    # ─────── FIX 4: ALWAYS resample if current_sr is None or wrong ───────
    if current_sr != sr:
        y = librosa.resample(y, orig_sr=current_sr or sr, target_sr=sr)

    return y.squeeze()


def preprocess_audio(
    audio: np.ndarray,
    processor: transformers.WhisperProcessor,
    sampling_rate: int = 16_000,
) -> ctranslate2.StorageView:
    """Convert raw waveform → log-Mel features for CTranslate2."""
    inputs = processor(audio, return_tensors="np", sampling_rate=sampling_rate)
    return ctranslate2.StorageView.from_array(inputs.input_features)


def detect_language(
    model: ctranslate2.models.Whisper,
    features: ctranslate2.StorageView,
) -> Tuple[str, float]:
    """Return (language_token like '<|fr|>', confidence)."""
    language_token, prob = model.detect_language(features)[0][0]
    return language_token, float(prob)


# ──────────────────────────────
# Separate transcribe & translate
# ──────────────────────────────

def transcribe(
    model: ctranslate2.models.Whisper,
    features: ctranslate2.StorageView,
    processor: transformers.WhisperProcessor,
    language_token: str,
) -> str:
    """
    Transcribe audio in its original detected language.
    
    Args:
        language_token: e.g. "<|es|>", "<|fr|>" – must be the detected one
    """
    tokenizer = processor.tokenizer
    prompt = tokenizer.convert_tokens_to_ids([
        "<|startoftranscript|>",
        language_token,
        "<|transcribe|>",
        "<|notimestamps|>"
    ])

    result = model.generate(features, [prompt])
    sequence = result[0].sequences_ids[0]
    return processor.decode(sequence, skip_special_tokens=True)


def transcribe_audio(
    audio: Any,
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict:
    model_size: QuantizedModelSizes = "small"
    model_dir = Path("~/.cache/hf_ctranslate2_models").expanduser() / f"whisper-{model_size}-ct2"

    # 1. Load once
    model, processor = load_whisper_ct2_model(model_size, str(model_dir))
    audio = load_audio(audio)
    features = preprocess_audio(audio, processor)

    # 2. Detect language (optional but recommended for transcription)
    lang_token, prob = detect_language(model, features)
    print(f"Detected: {lang_token} ({prob:.2%})")

    # 3. Transcribe in original language
    text_original = transcribe(model, features, processor, language_token=lang_token)

    return {
        "language": lang_token.strip("<|>").strip(),
        "language_prob": prob,
        "text": text_original,
    }


def translate_to_english(
    audio: Any,
) -> str:
    model_size: QuantizedModelSizes = "small"
    model_dir = Path("~/.cache/hf_ctranslate2_models").expanduser() / f"whisper-{model_size}-ct2"

    # 1. Load once
    model, processor = load_whisper_ct2_model(model_size, str(model_dir))
    audio = load_audio(audio)
    features = preprocess_audio(audio, processor)

    tokenizer = processor.tokenizer
    prompt = tokenizer.convert_tokens_to_ids([
        "<|startoftranscript|>",
        "<|en|>",
        "<|translate|>",
        "<|notimestamps|>"
    ])

    result = model.generate(features, [prompt])
    sequence = result[0].sequences_ids[0]
    return processor.decode(sequence, skip_special_tokens=True)


if __name__ == "__main__":
    model_size: QuantizedModelSizes = "small"
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"

    # Transcribe in original language
    result = transcribe_audio(audio_path)
    print("\nTranscription:")
    print(result)
