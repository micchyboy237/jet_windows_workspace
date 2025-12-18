# transcribe_quantized_model.py
import gc
import io
import itertools
from pathlib import Path
from typing import Union, BinaryIO, Optional, Literal

import av
import numpy as np
import torch
import torchaudio
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from tqdm import tqdm
import logging

import tokenizers
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Tokenizer, get_suppressed_tokens
from faster_whisper.audio import decode_audio, pad_or_trim

# ──────────────────────────────────────────────────────────────────────────────
# Logging & Console Setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Fallback: Exact large-v3 Mel extraction using torchaudio (128 bins)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_mel_torchaudio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Compute log-Mel spectrogram exactly like Whisper large-v3:
    - 128 mel bins
    - n_fft=400, hop_length=160, win_length=400
    - f_min=0, f_max=8000
    - Slaney normalization + Whisper-style log scaling
    """
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T)

    # STFT
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=400,
        win_length=400,
        hop_length=160,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )(waveform)

    # Mel scale (Slaney norm = constant energy per band)
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=128,
        sample_rate=sr,
        f_min=0.0,
        f_max=8000.0,
        n_stft=201,
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spec = mel_scale(spectrogram)  # (1, 128, T)

    # Log + Whisper normalization
    log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0  # → [0, 1]

    return log_spec.squeeze(0).cpu().numpy()  # (128, T)


# ──────────────────────────────────────────────────────────────────────────────
# Main Function: Compute 128-bin Mel for large-v3 (with auto-fix)
# ──────────────────────────────────────────────────────────────────────────────
def compute_log_mel_spectrogram(
    input_file: Union[str, BinaryIO, Path, np.ndarray],
    model_path: str = r"C:\Users\druiv\.cache\hf_ctranslate2_models\faster-whisper-large-v3-int8_float16",
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
    compute_type: Literal["int8_float16", "float16", "int8"] = "int8_float16",
    chunk_length: Optional[int] = 480000,  # 30s @ 16kHz
    return_padded: bool = True,
) -> np.ndarray:
    """
    Returns log-Mel spectrogram with 128 bins (large-v3 correct) from quantized model.
    Automatically fixes older faster-whisper versions that use 80 bins.
    """
    console.log(f"[bold cyan]Loading quantized model from[/] [green]{model_path}[/]")
    model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    extractor = model.feature_extractor

    # Detect current number of mel bins
    current_n_mels = extractor.mel_filters.shape[0]
    use_torchaudio_fallback = current_n_mels != 128

    # Config table
    table = Table(title="Mel Spectrogram Config (large-v3)")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Extractor n_mels", str(current_n_mels))
    table.add_row("n_fft", str(extractor.n_fft))
    table.add_row("hop_length", str(extractor.hop_length))
    table.add_row("Effective n_mels", "[bold green]128[/]" if use_torchaudio_fallback else str(current_n_mels))
    table.add_row("Method", "torchaudio (v3 fix)" if use_torchaudio_fallback else "native extractor")
    console.print(table)

    if use_torchaudio_fallback:
        logger.info("Detected 80-bin extractor → forcing 128-bin large-v3 behavior via torchaudio")

    # Decode audio
    if isinstance(input_file, np.ndarray):
        audio = input_file.astype(np.float32)
    else:
        audio = decode_audio(input_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)  # Mono

    # Extract features (chunked if needed)
    if chunk_length and len(audio) > chunk_length:
        features_list = []
        for i in tqdm(range(0, len(audio), chunk_length), desc="Extracting Mel chunks"):
            chunk = audio[i:i + chunk_length]
            if use_torchaudio_fallback:
                chunk_feat = _extract_mel_torchaudio(chunk)
            else:
                chunk_feat = extractor(chunk)
            features_list.append(chunk_feat)
        features = np.concatenate(features_list, axis=-1)
    else:
        if use_torchaudio_fallback:
            features = _extract_mel_torchaudio(audio)
        else:
            features = extractor(audio)

    # Pad/trim to encoder length
    if return_padded:
        features = pad_or_trim(features, length=3000, axis=-1)

    logger.info(f"Extracted Mel spectrogram → shape: {features.shape}")
    console.log(f"[bold green]Success! Final shape: {features.shape} (128 Mel bins confirmed)[/]")

    is_multilingual = True # model.is_multilingual

    hf_tokenizer = tokenizers.Tokenizer.from_pretrained("openai/whisper-large-v3")
    tokenizer = Tokenizer(
        hf_tokenizer,
        is_multilingual,
        task="translate",
        language="ja",
    )
    log_progress = True
    options = TranscriptionOptions(
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        log_prob_threshold=None,
        no_speech_threshold=None,
        compression_ratio_threshold=None,
        temperatures=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=(
            get_suppressed_tokens(tokenizer, [-1])
        ),
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:：”)]}、",
        max_new_tokens=None,
        hotwords=None,
        word_timestamps=False,
        hallucination_silence_threshold=None,
        condition_on_previous_text=False,
        clip_timestamps=[0.0, 999999.0],  # ← Fixed: default full audio ranges
        prompt_reset_on_temperature=0.5,
        multilingual=False,
        without_timestamps=True,
        max_initial_timestamp=0.0,
    )
    encoder_output = None
    segments = model.generate_segments(
        features, tokenizer, options, log_progress, encoder_output
    )
    # Collect translated text
    translated_en = " ".join(segment.text for segment in segments).strip()

    # Final result
    console.rule("[bold green]FINAL ENGLISH TRANSLATION[/]")
    console.print(f"[bold white]{translated_en}[/]")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return features



# Add this function to your existing file (e.g. transcribe_quantized_model.py)

from typing import Literal
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionOptions

def translate_text(
    audio_input: Union[str, Path, BinaryIO, np.ndarray],
    model: WhisperModel,
    beam_size: int = 5,
    patience: float = 1.0,
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: float = 2.4,
    logprob_threshold: float = -0.8,
    no_speech_threshold: float = 0.6,
    vad_filter: bool = True,
    vad_min_speech_duration_ms: int = 250,
) -> str:
    """
    Translates Japanese audio directly to English text using your quantized large-v3 model.
    
    Returns clean English text (no timestamps, no confidence scores).
    """
    console.log(f"[bold cyan]Loading model for Japanese → English translation...[/]")

    console.log("[bold green]Transcribing + translating Japanese → English[/]")

    segments, info = model.transcribe(
        audio_input,
        language="ja",                  # Force Japanese detection (critical for accuracy)
        task="translate",               # Direct translation to English
        beam_size=beam_size,
        patience=patience,
        temperature=temperature,
        best_of=5,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        vad_filter=vad_filter,
        vad_parameters=dict(min_speech_duration_ms=vad_min_speech_duration_ms),
        word_timestamps=False,          # Set True if you want per-word timing later
    )

    # Collect translated text
    translated_text = " ".join(segment.text for segment in segments).strip()

    # Rich summary
    table = Table(title="Japanese → English Translation Result")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Detected Language", info.language)
    table.add_row("Language Probability", f"{info.language_probability:.3f}")
    table.add_row("Duration", f"{info.duration:.1f}s")
    table.add_row("Segments", str(len(list(segments))))
    console.print(table)

    console.log(f"[bold green]Translation complete:[/] [white]{translated_text}[/]")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return translated_text


# ──────────────────────────────────────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\1.wav"
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\generated\preprocess_audio\cleaned.wav"

    mel_spec = compute_log_mel_spectrogram(
        input_file=audio_path,
        model_path=r"C:\Users\druiv\.cache\hf_ctranslate2_models\faster-whisper-large-v3-int8_float16",
        device="cuda",           # GTX 1660 → fast!
        compute_type="int8_float16",
        chunk_length=480000,     # 30s chunks
        return_padded=True,
    )

    # Optional: Save or visualize
    np.save("mel_128.npy", mel_spec)
    console.log("[bold blue]Saved mel_128.npy[/]")

