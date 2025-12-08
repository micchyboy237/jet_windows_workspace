# whisper_ct2_transcriber.py
from __future__ import annotations

import os
import ctranslate2
import librosa
import numpy as np
import transformers
from typing import Literal, Optional, Tuple, List
from pathlib import Path

# Supported quantized Whisper models (int8 or int8_float16)
QuantizedModelSizes = Literal[
    "tiny", "base", "small", "medium", "large-v2", "large-v3"
]

class WhisperCT2Transcriber:
    """
    Reusable wrapper around CTranslate2 + Hugging Face Whisper processor
    for fast inference using quantized int8/float16 models.

    Features:
    - Language detection
    - Transcription in original language
    - Translation to English
    - Fully typed, modular, no hardcoded paths
    """

    SUPPORTED_SIZES: List[QuantizedModelSizes] = [
        "tiny", "base", "small", "medium", "large-v2", "large-v3"
    ]

    def __init__(
        self,
        model_size: QuantizedModelSizes = "large-v2",
        model_dir: Optional[str | Path] = None,
        device: str = "cpu",
        compute_type: str = "int8",  # or "int8" on CPU
    ) -> None:
        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(f"model_size must be one of {self.SUPPORTED_SIZES}")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # Resolve model path
        default_cache = Path(os.path.expanduser("~/.cache/hf_ctranslate2_models"))
        self.model_dir = Path(model_dir) if model_dir else default_cache / f"whisper-{model_size}-ct2"

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Quantized model not found at: {self.model_dir}\n"
                "Convert it first with:\n"
                f"  ct2-transformers-converter --model openai/whisper-{model_size} "
                f"--output_dir {self.model_dir} --quantization {compute_type} --force"
            )

        # Load processor and model once
        self.processor = transformers.WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = ctranslate2.models.Whisper(
            str(self.model_dir),
            # device=device,
            # compute_type=compute_type,
            # intra_threads=4,
            # inter_threads=1,
        )

    def load_audio(self, audio_path: str | Path, sr: int = 16000) -> np.ndarray:
        """Load and resample audio to 16kHz mono."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        return np.asarray(audio, dtype=np.float32)

    def preprocess(self, audio: np.ndarray) -> ctranslate2.StorageView:
        """Convert raw audio waveform → log-Mel features expected by Whisper."""
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="np")
        return ctranslate2.StorageView.from_array(inputs.input_features)

    def detect_language(self, features: ctranslate2.StorageView) -> Tuple[str, float]:
        """Return most likely language token and its probability."""
        results = self.model.detect_language(features)
        language_token, prob = results[0][0]
        return language_token, prob

    def build_prompt(
        self,
        language_token: Optional[str] = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        target_lang: str = "en",
    ) -> List[int]:
        """Build the prompt token sequence required by Whisper."""
        tokens = ["<|startoftranscript|>"]

        if task == "transcribe":
            assert language_token is not None, "language_token required for transcription"
            tokens.append(language_token)
            tokens.append("<|transcribe|>")
        else:  # translate
            tokens.append(f"<|{target_lang}|>")
            tokens.append("<|translate|>")

        tokens.append("<|notimestamps|>")
        return self.processor.tokenizer.convert_tokens_to_ids(tokens)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        detect_language: bool = True,
        translate_to_english: bool = False,
    ) -> dict:
        """
        High-level method: full pipeline from file → text (transcribe or translate).

        Returns a dict with all results for easy inspection or further processing.
        """
        # 1. Load & preprocess
        audio = self.load_audio(audio_path)
        features = self.preprocess(audio)

        result = {
            "audio_path": str(Path(audio_path).resolve()),
            "duration_sec": len(audio) / 16000,
            "detected_language": None,
            "detected_language_prob": None,
            "transcription": None,
            "translation": None,
        }

        # 2. Language detection (optional but recommended)
        if detect_language:
            lang_token, prob = self.detect_language(features)
            result["detected_language"] = lang_token
            result["detected_language_prob"] = round(float(prob), 4)

        # 3. Transcription in original language
        transcribe_prompt = self.build_prompt(
            language_token=result["detected_language"],
            task="transcribe",
        )
        transcription_res = self.model.generate(features, [transcribe_prompt])
        transcription = self.processor.decode(
            transcription_res[0].sequences_ids[0],
            skip_special_tokens=True,
        )
        result["transcription"] = transcription.strip()

        # 4. Optional translation to English
        if translate_to_english:
            translate_prompt = self.build_prompt(task="translate")
            translation_res = self.model.generate(features, [translate_prompt])
            translation = self.processor.decode(
                translation_res[0].sequences_ids[0],
                skip_special_tokens=True,
            )
            result["translation"] = translation.strip()

        return result

    # Convenience aliases
    def __call__(self, audio_path: str | Path, **kwargs) -> dict:
        return self.transcribe(audio_path, **kwargs)
    
if __name__ == "__main__":
    model_size: QuantizedModelSizes = "large-v2"
    quantized_model_path = f"C:/Users/druiv/.cache/hf_ctranslate2_models/whisper-{model_size}-ct2"
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\1.wav"

    transcriber = WhisperCT2Transcriber(
        model_size=model_size,
        model_dir=quantized_model_path
    )

    result = transcriber(
        audio_path=audio_path,
        translate_to_english=True
    )

    print(f"Detected: {result['detected_language']} ({result['detected_language_prob']})")
    print("\nTranscription:\n", result["transcription"])
    print("\nEnglish translation:\n", result["translation"])
