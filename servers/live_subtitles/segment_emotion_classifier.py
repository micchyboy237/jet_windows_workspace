import os
import io
import tempfile
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
from transformers import pipeline
from typing import Sequence, Mapping, Union
import torch

AudioInput = Union[np.ndarray, bytes, bytearray, io.BytesIO, str, Path]


class SegmentEmotionClassifier:
    """
    Wraps a Hugging Face audio-classification pipeline for
    litagin/anime_speech_emotion_classification.
    """

    def __init__(
        self,
        model_id: str = "litagin/anime_speech_emotion_classification",
        device: int | str | None = None,
    ):
        """
        Args:
            model_id: HF model identifier
            device: Pipeline device ("cpu" or CUDA device index)
        """
        resolved_device = self._resolve_device(device)

        self.pipe = pipeline(
            task="audio-classification",
            model=model_id,
            feature_extractor=model_id,
            trust_remote_code=True,
            device=resolved_device,
        )

    @staticmethod
    def _resolve_device(device: int | str | None) -> int | str:
        """
        Resolve execution device for Hugging Face pipelines.

        Priority:
          1. Explicit device argument
          2. CUDA
          3. Apple MPS
          4. CPU
        """
        # Explicit override
        if device is not None:
            return device

        # CUDA (Windows / Linux / NVIDIA)
        if torch.cuda.is_available():
            return 0  # cuda:0

        # Apple Silicon (macOS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        # Fallback
        return -1  # CPU

    def _write_temp_wav(self, pcm_bytes: bytes, sr: int) -> str:
        """
        Writes raw PCM int16 to a temp WAV for the classifier.
        Returns the file path.
        """
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        wavfile.write(tmp, sr, arr)
        return tmp

    def _normalize_input(
        self,
        audio: AudioInput,
        sample_rate: int | None,
    ) -> str:
        """
        Normalize supported audio input types into a temporary WAV file path.
        """
        # Path / str → assume WAV file already
        if isinstance(audio, (str, Path)):
            return str(audio)

        # BytesIO → raw PCM bytes
        if isinstance(audio, io.BytesIO):
            if sample_rate is None:
                raise ValueError("sample_rate is required for BytesIO input")
            return self._write_temp_wav(audio.getvalue(), sample_rate)

        # bytes / bytearray → raw PCM16
        if isinstance(audio, (bytes, bytearray)):
            if sample_rate is None:
                raise ValueError("sample_rate is required for PCM byte input")
            return self._write_temp_wav(bytes(audio), sample_rate)

        # numpy array → waveform
        if isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required for ndarray input")

            # Ensure float32 waveform in [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            wavfile.write(tmp, sample_rate, audio)
            return tmp

        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    def classify(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> Sequence[Mapping[str, object]]:
        """
        Classify emotions for a single audio segment.

        Args:
            audio: AudioInput (ndarray, PCM bytes, BytesIO, path)
            sample_rate: required for non-file inputs

        Returns:
            List of {label: str, score: float} sorted by score descending
        """
        wav_path = self._normalize_input(audio, sample_rate)
        try:
            results = self.pipe(wav_path)
        finally:
            # Cleanup only if we created a temp file
            if not isinstance(audio, (str, Path)):
                try:
                    Path(wav_path).unlink()
                except Exception:
                    pass

        sorted_results = sorted(
            results, key=lambda r: r.get("score", 0.0), reverse=True
        )
        return sorted_results

if __name__ == "__main__":
    import json
    from utils import resolve_audio_paths

    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server_spyxfamily_intro"
    segment_paths = resolve_audio_paths(audio_dir)

    def load_wav_as_pcm_bytes(path: Path) -> tuple[bytes, int]:
        """
        Utility: load WAV file and return PCM16 bytes + sample rate.
        """
        sr, audio = wavfile.read(path)

        # Ensure int16 PCM (expected by classifier)
        if audio.dtype != np.int16:
            audio = (audio * 32768.0).astype(np.int16)

        return audio.tobytes(), sr

    # --------------------------------------------------
    # Usage
    # --------------------------------------------------

    audio_file = segment_paths[1]
    # Initialize once (reuse across utterances)
    emotion_classifier = SegmentEmotionClassifier()

    # Load example audio
    pcm_bytes, sample_rate = load_wav_as_pcm_bytes(
        Path(audio_file)
    )

    # Run emotion classification with new API
    results = emotion_classifier.classify(
        audio=pcm_bytes,
        sample_rate=sample_rate,
    )

    # Inspect results
    print("All emotion scores:")
    for r in results:
        print(f"  {r['label']:<12} {r['score']:.4f}")

    # Top emotion
    if results:
        top = results[0]
        print("\nTop emotion:")
        print(f"  label={top['label']} score={top['score']:.4f}")
