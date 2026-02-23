"""Audio processing utilities for streaming, including buffer and VAD."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.asr import ASRTranscriber
from utils.translation import JapaneseToEnglishTranslator


class AudioStreamProcessor:
    """Processes audio stream with VAD to trigger ASR + translation."""

    def __init__(
        self,
        asr: ASRTranscriber,
        translator: JapaneseToEnglishTranslator,
        sample_rate: int = 16000,
    ):
        self.asr = asr
        self.translator = translator
        self.sample_rate = sample_rate
        self.current_speech: list[np.ndarray] = []
        self.is_speaking: bool = False
        self.silence_frames: int = 0
        self.max_silence: int = int(sample_rate * 0.7)  # 0.7s silence to end utterance
        self.chunk_samples: int = 512

    def _calculate_rms(self, chunk: np.ndarray) -> float:
        """Small method for energy calculation."""
        return np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    def _handle_vad(self, chunk: np.ndarray) -> bool:
        """Simple energy VAD. Returns if speech detected."""
        rms = self._calculate_rms(chunk)
        return rms > 0.015  # adjustable threshold

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[str, str]]:
        """Process one audio chunk, return (en, jp) if utterance ready."""
        if len(chunk) == 0:
            return None

        is_speech = self._handle_vad(chunk)

        if is_speech:
            self.is_speaking = True
            self.silence_frames = 0
            self.current_speech.append(chunk.copy())
        elif self.is_speaking:
            self.current_speech.append(chunk.copy())
            self.silence_frames += len(chunk)
            if self.silence_frames > self.max_silence:
                # End of utterance
                if self.current_speech:
                    utterance = np.concatenate(self.current_speech)
                    jp_text, _ = self.asr.transcribe_japanese_asr(
                        utterance, self.sample_rate
                    )
                    if jp_text.strip():
                        en_text = self.translator.translate_japanese_to_english(jp_text)
                        self.current_speech = []
                        self.is_speaking = False
                        self.silence_frames = 0
                        return en_text, jp_text
                self.current_speech = []
                self.is_speaking = False
                self.silence_frames = 0
        return None
