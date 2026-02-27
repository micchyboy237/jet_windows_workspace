"""Audio processing utilities for streaming, including buffer and VAD (with Silero-based partial/final utterance support)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from utils.asr import ASRTranscriber
from utils.translation import JapaneseToEnglishTranslator


class AudioStreamProcessor:
    def __init__(
        self,
        asr: ASRTranscriber,
        translator: JapaneseToEnglishTranslator,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = 20.0,
        partial_interval_s: float = 4.5,
    ):
        self.asr = asr
        self.translator = translator
        self.sample_rate = sample_rate

        # VAD fixed window
        self.window_size = 512 if sample_rate == 16000 else 256

        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_duration_s)
        self.partial_interval_samples = int(sample_rate * partial_interval_s)

        self.vad_iterator = None
        self.current_speech: list[
            np.ndarray
        ] = []  # accumulated speech chunks (any size)
        self.buffer: np.ndarray = np.array(
            [], dtype=np.float32
        )  # temp buffer for partial windows
        self.last_partial_samples: int = 0
        self.total_speech_samples: int = 0

        self._init_vad()

    def _init_vad(self):
        torch.set_num_threads(1)
        model = load_silero_vad(onnx=True)
        self.vad_iterator = VADIterator(
            model,
            threshold=0.5,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=350,
            speech_pad_ms=30,
        )

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[str, str, bool]]:
        if len(chunk) == 0:
            return None

        # Append new chunk to internal buffer
        self.buffer = np.concatenate([self.buffer, chunk])

        results = None

        # Process as many full 512-sample windows as possible
        while len(self.buffer) >= self.window_size:
            window = self.buffer[: self.window_size]
            self.buffer = self.buffer[self.window_size :]

            # Feed exactly 512 samples (1D float32)
            window_tensor = (
                torch.from_numpy(window).float().unsqueeze(0)
            )  # shape (1, 512)

            speech_dict = self.vad_iterator(
                window_tensor
            )  # returns dict with 'start'/'end' or None

            self.total_speech_samples += self.window_size

            if speech_dict:
                # Speech END detected → finalize current utterance
                if self.current_speech:
                    utterance = np.concatenate(self.current_speech)
                    if len(utterance) >= self.min_speech_samples:
                        jp_text, _ = self.asr.transcribe_japanese_asr(
                            utterance, self.sample_rate
                        )
                        jp_text = jp_text.strip()
                        if jp_text:
                            en_text = self.translator.translate_japanese_to_english(
                                jp_text
                            )
                            results = (en_text, jp_text, False)  # final
                            self._reset_state()
                            break  # or continue to process more if you want
                self._reset_state()
            else:
                # Accumulate (VAD considers this part speech or transition)
                self.current_speech.append(window.copy())

                # Partial trigger logic (same as before, but based on accumulated samples)
                current_len = sum(c.shape[0] for c in self.current_speech)
                since_last = current_len - self.last_partial_samples

                if (
                    since_last >= self.partial_interval_samples
                    and current_len >= self.min_speech_samples
                ):
                    utterance = np.concatenate(self.current_speech)
                    jp_text, _ = self.asr.transcribe_japanese_asr(
                        utterance, self.sample_rate
                    )
                    jp_text = jp_text.strip()
                    if jp_text:
                        en_text = self.translator.translate_japanese_to_english(jp_text)
                        self.last_partial_samples = current_len
                        results = (en_text, jp_text, True)  # partial
                        # Do NOT reset here — continue accumulating

                # Max length safety
                if current_len > self.max_speech_samples:
                    utterance = np.concatenate(self.current_speech)
                    jp_text, _ = self.asr.transcribe_japanese_asr(
                        utterance, self.sample_rate
                    )
                    jp_text = jp_text.strip()
                    if jp_text:
                        en_text = self.translator.translate_japanese_to_english(jp_text)
                        results = (en_text, jp_text, False)
                        self._reset_state()

        if results:
            return results

        return None

    def _reset_state(self):
        self.current_speech.clear()
        self.last_partial_samples = 0
        self.total_speech_samples = 0
        if self.vad_iterator:
            self.vad_iterator.reset_states()

    def reset(self):
        self._reset_state()
        self.buffer = np.array([], dtype=np.float32)
