import numpy as np
import torch
from faster_whisper import WhisperModel
from typing import List
import logging

log = logging.getLogger("subtitle_server")

class TranscriptionSession:
    def __init__(
        self,
        model_name: str = "large-v3",  # Excellent balance: ~6x faster than large-v3, near accuracy
        device: str = "cuda",
        compute_type: str = "int8",       # GTX 1660 supports fp16 well
        language: str = "ja",                # Change if needed; None for auto-detect
    ):
        log.info(f"Loading faster-whisper model '{model_name}' on CUDA ({compute_type})...")
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        log.info("Model loaded")

        # Load Silero VAD (lightweight, accurate, torch hub is simple & fast enough)
        torch.set_num_threads(4)
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )
        (
            self.get_speech_timestamps,
            _,
            self.read_audio,
            _,
            _,
        ) = utils

        self.sample_rate = 16000
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.min_speech_ms = 250
        self.min_silence_ms = 400  # Tune for natural pause detection

    async def process_audio_chunk(self, chunk: bytes) -> List[str]:
        # Convert incoming int16 bytes to float32 [-1.0, 1.0]
        audio_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

        # Need at least ~1 second for meaningful VAD
        if len(self.audio_buffer) < self.sample_rate:
            return []

        # Run VAD on current buffer
        speech_timestamps = self.get_speech_timestamps(
            self.audio_buffer,
            self.vad_model,
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
        )

        subtitles: List[str] = []

        if speech_timestamps:
            # Check if last speech segment is complete (ended with silence)
            last_segment = speech_timestamps[-1]
            end_sample = last_segment["end"]
            buffer_samples = len(self.audio_buffer)

            # Require some silence after speech end
            if buffer_samples - end_sample > self.sample_rate * 0.3:
                utterance = self.audio_buffer[:end_sample]
                self.audio_buffer = self.audio_buffer[end_sample:]

                # Transcribe utterance
                segments, _ = self.model.transcribe(
                    utterance,
                    language=self.language,
                    beam_size=5,
                    vad_filter=False,  # Already handled by Silero
                    word_timestamps=False,  # Set True if you want word-level sync later
                )
                text = " ".join(seg.text for seg in segments).strip()
                if text:
                    subtitles.append(text)
                    log.info(f"[bold green]Subtitle:[/bold green] {text[:100]}...")

        return subtitles
