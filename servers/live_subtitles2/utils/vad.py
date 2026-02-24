import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from jet.audio.audio_types import AudioInput
from jet.logger import logger
from speechbrain.inference.VAD import VAD


class SpeechBrainVAD:
    def __init__(
        self,
        target_sample_rate: int = 16000,
        context_seconds: float = 1.6,
        inference_every_seconds: float = 0.32,
        model_source: str = "speechbrain/vad-crdnn-libriparty",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("[VAD] Loading SpeechBrain %s on %s...", model_source, self.device)
        self.vad = VAD.from_hparams(
            source=model_source,
            savedir="pretrained_models/vad-crdnn-libriparty",
            run_opts={"device": str(self.device)},
        )
        logger.success("[VAD] SpeechBrain VAD loaded")

        self.sample_rate = target_sample_rate
        self.context_samples = int(context_seconds * target_sample_rate)
        self.audio_ring = torch.zeros(
            self.context_samples, dtype=torch.float32, device=self.device
        )
        self.write_pos = 0
        self.last_prob = 0.0
        self.min_time_between_inferences = inference_every_seconds
        self.last_inference_time = -999.0

    @torch.inference_mode()
    def get_prob(self, chunk_bytes: bytes, mono: bool = True) -> float:
        """Legacy streaming-style inference (expects raw PCM 16-bit bytes)"""
        now = time.monotonic()
        if now - self.last_inference_time < self.min_time_between_inferences:
            return self.last_prob

        chunk_np = (
            np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        chunk_t = torch.from_numpy(chunk_np).to(self.device)
        chunk_len = len(chunk_t)

        space = self.context_samples - self.write_pos
        if chunk_len <= space:
            self.audio_ring[self.write_pos : self.write_pos + chunk_len] = chunk_t
            self.write_pos += chunk_len
        else:
            self.audio_ring[:chunk_len] = chunk_t[-self.context_samples :]
            self.write_pos = chunk_len

        prob_tensor = self.vad.get_speech_prob_chunk(self.audio_ring.unsqueeze(0))
        prob = float(prob_tensor[-1, -1].item())

        self.last_prob = prob
        self.last_inference_time = now
        return prob

    # ──────────────────────────────────────────────────────────────────────────────
    # New flexible method – does NOT touch speechbrain.inference.VAD
    # ──────────────────────────────────────────────────────────────────────────────

    def _normalize_audio_input(self, audio: AudioInput) -> torch.Tensor:
        """Convert any AudioInput → mono float32 waveform at self.sample_rate on device"""
        if isinstance(audio, (str, os.PathLike, Path)):
            waveform, sr = torchaudio.load(str(audio))
        elif isinstance(audio, bytes):
            # Assume raw 16-bit PCM little-endian, mono
            arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(arr).float().unsqueeze(0)
            sr = self.sample_rate  # trust input rate
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            sr = self.sample_rate
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        elif isinstance(audio, torch.Tensor):
            waveform = audio.float()
            sr = self.sample_rate
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise TypeError(f"Unsupported AudioInput type: {type(audio).__name__}")

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        return waveform.to(self.device)

    @torch.inference_mode()
    def get_speech_probs(
        self,
        audio: AudioInput,
        chunk_size_sec: float = 10.0,
        overlap: bool = False,
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute frame-level speech probabilities from file, bytes, numpy array or tensor.

        Returns shape: (1, n_frames, 1)   probabilities ∈ [0,1]
        Time resolution ≈ self.vad.time_resolution (usually ~10–20 ms)
        """
        waveform = self._normalize_audio_input(audio)

        duration_sec = waveform.shape[1] / self.sample_rate

        # For short audio or when chunking is disabled → process whole
        if chunk_size_sec <= 0 or duration_sec <= chunk_size_sec * 1.5:
            probs = self.vad.get_speech_prob_chunk(waveform)
        else:
            # For longer audio: fall back to the original file-based chunked method
            # (this requires a path – we create a temporary file if needed)
            if isinstance(audio, (str, os.PathLike, Path)):
                probs = self.vad.get_speech_prob_file(
                    str(audio),
                    large_chunk_size=chunk_size_sec,
                    small_chunk_size=10.0,
                    overlap_small_chunk=overlap,
                )
            else:
                # Non-file input → temporary file (not ideal, but safe fallback)
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    torchaudio.save(tmp.name, waveform.cpu(), self.sample_rate)
                    probs = self.vad.get_speech_prob_file(
                        tmp.name,
                        large_chunk_size=chunk_size_sec,
                        small_chunk_size=10.0,
                        overlap_small_chunk=overlap,
                    )
                os.unlink(tmp.name)

        if not return_tensor:
            probs = probs.cpu().numpy()

        return probs
