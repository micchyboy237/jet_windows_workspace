import numpy as np
import torch
from typing import List

from pyannote.audio import Inference, Model


class PyannoteEmbeddingModel:
    """
    Modern pyannote embedding wrapper (v4+ API).
    """

    def __init__(self, device: str | None = None,):
        # Load model explicitly (new API requirement)
        model = Model.from_pretrained("pyannote/embedding")

        self.inference = Inference(
            model,
            window="whole",
        )

        if device is not None:
            _device = torch.device(device)
            device_str = str(_device)
        else:
            if torch.backends.mps.is_available():
                device_str = "mps"
            elif torch.cuda.is_available():
                device_str = "cuda"
            else:
                device_str = "cpu"
            _device = torch.device(device_str)

        self.inference.to(_device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Args:
            audio: mono float32 waveform
            sample_rate: expected 16000

        Returns:
            embedding vector (np.ndarray)
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Convert numpy → torch (required by latest pyannote)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        embedding = self.inference(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        )

        # convert to numpy if needed
        if hasattr(embedding, "data"):
            embedding = embedding.data

        return np.asarray(embedding).flatten()


class SpeakerManager:
    """
    Incremental speaker clustering using cosine similarity.
    """

    def __init__(self, threshold: float = 0.75, max_speakers: int = 10):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.embeddings: List[np.ndarray] = []

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def assign_speaker(self, embedding: np.ndarray) -> int:
        if not self.embeddings:
            self.embeddings.append(embedding)
            return 0

        sims = [self._cosine_similarity(embedding, e) for e in self.embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.threshold:
            return best_idx

        if len(self.embeddings) < self.max_speakers:
            self.embeddings.append(embedding)
            return len(self.embeddings) - 1

        return best_idx
