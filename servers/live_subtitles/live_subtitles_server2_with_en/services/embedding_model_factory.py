"""
Speaker embedding model abstraction layer.

Provides a unified interface for four embedding model backends:
  - pyannote/embedding  (default, current)
  - SpeechBrain ECAPA-TDNN
  - SpeechBrain x-vector
  - NeMo TitaNet Large
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from rich.console import Console

console = Console()
_LOGGER_PREFIX = "[dim cyan]EmbeddingFactory[/dim cyan]"


# ---------------------------------------------------------------------------
# Model type enumeration
# ---------------------------------------------------------------------------

class EmbeddingModelType(str, Enum):
    """Supported speaker embedding model backends."""
    PYANNOTE = "pyannote"
    """pyannote/embedding — current default."""
    SPEECHBRAIN_ECAPA = "speechbrain_ecapa"
    """SpeechBrain ECAPA-TDNN: speechbrain/spkrec-ecapa-voxceleb"""
    SPEECHBRAIN_XVECT = "speechbrain_xvect"
    """SpeechBrain x-vector: speechbrain/spkrec-xvect-voxceleb"""
    NEMO_TITANET = "nemo_titanet"
    """NeMo TitaNet Large: titanet_large"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Container for a computed speaker embedding."""
    vector: np.ndarray          # shape (1, dim) or (dim,)
    model_type: EmbeddingModelType
    embedding_dim: int
    compute_time_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseEmbeddingModel(ABC):
    """Abstract interface for speaker embedding models.

    All backends must implement `encode()` which accepts a waveform and
    returns a numpy embedding array.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    @abstractmethod
    def model_type(self) -> EmbeddingModelType:
        """Return the enum identifying this backend."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""

    @abstractmethod
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Compute speaker embedding from raw audio.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``(samples,)`` or ``(1, samples)``.
        sample_rate : int
            Expected sample rate (model may resample internally).

        Returns
        -------
        np.ndarray
            Embedding vector with shape ``(1, dim)``.
        """

    def __call__(self, audio: Union[str, Path, Dict[str, Any]]) -> np.ndarray:
        """Convenience wrapper: accept path or dict like pyannote Inference."""
        if isinstance(audio, dict):
            # Backward-compatible dict interface
            waveform = audio["waveform"]
            sr = audio["sample_rate"]
            return self.encode(waveform, sr)

        # File path interface — load and send to device
        import librosa
        signal, sr = librosa.load(str(audio), sr=None, mono=True)
        waveform = torch.from_numpy(signal).float().to(self._device)
        return self.encode(waveform, sr)

    def to(self, device: torch.device) -> "BaseEmbeddingModel":
        """Move model to *device* (no-op by default)."""
        self._device = device
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.model_type.value}, dim={self.embedding_dim})"


# ---------------------------------------------------------------------------
# Pyannote backend (current default)
# ---------------------------------------------------------------------------

class PyannoteEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around ``pyannote/embedding`` via ``Inference``."""

    _MODEL_TYPE = EmbeddingModelType.PYANNOTE
    _EMBEDDING_DIM = 512  # pyannote/embedding default

    def __init__(
        self,
        model_name: str = "pyannote/embedding",
        window: str = "whole",
        device: Optional[torch.device] = None,
        **inference_kwargs,
    ) -> None:
        super().__init__(device=device)
        self._model_name = model_name
        self._window = window
        self._inference_kwargs = inference_kwargs
        self._inference = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        """Import and instantiate pyannote components."""
        try:
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for PyannoteEmbeddingModel. "
                "Install with: pip install pyannote.audio"
            ) from exc

        console.log(f"{_LOGGER_PREFIX} Loading pyannote model '{self._model_name}'...")
        model = Model.from_pretrained(self._model_name)
        # Pass device explicitly so Inference uses our target device
        self._inference = Inference(
            model, 
            window=self._window, 
            device=self._device,
            **self._inference_kwargs
        )
        console.log(f"{_LOGGER_PREFIX} Pyannote model ready on {self._device}")

    @property
    def model_type(self) -> EmbeddingModelType:
        return self._MODEL_TYPE

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Compute embedding via pyannote Inference.infer().

        Uses ``Inference.infer()`` which handles device placement internally
        and returns raw model outputs.
        """
        if self._inference is None:
            self._lazy_init()

        # Ensure waveform is (batch=1, channels=1, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1, channels, samples)

        # 🔧 infer() handles device placement internally via chunks.to(self.device)
        result = self._inference.infer(waveform)

        # Handle different return types from infer()
        # - np.ndarray for single-task models
        # - tuple of np.ndarray for multi-task models
        if isinstance(result, tuple):
            emb = result[0]
        else:
            emb = result

        # Convert to numpy array if needed
        if isinstance(emb, memoryview):
            emb = np.asarray(emb)
        elif isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        elif not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)

        # Ensure shape is (1, embedding_dim)
        if emb.ndim == 0:
            emb = emb.reshape(1, -1)
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim > 2:
            # Take the first sample in the batch and flatten
            emb = emb[0:1].reshape(1, -1)

        return emb

    def to(self, device: torch.device) -> "PyannoteEmbeddingModel":
        super().to(device)
        if self._inference is not None:
            self._inference.to(device)
        return self


# ---------------------------------------------------------------------------
# SpeechBrain ECAPA-TDNN backend
# ---------------------------------------------------------------------------

class SpeechBrainECAPAEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around SpeechBrain ECAPA-TDNN."""

    _MODEL_TYPE = EmbeddingModelType.SPEECHBRAIN_ECAPA
    _EMBEDDING_DIM = 192  # ECAPA-TDNN

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self._source = source
        self._classifier = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "speechbrain is required for SpeechBrainECAPAEmbeddingModel. "
                "Install with: pip install speechbrain"
            ) from exc

        console.log(f"{_LOGGER_PREFIX} Loading SpeechBrain ECAPA from '{self._source}'...")
        self._classifier = EncoderClassifier.from_hparams(
            source=self._source,
            run_opts={"device": str(self._device)},
        )
        console.log(f"{_LOGGER_PREFIX} SpeechBrain ECAPA ready on {self._device}")

    @property
    def model_type(self) -> EmbeddingModelType:
        return self._MODEL_TYPE

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        if self._classifier is None:
            self._lazy_init()

        # SpeechBrain expects (batch, time) float tensor
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.unsqueeze(0).float().to(self._device)

        emb = self._classifier.encode_batch(waveform)  # (1, 1, dim)
        emb = emb.squeeze(0).cpu().numpy()              # (1, dim)
        return emb


# ---------------------------------------------------------------------------
# SpeechBrain x-vector backend
# ---------------------------------------------------------------------------

class SpeechBrainXVectEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around SpeechBrain x-vector."""

    _MODEL_TYPE = EmbeddingModelType.SPEECHBRAIN_XVECT
    _EMBEDDING_DIM = 512  # x-vector

    def __init__(
        self,
        source: str = "speechbrain/spkrec-xvect-voxceleb",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self._source = source
        self._classifier = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "speechbrain is required for SpeechBrainXVectEmbeddingModel. "
                "Install with: pip install speechbrain"
            ) from exc

        console.log(f"{_LOGGER_PREFIX} Loading SpeechBrain x-vector from '{self._source}'...")
        self._classifier = EncoderClassifier.from_hparams(
            source=self._source,
            run_opts={"device": str(self._device)},
        )
        console.log(f"{_LOGGER_PREFIX} SpeechBrain x-vector ready on {self._device}")

    @property
    def model_type(self) -> EmbeddingModelType:
        return self._MODEL_TYPE

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        if self._classifier is None:
            self._lazy_init()

        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.unsqueeze(0).float().to(self._device)

        emb = self._classifier.encode_batch(waveform)  # (1, 1, dim)
        emb = emb.squeeze(0).cpu().numpy()              # (1, dim)
        return emb


# ---------------------------------------------------------------------------
# NeMo TitaNet backend
# ---------------------------------------------------------------------------

class NeMoTitaNetEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around NeMo TitaNet Large."""

    _MODEL_TYPE = EmbeddingModelType.NEMO_TITANET
    _EMBEDDING_DIM = 192  # TitaNet Large

    def __init__(
        self,
        model_name: str = "titanet_large",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self._model_name = model_name
        self._speaker_model = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError(
                "nemo_toolkit is required for NeMoTitaNetEmbeddingModel. "
                "Install with: pip install nemo_toolkit"
            ) from exc

        console.log(f"{_LOGGER_PREFIX} Loading NeMo '{self._model_name}'...")
        self._speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self._model_name
        )
        self._speaker_model = self._speaker_model.to(self._device)
        self._speaker_model.eval()
        console.log(f"{_LOGGER_PREFIX} NeMo TitaNet ready on {self._device}")

    @property
    def model_type(self) -> EmbeddingModelType:
        return self._MODEL_TYPE

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """NeMo's get_embedding() accepts a file path or tensor.

        Because we already have a waveform tensor, we save to a temp file.
        A more efficient path would use internal methods, but this is the
        public API.
        """
        if self._speaker_model is None:
            self._lazy_init()

        import tempfile
        import soundfile as sf

        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        audio_np = waveform.cpu().numpy()

        # NeMo's get_embedding expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, sample_rate)
            emb = self._speaker_model.get_embedding(tmp.name)

        emb = emb.cpu().numpy()
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Registry of available models
_MODEL_REGISTRY: Dict[EmbeddingModelType, type] = {
    EmbeddingModelType.PYANNOTE: PyannoteEmbeddingModel,
    EmbeddingModelType.SPEECHBRAIN_ECAPA: SpeechBrainECAPAEmbeddingModel,
    EmbeddingModelType.SPEECHBRAIN_XVECT: SpeechBrainXVectEmbeddingModel,
    EmbeddingModelType.NEMO_TITANET: NeMoTitaNetEmbeddingModel,
}


def create_embedding_model(
    model_type: Union[str, EmbeddingModelType] = EmbeddingModelType.PYANNOTE,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BaseEmbeddingModel:
    """Factory: create a speaker embedding model by type.

    Parameters
    ----------
    model_type : str or EmbeddingModelType
        One of "pyannote", "speechbrain_ecapa", "speechbrain_xvect",
        "nemo_titanet".
    device : torch.device, optional
        Target device.
    **kwargs
        Forwarded to the model constructor (e.g. ``model_name``,
        ``source``, ``window``).

    Returns
    -------
    BaseEmbeddingModel
        Ready-to-use embedding model instance.

    Raises
    ------
    ValueError
        If *model_type* is not recognised.

    Examples
    --------
    >>> model = create_embedding_model("pyannote")
    >>> model = create_embedding_model("speechbrain_ecapa")
    >>> model = create_embedding_model(EmbeddingModelType.NEMO_TITANET)
    """
    if isinstance(model_type, str):
        try:
            model_type = EmbeddingModelType(model_type)
        except ValueError:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Choose from: {[e.value for e in EmbeddingModelType]}"
            )

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' is not registered. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    klass = _MODEL_REGISTRY[model_type]
    console.log(f"{_LOGGER_PREFIX} Creating {klass.__name__} (device={device})")
    return klass(device=device, **kwargs)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """Return a summary of all registered embedding models."""
    return {
        model_type.value: {
            "class": klass.__name__,
            "embedding_dim": klass._EMBEDDING_DIM,
        }
        for model_type, klass in _MODEL_REGISTRY.items()
    }
