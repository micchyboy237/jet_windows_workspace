"""
Speaker embedding model abstraction layer.
Provides a unified interface for five embedding model backends:
  - pyannote/embedding  (default, current)
  - SpeechBrain ECAPA-TDNN
  - SpeechBrain x-vector
  - NeMo TitaNet Large
  - ModelScope ERes2NetV2 (iic/speech_eres2netv2_sv_zh-cn_16k-common)
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
    MODELSCOPE_ERES2NETV2 = "modelscope_eres2netv2"
    """ModelScope ERes2NetV2: iic/speech_eres2netv2_sv_zh-cn_16k-common"""

@dataclass
class EmbeddingResult:
    """Container for a computed speaker embedding."""
    vector: np.ndarray
    model_type: EmbeddingModelType
    embedding_dim: int
    compute_time_ms: Optional[float] = None

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
            waveform = audio["waveform"]
            sr = audio["sample_rate"]
            return self.encode(waveform, sr)
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


class PyannoteEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around ``pyannote/embedding`` via ``Inference``."""
    _MODEL_TYPE = EmbeddingModelType.PYANNOTE
    _EMBEDDING_DIM = 512

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
        """Compute embedding via pyannote Inference.infer()."""
        if self._inference is None:
            self._lazy_init()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        result = self._inference.infer(waveform)
        if isinstance(result, tuple):
            emb = result[0]
        else:
            emb = result
        if isinstance(emb, memoryview):
            emb = np.asarray(emb)
        elif isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        elif not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)
        if emb.ndim == 0:
            emb = emb.reshape(1, -1)
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim > 2:
            emb = emb[0:1].reshape(1, -1)
        return emb

    def to(self, device: torch.device) -> "PyannoteEmbeddingModel":
        super().to(device)
        if self._inference is not None:
            self._inference.to(device)
        return self


class SpeechBrainECAPAEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around SpeechBrain ECAPA-TDNN."""
    _MODEL_TYPE = EmbeddingModelType.SPEECHBRAIN_ECAPA
    _EMBEDDING_DIM = 192

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
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.unsqueeze(0).float().to(self._device)
        emb = self._classifier.encode_batch(waveform)
        emb = emb.squeeze(0).cpu().numpy()
        return emb


class SpeechBrainXVectEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around SpeechBrain x-vector."""
    _MODEL_TYPE = EmbeddingModelType.SPEECHBRAIN_XVECT
    _EMBEDDING_DIM = 512

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
        emb = self._classifier.encode_batch(waveform)
        emb = emb.squeeze(0).cpu().numpy()
        return emb


class NeMoTitaNetEmbeddingModel(BaseEmbeddingModel):
    """Wrapper around NeMo TitaNet Large."""
    _MODEL_TYPE = EmbeddingModelType.NEMO_TITANET
    _EMBEDDING_DIM = 192

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, sample_rate)
            emb = self._speaker_model.get_embedding(tmp.name)
        emb = emb.cpu().numpy()
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb


class ModelScopeEres2Netv2EmbeddingModel(BaseEmbeddingModel):
    """Wrapper around ModelScope ERes2NetV2 speaker verification model.
    
    Uses the ModelScope pipeline API with ``output_emb=True`` to extract
    speaker embeddings. The pipeline handles resampling internally.
    
    Model: iic/speech_eres2netv2_sv_zh-cn_16k-common
    Expected sample rate: 16000 Hz
    """
    _MODEL_TYPE = EmbeddingModelType.MODELSCOPE_ERES2NETV2
    _EMBEDDING_DIM = 512  # ERes2NetV2 typically outputs 512-dim embeddings

    def __init__(
        self,
        model_name: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self._model_name = model_name
        self._pipeline = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        """Import and instantiate ModelScope speaker verification pipeline."""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as exc:
            raise ImportError(
                "modelscope is required for ModelScopeEres2Netv2EmbeddingModel. "
                "Install with: pip install modelscope"
            ) from exc
        console.log(
            f"{_LOGGER_PREFIX} Loading ModelScope ERes2NetV2 "
            f"'{self._model_name}'..."
        )
        # Note: ModelScope pipeline doesn't accept a device parameter directly;
        # it uses the default device (usually CPU). We'll handle device placement
        # by moving tensors after extraction if needed.
        self._pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=self._model_name,
        )
        console.log(
            f"{_LOGGER_PREFIX} ModelScope ERes2NetV2 ready "
            f"(pipeline device is managed by ModelScope)"
        )

    @property
    def model_type(self) -> EmbeddingModelType:
        return self._MODEL_TYPE

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Compute speaker embedding from raw audio waveform.
        
        The ModelScope pipeline expects a list of two audio paths for verification,
        but we extract a single embedding by providing the same audio twice and
        taking only the first embedding from the result.
        
        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``(samples,)`` or ``(1, samples)``.
        sample_rate : int
            Sample rate of the input waveform.
        
        Returns
        -------
        np.ndarray
            Embedding vector with shape ``(1, dim)``.
        """
        if self._pipeline is None:
            self._lazy_init()
        
        import tempfile
        import os
        import soundfile as sf
        
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        audio_np = waveform.cpu().numpy().astype(np.float32)
        
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
            sf.write(tmp.name, audio_np, sample_rate)
            tmp_path = tmp.name
        
        try:
            result = self._pipeline(
                [tmp_path, tmp_path],
                output_emb=True,
            )
            
            # Extract embedding from result
            if 'embs' in result:
                emb = np.asarray(result['embs'], dtype=np.float32)
            elif 'outputs' in result and 'embs' in result.get('outputs', {}):
                emb = np.asarray(result['outputs']['embs'], dtype=np.float32)
            else:
                raise KeyError(
                    f"Expected 'embs' in pipeline result, got keys: "
                    f"{list(result.keys())}"
                )
            
            # CRITICAL FIX: The pipeline returns embeddings for BOTH inputs
            # (shape: 2, dim). We only need ONE embedding, so take the first.
            if emb.ndim == 2 and emb.shape[0] == 2:
                emb = emb[0:1]  # Take first embedding, keep 2D shape (1, dim)
                console.log(
                    f"{_LOGGER_PREFIX} Extracted first embedding from batch: "
                    f"input shape (2, {emb.shape[1]}) -> output shape (1, {emb.shape[1]})"
                )
            elif emb.ndim == 1:
                emb = emb.reshape(1, -1)
            elif emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1)
                emb = emb[0:1]  # Take first embedding
            
            # Final safety check: ensure 2D shape (1, dim)
            if emb.ndim != 2 or emb.shape[0] != 1:
                console.log(
                    f"{_LOGGER_PREFIX} WARNING: Unexpected embedding shape "
                    f"{emb.shape}, forcing to (1, -1)"
                )
                emb = emb.reshape(1, -1)
            
            # Update expected dimension if needed
            if emb.shape[1] != self._EMBEDDING_DIM:
                console.log(
                    f"{_LOGGER_PREFIX} Updating embedding_dim from "
                    f"{self._EMBEDDING_DIM} to {emb.shape[1]}"
                )
                type(self)._EMBEDDING_DIM = emb.shape[1]
            
            return emb
            
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def __call__(self, audio: Union[str, Path, Dict[str, Any]]) -> np.ndarray:
        """Override to use pipeline directly with file paths when possible."""
        if isinstance(audio, dict):
            waveform = audio["waveform"]
            sr = audio["sample_rate"]
            return self.encode(waveform, sr)
        
        if self._pipeline is None:
            self._lazy_init()
        
        audio_path = str(audio)
        result = self._pipeline(
            [audio_path, audio_path],
            output_emb=True,
        )
        
        if 'embs' in result:
            emb = np.asarray(result['embs'], dtype=np.float32)
        elif 'outputs' in result and 'embs' in result.get('outputs', {}):
            emb = np.asarray(result['outputs']['embs'], dtype=np.float32)
        else:
            raise KeyError(
                f"Expected 'embs' in pipeline result, got keys: "
                f"{list(result.keys())}"
            )
        
        # CRITICAL FIX: Take only first embedding from batch
        if emb.ndim == 2 and emb.shape[0] == 2:
            emb = emb[0:1]  # Take first embedding, keep 2D shape (1, dim)
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
            emb = emb[0:1]
        
        if emb.ndim != 2 or emb.shape[0] != 1:
            emb = emb.reshape(1, -1)
        
        return emb


_MODEL_REGISTRY: Dict[EmbeddingModelType, type] = {
    EmbeddingModelType.PYANNOTE: PyannoteEmbeddingModel,
    EmbeddingModelType.SPEECHBRAIN_ECAPA: SpeechBrainECAPAEmbeddingModel,
    EmbeddingModelType.SPEECHBRAIN_XVECT: SpeechBrainXVectEmbeddingModel,
    EmbeddingModelType.NEMO_TITANET: NeMoTitaNetEmbeddingModel,
    EmbeddingModelType.MODELSCOPE_ERES2NETV2: ModelScopeEres2Netv2EmbeddingModel,
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
        "nemo_titanet", "modelscope_eres2netv2".
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
    >>> model = create_embedding_model("modelscope_eres2netv2")
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
