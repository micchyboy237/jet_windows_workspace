"""
Speaker embedding model abstraction layer.
Provides a unified interface for four embedding model backends:
  - pyannote/embedding  (default, current)
  - SpeechBrain ECAPA-TDNN
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

try:
    from services.audio_utils import load_audio, SAMPLE_RATE
except ImportError:
    from audio_utils import load_audio, SAMPLE_RATE

console = Console()
_LOGGER_PREFIX = "[dim cyan]EmbeddingFactory[/dim cyan]"


# ── Model type enum ───────────────────────────────────────────────────────────

class EmbeddingModelType(str, Enum):
    """Supported speaker embedding model backends."""

    PYANNOTE = "pyannote"
    """pyannote/embedding — current default."""

    SPEECHBRAIN_ECAPA = "speechbrain_ecapa"
    """SpeechBrain ECAPA-TDNN: speechbrain/spkrec-ecapa-voxceleb"""

    NEMO_TITANET = "nemo_titanet"
    """NeMo TitaNet Large: titanet_large"""

    MODELSCOPE_ERES2NETV2 = "modelscope_eres2netv2"
    """ModelScope ERes2NetV2: iic/speech_eres2netv2_sv_zh-cn_16k-common"""


# ── Threshold data classes ────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Container for a computed speaker embedding."""
    vector: np.ndarray
    model_type: EmbeddingModelType
    embedding_dim: int
    compute_time_ms: Optional[float] = None


@dataclass
class EmbeddingThresholds:
    """Optimal thresholds for a specific embedding model."""
    same: float
    possible: float
    new_speaker: float
    promotion: float  # NEW: minimum similarity to consider outliers a match


class EmbeddingThresholdProvider:
    """Provides model-specific thresholds for speaker matching.

    Each embedding model produces embeddings in a different similarity space.
    This class centralizes the empirically-determined optimal thresholds
    for each backend so callers don't need to guess.

    Thresholds are calibrated from evaluation results (Jun 2026) on a
    4-speaker, 64-trial dataset.  Derivation per model:
        same         ≈ EER_threshold + margin  (conservative accept boundary)
        possible     ≈ (intra_sim + inter_sim) / 2  (uncertain zone midpoint)
        new_speaker  ≈ inter_sim + 0.03  (just above noise floor)

    Observed similarity ranges:
        pyannote:      intra=0.274, inter=0.108, EER_thresh=0.153
        ecapa:         intra=0.320, inter=0.142, EER_thresh=0.105
        nemo_titanet:  intra=0.620, inter=0.204, EER_thresh=0.357
        modelscope:    intra=0.640, inter=0.271, EER_thresh=0.465

    Usage:
        provider = EmbeddingThresholdProvider()
        thresholds = provider.get_thresholds(EmbeddingModelType.NEMO_TITANET)
        # thresholds.same -> 0.50
    """

    _THRESHOLDS: Dict[EmbeddingModelType, EmbeddingThresholds] = {
        EmbeddingModelType.PYANNOTE: EmbeddingThresholds(
            same=0.75,
            possible=0.50,
            new_speaker=0.30,
            promotion=0.55,
        ),
        EmbeddingModelType.SPEECHBRAIN_ECAPA: EmbeddingThresholds(
            same=0.65,
            possible=0.40,
            new_speaker=0.25,
            promotion=0.55,
        ),
        EmbeddingModelType.NEMO_TITANET: EmbeddingThresholds(
            same=0.70,
            possible=0.45,
            new_speaker=0.25,
            promotion=0.55,
        ),
        EmbeddingModelType.MODELSCOPE_ERES2NETV2: EmbeddingThresholds(
            same=0.70,
            possible=0.55,
            new_speaker=0.35,
            promotion=0.55,
        ),
    }

    @classmethod
    def get_thresholds(
        cls,
        model_type: Union[str, EmbeddingModelType],
    ) -> EmbeddingThresholds:
        """Get the recommended thresholds for a given embedding model.

        Parameters
        ----------
        model_type : str or EmbeddingModelType
            The embedding model backend identifier.

        Returns
        -------
        EmbeddingThresholds
            Dataclass with same, possible, and new_speaker thresholds.

        Raises
        ------
        ValueError
            If the model_type is not recognized.
        """
        if isinstance(model_type, str):
            try:
                model_type = EmbeddingModelType(model_type)
            except ValueError:
                raise ValueError(
                    f"Unknown model type '{model_type}'. "
                    f"Choose from: {[e.value for e in EmbeddingModelType]}"
                )

        if model_type not in cls._THRESHOLDS:
            raise ValueError(
                f"No thresholds defined for model type '{model_type}'. "
                f"Available: {list(cls._THRESHOLDS.keys())}"
            )

        thresholds = cls._THRESHOLDS[model_type]
        console.log(
            f"{_LOGGER_PREFIX} Thresholds for {model_type.value}: "
            f"same={thresholds.same}, possible={thresholds.possible}, "
            f"new_speaker={thresholds.new_speaker}"
        )
        return thresholds

    @classmethod
    def resolve_thresholds(
        cls,
        model_type: Union[str, EmbeddingModelType],
        threshold_same: Optional[float] = None,
        threshold_possible: Optional[float] = None,
        threshold_new_speaker: Optional[float] = None,
        threshold_promotion: Optional[float] = None,  # NEW
    ) -> EmbeddingThresholds:
        """Resolve thresholds, using provided values or falling back to defaults.

        If any threshold is None, the model-specific default is used.

        Parameters
        ----------
        model_type : str or EmbeddingModelType
            The embedding model backend identifier.
        threshold_same : float, optional
            User-provided same-speaker threshold.
        threshold_possible : float, optional
            User-provided possible-match threshold.
        threshold_new_speaker : float, optional
            User-provided new-speaker threshold.

        Returns
        -------
        EmbeddingThresholds
            Resolved thresholds with all values populated.
        """
        defaults = cls.get_thresholds(model_type)
        return EmbeddingThresholds(
            same=(threshold_same if threshold_same is not None else defaults.same),
            possible=(threshold_possible if threshold_possible is not None else defaults.possible),
            new_speaker=(threshold_new_speaker if threshold_new_speaker is not None else defaults.new_speaker),
            promotion=(threshold_promotion if threshold_promotion is not None else defaults.promotion),  # NEW
        )


# ── Base model interface ──────────────────────────────────────────────────────

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
        return (
            f"{self.__class__.__name__}"
            f"(type={self.model_type.value}, dim={self.embedding_dim})"
        )


# ── Backend implementations ───────────────────────────────────────────────────

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
            **self._inference_kwargs,
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

        console.log(
            f"{_LOGGER_PREFIX} Loading SpeechBrain ECAPA from '{self._source}'..."
        )
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
        """NeMo's get_embedding() accepts a file path; we write a temp file."""
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
    speaker embeddings.

    Model: iic/speech_eres2netv2_sv_zh-cn_16k-common
    Expected sample rate: 16000 Hz

    Notes
    -----
    torchaudio.sox_effects was removed in torchaudio >= 0.13.  ModelScope
    uses it internally when the input sample rate differs from 16 kHz.
    To avoid the AttributeError, both ``encode()`` and ``__call__()`` now
    pre-resample to 16 kHz via librosa before writing the temp WAV file,
    so the pipeline never needs to resample.
    """

    _MODEL_TYPE = EmbeddingModelType.MODELSCOPE_ERES2NETV2
    _EMBEDDING_DIM = 192
    _TARGET_SR = 16_000

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

    def _resample_if_needed(
        self, audio_np: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, int]:
        """Resample audio to TARGET_SR using librosa if needed.

        This avoids torchaudio.sox_effects (removed in torchaudio >= 0.13)
        which ModelScope calls internally when the sample rate doesn't match.

        Parameters
        ----------
        audio_np : np.ndarray
            Mono float32 audio array.
        sample_rate : int
            Current sample rate of audio_np.

        Returns
        -------
        (resampled_audio, new_sample_rate)
        """
        if sample_rate == self._TARGET_SR:
            return audio_np, sample_rate

        import librosa
        console.log(
            f"{_LOGGER_PREFIX} Pre-resampling {sample_rate} Hz → "
            f"{self._TARGET_SR} Hz (avoids torchaudio.sox_effects)"
        )
        resampled = librosa.resample(
            audio_np, orig_sr=sample_rate, target_sr=self._TARGET_SR
        )
        return resampled, self._TARGET_SR

    def _extract_emb_from_result(self, result: dict) -> np.ndarray:
        """Pull embedding array out of the ModelScope pipeline result dict."""
        if "embs" in result:
            emb = np.asarray(result["embs"], dtype=np.float32)
        elif "outputs" in result and "embs" in result.get("outputs", {}):
            emb = np.asarray(result["outputs"]["embs"], dtype=np.float32)
        else:
            raise KeyError(
                f"Expected 'embs' in pipeline result, got keys: "
                f"{list(result.keys())}"
            )

        # Pipeline returns shape (2, dim) when given two identical paths;
        # we only need the first embedding.
        if emb.ndim == 2 and emb.shape[0] == 2:
            emb = emb[0:1]
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)[0:1]

        if emb.ndim != 2 or emb.shape[0] != 1:
            console.log(
                f"{_LOGGER_PREFIX} WARNING: unexpected embedding shape "
                f"{emb.shape}, forcing to (1, -1)"
            )
            emb = emb.reshape(1, -1)

        # Update class-level dim if the model changed it
        if emb.shape[1] != self._EMBEDDING_DIM:
            console.log(
                f"{_LOGGER_PREFIX} Updating embedding_dim from "
                f"{self._EMBEDDING_DIM} to {emb.shape[1]}"
            )
            type(self)._EMBEDDING_DIM = emb.shape[1]

        return emb

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Compute speaker embedding from raw audio waveform.

        Pre-resamples to 16 kHz before writing the temp file so ModelScope
        never needs to call torchaudio.sox_effects internally.

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

        # Pre-resample to 16 kHz — prevents torchaudio.sox_effects AttributeError
        audio_np, sample_rate = self._resample_if_needed(audio_np, sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, sample_rate)
            tmp_path = tmp.name

        try:
            result = self._pipeline([tmp_path, tmp_path], output_emb=True)
            return self._extract_emb_from_result(result)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def __call__(self, audio: Union[str, Path, Dict[str, Any]]) -> np.ndarray:
        """Override to use pipeline directly with file paths when possible.

        If the file is already 16 kHz, passes the path directly to the
        pipeline (fastest path).  If the sample rate differs, loads via
        librosa, resamples, and routes through encode() to write a
        correctly-resampled temp file.  This prevents torchaudio.sox_effects
        errors in both code paths.
        """
        if isinstance(audio, dict):
            waveform = audio["waveform"]
            sr = audio["sample_rate"]
            return self.encode(waveform, sr)

        if self._pipeline is None:
            self._lazy_init()

        # Check the actual sample rate before deciding which path to take
        import librosa
        audio_np, sr = librosa.load(str(audio), sr=None, mono=True)

        if sr != self._TARGET_SR:
            # Route through encode() which pre-resamples correctly
            console.log(
                f"{_LOGGER_PREFIX} __call__: {sr} Hz input detected — "
                f"routing through encode() for pre-resampling "
                f"(avoids torchaudio.sox_effects)"
            )
            waveform = torch.from_numpy(audio_np).float()
            return self.encode(waveform, sr)

        # Already 16 kHz — pass the original file path directly (no temp file)
        audio_path = str(audio)
        result = self._pipeline([audio_path, audio_path], output_emb=True)
        return self._extract_emb_from_result(result)


# ── Registry and factory ──────────────────────────────────────────────────────

_MODEL_REGISTRY: Dict[EmbeddingModelType, type] = {
    EmbeddingModelType.PYANNOTE: PyannoteEmbeddingModel,
    EmbeddingModelType.SPEECHBRAIN_ECAPA: SpeechBrainECAPAEmbeddingModel,
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
        One of "pyannote", "speechbrain_ecapa", "nemo_titanet",
        "modelscope_eres2netv2".
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


def preprocess_audio(audio: Any, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample audio to target sample rate and downmix to mono if needed.

    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        sr: Target sample rate (default: 16000)

    Returns:
        numpy array of preprocessed audio (mono, target sample rate)
    """
    audio_array, actual_sr = load_audio(audio, sr=sr, mono=True)
    return audio_array
