from functools import cached_property
from pathlib import Path
from typing import Optional, Text, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from pyannote.audio.core.inference import BaseInference

try:
    from speechbrain.inference import EncoderClassifier as SpeechBrain_EncoderClassifier

    SPEECHBRAIN_IS_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_IS_AVAILABLE = False

# File: speaker_verification.py (updated functions)

class SpeechBrainPretrainedSpeakerEmbedding(BaseInference):
    """Pretrained SpeechBrain speaker embedding
    
    Parameters
    ----------
    embedding : str
        Name of SpeechBrain model
    device : torch.device, optional
        Device
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.
    
    Usage
    -----
    >>> get_embedding = SpeechBrainPretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size
    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    """
    
    def __init__(
        self,
        embedding: Text = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[torch.device] = None,
        token: Union[Text, None] = None,
        cache_dir: Union[Path, Text, None] = None,
    ):
        if not SPEECHBRAIN_IS_AVAILABLE:
            raise ImportError(
                f"'speechbrain' must be installed to use '{embedding}' embeddings. "
                "Visit https://speechbrain.github.io for installation instructions."
            )
        super().__init__()
        
        if "@" in embedding:
            self.embedding = embedding.split("@")[0]
            self.revision = embedding.split("@")[1]
        else:
            self.embedding = embedding
            self.revision = None
        
        self.device = device or torch.device("cpu")
        self.token = token
        self.cache_dir = cache_dir
        
        # Import LocalStrategy and FetchConfig for proper configuration
        from speechbrain.utils.fetching import LocalStrategy, FetchConfig
        import tempfile
        import os
        
        # Create a proper temporary directory if no cache_dir specified
        if self.cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "speechbrain_embeddings")
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure fetch to use COPY instead of SYMLINK (Windows compatibility)
        fetch_config = FetchConfig(
            token=token,
            revision=self.revision,
            huggingface_cache_dir=self.cache_dir,
        )
        
        # Use COPY strategy to avoid symlink issues on Windows
        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=os.path.join(self.cache_dir, "speechbrain"),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,  # Use COPY instead of SYMLINK
            fetch_config=fetch_config,
        )
    
    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )
        
        # Import LocalStrategy and FetchConfig
        from speechbrain.utils.fetching import LocalStrategy, FetchConfig
        import os
        
        fetch_config = FetchConfig(
            token=self.token,
            revision=self.revision,
            huggingface_cache_dir=self.cache_dir,
        )
        
        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=os.path.join(self.cache_dir, "speechbrain"),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,  # Use COPY instead of SYMLINK
            fetch_config=fetch_config,
        )
        self.device = device
        return self
    
    # Rest of the methods remain unchanged
    @cached_property
    def sample_rate(self) -> int:
        return self.classifier_.audio_normalizer.sample_rate
    
    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = torch.rand(1, 16000).to(self.device)
        *_, dimension = self.classifier_.encode_batch(dummy_waveforms).shape
        return dimension
    
    @cached_property
    def metric(self) -> str:
        return "cosine"
    
    @cached_property
    def min_num_samples(self) -> int:
        with torch.inference_mode():
            lower, upper = 2, round(0.5 * self.sample_rate)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self.classifier_.encode_batch(
                        torch.randn(1, middle).to(self.device)
                    )
                    upper = middle
                except RuntimeError:
                    lower = middle
                middle = (lower + upper) // 2
        return upper
    
    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional
        
        Returns
        -------
        embeddings : (batch_size, dimension)
        """
        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1
        
        waveforms = waveforms.squeeze(dim=1)
        
        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)
        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks
            
            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)
            imasks = imasks > 0.5
            
            signals = pad_sequence(
                [
                    waveform[imask].contiguous()
                    for waveform, imask in zip(waveforms, imasks)
                ],
                batch_first=True,
            )
            wav_lens = imasks.sum(dim=1)
        
        max_len = wav_lens.max()
        if max_len < self.min_num_samples:
            return np.nan * np.zeros((batch_size, self.dimension))
        
        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0
        
        embeddings = (
            self.classifier_.encode_batch(signals, wav_lens=wav_lens)
            .squeeze(dim=1)
            .cpu()
            .numpy()
        )
        
        embeddings[too_short.cpu().numpy()] = np.nan
        
        return embeddings
