"""
Audio IO

pyannote.audio supports multiple backends for reading:
- torchcodec (primary, if available)
- librosa/soundfile (fallback)
- torchaudio for resampling (always used)

The module automatically selects the best available backend.
"""

import random
import warnings
from io import IOBase
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from pyannote.core import Segment
from torch import Tensor

# Try to import torchcodec first
try:
    import torchcodec
    from torchcodec import AudioSamples
    from torchcodec.decoders import AudioDecoder, AudioStreamMetadata
    
    _USING_TORCHCODEC = True
    _BACKEND_NAME = "torchcodec"
    
except Exception as e:
    _USING_TORCHCODEC = False
    
    # Try librosa/soundfile as fallback
    try:
        import soundfile as sf
        import numpy as np
        from dataclasses import dataclass
        
        _BACKEND_NAME = "librosa"
        
        # Define compatible classes locally
        @dataclass
        class AudioStreamMetadata:
            """Compatible replacement for torchcodec's AudioStreamMetadata"""
            sample_rate: int
            num_channels: int
            duration_seconds_from_header: float
            sample_format: Optional[str] = None
            codec: Optional[str] = None
            bit_rate: Optional[float] = None
            stream_index: int = 0
            
            def __repr__(self):
                fields = [
                    f"  sample_rate: {self.sample_rate}",
                    f"  num_channels: {self.num_channels}",
                    f"  duration_seconds_from_header: {self.duration_seconds_from_header:.3f}",
                ]
                return "AudioStreamMetadata:\n" + "\n".join(fields)

        class AudioSamples:
            """Compatible replacement for torchcodec's AudioSamples"""
            def __init__(self, data: Tensor, sample_rate: int):
                self.data = data
                self.sample_rate = sample_rate

        class AudioDecoder:
            """Compatible replacement for torchcodec's AudioDecoder using soundfile"""
            def __init__(self, source):
                self.source = source
                self._sf_file = None
                self._metadata = None
                
                if isinstance(source, (str, Path)):
                    self._sf_file = sf.SoundFile(str(source))
                elif isinstance(source, IOBase):
                    # Handle IOBase objects
                    import io
                    if hasattr(source, 'seekable') and source.seekable():
                        source.seek(0)
                        data_bytes = source.read()
                        source.seek(0)
                        self._sf_file = sf.SoundFile(io.BytesIO(data_bytes))
                    else:
                        raise ValueError("IOBase object must be seekable")
            
            @property
            def metadata(self) -> AudioStreamMetadata:
                if self._metadata is None:
                    self._metadata = AudioStreamMetadata(
                        sample_rate=self._sf_file.samplerate,
                        num_channels=self._sf_file.channels,
                        duration_seconds_from_header=len(self._sf_file) / self._sf_file.samplerate,
                        sample_format=self._sf_file.subtype
                    )
                return self._metadata
            
            def get_all_samples(self) -> AudioSamples:
                self._sf_file.seek(0)
                data = self._sf_file.read()
                
                if data.ndim == 1:
                    tensor_data = torch.from_numpy(data.copy()).unsqueeze(0).float()
                else:
                    tensor_data = torch.from_numpy(data.copy()).T.float()
                
                return AudioSamples(tensor_data, self._sf_file.samplerate)
            
            def get_samples_played_in_range(self, start_time: float, end_time: float) -> AudioSamples:
                sr = self._sf_file.samplerate
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                self._sf_file.seek(max(0, start_sample))
                num_samples = end_sample - start_sample
                data = self._sf_file.read(num_samples)
                
                if data.ndim == 1:
                    tensor_data = torch.from_numpy(data.copy()).unsqueeze(0).float()
                else:
                    tensor_data = torch.from_numpy(data.copy()).T.float()
                
                return AudioSamples(tensor_data, sr)
            
            def __del__(self):
                if self._sf_file is not None:
                    try:
                        self._sf_file.close()
                    except:
                        pass
        
        warnings.warn(
            f"torchcodec is not installed ({e}). Using soundfile backend instead.\n"
            "For better performance and format support, consider installing torchcodec."
        )
        
    except ImportError as e2:
        warnings.warn(
            "\nNeither torchcodec nor soundfile is available. Audio decoding will fail.\n"
            "Solutions are:\n"
            "* use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary;\n"
            "* install torchcodec: pip install torchcodec\n"
            "* install soundfile: pip install soundfile\n"
            f"torchcodec error: {e}\n"
            f"soundfile error: {e2}"
        )
        
        # Define dummy classes for type hints
        class AudioSamples:
            pass
        
        class AudioDecoder:
            pass
        
        class AudioStreamMetadata:
            pass
        
        _BACKEND_NAME = "none"


AudioFile = str | Path | IOBase | Mapping
AudioFileDocString = """
Audio files can be provided to the Audio class using different types:
    - a "str" or "Path" instance: "audio.wav" or Path("audio.wav")
    - a "IOBase" instance with "read" and "seek" support: open("audio.wav", "rb")
    - a "Mapping" with any of the above as "audio" key: {"audio": ...}
    - a "Mapping" with both "waveform" and "sample_rate" key:
        {"waveform": (channel, time) torch.Tensor, "sample_rate": 44100}
For last two options, an additional "channel" key can be provided as a zero-indexed
integer to load a specific channel: {"audio": "stereo.wav", "channel": 0}
"""


def get_audio_metadata(file: AudioFile) -> "AudioStreamMetadata":
    """Protocol preprocessor used to cache audio metadata
    
    This is useful to speed future random access to this file, e.g.
    in dataloaders using Audio.crop a lot.
    
    Parameters
    ----------
    file : AudioFile
    
    Returns
    -------
    metadata : AudioStreamMetadata
        Audio file metadata
    """
    if _BACKEND_NAME == "none":
        raise RuntimeError("No audio backend available")
    
    # Handle different input formats
    if isinstance(file, (str, Path)):
        # Convert string/path to dict format
        file = {"audio": str(file)}
    elif isinstance(file, IOBase):
        file = {"audio": file}
    
    metadata = AudioDecoder(file["audio"]).metadata
    if isinstance(file["audio"], IOBase):
        file["audio"].seek(0)
    return metadata


class Audio:
    """Audio IO
    
    Parameters
    ----------
    sample_rate : int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : {'random', 'downmix'}, optional
        In case of multi-channel audio, convert to single-channel audio
        using one of the following strategies: select one channel at
        'random' or 'downmix' by averaging all channels.
    
    Usage
    -----
    >>> audio = Audio(sample_rate=16000, mono='downmix')
    >>> waveform, sample_rate = audio({"audio": "/path/to/audio.wav"})
    >>> assert sample_rate == 16000
    >>> sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * sample_rate)
    >>> waveform, sample_rate = audio({"waveform": two_seconds_stereo, "sample_rate": sample_rate})
    >>> assert sample_rate == 16000
    >>> assert waveform.shape[0] == 1
    """
    
    PRECISION = 0.001
    
    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        """Power-normalize waveform
        
        Parameters
        ----------
        waveform : (..., time) Tensor
            Waveform(s)
        
        Returns
        -------
        waveform : (..., time) Tensor
            Power-normalized waveform(s)
        """
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
        return waveform / (rms + 1e-8)
    
    @staticmethod
    def validate_file(file: AudioFile) -> Mapping:
        """Validate file for use with the other Audio methods
        
        Parameter
        ---------
        file : AudioFile
        
        Returns
        -------
        validated_file : Mapping
            {"audio": str, "uri": str, ...}
            {"waveform": tensor, "sample_rate": int, "uri": str, ...}
            {"audio": file, "uri": "stream"} if `file` is an IOBase instance
        
        Raises
        ------
        ValueError if file format is not valid or file does not exist.
        """
        if isinstance(file, Mapping):
            pass
        elif isinstance(file, (str, Path)):
            file = {"audio": str(file), "uri": Path(file).stem}
        elif isinstance(file, IOBase):
            return {"audio": file, "uri": "stream"}
        else:
            raise ValueError(AudioFileDocString)
        
        if "waveform" in file:
            waveform: Tensor = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                raise ValueError(
                    "'waveform' must be provided as a (channel, time) torch Tensor."
                )
            sample_rate: int | None = file.get("sample_rate", None)
            if sample_rate is None:
                raise ValueError(
                    "'waveform' must be provided with their 'sample_rate'."
                )
            file.setdefault("uri", "waveform")
        elif "audio" in file:
            if isinstance(file["audio"], IOBase):
                return file
            path = Path(file["audio"])
            if not path.is_file():
                raise ValueError(f"File {path} does not exist")
            file.setdefault("uri", path.stem)
        else:
            raise ValueError(
                "Neither 'waveform' nor 'audio' is available for this file."
            )
        
        return file
    
    def __init__(self, sample_rate: int = None, mono=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono
        
        if _BACKEND_NAME == "none":
            warnings.warn(
                "No audio backend available. Only in-memory waveforms can be processed."
            )
    
    def downmix_and_resample(
        self, waveform: Tensor, sample_rate: int, channel: int | None = None
    ) -> Tuple[Tensor, int]:
        """Downmix and resample
        
        Parameters
        ----------
        waveform : (channel, time) Tensor
            Waveform.
        sample_rate : int
            Sample rate.
        channel : int, optional
            Channel to use.
        
        Returns
        -------
        waveform : (channel, time) Tensor
            Remixed and resampled waveform
        sample_rate : int
            New sample rate
        """
        if channel is not None:
            waveform = waveform[channel : channel + 1]
        
        num_channels = waveform.shape[0]
        if num_channels > 1:
            if self.mono == "random":
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix":
                waveform = waveform.mean(dim=0, keepdim=True)
        
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate
        
        return waveform, sample_rate
    
    def get_duration(self, file: AudioFile) -> float:
        """Get audio file duration in seconds
        
        Parameters
        ----------
        file : AudioFile
            Audio file.
        
        Returns
        -------
        duration : float
            Duration in seconds.
        """
        file = self.validate_file(file)
        if "waveform" in file:
            frames = len(file["waveform"].T)
            sample_rate = file["sample_rate"]
            return frames / sample_rate
        
        metadata: AudioStreamMetadata = get_audio_metadata(file)
        return metadata.duration_seconds_from_header
    
    def get_num_samples(
        self, duration: float, sample_rate: Optional[int] = None
    ) -> int:
        """Deterministic number of samples from duration and sample rate"""
        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None:
            raise ValueError(
                "`sample_rate` must be provided to compute number of samples."
            )
        return round(duration * sample_rate)
    
    def __call__(self, file: AudioFile) -> Tuple[Tensor, int]:
        """Obtain waveform
        
        Parameters
        ----------
        file : AudioFile
        
        Returns
        -------
        waveform : (channel, time) torch.Tensor
            Waveform
        sample_rate : int
            Sample rate
        
        See also
        --------
        AudioFile
        """
        file = self.validate_file(file)
        channel = file.get("channel", None)
        
        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]
            return self.downmix_and_resample(waveform, sample_rate, channel=channel)
        
        if _BACKEND_NAME == "none":
            raise RuntimeError(
                "Cannot load audio file: no backend available. "
                "Install torchcodec or soundfile."
            )
        
        decoder = AudioDecoder(file["audio"])
        samples: AudioSamples = decoder.get_all_samples()
        waveform = samples.data
        sample_rate = samples.sample_rate
        
        if isinstance(file["audio"], IOBase):
            file["audio"].seek(0)
        
        return self.downmix_and_resample(waveform, sample_rate, channel=channel)
    
    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        mode: str = "raise",
    ) -> Tuple[Tensor, int]:
        """Fast version of self(file).crop(segment, **kwargs)
        
        Parameters
        ----------
        file : AudioFile
            Audio file.
        segment : `pyannote.core.Segment`
            Temporal segment to load.
        mode : {'raise', 'pad'}, optional
            Specifies how out-of-bounds segments will behave.
            * 'raise' -- raise an error (default)
            * 'pad' -- zero pad
        
        Returns
        -------
        waveform : (channel, time) torch.Tensor
            Waveform
        sample_rate : int
            Sample rate
        """
        file = self.validate_file(file)
        channel = file.get("channel", None)
        
        if "waveform" in file:
            waveform = file["waveform"]
            _, num_samples = waveform.shape
            sample_rate = file["sample_rate"]
            duration = num_samples / sample_rate
            
            start_sample: int = self.get_num_samples(segment.start, sample_rate)
            pad_start: int = max(0, -start_sample)
            
            if start_sample < 0:
                if mode == "raise":
                    raise ValueError(
                        f"requested chunk with negative start time (t={segment.start:.3f}s)"
                    )
                else:
                    start_sample = 0
            
            end_sample: int = self.get_num_samples(segment.end, sample_rate)
            pad_end: int = max(end_sample, num_samples) - num_samples
            
            if end_sample >= num_samples:
                if mode == "raise":
                    raise ValueError(
                        f"requested chunk with end time (t={segment.end:.3f}s) greater than "
                        f"{file.get('uri', 'in-memory')} file duration ({duration:.3f}s)."
                    )
                else:
                    end_sample = num_samples
            
            data = waveform[:, start_sample:end_sample]
            data = F.pad(data, (pad_start, pad_end))
            
            return self.downmix_and_resample(data, sample_rate, channel=channel)
        
        if _BACKEND_NAME == "none":
            raise RuntimeError(
                "Cannot load audio file: no backend available. "
                "Install torchcodec or soundfile."
            )
        
        decoder = AudioDecoder(file["audio"])
        metadata: AudioStreamMetadata = decoder.metadata
        sample_rate = metadata.sample_rate
        duration = metadata.duration_seconds_from_header
        num_samples = self.get_num_samples(
            metadata.duration_seconds_from_header, sample_rate
        )
        
        start: float = float(segment.start)
        end: float = float(segment.end)
        pad_start: int = max(0, self.get_num_samples(-start, sample_rate))
        
        if start < 0:
            if mode == "raise":
                raise ValueError(
                    f"requested chunk with negative start time (t={start:.3f}s)"
                )
            else:
                start = 0.0
        
        pad_end: int = (
            max(self.get_num_samples(end, sample_rate), num_samples) - num_samples
        )
        
        if end > duration:
            if mode == "raise":
                raise ValueError(
                    f"requested chunk with end time (t={end:.3f}s) greater than "
                    f"{file.get('uri', 'in-memory')} file duration ({duration:.3f}s)."
                )
            else:
                end = duration
        
        samples: AudioSamples = decoder.get_samples_played_in_range(start, end)
        data = samples.data
        sample_rate = samples.sample_rate
        
        if isinstance(file["audio"], IOBase):
            file["audio"].seek(0)
        
        expected_num_samples = self.get_num_samples(segment.duration, sample_rate)
        _, actual_num_samples = data.shape
        difference = pad_start + actual_num_samples + pad_end - expected_num_samples
        
        if abs(difference) > 1:
            raise ValueError(
                f"requested chunk {segment} from {file.get('uri', 'in-memory')} file resulted in {actual_num_samples} samples "
                f"instead of the expected {expected_num_samples} samples."
            )
        
        if difference == 1:
            data = data[:, :-1]
        elif difference == -1:
            pad_end += 1
        
        data = F.pad(data, (pad_start, pad_end))
        
        return self.downmix_and_resample(data, sample_rate, channel=channel)
