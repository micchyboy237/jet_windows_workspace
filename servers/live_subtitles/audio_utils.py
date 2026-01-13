import numpy as np
import numpy.typing as npt

def load_audio_bytes(
    audio_bytes: bytes,
    expected_sample_rate: int,
    channels: int = 1,
    dtype: npt.DTypeLike = np.float32,  # ← modern way, accepts dtype or type
) -> tuple[np.ndarray, int]:
    """
    Load raw PCM bytes from live capture / microphone stream
    
    Args:
        audio_bytes: Raw PCM bytes
        expected_sample_rate: Sample rate of the audio (16000, 44100, etc.)
        channels: Number of channels in the buffer (usually 1 for mono)
        dtype: Data type of samples (np.float32, np.int16, etc.)
    """
    # Get the actual item size (bytes per sample)
    itemsize = np.dtype(dtype).itemsize   # ← this is the key fix
    
    byte_count = len(audio_bytes)
    sample_count = byte_count // (channels * itemsize)
    
    if byte_count % (channels * itemsize) != 0:
        raise ValueError(
            f"Audio bytes length {byte_count} is not divisible by "
            f"(channels={channels} × itemsize={itemsize}) → incomplete frame?"
        )

    array = np.frombuffer(audio_bytes, dtype=dtype)
    
    # Reshape if multi-channel, then downmix to mono
    if channels > 1:
        array = array.reshape(-1, channels).mean(axis=1).astype(np.float32)
    
    return array, expected_sample_rate