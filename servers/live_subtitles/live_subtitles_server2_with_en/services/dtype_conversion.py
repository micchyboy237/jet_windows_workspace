from typing import Optional, Union
import numpy as np
try:
    from services.audio_utils import AudioInput, load_audio, SAMPLE_RATE
except ImportError:
    from audio_utils import AudioInput, load_audio, SAMPLE_RATE
import logging

logger = logging.getLogger(__name__)


def _ensure_numpy(audio: AudioInput) -> np.ndarray:
    """
    Convert any AudioInput to a numpy array for dtype conversion.
    
    Parameters
    ----------
    audio : AudioInput
        Flexible audio input (file path, bytes, numpy array, or torch tensor)
    
    Returns
    -------
    np.ndarray
        Audio as numpy float32 array
    """
    if isinstance(audio, np.ndarray):
        logger.debug(f"_ensure_numpy: received numpy array with shape {audio.shape}")
        return audio
    else:
        logger.debug(f"_ensure_numpy: loading audio from {type(audio).__name__}")
        array, _ = load_audio(audio)
        return array


def convert_audio_dtype(
    audio: AudioInput,
    target_dtype: Union[np.dtype, str],
    scale: Optional[bool] = True,
    preserve_range: bool = False,
) -> np.ndarray:
    """
    Safely convert audio array between different dtypes without corruption.
    Accepts flexible audio inputs (file paths, bytes, numpy arrays, torch tensors).
    
    Parameters
    ----------
    audio : AudioInput
        Input audio (file path, bytes, numpy array, or torch tensor)
    target_dtype : np.dtype or str
        Target dtype (e.g., np.int16, 'float32', np.int32)
    scale : bool, default=True
        If True, scale to target dtype's full range
        If False, assume values are already in target range
    preserve_range : bool, default=False
        If True, don't normalize float audio (assume already in [-1, 1])
    
    Returns
    -------
    np.ndarray
        Converted audio array
    
    Examples
    --------
    >>> # Float [-1, 1] to int16
    >>> audio_int16 = convert_audio_dtype(audio_float, np.int16)
    >>> # int16 to float32 (preserving range)
    >>> audio_float = convert_audio_dtype(audio_int16, np.float32)
    >>> # int16 to int32 without scaling
    >>> audio_int32 = convert_audio_dtype(audio_int16, np.int32, scale=False)
    >>> # From file path to int16
    >>> audio_int16 = convert_audio_dtype("audio.wav", np.int16)
    """
    logger.info(
        f"convert_audio_dtype: converting to {target_dtype} "
        f"(scale={scale}, preserve_range={preserve_range})"
    )
    
    # Convert flexible audio input to numpy array
    audio_array = _ensure_numpy(audio)
    logger.debug(f"convert_audio_dtype: input shape {audio_array.shape}, dtype {audio_array.dtype}")
    
    target_dtype = np.dtype(target_dtype)
    
    if audio_array.dtype == target_dtype:
        logger.debug("convert_audio_dtype: source and target dtypes match, returning copy")
        return audio_array.copy()
    
    src_info = np.iinfo(audio_array.dtype) if np.issubdtype(audio_array.dtype, np.integer) else None
    tgt_info = (
        np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None
    )
    
    is_int_input = np.issubdtype(audio_array.dtype, np.integer)
    is_int_output = np.issubdtype(target_dtype, np.integer)
    
    if not is_int_input and not is_int_output:
        logger.debug("convert_audio_dtype: float to float conversion")
        return audio_array.astype(target_dtype)
    
    if is_int_input and is_int_output:
        if scale:
            if src_info and tgt_info:
                if src_info.bits > tgt_info.bits:
                    scale_factor = tgt_info.max / src_info.max
                    logger.debug(f"convert_audio_dtype: int{src_info.bits} to int{tgt_info.bits} (downscale: {scale_factor})")
                    return (audio_array.astype(np.float64) * scale_factor).astype(
                        target_dtype
                    )
                elif src_info.bits < tgt_info.bits:
                    scale_factor = tgt_info.max / src_info.max
                    logger.debug(f"convert_audio_dtype: int{src_info.bits} to int{tgt_info.bits} (upscale: {scale_factor})")
                    return (audio_array.astype(np.float64) * scale_factor).astype(
                        target_dtype
                    )
                else:
                    logger.debug(f"convert_audio_dtype: int{src_info.bits} to int{tgt_info.bits} (same bits)")
                    return audio_array.astype(target_dtype)
        else:
            logger.debug("convert_audio_dtype: int to int without scaling")
            return audio_array.astype(target_dtype)
    
    if not is_int_input and is_int_output:
        logger.debug(f"convert_audio_dtype: float to int{target_dtype.itemsize * 8} conversion")
        if scale and not preserve_range:
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_scaled = audio_array / max_val
            else:
                audio_scaled = audio_array
            return (audio_scaled * tgt_info.max).astype(target_dtype)
        else:
            return np.clip(audio_array * tgt_info.max, tgt_info.min, tgt_info.max).astype(
                target_dtype
            )
    
    if is_int_input and not is_int_output:
        logger.debug(f"convert_audio_dtype: int to float conversion")
        if scale:
            if src_info:
                return (audio_array.astype(np.float64) / src_info.max).astype(target_dtype)
        else:
            return audio_array.astype(target_dtype)
    
    return audio_array.astype(target_dtype)


def to_int16(audio: AudioInput, normalize: bool = True) -> np.ndarray:
    """Convert audio to int16 safely. Accepts file paths, bytes, numpy arrays, or torch tensors."""
    logger.info(f"to_int16: converting audio to int16 (normalize={normalize})")
    return convert_audio_dtype(
        audio, np.int16, scale=True, preserve_range=not normalize
    )


def to_float32(audio: AudioInput, normalize: bool = True) -> np.ndarray:
    """Convert audio to float32 safely. Accepts file paths, bytes, numpy arrays, or torch tensors."""
    logger.info(f"to_float32: converting audio to float32 (normalize={normalize})")
    return convert_audio_dtype(
        audio, np.float32, scale=True, preserve_range=not normalize
    )


def to_int32(audio: AudioInput, normalize: bool = True) -> np.ndarray:
    """Convert audio to int32 safely. Accepts file paths, bytes, numpy arrays, or torch tensors."""
    logger.info(f"to_int32: converting audio to int32 (normalize={normalize})")
    return convert_audio_dtype(
        audio, np.int32, scale=True, preserve_range=not normalize
    )


if __name__ == "__main__":
    print("Testing audio dtype conversion:")
    
    # Test with numpy array (original behavior)
    audio_float = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype="float32")
    audio_int16 = convert_audio_dtype(audio_float, "int16")
    print(f"Float to int16: {audio_int16}")
    
    audio_int16_orig = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    audio_int16_new = convert_audio_dtype(audio_int16_orig, np.int16)
    print(f"int16 to int16 (same): {audio_int16_new}")
    
    audio_float2 = convert_audio_dtype(audio_int16_orig, np.float32)
    print(f"int16 to float32: {audio_float2}")
    
    audio_int32 = convert_audio_dtype(audio_int16_orig, np.int32, scale=True)
    print(f"int16 to int32 (scaled): {audio_int32}")
    
    audio_int32_preserve = convert_audio_dtype(audio_int16_orig, np.int32, scale=False)
    print(f"int16 to int32 (preserve): {audio_int32_preserve}")
    
    audio_float_norm = np.array([-0.5, 0.5], dtype=np.float32)
    audio_int16_norm = convert_audio_dtype(
        audio_float_norm, np.int16, preserve_range=True
    )
    print(f"Float [-0.5, 0.5] to int16 (preserve range): {audio_int16_norm}")
    
    # Test with AudioInput types
    print("\nTesting with AudioInput types:")
    
    # Test with bytes
    audio_bytes = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    try:
        result = to_float32(audio_bytes)
        print(f"Bytes to float32: {result}")
    except Exception as e:
        print(f"Bytes test skipped: {e}")
    
    # Test with file path (if exists)
    import os
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        result = to_int16(test_file)
        print(f"File to int16: shape={result.shape}")
    else:
        print("No test audio file found, skipping file path test")
    
    print("\nAll tests completed!")
