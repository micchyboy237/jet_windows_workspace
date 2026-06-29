from typing import Optional, Union
import numpy as np
try:
    from services.audio_utils import AudioInput, load_audio, SAMPLE_RATE
except ImportError:
    from audio_utils import AudioInput, load_audio, SAMPLE_RATE
import logging

logger = logging.getLogger(__name__)

# Optional torch support
try:
    import torch
    TORCH_AVAILABLE = True
    logger.debug("torch is available for tensor conversion")
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    logger.debug("torch is not available; tensor conversion disabled")


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


def _to_tensor(
    array: np.ndarray,
    device: Optional[Union[str, "torch.device"]] = None
) -> "torch.Tensor":
    """
    Convert numpy array to torch tensor with optional device placement.
    
    Parameters
    ----------
    array : np.ndarray
        Input numpy array
    device : str or torch.device, optional
        Target device for the tensor (e.g., 'cpu', 'cuda', 'mps')
        If None, defaults to 'cpu'
    
    Returns
    -------
    torch.Tensor
        Tensor version of the input array
    
    Raises
    ------
    ImportError
        If torch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for tensor conversion. "
            "Install it with: pip install torch"
        )
    
    if device is None:
        device = "cpu"
    
    logger.debug(f"_to_tensor: converting numpy array (shape={array.shape}) to tensor on device={device}")
    tensor = torch.from_numpy(array.copy()).to(device)
    logger.debug(f"_to_tensor: tensor created with shape {tensor.shape}, device={tensor.device}")
    return tensor


def convert_audio_dtype(
    audio: AudioInput,
    target_dtype: Union[np.dtype, str],
    scale: Optional[bool] = True,
    preserve_range: bool = False,
    return_tensor: bool = False,
    device: Optional[Union[str, "torch.device"]] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
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
    return_tensor : bool, default=False
        If True, return as torch.Tensor instead of numpy array
    device : str or torch.device, optional
        Target device for tensor output (only used when return_tensor=True)
        Examples: 'cpu', 'cuda', 'cuda:0', 'mps'
        If None, defaults to 'cpu'
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Converted audio array (as tensor if return_tensor=True)
    
    Raises
    ------
    ImportError
        If return_tensor=True but torch is not installed
    
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
    >>> # Convert to float32 and return as CUDA tensor
    >>> audio_tensor = convert_audio_dtype("audio.wav", np.float32, return_tensor=True, device='cuda')
    """
    logger.info(
        f"convert_audio_dtype: converting to {target_dtype} "
        f"(scale={scale}, preserve_range={preserve_range}, "
        f"return_tensor={return_tensor}, device={device})"
    )
    
    audio_array = _ensure_numpy(audio)
    logger.debug(f"convert_audio_dtype: input shape {audio_array.shape}, dtype {audio_array.dtype}")
    
    target_dtype = np.dtype(target_dtype)
    
    if audio_array.dtype == target_dtype:
        logger.debug("convert_audio_dtype: source and target dtypes match, returning copy")
        result = audio_array.copy()
    else:
        src_info = np.iinfo(audio_array.dtype) if np.issubdtype(audio_array.dtype, np.integer) else None
        tgt_info = (
            np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None
        )
        
        is_int_input = np.issubdtype(audio_array.dtype, np.integer)
        is_int_output = np.issubdtype(target_dtype, np.integer)
        
        if not is_int_input and not is_int_output:
            # Float → Float
            logger.debug("convert_audio_dtype: float to float conversion")
            result = audio_array.astype(target_dtype)
            
        elif is_int_input and is_int_output:
            # Int → Int
            if scale:
                if src_info and tgt_info:
                    if src_info.bits > tgt_info.bits:
                        scale_factor = tgt_info.max / src_info.max
                        logger.debug(
                            f"convert_audio_dtype: int{src_info.bits} to int{tgt_info.bits} "
                            f"(downscale: {scale_factor})"
                        )
                        result = (audio_array.astype(np.float64) * scale_factor).astype(target_dtype)
                    elif src_info.bits < tgt_info.bits:
                        scale_factor = tgt_info.max / src_info.max
                        logger.debug(
                            f"convert_audio_dtype: int{src_info.bits} to int{tgt_info.bits} "
                            f"(upscale: {scale_factor})"
                        )
                        result = (audio_array.astype(np.float64) * scale_factor).astype(target_dtype)
                    else:
                        logger.debug(
                            f"convert_audio_dtype: int{src_info.bits} to int{tgt_info.bits} "
                            f"(same bits)"
                        )
                        result = audio_array.astype(target_dtype)
            else:
                logger.debug("convert_audio_dtype: int to int without scaling")
                result = audio_array.astype(target_dtype)
                
        elif not is_int_input and is_int_output:
            # Float → Int
            logger.debug(f"convert_audio_dtype: float to int{target_dtype.itemsize * 8} conversion")
            if scale and not preserve_range:
                max_val = np.max(np.abs(audio_array))
                if max_val > 1.0:
                    audio_scaled = audio_array / max_val
                else:
                    audio_scaled = audio_array
                result = (audio_scaled * tgt_info.max).astype(target_dtype)
            else:
                result = np.clip(
                    audio_array * tgt_info.max, tgt_info.min, tgt_info.max
                ).astype(target_dtype)
                
        elif is_int_input and not is_int_output:
            # Int → Float
            logger.debug("convert_audio_dtype: int to float conversion")
            if scale:
                if src_info:
                    result = (audio_array.astype(np.float64) / src_info.max).astype(target_dtype)
            else:
                result = audio_array.astype(target_dtype)
        else:
            result = audio_array.astype(target_dtype)
    
    # Return as tensor if requested
    if return_tensor:
        logger.info("convert_audio_dtype: returning result as torch.Tensor")
        return _to_tensor(result, device=device)
    
    logger.debug(f"convert_audio_dtype: returning numpy array with shape {result.shape}")
    return result


def to_int16(
    audio: AudioInput,
    normalize: bool = True,
    return_tensor: bool = False,
    device: Optional[Union[str, "torch.device"]] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convert audio to int16 safely.
    Accepts file paths, bytes, numpy arrays, or torch tensors.
    
    Parameters
    ----------
    audio : AudioInput
        Input audio (file path, bytes, numpy array, or torch tensor)
    normalize : bool, default=True
        If True, normalize audio to full int16 range
        If False, preserve current range
    return_tensor : bool, default=False
        If True, return as torch.Tensor instead of numpy array
    device : str or torch.device, optional
        Target device for tensor output (only used when return_tensor=True)
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Audio as int16 array or tensor
    """
    logger.info(f"to_int16: converting audio to int16 (normalize={normalize}, return_tensor={return_tensor})")
    return convert_audio_dtype(
        audio,
        np.int16,
        scale=True,
        preserve_range=not normalize,
        return_tensor=return_tensor,
        device=device,
    )


def to_float32(
    audio: AudioInput,
    normalize: bool = True,
    return_tensor: bool = False,
    device: Optional[Union[str, "torch.device"]] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convert audio to float32 safely.
    Accepts file paths, bytes, numpy arrays, or torch tensors.
    
    Parameters
    ----------
    audio : AudioInput
        Input audio (file path, bytes, numpy array, or torch tensor)
    normalize : bool, default=True
        If True, normalize audio to [-1, 1] range
        If False, preserve current range
    return_tensor : bool, default=False
        If True, return as torch.Tensor instead of numpy array
    device : str or torch.device, optional
        Target device for tensor output (only used when return_tensor=True)
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Audio as float32 array or tensor
    """
    logger.info(f"to_float32: converting audio to float32 (normalize={normalize}, return_tensor={return_tensor})")
    return convert_audio_dtype(
        audio,
        np.float32,
        scale=True,
        preserve_range=not normalize,
        return_tensor=return_tensor,
        device=device,
    )


def to_int32(
    audio: AudioInput,
    normalize: bool = True,
    return_tensor: bool = False,
    device: Optional[Union[str, "torch.device"]] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convert audio to int32 safely.
    Accepts file paths, bytes, numpy arrays, or torch tensors.
    
    Parameters
    ----------
    audio : AudioInput
        Input audio (file path, bytes, numpy array, or torch tensor)
    normalize : bool, default=True
        If True, normalize audio to full int32 range
        If False, preserve current range
    return_tensor : bool, default=False
        If True, return as torch.Tensor instead of numpy array
    device : str or torch.device, optional
        Target device for tensor output (only used when return_tensor=True)
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Audio as int32 array or tensor
    """
    logger.info(f"to_int32: converting audio to int32 (normalize={normalize}, return_tensor={return_tensor})")
    return convert_audio_dtype(
        audio,
        np.int32,
        scale=True,
        preserve_range=not normalize,
        return_tensor=return_tensor,
        device=device,
    )


if __name__ == "__main__":
    print("Testing audio dtype conversion:")
    
    # Test 1: Basic numpy conversions (original tests)
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
    
    # Test 2: Tensor conversion tests
    print("\n--- Testing tensor conversion ---")
    if TORCH_AVAILABLE:
        # Test convert_audio_dtype with return_tensor=True
        audio_float = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype="float32")
        audio_tensor = convert_audio_dtype(audio_float, np.float32, return_tensor=True)
        print(f"Numpy to float32 tensor: {audio_tensor}, dtype={audio_tensor.dtype}, device={audio_tensor.device}")
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            audio_tensor_cuda = convert_audio_dtype(audio_float, np.float32, return_tensor=True, device='cuda')
            print(f"Numpy to float32 CUDA tensor: {audio_tensor_cuda}, device={audio_tensor_cuda.device}")
        
        # Test convenience functions with tensor output
        tensor_int16 = to_int16(audio_float, return_tensor=True)
        print(f"to_int16 as tensor: {tensor_int16}, dtype={tensor_int16.dtype}")
        
        tensor_float32 = to_float32(audio_int16_orig, return_tensor=True)
        print(f"to_float32 as tensor: {tensor_float32}, dtype={tensor_float32.dtype}")
        
        tensor_int32 = to_int32(audio_float, return_tensor=True)
        print(f"to_int32 as tensor: {tensor_int32}, dtype={tensor_int32.dtype}")
        
        # Test backward compatibility (return_tensor=False by default)
        result_numpy = to_float32(audio_float)
        print(f"to_float32 default (numpy): type={type(result_numpy).__name__}, shape={result_numpy.shape}")
    else:
        print("torch is not available. Skipping tensor tests.")
        
        # Test that it raises properly
        try:
            convert_audio_dtype(audio_float, np.float32, return_tensor=True)
        except ImportError as e:
            print(f"Correctly raised ImportError: {e}")
    
    # Test 3: AudioInput type tests
    print("\n--- Testing with AudioInput types ---")
    audio_bytes = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    try:
        result = to_float32(audio_bytes)
        print(f"Bytes to float32: {result}")
    except Exception as e:
        print(f"Bytes test skipped: {e}")
    
    import os
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        result = to_int16(test_file)
        print(f"File to int16: shape={result.shape}")
    else:
        print("No test audio file found, skipping file path test")
    
    print("\nAll tests completed!")
