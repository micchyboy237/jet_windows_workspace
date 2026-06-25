from typing import Optional, Tuple, Union

import numpy as np
import torch


def quantize_audio(
    audio: Union[np.ndarray, torch.Tensor],
    target_dtype: Union[str, np.dtype] = "int16",  # Sensible default for WAV files
    sr: Optional[int] = 22050,  # Sensible default (librosa's default)
    dither: bool = True,
    dither_amount: float = 0.5,
    normalize: bool = True,
    verbose: bool = True,
    force_mono: bool = True,  # New parameter to handle multi-dimensional audio
) -> Tuple[Union[np.ndarray, torch.Tensor], dict]:
    """
    Robustly quantize audio to a target dtype with safety checks.

    Parameters
    ----------
    audio : np.ndarray or torch.Tensor
        Input audio array (any dtype, any shape).
    target_dtype : str or np.dtype
        Target dtype. Default 'int16' (standard for 16-bit WAV).
        Options: 'float32', 'float16', 'int16', 'int32', 'int8', 'uint8'.
    sr : int, optional
        Sample rate. Default 22050 (librosa's default).
        Set to None if sample rate is unknown.
    dither : bool
        Apply triangular dithering when quantizing float→int (reduces distortion).
    dither_amount : float
        Amount of dither in LSBs (0.5–2.0 typical, 0.5 is standard TPDF).
    normalize : bool
        If True, normalize float audio to full integer range before quantizing.
        If False, assumes float audio is already in [-1, 1].
    verbose : bool
        Print quantization details.
    force_mono : bool
        If True and audio is multi-channel with >2 dimensions, automatically
        convert to mono by averaging channels. If False, attempt to preserve shape.

    Returns
    -------
    quantized : np.ndarray or torch.Tensor
        Quantized audio array (same type as input).
    metadata : dict
        Information about the quantization process.

    Examples
    --------
    >>> audio, sr = librosa.load('file.wav', sr=22050)
    >>> quantized, meta = quantize_audio(audio)  # Uses defaults: int16, 22050
    >>>
    >>> # Multi-dimensional audio auto-converted to mono
    >>> multi_chan = np.random.randn(2, 4, 44100)
    >>> quantized, meta = quantize_audio(multi_chan)
    """
    # Track input type for return conversion
    is_torch = isinstance(audio, torch.Tensor)

    # Convert to numpy for processing
    if is_torch:
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio

    # --- 1. Validate and reshape inputs ---
    audio_np = np.asarray(
        audio_np, dtype=np.float64
    )  # Work in double precision internally

    # Handle multi-dimensional audio gracefully
    original_shape = audio_np.shape
    original_ndim = audio_np.ndim

    if audio_np.ndim > 2:
        if verbose:
            print(f"⚠️  Got {audio_np.ndim}D audio with shape {audio_np.shape}")

        if force_mono:
            # Strategy 1: Flatten everything to 1D mono
            if verbose:
                print("→ Flattening to 1D mono (averaging all dimensions)")

            # Keep flattening until we get 1D
            while audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=0)

            if verbose:
                print(f"→ Resulting shape: {audio_np.shape}")
        else:
            # Strategy 2: Try to preserve as 2D (channels, samples)
            if audio_np.ndim == 3:
                # Could be (batch, channels, samples) or (channels, samples, something)
                # Guess: if last dimension is largest, it's likely (..., samples)
                if audio_np.shape[-1] > audio_np.shape[-2]:
                    # Likely (batch, channels, samples) → (channels, samples)
                    if verbose:
                        print(
                            "→ Assuming shape (batch, channels, samples), averaging batch dimension"
                        )
                    audio_np = np.mean(audio_np, axis=0)
                else:
                    # Unknown layout, flatten to 2D by merging leading dims
                    if verbose:
                        print("→ Reshaping to 2D by merging leading dimensions")
                    audio_np = audio_np.reshape(-1, audio_np.shape[-1])
            else:
                # For >3D, merge all but last dimension
                audio_np = audio_np.reshape(-1, audio_np.shape[-1])
                if verbose:
                    print(f"→ Reshaped to {audio_np.shape}")

    elif audio_np.ndim == 1:
        # Add channel dimension for consistent processing
        audio_np = audio_np.reshape(1, -1)

    # Now audio is guaranteed to be 2D: (channels, samples)
    n_channels, n_samples = audio_np.shape

    # --- 2. Validate dtype ---
    dtype_map = {
        "float32": (np.float32, "float"),
        "float64": (np.float64, "float"),
        "float16": (np.float16, "float"),
        "int16": (np.int16, "int"),
        "int32": (np.int32, "int"),
        "int8": (np.int8, "int"),
        "uint8": (np.uint8, "int"),
    }

    if isinstance(target_dtype, str):
        target_dtype_str = target_dtype.lower()
    else:
        target_dtype_str = str(np.dtype(target_dtype))

    if target_dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported target dtype: {target_dtype_str}. "
            f"Choose from: {list(dtype_map.keys())}"
        )

    target_dtype_obj, dtype_family = dtype_map[target_dtype_str]

    # --- 3. Build initial metadata ---
    metadata = {
        "original_dtype": str(audio_np.dtype),
        "original_shape": original_shape,
        "processed_shape": audio_np.shape,
        "was_reshaped": original_shape != audio_np.shape,
        "target_dtype": target_dtype_str,
        "original_range": (float(np.min(audio_np)), float(np.max(audio_np))),
        "was_normalized": False,
        "was_dithered": False,
        "sample_rate": sr if sr else "unknown",
        "n_channels": n_channels,
    }

    # --- 4. Check if quantization is even needed ---
    if np.dtype(target_dtype_obj) == audio_np.dtype:
        if verbose:
            print(f"✓ Audio already in {target_dtype_str}, no quantization needed.")
        # Reshape back if originally 1D
        if original_ndim == 1:
            audio_np = audio_np.flatten()
        result = audio_np.astype(target_dtype_obj)
        if is_torch:
            return torch.from_numpy(result), metadata
        return result, metadata

    # --- 5. Float-to-Float (just type conversion) ---
    if dtype_family == "float":
        if verbose:
            print(f"✓ Converting float→{target_dtype_str} (type cast only)")
        quantized = audio_np.astype(target_dtype_obj)
        metadata["quantized_range"] = (
            float(np.min(quantized)),
            float(np.max(quantized)),
        )
        if original_ndim == 1:
            quantized = quantized.flatten()
        if is_torch:
            return torch.from_numpy(quantized), metadata
        return quantized, metadata

    # --- 6. Float-to-Integer (actual quantization) ---
    int_info = np.iinfo(target_dtype_obj)
    int_min, int_max = int_info.min, int_info.max

    if verbose:
        print(f"Target integer range: {int_min} to {int_max}")

    # Check if audio is already integer
    if np.issubdtype(audio_np.dtype, np.integer):
        if verbose:
            print("→ Audio is already integer, converting type...")
        quantized = audio_np.astype(target_dtype_obj)
        metadata["quantized_range"] = (
            float(np.min(quantized)),
            float(np.max(quantized)),
        )
        if original_ndim == 1:
            quantized = quantized.flatten()
        if is_torch:
            return torch.from_numpy(quantized), metadata
        return quantized, metadata

    # --- 7. Normalize if needed ---
    peak = np.max(np.abs(audio_np))

    if peak == 0:
        if verbose:
            print("⚠️  Audio is silence (all zeros). Returning zeros.")
        quantized = np.zeros_like(audio_np, dtype=target_dtype_obj)
        if original_ndim == 1:
            quantized = quantized.flatten()
        if is_torch:
            return torch.from_numpy(quantized), metadata
        return quantized, metadata

    if normalize or peak > 1.0:
        if peak > 1.0:
            if verbose:
                print(
                    f"⚠️  Peak amplitude {peak:.2f} exceeds 1.0. Normalizing to prevent clipping."
                )
            audio_np = audio_np / peak
            metadata["was_normalized"] = True
        elif normalize and peak < 0.01:
            if verbose:
                print(
                    f"⚠️  Peak amplitude {peak:.6f} is very quiet. Normalizing to full scale."
                )
            audio_np = audio_np / peak
            metadata["was_normalized"] = True
        elif normalize:
            if verbose:
                print(f"→ Normalizing to full scale (peak: {peak:.4f})")
            audio_np = audio_np / peak
            metadata["was_normalized"] = True

    # Clip to [-1, 1] safety
    if np.max(np.abs(audio_np)) > 1.0:
        if verbose:
            print("→ Clipping audio to [-1, 1]")
        np.clip(audio_np, -1.0, 1.0, out=audio_np)

    # --- 8. Scale to integer range ---
    if target_dtype_obj == np.uint8:
        scaled = (audio_np * 127.5 + 127.5).astype(np.float64)
    else:
        scaled = audio_np * int_max

    # --- 9. Apply dithering ---
    if dither:
        dither_noise = (
            (
                np.random.uniform(-1, 1, audio_np.shape).astype(np.float64)
                + np.random.uniform(-1, 1, audio_np.shape).astype(np.float64)
            )
            * dither_amount
            * 0.5
        )

        scaled += dither_noise
        metadata["was_dithered"] = True
        if verbose:
            print(f"→ Applied triangular dither ({dither_amount} LSB)")

    # --- 10. Round and convert ---
    quantized = np.rint(scaled).astype(target_dtype_obj)
    np.clip(quantized, int_min, int_max, out=quantized)

    # --- 11. Restore original shape if needed ---
    if original_ndim == 1:
        quantized = quantized.flatten()

    # --- 12. Final metadata ---
    metadata["quantized_range"] = (float(np.min(quantized)), float(np.max(quantized)))
    metadata["quantized_dtype"] = str(quantized.dtype)
    metadata["peak_db"] = float(20 * np.log10(peak)) if peak > 0 else -np.inf
    metadata["final_shape"] = quantized.shape

    if verbose:
        print(
            f"✓ Quantized to {target_dtype_str} | "
            f"Shape: {quantized.shape} | "
            f"Range: [{metadata['quantized_range'][0]}, {metadata['quantized_range'][1]}]"
        )
        if metadata["was_normalized"]:
            print("  (Audio was normalized during quantization)")
        if metadata["was_dithered"]:
            print("  (Dithering applied)")
        if metadata["was_reshaped"]:
            print(f"  (Reshaped from {original_shape} to {quantized.shape})")

    # Return as torch tensor if input was torch
    if is_torch:
        return torch.from_numpy(quantized), metadata
    return quantized, metadata


def is_quantization_needed(
    audio: Union[np.ndarray, torch.Tensor], target_dtype: str = "int16"
) -> bool:
    """
    Check if quantization is needed.

    Parameters
    ----------
    audio : np.ndarray or torch.Tensor
        Input audio.
    target_dtype : str
        Desired dtype. Default 'int16'.

    Returns
    -------
    bool
        True if quantization is required.
    """
    # Get dtype from tensor or array
    if isinstance(audio, torch.Tensor):
        current_dtype = str(audio.dtype).replace("torch.", "")
    else:
        current_dtype = str(audio.dtype)

    return np.dtype(current_dtype) != np.dtype(target_dtype)


def quantize_safe(
    audio: Union[np.ndarray, torch.Tensor],
    target_dtype: str = "int16",
    sr: Optional[int] = 22050,
    **kwargs,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Simplified version that only returns the quantized array.

    Parameters
    ----------
    audio : np.ndarray or torch.Tensor
        Input audio array.
    target_dtype : str
        Target dtype. Default 'int16'.
    sr : int, optional
        Sample rate. Default 22050.
    **kwargs
        Additional arguments passed to quantize_audio.

    Returns
    -------
    np.ndarray or torch.Tensor
        Quantized audio array (same type as input).
    """
    quantized, _ = quantize_audio(audio, target_dtype, sr=sr, **kwargs)
    return quantized


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path

    import librosa
    import numpy as np
    import soundfile as sf
    from custom_logging import linkify
    from rich.console import Console

    console = Console()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Quantize and save audio.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    args = parser.parse_args()

    # Simplest usage - just use defaults
    audio, sr = librosa.load(args.audio_path, sr=None)
    quantized, meta = quantize_audio(audio, sr=sr)

    output_path_wav = OUTPUT_DIR / "quantized.wav"
    output_path_json = OUTPUT_DIR / "quantized_meta.json"

    sf.write(output_path_wav, quantized, sr)
    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Use rich console.print for logs
    console.print(
        f"[bold green]Quantized audio saved to:[/bold green] {linkify(output_path_wav)}"
    )
    console.print(
        f"[bold blue]Metadata saved to:[/bold blue] {linkify(output_path_json)}"
    )

    # Example 3: Multi-dimensional audio is auto-flattened
    # multi_dim = np.random.randn(2, 4, 44100)  # 3D: (batch, channels, samples)
    # quantized, meta = quantize_audio(multi_dim)
    # print(meta["was_reshaped"])  # True
    # print(meta["final_shape"])  # (44100,) - flattened to mono 1D

    # # Example 4: Keep multi-dimensional as 2D
    # quantized, meta = quantize_audio(multi_dim, force_mono=False)
    # print(meta["final_shape"])  # (4, 44100) - preserved as (channels, samples)

    # # Example 5: Integer input passes through
    # int_audio = np.array([0, 100, 200, -100], dtype=np.int32)
    # quantized, meta = quantize_audio(int_audio, "int16")
    # print(meta["original_range"])  # (-100, 200)

    # # Example 6: Check if quantization needed
    # if is_quantization_needed(audio):  # Checks against default 'int16'
    #     quantized, meta = quantize_audio(audio)
    # else:
    #     print("Already int16!")

    # # Example 7: Unknown sample rate
    # quantized, meta = quantize_audio(audio, sr=None)
    # print(meta["sample_rate"])  # 'unknown'
