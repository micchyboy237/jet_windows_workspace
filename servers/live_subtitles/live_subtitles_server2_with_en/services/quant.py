from typing import Optional, Tuple, Union

import numpy as np


def quantize_audio(
    audio: np.ndarray,
    target_dtype: Union[str, np.dtype] = "int16",  # Sensible default for WAV files
    sr: Optional[int] = 22050,  # Sensible default (librosa's default)
    dither: bool = True,
    dither_amount: float = 0.5,
    normalize: bool = True,
    verbose: bool = True,
    force_mono: bool = True,  # New parameter to handle multi-dimensional audio
) -> Tuple[np.ndarray, dict]:
    """
    Robustly quantize audio to a target dtype with safety checks.

    Parameters
    ----------
    audio : np.ndarray
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
    quantized : np.ndarray
        Quantized audio array.
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

    # --- 1. Validate and reshape inputs ---
    audio = np.asarray(audio, dtype=np.float64)  # Work in double precision internally

    # Handle multi-dimensional audio gracefully
    original_shape = audio.shape
    original_ndim = audio.ndim

    if audio.ndim > 2:
        if verbose:
            print(f"⚠️  Got {audio.ndim}D audio with shape {audio.shape}")

        if force_mono:
            # Strategy 1: Flatten everything to 1D mono
            if verbose:
                print("→ Flattening to 1D mono (averaging all dimensions)")

            # Keep flattening until we get 1D
            while audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            if verbose:
                print(f"→ Resulting shape: {audio.shape}")
        else:
            # Strategy 2: Try to preserve as 2D (channels, samples)
            if audio.ndim == 3:
                # Could be (batch, channels, samples) or (channels, samples, something)
                # Guess: if last dimension is largest, it's likely (..., samples)
                if audio.shape[-1] > audio.shape[-2]:
                    # Likely (batch, channels, samples) → (channels, samples)
                    if verbose:
                        print(
                            "→ Assuming shape (batch, channels, samples), averaging batch dimension"
                        )
                    audio = np.mean(audio, axis=0)
                else:
                    # Unknown layout, flatten to 2D by merging leading dims
                    if verbose:
                        print("→ Reshaping to 2D by merging leading dimensions")
                    audio = audio.reshape(-1, audio.shape[-1])
            else:
                # For >3D, merge all but last dimension
                audio = audio.reshape(-1, audio.shape[-1])
                if verbose:
                    print(f"→ Reshaped to {audio.shape}")

    elif audio.ndim == 1:
        # Add channel dimension for consistent processing
        audio = audio.reshape(1, -1)

    # Now audio is guaranteed to be 2D: (channels, samples)
    n_channels, n_samples = audio.shape

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
        target_dtype = target_dtype.lower()
    else:
        target_dtype = str(np.dtype(target_dtype))

    if target_dtype not in dtype_map:
        raise ValueError(
            f"Unsupported target dtype: {target_dtype}. "
            f"Choose from: {list(dtype_map.keys())}"
        )

    target_dtype_obj, dtype_family = dtype_map[target_dtype]

    # --- 3. Build initial metadata ---
    metadata = {
        "original_dtype": str(audio.dtype),
        "original_shape": original_shape,
        "processed_shape": audio.shape,
        "was_reshaped": original_shape != audio.shape,
        "target_dtype": target_dtype,
        "original_range": (float(np.min(audio)), float(np.max(audio))),
        "was_normalized": False,
        "was_dithered": False,
        "sample_rate": sr if sr else "unknown",
        "n_channels": n_channels,
    }

    # --- 4. Check if quantization is even needed ---
    if np.dtype(target_dtype_obj) == audio.dtype:
        if verbose:
            print(f"✓ Audio already in {target_dtype}, no quantization needed.")
        # Reshape back if originally 1D
        if original_ndim == 1:
            audio = audio.flatten()
        return audio.astype(target_dtype_obj), metadata

    # --- 5. Float-to-Float (just type conversion) ---
    if dtype_family == "float":
        if verbose:
            print(f"✓ Converting float→{target_dtype} (type cast only)")
        quantized = audio.astype(target_dtype_obj)
        metadata["quantized_range"] = (
            float(np.min(quantized)),
            float(np.max(quantized)),
        )
        if original_ndim == 1:
            quantized = quantized.flatten()
        return quantized, metadata

    # --- 6. Float-to-Integer (actual quantization) ---
    int_info = np.iinfo(target_dtype_obj)
    int_min, int_max = int_info.min, int_info.max

    if verbose:
        print(f"Target integer range: {int_min} to {int_max}")

    # Check if audio is already integer
    if np.issubdtype(audio.dtype, np.integer):
        if verbose:
            print("→ Audio is already integer, converting type...")
        quantized = audio.astype(target_dtype_obj)
        metadata["quantized_range"] = (
            float(np.min(quantized)),
            float(np.max(quantized)),
        )
        if original_ndim == 1:
            quantized = quantized.flatten()
        return quantized, metadata

    # --- 7. Normalize if needed ---
    peak = np.max(np.abs(audio))

    if peak == 0:
        if verbose:
            print("⚠️  Audio is silence (all zeros). Returning zeros.")
        quantized = np.zeros_like(audio, dtype=target_dtype_obj)
        if original_ndim == 1:
            quantized = quantized.flatten()
        return quantized, metadata

    if normalize or peak > 1.0:
        if peak > 1.0:
            if verbose:
                print(
                    f"⚠️  Peak amplitude {peak:.2f} exceeds 1.0. Normalizing to prevent clipping."
                )
            audio = audio / peak
            metadata["was_normalized"] = True
        elif normalize and peak < 0.01:
            if verbose:
                print(
                    f"⚠️  Peak amplitude {peak:.6f} is very quiet. Normalizing to full scale."
                )
            audio = audio / peak
            metadata["was_normalized"] = True
        elif normalize:
            if verbose:
                print(f"→ Normalizing to full scale (peak: {peak:.4f})")
            audio = audio / peak
            metadata["was_normalized"] = True

    # Clip to [-1, 1] safety
    if np.max(np.abs(audio)) > 1.0:
        if verbose:
            print("→ Clipping audio to [-1, 1]")
        np.clip(audio, -1.0, 1.0, out=audio)

    # --- 8. Scale to integer range ---
    if target_dtype_obj == np.uint8:
        scaled = (audio * 127.5 + 127.5).astype(np.float64)
    else:
        scaled = audio * int_max

    # --- 9. Apply dithering ---
    if dither:
        dither_noise = (
            (
                np.random.uniform(-1, 1, audio.shape).astype(np.float64)
                + np.random.uniform(-1, 1, audio.shape).astype(np.float64)
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
            f"✓ Quantized to {target_dtype} | "
            f"Shape: {quantized.shape} | "
            f"Range: [{metadata['quantized_range'][0]}, {metadata['quantized_range'][1]}]"
        )
        if metadata["was_normalized"]:
            print("  (Audio was normalized during quantization)")
        if metadata["was_dithered"]:
            print("  (Dithering applied)")
        if metadata["was_reshaped"]:
            print(f"  (Reshaped from {original_shape} to {quantized.shape})")

    return quantized, metadata


def is_quantization_needed(audio: np.ndarray, target_dtype: str = "int16") -> bool:
    """
    Check if quantization is needed.

    Parameters
    ----------
    audio : np.ndarray
        Input audio.
    target_dtype : str
        Desired dtype. Default 'int16'.

    Returns
    -------
    bool
        True if quantization is required.
    """
    return np.dtype(audio.dtype) != np.dtype(target_dtype)


def quantize_safe(
    audio: np.ndarray, target_dtype: str = "int16", sr: Optional[int] = 22050, **kwargs
) -> np.ndarray:
    """
    Simplified version that only returns the quantized array.

    Parameters
    ----------
    audio : np.ndarray
        Input audio array.
    target_dtype : str
        Target dtype. Default 'int16'.
    sr : int, optional
        Sample rate. Default 22050.
    **kwargs
        Additional arguments passed to quantize_audio.

    Returns
    -------
    np.ndarray
        Quantized audio array.
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
