# preprocessors.py

"""
Standalone loudness normalization utility.

Provides a single, reusable function to normalize audio to a target LUFS value
using pyloudnorm (ITU-R BS.1770-4 compliant). Handles caching of meters for
performance, clipping prevention, and graceful fallback for silent/failed cases.
"""

from __future__ import annotations

import logging
from typing import Union
from typing import Optional, Literal
import os

import numpy as np
import numpy.typing as npt
import torch
import soundfile as sf
from io import BytesIO

import pyloudnorm as pyln

logger = logging.getLogger(__name__)

# Cache meters per sample rate to avoid expensive recreation
_METER_CACHE: dict[int, pyln.Meter] = {}

# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]

VALID_DTYPE_STRINGS = Literal["float32", "float64", "int16", "int32"]

def normalize_loudness(
    audio: AudioInput,
    sample_rate: int,
    target_lufs: float = -14.0,
    min_lufs_threshold: float = -70.0,
    headroom_factor: float = 1.05,
    mode: Optional[Literal["general", "speech"]] = None,
    max_loudness_threshold: Optional[float] = None,
    return_dtype: Optional[Union[VALID_DTYPE_STRINGS, np.dtype]] = None,
) -> np.ndarray:
    """
    Normalize audio to a target integrated loudness (LUFS).

    Args:
        audio: Input audio – can be:
               - File path (str or os.PathLike)
               - Raw audio bytes
               - NumPy array (floating or integer dtype)
               - torch.Tensor
               When path/bytes provided, sample_rate is inferred from file.
        sample_rate: Sample rate of the audio in Hz.
        target_lufs: Desired integrated loudness in LUFS (default -14.0).
        min_lufs_threshold: If measured loudness is below this, skip normalization
                            to avoid amplifying pure noise/silence.
        headroom_factor: Multiplier applied after normalization to prevent clipping
                         (default 1.05 → ~0.87 peak).
        mode: Optional preset mode.
              - "speech": Optimized for spoken word – louder target (-13.0 LUFS)
                and minimal headroom (1.0) for maximum clarity.
              - "general" or None: Standard music/streaming settings.
        max_loudness_threshold: Optional upper bound (in LUFS).
            If provided and measured loudness exceeds this value,
            normalization will not amplify (only attenuate if needed).
            Useful to prevent boosting already-loud speech content.
        return_dtype: Desired dtype of returned array.
            - None: preserve input dtype if ndarray, else float32
            - "float32", "float64", np.float32, np.float64
            - "int16", "int32", np.int16, np.int32 → scales [-1,1] → integer range

    Returns:
        Normalized audio array with the same shape as input.

    Raises:
        ImportError: If pyloudnorm is not installed.
        ValueError: If input parameters are invalid.
        RuntimeError: If audio loading fails (for path/bytes inputs).
    """
    # Load and unify input to np.float32 ndarray
    original_dtype = None
    if isinstance(audio, (str, os.PathLike)):
        # File path input
        audio_path = str(audio)
        loaded_audio, loaded_sr = sf.read(audio_path, always_2d=False)
        sample_rate = loaded_sr
    elif isinstance(audio, bytes):
        # Raw bytes input
        with BytesIO(audio) as buf:
            loaded_audio, loaded_sr = sf.read(buf, always_2d=False)
        sample_rate = loaded_sr
    elif isinstance(audio, torch.Tensor):
        loaded_audio = audio.cpu().numpy()
    elif isinstance(audio, np.ndarray):
        loaded_audio = audio
        original_dtype = loaded_audio.dtype
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # Convert to float32 for processing
    if not isinstance(loaded_audio, np.ndarray) or loaded_audio.dtype != np.float32:
        loaded_audio = np.asarray(loaded_audio, dtype=np.float32)

    audio = loaded_audio  # Now unified

    if audio.ndim == 1:
        pass  # mono – already good
    elif audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            logger.warning("Audio appears to have channels as first dimension – transposing")
            audio = audio.T
    else:
        raise ValueError("Audio must be 1D (mono) or 2D (samples, channels)")

    # Cache meter per sample rate
    meter = _METER_CACHE.get(sample_rate)
    if meter is None:
        meter = pyln.Meter(sample_rate)
        _METER_CACHE[sample_rate] = meter

    # --- main LUFS logic + short-audio fallback ---
    # Prepare variables for fallback use
    effective_target_lufs = target_lufs
    effective_headroom_factor = headroom_factor
    apply_peak_norm = False

    if mode == "speech":
        if target_lufs == -14.0:
            effective_target_lufs = -13.0
        effective_headroom_factor = 1.0
        apply_peak_norm = True
        logger.debug("Speech mode activated: target_lufs=%.1f, headroom_factor=1.0", effective_target_lufs)
    elif mode == "general":
        pass
    elif mode is not None:
        raise ValueError(f"Invalid mode: {mode!r}. Allowed: 'general', 'speech', or None.")

    try:
        measured_lufs = meter.integrated_loudness(audio)
        logger.debug(f"Measured integrated loudness: {measured_lufs:.2f} LUFS")
    except Exception as exc:
        # pyloudnorm raises ValueError when audio is shorter than one gating block (~0.4s)
        if "Audio must have length greater than the block size" in str(exc):
            logger.info(
                "Audio too short for reliable LUFS measurement (< ~0.4s). "
                "Falling back to peak normalization."
            )
            # Use same headroom logic as main path

            peak = np.max(np.abs(audio))
            if peak == 0:
                logger.debug("Silent audio detected in short clip fallback")
                return audio.copy()

            normalized = audio / peak  # bring peak to 1.0
            # Apply headroom (same as main path)
            normalized /= effective_headroom_factor

            # Speech mode secondary peak boost
            if apply_peak_norm:
                current_peak = np.max(np.abs(normalized))
                if current_peak < 0.95:
                    target_peak = 0.99
                    gain = target_peak / current_peak
                    normalized *= gain
                    logger.debug(
                        "Speech mode (short audio): applied secondary peak normalization "
                        "(gain=%.3f, final peak=%.3f)", gain, np.max(np.abs(normalized))
                    )
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_audio = normalized.astype(np.float32).copy()
            # Shared dtype handling is applied later
        else:
            logger.warning(f"Unexpected LUFS measurement failure ({exc}), returning original audio")
            return audio.copy()
    else:
        if measured_lufs <= min_lufs_threshold:
            logger.debug("Audio too quiet – skipping loudness normalization")
            return audio.copy()

        # Optional cap: prevent amplification of already-loud content
        if max_loudness_threshold is not None:
            if measured_lufs > max_loudness_threshold:
                effective_target_lufs = min(effective_target_lufs, measured_lufs)
                logger.debug(
                    "Measured loudness %.2f exceeds max threshold %.2f – "
                    "preventing amplification (effective target: %.2f)",
                    measured_lufs, max_loudness_threshold, effective_target_lufs
                )

        try:
            normalized = pyln.normalize.loudness(audio, measured_lufs, effective_target_lufs)
        except Exception as exc:
            logger.warning(f"LUFS normalization failed ({exc}), returning original audio")
            return audio.copy()

        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized /= peak * effective_headroom_factor
            logger.debug(f"Applied headroom – post-norm peak reduced to {np.max(np.abs(normalized)):.3f}")
        else:
            if apply_peak_norm:
                current_peak = np.max(np.abs(normalized))
                if current_peak < 0.95:
                    target_peak = 0.99
                    gain = target_peak / current_peak
                    normalized *= gain
                    logger.debug(
                        "Speech mode: applied secondary peak normalization (gain=%.3f, final peak=%.3f)",
                        gain,
                        np.max(np.abs(normalized))
                    )
                normalized = np.clip(normalized, -1.0, 1.0)

        normalized_audio = normalized.astype(np.float32).copy()

    # Shared final dtype handling (works for both LUFS-path and peak fallback)
    ALLOWED_OUTPUT_TYPES = {np.float32, np.float64, np.int16, np.int32}

    if return_dtype is not None:
        target_dtype = np.dtype(return_dtype)
        target_type = target_dtype.type

        if target_type not in ALLOWED_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported return_dtype: {return_dtype!r}. "
                "Supported: 'float32', 'float64', 'int16', 'int32' (or their np.dtype equivalents)."
            )

        if target_type is np.float64:
            normalized_audio = normalized_audio.astype(np.float64)
        elif target_type is np.int16:
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            normalized_audio = (normalized_audio * 32767.0).astype(np.int16)
        elif target_type is np.int32:
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            normalized_audio = (normalized_audio * 2147483647.0).astype(np.int32)
        # else: np.float32 is default
    else:
        if original_dtype is not None:
            if np.issubdtype(original_dtype, np.floating):
                normalized_audio = normalized_audio.astype(original_dtype)
            elif np.issubdtype(original_dtype, np.integer):
                if original_dtype == np.int16:
                    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                    normalized_audio = (normalized_audio * 32767.0).astype(np.int16)
                elif original_dtype == np.int32:
                    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                    normalized_audio = (normalized_audio * 2147483647.0).astype(np.int32)
                else:
                    normalized_audio = normalized_audio.astype(np.float32)
            else:
                normalized_audio = normalized_audio.astype(np.float32)
        else:
            normalized_audio = normalized_audio.astype(np.float32)

    return normalized_audio


# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]

def get_audio_energy(audio: AudioInput) -> float:
    """
    Compute the root mean square (RMS) energy of a given audio input.
    The RMS is calculated across all samples and channels, providing a single
    measure of overall signal energy.

    Args:
        audio: AudioInput
            Audio signal as NumPy array, torch.Tensor, file path (str/os.PathLike),
            or raw bytes (WAV-compatible).

    Returns:
        float: RMS energy value.
            - For normalized audio (range [-1, 1]), typically [0.0, 0.707] for typical signals.
            - Returns 0.0 for silent/empty audio.

    Raises:
        ValueError: If the input type is not supported.
    """
    import numpy as np
    import soundfile as sf
    import os
    from io import BytesIO
    try:
        import torch
    except ImportError:
        torch = None

    # Load audio to np.float32 array
    if isinstance(audio, np.ndarray):
        audio_np = np.asarray(audio, dtype=np.float32)
    elif torch is not None and isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().float().numpy()
    elif isinstance(audio, (str, os.PathLike)):
        audio_np, _ = sf.read(str(audio), always_2d=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)
    elif isinstance(audio, bytes):
        with BytesIO(audio) as buf:
            audio_np, _ = sf.read(buf, always_2d=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio)!r}")

    # Handle empty or zero-length audio
    if audio_np.size == 0:
        return 0.0

    # Compute global RMS across all samples and all channels (standard in audio)
    # We explicitly flatten to avoid any ambiguity
    flattened = audio_np.reshape(-1)
    mean_square = np.mean(np.square(flattened))
    rms = float(np.sqrt(mean_square))
    return rms

def has_sound(
    audio: AudioInput,
    *,
    threshold: float = 0.01
) -> bool:
    """
    Determine if the audio contains perceptible sound.

    Uses RMS energy to decide. Audio with RMS energy above the threshold
    is considered to have sound (e.g., speech, music). Below the threshold
    it is treated as silence or near-silence.

    Args:
        audio: AudioInput
            Same supported formats as :func:`get_audio_energy`.
        threshold: float, optional
            RMS threshold above which audio is considered to have sound.
            Default is 0.01, which corresponds to approximately -40 dBFS
            (a common practical value for distinguishing silence from content).

    Returns:
        bool: True if the audio has perceptible sound, False otherwise.

    Example:
        >>> has_sound("speech.wav")          # True for normal speech
        >>> has_sound("silent.wav")          # False
        >>> has_sound(noisy_but_quiet, threshold=0.005)  # more sensitive
    """
    rms_energy = get_audio_energy(audio)
    return rms_energy > threshold


if __name__ == "__main__":
    import argparse
    import sys
    import soundfile as sf
    import shutil
    from pathlib import Path
    from rich import print as rprint

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_missav_5mins.wav"
    OUTPUT_AUDIO = OUTPUT_DIR / (Path(INPUT_AUDIO).stem + "_norm" + Path(INPUT_AUDIO).suffix)

    parser = argparse.ArgumentParser(
        description="Normalize loudness of a WAV file to target LUFS (ITU-R BS.1770-4)."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",  # Make positional input optional
        default=Path(INPUT_AUDIO),
        help="Input WAV file path"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",  # Make it optional
        default=OUTPUT_AUDIO,
        help="Output WAV file path (default: <input>_norm.wav)"
    )
    parser.add_argument(
        "-t",
        "--target",
        type=float,
        default=-14.0,
        help="Target integrated loudness in LUFS (default: -14.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64", "int16", "int32"],
        default=None,
        help="Output data type: float32, float64, int16, int32. "
             "If not specified, automatically matches the input file's native subtype when possible "
             "(e.g., int16 input → int16 output). Falls back to float32.",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.is_file():
        rprint(f"[red]Error: Input file not found: {input_path}[/red]")
        sys.exit(1)

    output_path = args.output or input_path.with_name(f"{input_path.stem}_norm{input_path.suffix}")

    rprint(f"[bold]Loading audio:[/bold] {input_path}")
    audio, sr = sf.read(input_path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    # Measure original loudness
    meter = pyln.Meter(sr)
    try:
        original_lufs = meter.integrated_loudness(audio)
    except Exception:
        original_lufs = float("-inf")

    rprint(f"[bold]Normalizing[/bold] to {args.target} LUFS...")
    normalized_audio = normalize_loudness(
        audio,
        sr,
        target_lufs=args.target,
        return_dtype=args.dtype,
        mode="speech",
    )

    # Measure final loudness
    try:
        final_lufs = meter.integrated_loudness(normalized_audio)
    except Exception:
        final_lufs = float("-inf")

    rprint(f"[green]Original:[/green] {original_lufs:.2f} LUFS → [green]Normalized:[/green] {final_lufs:.2f} LUFS")
    rprint(f"[bold]Writing output:[/bold] {output_path}")
    sf.write(output_path, normalized_audio, sr)
    rprint("[bold green]Done![/bold green]")
