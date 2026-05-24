from __future__ import annotations

import logging
from typing import Optional, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import torch

logger = logging.getLogger(__name__)

_SILERO_MODEL = None


def _load_silero_vad():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _SILERO_MODEL = (model, utils)
    return _SILERO_MODEL


def _speech_probability(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Compute per-sample speech probability using Silero VAD.

    Silero requires fixed-size frames:
    - 512 samples @ 16kHz
    - 256 samples @ 8kHz
    """
    if sample_rate not in (8000, 16000):
        raise ValueError(
            f"Unsupported sample_rate={sample_rate}. "
            "Silero VAD supports only 8000 or 16000 Hz."
        )

    model, utils = _load_silero_vad()
    frame_size = 512 if sample_rate == 16000 else 256

    audio_tensor = torch.from_numpy(audio).float()

    num_samples = audio_tensor.shape[0]
    num_frames = int(np.ceil(num_samples / frame_size))

    # Pad to full frames
    padded_len = num_frames * frame_size
    if padded_len > num_samples:
        pad = padded_len - num_samples
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))

    probs_per_frame = []

    with torch.no_grad():
        for i in range(num_frames):
            frame = audio_tensor[i * frame_size : (i + 1) * frame_size]
            frame = frame.unsqueeze(0)  # shape: (1, frame_size)
            prob = model(frame, sample_rate)
            probs_per_frame.append(prob.item())

    frame_probs = np.array(probs_per_frame, dtype=np.float32)

    # Upsample frame probabilities to sample-level
    sample_probs = np.repeat(frame_probs, frame_size)
    sample_probs = sample_probs[:num_samples]

    return sample_probs


def normalize_speech_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -13.0,
    min_lufs_threshold: float = -70.0,
    max_loudness_threshold: float | None = -10.0,
    peak_target: float = 0.99,
    return_dtype=None,
) -> np.ndarray:
    """
    Normalize speech audio using speech-probability-weighted LUFS.
    """

    # Accept and repair common multichannel input
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            audio = audio[:, 0]  # squeeze trivial stereo
        else:
            # Average channels → simple downmix
            audio = np.mean(audio.astype(np.float64), axis=1).astype(audio.dtype)
    elif audio.ndim > 2:
        raise ValueError(
            f"Unsupported audio shape {audio.shape} — "
            "expected 1D (mono) or 2D (frames, channels)"
        )

    orig_dtype = audio.dtype

    meter = pyln.Meter(sample_rate)

    # 1. Speech probabilities
    probs = _speech_probability(audio, sample_rate)

    if np.max(probs) < 0.1:
        return audio.astype(return_dtype or orig_dtype, copy=True)

    # 2. Weighted audio for LUFS measurement
    weighted_audio = audio * probs

    try:
        speech_lufs = meter.integrated_loudness(weighted_audio)
    except Exception:
        peak = np.max(np.abs(audio))
        if peak == 0:
            result = audio.copy()
        else:
            result = audio / peak * peak_target

        target_dtype = return_dtype or orig_dtype
        return _cast_audio_dtype(result, target_dtype)

    if speech_lufs <= min_lufs_threshold:
        return audio.astype(return_dtype or orig_dtype, copy=True)

    if max_loudness_threshold is not None:
        target_lufs = min(target_lufs, speech_lufs, max_loudness_threshold)

    # 3. Normalize ORIGINAL audio using speech LUFS
    normalized = pyln.normalize.loudness(
        audio,
        speech_lufs,
        target_lufs,
    )

    # 4. Speech peak normalization (AMPLIFICATION ALLOWED)
    peak = np.max(np.abs(normalized))
    if peak > 0:
        gain = peak_target / peak
        normalized *= gain

    normalized = np.clip(normalized, -1.0, 1.0)

    # 5. Respect return dtype
    target_dtype = return_dtype or orig_dtype
    return _cast_audio_dtype(normalized, target_dtype)


def _cast_audio_dtype(audio: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Cast normalized float audio back to target dtype.
    Integers are scaled from [-1, 1] to full-scale range.
    """
    if np.issubdtype(dtype, np.floating):
        return audio.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = audio * info.max
        return np.clip(scaled, info.min, info.max).astype(dtype)

    raise TypeError(f"Unsupported audio dtype: {dtype}")


def normalize_audio_for_vad(
    y: np.ndarray,
    sr: Optional[int] = None,
    method: str = "hybrid",
    target_rms_db: float = -20.0,
    max_peak: float = 0.95,
    eps: float = 1e-8,
    min_signal_db: float = -60.0,
    remove_dc: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize audio specifically for Voice Activity Detection (VAD).

    Recommended for most pipelines: 'hybrid' method with target_rms_db=-20.
    This provides consistent levels for energy-based, WebRTC, Silero, and
    neural VADs.

    Args:
        y:               Input audio array (any dtype; converted to float32).
        sr:              Sample rate in Hz. Currently used for documentation
                         and future extensions (e.g., resampling, pre-emphasis
                         cutoff). Pass it for forward-compatibility.
        method:          Normalization strategy:
                           'peak'   – scale so the loudest sample hits ±1.0.
                           'rms'    – scale to target_rms_db; no peak limit.
                           'hybrid' – RMS target + peak ceiling (recommended).
        target_rms_db:   Desired RMS level in dBFS for 'rms' / 'hybrid'.
        max_peak:        Peak ceiling for 'hybrid' (0 < max_peak ≤ 1.0).
        eps:             Small constant to guard log/division of silent frames.
        min_signal_db:   Signals whose RMS is below this threshold are treated
                         as silent and returned unchanged (avoids boosting pure
                         noise by 50+ dB).
        remove_dc:       If True, subtract the mean before normalizing.
                         Recommended for energy-based and WebRTC VADs.

    Returns:
        y_norm:  Normalized float32 audio array.
        info:    Diagnostic dict with original/final statistics.
    """

    # ------------------------------------------------------------------ #
    # 0. Empty-array guard                                                 #
    # ------------------------------------------------------------------ #
    if len(y) == 0:
        return y.astype(np.float32), {
            "method": method,
            "original_rms_db": -np.inf,
            "final_rms_db": -np.inf,
            "original_peak": 0.0,
            "final_peak": 0.0,
            "applied_gain_db": 0.0,
            "skipped_reason": "empty_input",
        }

    # ------------------------------------------------------------------ #
    # 1. Convert to float32                                                #
    # ------------------------------------------------------------------ #
    y_norm = y.astype(np.float32).copy()

    # ------------------------------------------------------------------ #
    # 2. DC offset removal (before any level measurement)                  #
    #    Eliminates bias that inflates RMS and confuses energy-based VADs. #
    # ------------------------------------------------------------------ #
    if remove_dc:
        y_norm -= np.mean(y_norm)

    # ------------------------------------------------------------------ #
    # 3. Original statistics                                               #
    # ------------------------------------------------------------------ #
    original_rms = np.sqrt(np.mean(y_norm**2) + eps)
    original_peak = float(np.max(np.abs(y_norm)))
    original_rms_db = (
        float(20 * np.log10(original_rms)) if original_rms > eps else -np.inf
    )

    # ------------------------------------------------------------------ #
    # 4. Silence guard                                                     #
    #    Very quiet signals (< min_signal_db) are mostly noise; boosting  #
    #    them by 50+ dB would make the noise floor dominate the VAD.      #
    # ------------------------------------------------------------------ #
    if original_rms_db < min_signal_db:
        info = {
            "method": method,
            "original_rms_db": round(original_rms_db, 2),
            "final_rms_db": round(original_rms_db, 2),
            "original_peak": round(original_peak, 4),
            "final_peak": round(original_peak, 4),
            "applied_gain_db": 0.0,
            "skipped_reason": "silent_input",
        }
        return y_norm, info

    # ------------------------------------------------------------------ #
    # 5. Normalization                                                     #
    # ------------------------------------------------------------------ #
    if method == "peak":
        # Scale so the loudest sample reaches ±1.0.
        # Measure the actual result instead of assuming librosa's output.
        y_norm = librosa.util.normalize(y_norm, norm=np.inf)
        # Re-measure: all-zeros edge case yields 0.0, not 1.0.
        final_peak = float(np.max(np.abs(y_norm)))

    elif method in ("rms", "hybrid"):
        target_rms = 10 ** (target_rms_db / 20.0)
        scale = target_rms / (original_rms + eps)
        y_norm *= scale

        # Post-scale peak (this is the true current peak, not pre-scale).
        current_peak = float(np.max(np.abs(y_norm)))

        if method == "hybrid" and current_peak > max_peak:
            # current_peak > max_peak > 0, so division is safe without eps.
            y_norm *= max_peak / current_peak
            final_peak = max_peak
        else:
            # 'rms' method, or 'hybrid' where peak is already within limit.
            final_peak = current_peak

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from 'peak', 'rms', or 'hybrid'."
        )

    # ------------------------------------------------------------------ #
    # 6. Final statistics                                                  #
    # ------------------------------------------------------------------ #
    final_rms = np.sqrt(np.mean(y_norm**2) + eps)
    final_rms_db = float(20 * np.log10(final_rms))

    info = {
        "method": method,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(original_peak, 4),
        "final_peak": round(final_peak, 4),
        "applied_gain_db": round(final_rms_db - original_rms_db, 2),
        "skipped_reason": None,
        # sr preserved for downstream traceability
        "sr": sr,
    }

    return y_norm, info


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path

    import librosa
    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table

    # Configure rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger(__name__)
    console = Console()

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract and analyse speech waves from audio using FireRedVAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help="Output results directory.",
    )

    args = parser.parse_args()

    # Load audio
    logger.info(f"Loading audio from: {args.audio_path}")
    y, sr = librosa.load(args.audio_path, sr=None)
    logger.info(f"Loaded audio: {len(y)/sr:.2f}s, {sr}Hz, shape: {y.shape}")

    # Create output directories
    speech_loudness_dir = args.output_dir / "normalize_speech_loudness"
    vad_dir = args.output_dir / "normalize_audio_for_vad"
    
    speech_loudness_dir.mkdir(parents=True, exist_ok=True)
    vad_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directories created:")
    logger.info(f"  Speech loudness: {speech_loudness_dir}")
    logger.info(f"  VAD normalization: {vad_dir}")

    # ── Process 1: normalize_audio_for_vad ─────────────────────────────────
    logger.info("Processing: normalize_audio_for_vad")
    
    y_vad, vad_stats = normalize_audio_for_vad(y, sr=sr)
    
    # Save VAD results
    vad_audio_path = vad_dir / f"{Path(args.audio_path).stem}_vad_normalized.wav"
    sf.write(vad_audio_path, y_vad, sr)
    
    # Save VAD stats
    vad_stats_path = vad_dir / f"{Path(args.audio_path).stem}_vad_stats.json"
    with open(vad_stats_path, 'w') as f:
        json.dump(vad_stats, f, indent=2, default=str)
    
    # Display VAD stats with rich
    console.print(f"\n[bold green]✓[/bold green] Normalize Audio for VAD - Complete")
    vad_table = Table(title="VAD Normalization Stats")
    vad_table.add_column("Metric", style="cyan")
    vad_table.add_column("Value", style="magenta")
    
    for key, value in vad_stats.items():
        vad_table.add_row(str(key), str(value))
    
    console.print(vad_table)

    # ── Process 2: normalize_speech_loudness ───────────────────────────────
    logger.info("Processing: normalize_speech_loudness")
    
    # Get speech probabilities for analysis
    try:
        probs = _speech_probability(y, sr)
        speech_regions = np.sum(probs > 0.5) / len(probs) * 100
        max_prob = float(np.max(probs))
        mean_prob = float(np.mean(probs))
        prob_threshold = 0.1
        has_speech = np.max(probs) >= prob_threshold
    except ValueError as e:
        logger.warning(f"Could not compute speech probabilities: {e}")
        probs = None
        speech_regions = None
        max_prob = None
        mean_prob = None
        has_speech = None
    
    # normalize_speech_loudness returns only the normalized audio array
    y_speech = normalize_speech_loudness(y, sr)
    
    # Calculate gain statistics
    if has_speech:
        # Calculate RMS before and after for gain estimation
        eps = 1e-8
        original_rms = np.sqrt(np.mean(y.astype(np.float64)**2) + eps)
        normalized_rms = np.sqrt(np.mean(y_speech.astype(np.float64)**2) + eps)
        gain_db = 20 * np.log10(normalized_rms / original_rms) if original_rms > eps else 0
        
        original_peak = float(np.max(np.abs(y)))
        normalized_peak = float(np.max(np.abs(y_speech)))
    else:
        gain_db = 0.0
        original_peak = float(np.max(np.abs(y)))
        normalized_peak = original_peak
    
    # Save speech loudness results
    speech_audio_path = speech_loudness_dir / f"{Path(args.audio_path).stem}_speech_normalized.wav"
    sf.write(speech_audio_path, y_speech, sr)
    
    # Compile comprehensive speech loudness info
    speech_loudness_info = {
        "normalization_params": {
            "target_lufs": -13.0,
            "min_lufs_threshold": -70.0,
            "max_loudness_threshold": -10.0,
            "peak_target": 0.99,
        },
        "speech_probability_stats": {
            "max_probability": max_prob,
            "mean_probability": mean_prob,
            "speech_percentage": speech_regions,
            "probabilities_available": probs is not None,
            "has_speech": has_speech,
        },
        "audio_stats": {
            "original_shape": y.shape,
            "normalized_shape": y_speech.shape,
            "original_dtype": str(y.dtype),
            "normalized_dtype": str(y_speech.dtype),
            "sample_rate": sr,
            "duration_seconds": len(y) / sr,
            "original_peak": round(original_peak, 4),
            "normalized_peak": round(normalized_peak, 4),
            "estimated_gain_db": round(gain_db, 2),
        }
    }
    
    # Save speech loudness stats
    speech_stats_path = speech_loudness_dir / f"{Path(args.audio_path).stem}_speech_loudness_stats.json"
    with open(speech_stats_path, 'w') as f:
        json.dump(speech_loudness_info, f, indent=2, default=str)
    
    # Save speech probabilities if available
    if probs is not None:
        prob_data = np.column_stack([
            np.arange(len(probs)) / sr,  # timestamps
            probs
        ])
        prob_path = speech_loudness_dir / f"{Path(args.audio_path).stem}_speech_probabilities.csv"
        np.savetxt(prob_path, prob_data, delimiter=',', 
                   header='time_seconds,speech_probability', comments='')
    else:
        prob_path = None
    
    # Display speech loudness stats with rich
    console.print(f"\n[bold green]✓[/bold green] Normalize Speech Loudness - Complete")
    speech_table = Table(title="Speech Loudness Normalization Stats")
    speech_table.add_column("Metric", style="cyan")
    speech_table.add_column("Value", style="magenta")
    
    speech_table.add_row("Target LUFS", "-13.0")
    speech_table.add_row("Sample Rate", f"{sr} Hz")
    speech_table.add_row("Duration", f"{len(y)/sr:.2f}s")
    speech_table.add_row("Original Peak", f"{original_peak:.4f}")
    speech_table.add_row("Normalized Peak", f"{normalized_peak:.4f}")
    speech_table.add_row("Estimated Gain", f"{gain_db:.2f} dB")
    
    if probs is not None:
        speech_table.add_row("Max Speech Probability", f"{max_prob:.4f}")
        speech_table.add_row("Mean Speech Probability", f"{mean_prob:.4f}")
        speech_table.add_row("Speech Regions", f"{speech_regions:.1f}%")
        speech_table.add_row("Speech Detected", str(has_speech))
    
    console.print(speech_table)

    # ── Save combined summary ──────────────────────────────────────────────
    summary = {
        "input_file": str(Path(args.audio_path).resolve()),
        "sample_rate": sr,
        "duration_seconds": len(y) / sr,
        "vad_normalization": vad_stats,
        "speech_loudness_normalization": speech_loudness_info,
        "output_directories": {
            "vad_normalization": str(vad_dir),
            "speech_loudness_normalization": str(speech_loudness_dir),
        },
        "saved_files": {
            "vad_audio": str(vad_audio_path),
            "vad_stats": str(vad_stats_path),
            "speech_audio": str(speech_audio_path),
            "speech_stats": str(speech_stats_path),
        }
    }
    
    if prob_path is not None:
        summary["saved_files"]["speech_probabilities"] = str(prob_path)
    
    summary_path = args.output_dir / f"{Path(args.audio_path).stem}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"[bold]Processing complete![/bold]")
    logger.info(f"Summary saved to: {summary_path}")
    
    # Final summary table
    final_table = Table(title="Processing Summary")
    final_table.add_column("Processing Step", style="cyan")
    final_table.add_column("Status", style="green")
    final_table.add_column("Output Directory", style="magenta")
    
    final_table.add_row("VAD Normalization", "✓", str(vad_dir))
    final_table.add_row("Speech Loudness Norm", "✓", str(speech_loudness_dir))
    
    console.print(f"\n")
    console.print(final_table)
