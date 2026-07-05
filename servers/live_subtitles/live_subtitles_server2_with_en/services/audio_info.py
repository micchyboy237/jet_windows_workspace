from __future__ import annotations

import io
import os
from typing import Optional

import librosa
import numpy as np
import torch

try:
    from services.audio_utils import AudioInput
    from services.audio_config import SAMPLE_RATE
except ImportError:
    from audio_utils import AudioInput
    from audio_config import SAMPLE_RATE

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Initialize Rich console
console = Console()


def amplitude_to_dbfs(amplitude: float, eps: float = 1e-10) -> float:
    """
    Convert linear amplitude to dBFS (decibels relative to Full Scale).
    
    In digital audio, 0 dBFS is the maximum possible level (amplitude = 1.0).
    Values above 0 dBFS indicate clipping. Each -6 dBFS represents half the amplitude.
    
    Args:
        amplitude: Linear amplitude value (typically 0.0 to 1.0)
        eps: Small value to prevent log(0)
    
    Returns:
        dBFS value (negative for valid signals, 0 for maximum, positive for clipped)
    
    Examples:
        >>> amplitude_to_dbfs(1.0)    # Full scale
        0.0
        >>> amplitude_to_dbfs(0.5)    # Half amplitude
        -6.02
        >>> amplitude_to_dbfs(0.1)    # Common RMS target for VAD
        -20.0
        >>> amplitude_to_dbfs(0.01)   # Very quiet
        -40.0
    """
    return float(20 * np.log10(max(amplitude, eps)))


def dbfs_to_amplitude(dbfs: float) -> float:
    """
    Convert dBFS to linear amplitude.
    
    Args:
        dbfs: Level in dBFS (negative values typical, 0 = full scale)
    
    Returns:
        Linear amplitude (0.0 to 1.0 for valid signals)
    
    Examples:
        >>> dbfs_to_amplitude(0.0)     # Full scale
        1.0
        >>> dbfs_to_amplitude(-6.0)    # Half amplitude
        0.501
        >>> dbfs_to_amplitude(-20.0)   # Common VAD target
        0.1
        >>> dbfs_to_amplitude(-3.0)    # Safe peak limit
        0.708
    """
    return float(10 ** (dbfs / 20.0))


def get_audio_info(audio: AudioInput, sr: Optional[int] = None) -> dict:
    """
    Extract detailed information about an audio input for debugging purposes.
    Analyzes audio files, arrays, or tensors to extract metadata and signal
    characteristics that can affect VAD performance, such as:
    - Sample rate and duration
    - Number of channels and channel layout
    - Data type and value ranges
    - Signal statistics (RMS, peak, DC offset) in both linear and dBFS
    - Silence ratio estimation
    - Potential issues (clipping, low amplitude, etc.)
    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        sr: Sample rate for array/tensor inputs. If None, defaults to SAMPLE_RATE.
           For file inputs, sr is detected automatically.
    Returns:
        Dictionary containing audio metadata and signal statistics
    Raises:
        TypeError: If audio input type is not supported
    """
    info = {
        "input_type": type(audio).__name__,
        "source": None,
        "dtype": None,
        "sample_rate": None,
        "num_channels": None,
        "num_samples": None,
        "duration_seconds": None,
        # Linear values
        "rms_amplitude": None,
        "peak_amplitude": None,
        "dc_offset": None,
        "amplitude_range": None,
        # dBFS values
        "rms_dbfs": None,
        "peak_dbfs": None,
        "crest_factor_db": None,  # Peak - RMS (indicates dynamic range)
        # Status flags
        "is_normalized": None,
        "has_clipping": None,
        "silence_ratio": None,
        "estimated_snr": None,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }
    if isinstance(audio, (str, os.PathLike)):
        info["source"] = str(audio)
        try:
            info["sample_rate"] = librosa.get_samplerate(audio)
            info["duration_seconds"] = librosa.get_duration(path=audio)
            y, native_sr = librosa.load(audio, sr=None, mono=False)
            audio_data = y
            if sr is None:
                sr = native_sr
            info["sample_rate"] = native_sr
        except Exception as e:
            info["issues"].append(f"Failed to load audio file: {str(e)}")
            return info
    elif isinstance(audio, bytes):
        info["source"] = f"Bytes object ({len(audio)} bytes)"
        try:
            y, native_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
            audio_data = y
            if sr is None:
                sr = native_sr
            info["sample_rate"] = native_sr
        except Exception:
            if sr is None:
                sr = SAMPLE_RATE
            try:
                audio_data = (
                    np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                )
                info["sample_rate"] = sr
                info["warnings"].append("Decoded as raw PCM int16")
            except Exception as e2:
                info["issues"].append(f"Failed to decode audio bytes: {str(e2)}")
                return info
    elif isinstance(audio, np.ndarray):
        info["source"] = f"NumPy array (shape: {audio.shape})"
        audio_data = audio
        if sr is None:
            sr = SAMPLE_RATE
        info["sample_rate"] = sr
    elif isinstance(audio, torch.Tensor):
        info["source"] = f"PyTorch tensor (shape: {audio.shape})"
        audio_data = audio.detach().cpu().numpy()
        if sr is None:
            sr = SAMPLE_RATE
        info["sample_rate"] = sr
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
    if audio_data.ndim == 1:
        info["num_channels"] = 1
        info["num_samples"] = len(audio_data)
    elif audio_data.ndim == 2:
        if audio_data.shape[0] < audio_data.shape[1]:
            info["num_channels"] = audio_data.shape[0]
            info["num_samples"] = audio_data.shape[1]
        else:
            info["num_channels"] = audio_data.shape[1]
            info["num_samples"] = audio_data.shape[0]
            audio_data = audio_data.T
    else:
        info["issues"].append(f"Unexpected audio dimensions: {audio_data.ndim}")
        return info
    if info["duration_seconds"] is None:
        info["duration_seconds"] = info["num_samples"] / sr
    info["dtype"] = str(audio_data.dtype)

    # Normalize integer audio to [-1, 1] float64 for safe statistics.
    # float64 prevents overflow when squaring large int values
    # (e.g., 31128² summed over 100k samples exceeds int32 range).
    # Original dtype is preserved for metadata display.
    if np.issubdtype(audio_data.dtype, np.integer):
        original_dtype = audio_data.dtype
        dtype_max = np.iinfo(original_dtype).max
        audio_data = audio_data.astype(np.float64) / dtype_max
        info["dtype"] = str(original_dtype)

    channel_info = []
    for ch in range(info["num_channels"]):
        ch_data = audio_data[ch] if info["num_channels"] > 1 else audio_data
        ch_stats = {
            "rms": np.sqrt(np.mean(ch_data**2)),
            "peak": np.max(np.abs(ch_data)),
            "min": np.min(ch_data),
            "max": np.max(ch_data),
            "dc_offset": np.mean(ch_data),
        }
        channel_info.append(ch_stats)
    
    # Linear amplitude values
    info["rms_amplitude"] = float(np.mean([ch["rms"] for ch in channel_info]))
    info["peak_amplitude"] = float(np.max([ch["peak"] for ch in channel_info]))
    info["dc_offset"] = float(np.mean([ch["dc_offset"] for ch in channel_info]))
    info["amplitude_range"] = [
        float(np.min([ch["min"] for ch in channel_info])),
        float(np.max([ch["max"] for ch in channel_info])),
    ]
    
    # dBFS values
    info["rms_dbfs"] = amplitude_to_dbfs(info["rms_amplitude"])
    info["peak_dbfs"] = amplitude_to_dbfs(info["peak_amplitude"])
    info["crest_factor_db"] = info["peak_dbfs"] - info["rms_dbfs"]  # Dynamic range indicator
    
    info["is_normalized"] = abs(info["peak_amplitude"] - 1.0) < 0.01
    clipping_threshold = 0.99
    if info["num_channels"] == 1:
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
    else:
        clipped_samples = np.sum(
            np.any(np.abs(audio_data) > clipping_threshold, axis=0)
        )
    info["has_clipping"] = clipped_samples > 0
    info["clipping_percentage"] = float(100 * clipped_samples / info["num_samples"])
    silence_threshold = 0.01
    if info["num_channels"] == 1:
        is_silent = np.abs(audio_data) < silence_threshold
    else:
        is_silent = np.all(np.abs(audio_data) < silence_threshold, axis=0)
    info["silence_ratio"] = float(np.mean(is_silent))
    if info["silence_ratio"] < 0.95 and info["silence_ratio"] > 0.05:
        signal_mask = ~is_silent
        noise_mask = is_silent
        if info["num_channels"] == 1:
            signal_rms = (
                np.sqrt(np.mean(audio_data[signal_mask] ** 2))
                if np.any(signal_mask)
                else 0
            )
            noise_rms = (
                np.sqrt(np.mean(audio_data[noise_mask] ** 2))
                if np.any(noise_mask)
                else 0
            )
        else:
            signal_data = (
                audio_data[:, signal_mask] if np.any(signal_mask) else np.array([0])
            )
            noise_data = (
                audio_data[:, noise_mask] if np.any(noise_mask) else np.array([0])
            )
            signal_rms = np.sqrt(np.mean(signal_data**2))
            noise_rms = np.sqrt(np.mean(noise_data**2))
        if noise_rms > 0 and signal_rms > 0:
            info["estimated_snr"] = float(20 * np.log10(signal_rms / noise_rms))
    if info["sample_rate"] != 16000:
        info["warnings"].append(
            f"Sample rate is {info['sample_rate']} Hz (VAD often expects 16000 Hz)"
        )
        info["recommendations"].append(
            "Resample to 16000 Hz for optimal VAD performance"
        )
    if info["num_channels"] > 1:
        info["warnings"].append(
            f"Multi-channel audio ({info['num_channels']} channels)"
        )
        info["recommendations"].append("Consider converting to mono for VAD")
    if info["rms_dbfs"] < -40:
        info["warnings"].append(
            f"Very low amplitude (RMS: {info['rms_dbfs']:.1f} dBFS)"
        )
        info["recommendations"].append(
            "Audio may be too quiet for reliable VAD - consider amplification"
        )
    elif info["rms_dbfs"] > -6:
        info["warnings"].append(f"High amplitude (RMS: {info['rms_dbfs']:.1f} dBFS)")
    if info["has_clipping"] and info["clipping_percentage"] > 1.0:
        info["warnings"].append(
            f"Clipping detected ({info['clipping_percentage']:.1f}% of samples)"
        )
        info["recommendations"].append(
            "Clipping can degrade VAD accuracy - reduce input gain"
        )
    if info["silence_ratio"] > 0.9:
        info["warnings"].append(
            f"Mostly silent ({info['silence_ratio'] * 100:.1f}% silence)"
        )
        info["recommendations"].append(
            "VAD may struggle with very sparse speech content"
        )
    if abs(info["dc_offset"]) > 0.01:
        info["warnings"].append(f"Significant DC offset ({info['dc_offset']:.4f})")
        info["recommendations"].append("Remove DC offset to improve signal quality")
    if info["estimated_snr"] is not None and info["estimated_snr"] < 10:
        info["warnings"].append(f"Low estimated SNR ({info['estimated_snr']:.1f} dB)")
        info["recommendations"].append(
            "High noise levels may cause VAD false positives"
        )
    if info["crest_factor_db"] is not None and info["crest_factor_db"] > 20:
        info["warnings"].append(
            f"High crest factor ({info['crest_factor_db']:.1f} dB) - large dynamic range"
        )
        info["recommendations"].append(
            "Consider compression or limiting to reduce peak-to-RMS ratio"
        )
    return info


def display_audio_info(
    audio: AudioInput,
    sr: Optional[int] = None,
    show_waveform: bool = False,
    waveform_width: int = 80,
    waveform_height: int = 10,
) -> None:
    """
    Display detailed audio information using Rich console formatting.

    Creates a visually appealing output with audio metadata, signal statistics,
    and potential issues that could affect VAD performance.

    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        sr: Sample rate for array/tensor inputs. If None, defaults to SAMPLE_RATE.
        show_waveform: Whether to display an ASCII waveform preview
        waveform_width: Width of the waveform display in characters
        waveform_height: Height of the waveform display in characters

    Example:
        display_audio_info("recording.wav")
        display_audio_info(audio_array, sr=16000, show_waveform=True)
    """
    info = get_audio_info(audio, sr)

    # Create header
    console.print()
    console.print(
        Panel(
            Text("Audio Analysis for VAD Debugging", style="bold white on blue"),
            border_style="blue",
            padding=(0, 2),
        )
    )

    # Basic metadata table
    metadata_table = Table(
        title="Audio Metadata",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    metadata_table.add_column("Property", style="bold", width=20)
    metadata_table.add_column("Value", width=40)

    metadata_table.add_row("Source", info.get("source", "Unknown"))
    metadata_table.add_row("Input Type", info["input_type"])
    metadata_table.add_row("Data Type", str(info.get("dtype", "N/A")))
    metadata_table.add_row("Sample Rate", f"{info.get('sample_rate', 'N/A')} Hz")
    metadata_table.add_row("Channels", str(info.get("num_channels", "N/A")))
    metadata_table.add_row("Samples", f"{info.get('num_samples', 'N/A'):,}")

    # Format duration
    duration = info.get("duration_seconds")
    if duration is not None:
        minutes = int(duration // 60)
        seconds = duration % 60
        duration_str = f"{duration:.3f}s ({minutes}:{seconds:05.2f})"
    else:
        duration_str = "N/A"
    metadata_table.add_row("Duration", duration_str)

    console.print(metadata_table)

    # Signal statistics table
    stats_table = Table(
        title="Signal Statistics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    stats_table.add_column("Metric", style="bold", width=25)
    stats_table.add_column("Linear", width=18)
    stats_table.add_column("dBFS", width=18)
    stats_table.add_column("Status", width=15)

    # RMS amplitude
    rms_linear = info.get("rms_amplitude")
    rms_dbfs = info.get("rms_dbfs")
    if rms_linear is not None and rms_dbfs is not None:
        rms_status = "✓ Normal" if -30 <= rms_dbfs <= -6 else "⚠"
        stats_table.add_row(
            "RMS Level",
            f"{rms_linear:.4f}",
            f"{rms_dbfs:.1f} dBFS",
            rms_status,
        )
    else:
        stats_table.add_row("RMS Level", "N/A", "N/A", "")

    # Peak amplitude
    peak_linear = info.get("peak_amplitude")
    peak_dbfs = info.get("peak_dbfs")
    if peak_linear is not None and peak_dbfs is not None:
        peak_status = "✓ Normal" if peak_dbfs < 0 else "⚠ Clipping"
        stats_table.add_row(
            "Peak Level",
            f"{peak_linear:.4f}",
            f"{peak_dbfs:.1f} dBFS",
            peak_status,
        )
    else:
        stats_table.add_row("Peak Level", "N/A", "N/A", "")

    # Crest factor (dynamic range indicator)
    crest = info.get("crest_factor_db")
    if crest is not None:
        if crest < 10:
            crest_status = "Compressed"
        elif crest < 20:
            crest_status = "Normal"
        else:
            crest_status = "Dynamic"
        stats_table.add_row(
            "Crest Factor (Peak-RMS)",
            f"{10**(crest/20):.2f}x",
            f"{crest:.1f} dB",
            crest_status,
        )
    else:
        stats_table.add_row("Crest Factor", "N/A", "N/A", "")

    # DC offset
    dc = info.get("dc_offset", 0)
    dc_status = "✓ OK" if abs(dc) < 0.001 else "⚠ Offset"
    stats_table.add_row("DC Offset", f"{dc:.6f}", f"{amplitude_to_dbfs(abs(dc)):.1f} dBFS", dc_status)

    # Normalization
    stats_table.add_row(
        "Peak Normalized",
        "Yes" if info.get("is_normalized") else "No",
        "0 dBFS" if info.get("is_normalized") else f"{peak_dbfs:.1f} dBFS" if peak_dbfs else "N/A",
        "✓" if info.get("is_normalized") else "⚠",
    )

    # Clipping
    if info.get("has_clipping"):
        clip_pct = info.get("clipping_percentage", 0)
        stats_table.add_row(
            "Clipping",
            f"Yes ({clip_pct:.1f}%)",
            "> 0 dBFS",
            "⚠" if clip_pct > 0.5 else "⚠ Minor",
        )
    else:
        stats_table.add_row("Clipping", "No", "< 0 dBFS", "✓")

    # Silence ratio
    silence = info.get("silence_ratio")
    if silence is not None:
        silence_status = "⚠" if silence > 0.5 else "✓"
        stats_table.add_row(
            "Silence Ratio",
            f"{silence * 100:.1f}%",
            f"{amplitude_to_dbfs(0.01):.0f} dBFS thresh",
            silence_status,
        )
    else:
        stats_table.add_row("Silence Ratio", "N/A", "", "")

    # SNR
    snr = info.get("estimated_snr")
    if snr is not None:
        snr_status = "✓ Good" if snr > 20 else "⚠ Low" if snr > 10 else "✗ Poor"
        stats_table.add_row("Est. SNR", f"{10**(snr/20):.1f}x", f"{snr:.1f} dB", snr_status)
    else:
        stats_table.add_row("Est. SNR", "N/A", "", "")

    console.print(stats_table)

    # dBFS Reference Guide
    dbfs_guide = Table(
        title="dBFS Quick Reference",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
    )
    dbfs_guide.add_column("dBFS", width=10)
    dbfs_guide.add_column("Linear", width=10)
    dbfs_guide.add_column("Description", width=40)
    
    dbfs_guide.add_row("0.0", "1.000", "Full scale - clipping threshold")
    dbfs_guide.add_row("-3.0", "0.708", "Safe peak limit (prevents intersample clipping)")
    dbfs_guide.add_row("-6.0", "0.501", "Half amplitude")
    dbfs_guide.add_row("-12.0", "0.251", "Quarter amplitude")
    dbfs_guide.add_row("-20.0", "0.100", "Common RMS target for VAD")
    dbfs_guide.add_row("-30.0", "0.032", "Very quiet speech")
    dbfs_guide.add_row("-40.0", "0.010", "Near silence")
    dbfs_guide.add_row("-60.0", "0.001", "Digital silence threshold")
    dbfs_guide.add_row("-∞", "0.000", "Absolute silence")
    
    console.print(dbfs_guide)
    console.print()

    # Waveform display (optional)
    if show_waveform and info.get("num_samples", 0) > 0:
        console.print()
        try:
            # Load audio data for waveform
            if isinstance(audio, (str, os.PathLike)):
                waveform_data, _ = librosa.load(
                    audio, sr=info["sample_rate"], mono=True
                )
            elif isinstance(audio, np.ndarray):
                waveform_data = audio.flatten() if audio.ndim > 1 else audio
            elif isinstance(audio, torch.Tensor):
                waveform_data = audio.flatten().cpu().numpy()
            else:
                waveform_data = np.array([])

            if len(waveform_data) > 0:
                # Downsample for display
                downsample_factor = max(1, len(waveform_data) // waveform_width)
                downsampled = waveform_data[::downsample_factor][:waveform_width]

                # Normalize for display
                if np.max(np.abs(downsampled)) > 0:
                    downsampled = downsampled / np.max(np.abs(downsampled))

                # Create ASCII waveform
                waveform_lines = []
                for level in range(waveform_height, 0, -1):
                    threshold = (level / waveform_height) * 2 - 1
                    line = ""
                    for sample in downsampled:
                        if abs(sample) >= abs(threshold):
                            if sample > 0 and threshold > 0:
                                line += "▀"
                            elif sample < 0 and threshold < 0:
                                line += "▄"
                            else:
                                line += "█"
                        else:
                            line += " "
                    waveform_lines.append(line)

                waveform_panel = Panel(
                    "\n".join(waveform_lines),
                    title=f"Waveform Preview (Peak Normalized, {info.get('peak_dbfs', 'N/A')} dBFS original)",
                    border_style="green",
                    padding=(0, 1),
                )
                console.print(waveform_panel)

        except Exception as e:
            console.print(f"[yellow]Could not generate waveform: {str(e)}[/yellow]")

    # Issues and recommendations
    if info.get("warnings"):
        console.print()
        warnings_panel = Panel(
            Text("\n".join(f"⚠ {w}" for w in info["warnings"]), style="yellow"),
            title="Warnings",
            border_style="yellow",
        )
        console.print(warnings_panel)

    if info.get("recommendations"):
        rec_panel = Panel(
            Text("\n".join(f"→ {r}" for r in info["recommendations"]), style="cyan"),
            title="Recommendations for VAD",
            border_style="cyan",
        )
        console.print(rec_panel)

    if info.get("issues"):
        issues_panel = Panel(
            Text("\n".join(f"✗ {i}" for i in info["issues"]), style="red"),
            title="Critical Issues",
            border_style="red",
        )
        console.print(issues_panel)

    # VAD compatibility summary
    console.print()
    vad_score = 100

    # Deduct points for each issue
    if info.get("sample_rate") != 16000:
        vad_score -= 20
    if info.get("num_channels", 1) > 1:
        vad_score -= 10
    if info.get("rms_dbfs", 0) < -40:
        vad_score -= 20
    elif info.get("rms_dbfs", 0) < -30:
        vad_score -= 10
    if info.get("silence_ratio", 0) > 0.5:
        vad_score -= 15
    if info.get("has_clipping"):
        vad_score -= 10
    if info.get("estimated_snr") is not None and info["estimated_snr"] < 10:
        vad_score -= 25
    if abs(info.get("dc_offset", 0)) > 0.01:
        vad_score -= 10

    vad_score = max(0, min(100, vad_score))

    if vad_score >= 80:
        score_color = "green"
        score_text = "Good - Audio is well-suited for VAD"
    elif vad_score >= 60:
        score_color = "yellow"
        score_text = "Fair - VAD may have occasional issues"
    elif vad_score >= 40:
        score_color = "orange1"
        score_text = "Poor - VAD is likely to be unreliable"
    else:
        score_color = "red"
        score_text = "Bad - VAD will struggle significantly"

    score_panel = Panel(
        Text(
            f"VAD Compatibility Score: {vad_score}/100\n{score_text}",
            style=f"bold {score_color}",
        ),
        title="VAD Readiness Assessment",
        border_style=score_color,
    )
    console.print(score_panel)
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from main._main_audio_info import main

    main()
