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


def get_audio_info(audio: AudioInput, sr: Optional[int] = None) -> dict:
    """
    Extract detailed information about an audio input for debugging purposes.

    Analyzes audio files, arrays, or tensors to extract metadata and signal
    characteristics that can affect VAD performance, such as:
    - Sample rate and duration
    - Number of channels and channel layout
    - Data type and value ranges
    - Signal statistics (RMS, peak, DC offset)
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
        "rms_amplitude": None,
        "peak_amplitude": None,
        "dc_offset": None,
        "amplitude_range": None,
        "is_normalized": None,
        "has_clipping": None,
        "silence_ratio": None,
        "estimated_snr": None,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }

    # Load audio data if it's a file or bytes
    if isinstance(audio, (str, os.PathLike)):
        info["source"] = str(audio)
        try:
            # Get metadata without loading full audio first
            info["sample_rate"] = librosa.get_samplerate(audio)
            info["duration_seconds"] = librosa.get_duration(path=audio)

            # Load audio for detailed analysis
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
            # Try decoding as raw PCM
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

    # Convert to numpy array if not already
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)

    # Determine shape and channels
    if audio_data.ndim == 1:
        info["num_channels"] = 1
        info["num_samples"] = len(audio_data)
    elif audio_data.ndim == 2:
        # Could be (channels, samples) or (samples, channels)
        if audio_data.shape[0] < audio_data.shape[1]:
            # (channels, samples)
            info["num_channels"] = audio_data.shape[0]
            info["num_samples"] = audio_data.shape[1]
        else:
            # (samples, channels) - transpose for analysis
            info["num_channels"] = audio_data.shape[1]
            info["num_samples"] = audio_data.shape[0]
            audio_data = audio_data.T
    else:
        info["issues"].append(f"Unexpected audio dimensions: {audio_data.ndim}")
        return info

    # Calculate duration
    if info["duration_seconds"] is None:
        info["duration_seconds"] = info["num_samples"] / sr

    # Get dtype
    info["dtype"] = str(audio_data.dtype)

    # Analyze signal characteristics per channel
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

    # Aggregate channel statistics
    info["rms_amplitude"] = float(np.mean([ch["rms"] for ch in channel_info]))
    info["peak_amplitude"] = float(np.max([ch["peak"] for ch in channel_info]))
    info["dc_offset"] = float(np.mean([ch["dc_offset"] for ch in channel_info]))
    info["amplitude_range"] = [
        float(np.min([ch["min"] for ch in channel_info])),
        float(np.max([ch["max"] for ch in channel_info])),
    ]

    # Check if audio is normalized
    info["is_normalized"] = abs(info["peak_amplitude"] - 1.0) < 0.01

    # Check for clipping
    clipping_threshold = 0.99
    if info["num_channels"] == 1:
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
    else:
        clipped_samples = np.sum(
            np.any(np.abs(audio_data) > clipping_threshold, axis=0)
        )
    info["has_clipping"] = clipped_samples > 0
    info["clipping_percentage"] = float(100 * clipped_samples / info["num_samples"])

    # Estimate silence ratio
    silence_threshold = 0.01  # -40dB
    if info["num_channels"] == 1:
        is_silent = np.abs(audio_data) < silence_threshold
    else:
        # Consider silent if all channels are below threshold
        is_silent = np.all(np.abs(audio_data) < silence_threshold, axis=0)
    info["silence_ratio"] = float(np.mean(is_silent))

    # Estimate SNR (simple approach using signal vs silence regions)
    if info["silence_ratio"] < 0.95 and info["silence_ratio"] > 0.05:
        # Separate signal and potential noise regions
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

    # Generate warnings and recommendations
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

    if info["rms_amplitude"] < 0.01:
        info["warnings"].append(
            f"Very low amplitude (RMS: {info['rms_amplitude']:.6f})"
        )
        info["recommendations"].append(
            "Audio may be too quiet for reliable VAD - consider amplification"
        )
    elif info["rms_amplitude"] > 0.5:
        info["warnings"].append(f"High amplitude (RMS: {info['rms_amplitude']:.3f})")

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
    stats_table.add_column("Value", width=35)
    stats_table.add_column("Status", width=15)

    # RMS amplitude
    rms = info.get("rms_amplitude")
    if rms is not None:
        rms_db = 20 * np.log10(max(rms, 1e-10))
        rms_status = "✓ Normal" if -30 <= rms_db <= -6 else "⚠"
        stats_table.add_row("RMS Amplitude", f"{rms:.4f} ({rms_db:.1f} dB)", rms_status)
    else:
        stats_table.add_row("RMS Amplitude", "N/A", "")

    # Peak amplitude
    peak = info.get("peak_amplitude")
    if peak is not None:
        peak_db = 20 * np.log10(max(peak, 1e-10))
        peak_status = "✓ Normal" if peak_db < 0 else "⚠ Clipping"
        stats_table.add_row(
            "Peak Amplitude", f"{peak:.4f} ({peak_db:.1f} dB)", peak_status
        )
    else:
        stats_table.add_row("Peak Amplitude", "N/A", "")

    # DC offset
    dc = info.get("dc_offset", 0)
    dc_status = "✓ OK" if abs(dc) < 0.001 else "⚠ Offset"
    stats_table.add_row("DC Offset", f"{dc:.6f}", dc_status)

    # Normalization
    stats_table.add_row(
        "Normalized",
        "Yes" if info.get("is_normalized") else "No",
        "✓" if info.get("is_normalized") else "⚠",
    )

    # Clipping
    if info.get("has_clipping"):
        clip_pct = info.get("clipping_percentage", 0)
        stats_table.add_row(
            "Clipping",
            f"Yes ({clip_pct:.1f}% samples)",
            "⚠" if clip_pct > 0.5 else "⚠ Minor",
        )
    else:
        stats_table.add_row("Clipping", "No", "✓")

    # Silence ratio
    silence = info.get("silence_ratio")
    if silence is not None:
        silence_status = "⚠" if silence > 0.5 else "✓"
        stats_table.add_row("Silence Ratio", f"{silence * 100:.1f}%", silence_status)
    else:
        stats_table.add_row("Silence Ratio", "N/A", "")

    # SNR
    snr = info.get("estimated_snr")
    if snr is not None:
        snr_status = "✓ Good" if snr > 20 else "⚠ Low" if snr > 10 else "✗ Poor"
        stats_table.add_row("Est. SNR", f"{snr:.1f} dB", snr_status)
    else:
        stats_table.add_row("Est. SNR", "N/A", "")

    console.print(stats_table)

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
                    title="Waveform Preview (Peak Normalized)",
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
    if info.get("rms_amplitude", 0) < 0.005:
        vad_score -= 20
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
