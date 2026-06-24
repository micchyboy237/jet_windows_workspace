from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Optional, List, Union
import librosa
import numpy as np
import soundfile as sf
import torch
import logging

from audio_utils import AudioInput, get_audio_duration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_audio_segments(
    audio_segments: List[AudioInput],
    output_path: Union[str, os.PathLike],
    *,
    sample_rate: Optional[int] = None,
    gap_duration: float = 0.0,
    crossfade_duration: float = 0.0,
    target_sample_rate: int = 44100,
    normalize: bool = True,
) -> None:
    """
    Combine multiple audio inputs into a single audio file.
    
    Args:
        audio_segments: List of audio inputs (file paths, bytes, numpy arrays, or torch tensors)
        output_path: Path to save the combined audio file
        sample_rate: Sample rate for array/tensor inputs (required if any segment is array/tensor)
        gap_duration: Silence duration (in seconds) between segments
        crossfade_duration: Crossfade duration (in seconds) between overlapping segments
        target_sample_rate: Output sample rate for the combined audio (default: 44100)
        normalize: Whether to normalize the final audio to prevent clipping (default: True)
    
    Raises:
        ValueError: If audio_segments is empty or sample_rate is missing for array inputs
        FileNotFoundError: If any audio file doesn't exist
        TypeError: If any audio input type is unsupported
    """
    if not audio_segments:
        raise ValueError("audio_segments list cannot be empty")
    
    logger.info(f"Starting audio combination with {len(audio_segments)} segments")
    logger.info(f"Target sample rate: {target_sample_rate} Hz")
    logger.info(f"Gap between segments: {gap_duration}s")
    logger.info(f"Crossfade duration: {crossfade_duration}s")
    
    # Step 1: Load and standardize all audio segments
    processed_segments = []
    segment_sample_rates = []
    
    for i, audio in enumerate(audio_segments):
        logger.info(f"Processing segment {i + 1}/{len(audio_segments)}")
        
        # Load audio data and get its sample rate
        audio_data, sr = _load_audio_to_mono(audio, sample_rate)
        segment_sample_rates.append(sr)
        
        # Resample to target sample rate if needed
        if sr != target_sample_rate:
            logger.info(f"Resampling segment {i + 1} from {sr}Hz to {target_sample_rate}Hz")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sr, 
                target_sr=target_sample_rate
            )
        
        processed_segments.append(audio_data)
        logger.info(f"Segment {i + 1} duration: {len(audio_data)/target_sample_rate:.2f}s")
    
    # Step 2: Calculate total duration and create output array
    segment_durations = [len(seg) / target_sample_rate for seg in processed_segments]
    total_gap_samples = int(gap_duration * target_sample_rate * (len(processed_segments) - 1))
    
    # Calculate crossfade adjustments
    crossfade_samples = int(crossfade_duration * target_sample_rate)
    
    if crossfade_samples > 0:
        # With crossfade, total length is shorter due to overlap
        overlap_reduction = crossfade_samples * (len(processed_segments) - 1)
        total_samples = sum(len(seg) for seg in processed_segments) + total_gap_samples - overlap_reduction
    else:
        total_samples = sum(len(seg) for seg in processed_segments) + total_gap_samples
    
    combined_audio = np.zeros(total_samples, dtype=np.float32)
    
    # Step 3: Combine segments with optional gap and crossfade
    current_position = 0
    
    for i, segment in enumerate(processed_segments):
        segment_length = len(segment)
        
        if i > 0 and crossfade_samples > 0:
            # Apply crossfade with previous segment
            logger.info(f"Applying crossfade between segment {i} and {i + 1}")
            
            # Calculate overlap region
            overlap_start = current_position - crossfade_samples
            
            # Create fade curves
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            
            # Apply crossfade
            combined_audio[overlap_start:overlap_start + crossfade_samples] *= fade_out
            combined_audio[overlap_start:overlap_start + crossfade_samples] += (
                segment[:crossfade_samples] * fade_in
            )
            
            # Add remaining part of current segment
            remaining_length = segment_length - crossfade_samples
            combined_audio[current_position:current_position + remaining_length] = (
                segment[crossfade_samples:]
            )
            
            current_position += remaining_length
            
        else:
            # No crossfade, just append the segment
            combined_audio[current_position:current_position + segment_length] = segment
            current_position += segment_length
        
        # Add gap after segment (except for the last one)
        if i < len(processed_segments) - 1 and gap_duration > 0:
            current_position += total_gap_samples // (len(processed_segments) - 1)
    
    # Step 4: Normalize if requested
    if normalize:
        max_amplitude = np.max(np.abs(combined_audio))
        if max_amplitude > 0:
            logger.info(f"Normalizing audio (max amplitude: {max_amplitude:.4f})")
            combined_audio = combined_audio / max_amplitude * 0.95  # Leave 5% headroom
        else:
            logger.warning("Audio contains only silence, skipping normalization")
    
    # Step 5: Save the combined audio
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving combined audio to: {output_path}")
    sf.write(output_path, combined_audio, target_sample_rate, subtype='PCM_16')
    
    total_duration = len(combined_audio) / target_sample_rate
    logger.info(f"Successfully created combined audio file")
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info(f"Sample rate: {target_sample_rate} Hz")
    logger.info(f"Number of segments: {len(audio_segments)}")


def _load_audio_to_mono(
    audio: AudioInput,
    sample_rate: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Load audio from various input types and convert to mono.
    
    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        sample_rate: Sample rate for array/tensor inputs
        
    Returns:
        Tuple of (audio_data as mono float32 array, sample_rate)
    
    Raises:
        ValueError: If sample_rate is missing for array inputs
        FileNotFoundError: If audio file doesn't exist
        TypeError: If audio input type is unsupported
    """
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        logger.debug(f"Loading audio file: {path}")
        data, sr = librosa.load(path, sr=None, mono=True)
        return data.astype(np.float32), sr
        
    elif isinstance(audio, bytes):
        logger.debug("Loading audio from bytes")
        buffer = io.BytesIO(audio)
        data, sr = librosa.load(buffer, sr=None, mono=True)
        return data.astype(np.float32), sr
        
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy array input")
        logger.debug(f"Processing numpy array input (shape: {audio.shape})")
        # Convert to mono if multi-channel
        if audio.ndim > 1:
            data = librosa.to_mono(audio.astype(np.float32, copy=False).T)
        else:
            data = audio.astype(np.float32, copy=False)
        return data, sample_rate
        
    elif torch is not None and isinstance(audio, torch.Tensor):
        if sample_rate is None:
            raise ValueError("sample_rate is required for torch.Tensor input")
        logger.debug(f"Processing torch tensor input (shape: {audio.shape})")
        data = audio.detach().cpu().numpy()
        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = librosa.to_mono(data.astype(np.float32, copy=False).T)
        else:
            data = data.astype(np.float32, copy=False)
        return data, sample_rate
        
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")


if __name__ == "__main__":
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(description="Combine multiple audio segments into one file")
    parser.add_argument(
        "input_audios",
        nargs="+",
        help="Paths to audio files to combine (space-separated)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for combined audio (default: generated/combine_audio_segments/combined.wav)"
    )
    parser.add_argument(
        "-g", "--gap",
        type=float,
        default=0.0,
        help="Gap duration between segments in seconds (default: 0.0)"
    )
    parser.add_argument(
        "-c", "--crossfade",
        type=float,
        default=0.0,
        help="Crossfade duration between segments in seconds (default: 0.0)"
    )
    parser.add_argument(
        "-r", "--sample-rate",
        type=int,
        default=44100,
        help="Target sample rate for combined audio (default: 44100)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio normalization"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output if args.output else str(OUTPUT_DIR / "combined.wav")
    
    # Demo: Create test audio segments if no input files provided
    if not args.input_audios:
        logger.info("No input files provided. Creating demo segments...")
        sample_rate = 44100
        
        # Create three test tones
        t1 = np.linspace(0, 1, sample_rate, endpoint=False)
        t2 = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        t3 = np.linspace(0, 0.5, int(0.5 * sample_rate), endpoint=False)
        
        segment1 = np.sin(2 * np.pi * 440 * t1).astype(np.float32)  # A4 note, 1 second
        segment2 = np.sin(2 * np.pi * 880 * t2).astype(np.float32)  # A5 note, 2 seconds
        segment3 = np.sin(2 * np.pi * 220 * t3).astype(np.float32)  # A3 note, 0.5 seconds
        
        logger.info("Combining test tones with gap and crossfade...")
        combine_audio_segments(
            audio_segments=[segment1, segment2, segment3],
            output_path=output_path,
            sample_rate=sample_rate,
            gap_duration=0.2,
            crossfade_duration=0.1,
            target_sample_rate=sample_rate,
            normalize=True
        )
    else:
        # Combine provided audio files
        combine_audio_segments(
            audio_segments=args.input_audios,
            output_path=output_path,
            gap_duration=args.gap,
            crossfade_duration=args.crossfade,
            target_sample_rate=args.sample_rate,
            normalize=not args.no_normalize
        )
    
    print(f"\nCombined audio saved at: {output_path}")
    
    # Verify the output
    if Path(output_path).exists():
        duration = get_audio_duration(output_path)
        file_size = Path(output_path).stat().st_size
        print(f"Output duration: {duration:.2f}s")
        print(f"File size: {file_size:,} bytes")
