# speech_waves.py

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from rich.console import Console

try:
    from services._types import AudioInput, SpeechWave
    from services.audio_config import HOP_SIZE, SAMPLE_RATE
    from services.energy import compute_rms_per_frame
    from services.audio_utils import load_audio
    from services.norm_speech_loudness import normalize_audio_for_vad
    from services.vad_firered import extract_speech_timestamps
    from services.dtype_conversion import convert_audio_dtype
except ImportError:
    from _types import AudioInput, SpeechWave
    from audio_config import HOP_SIZE, SAMPLE_RATE
    from energy import compute_rms_per_frame
    from audio_utils import load_audio
    from norm_speech_loudness import normalize_audio_for_vad
    from vad_firered import extract_speech_timestamps
    from dtype_conversion import convert_audio_dtype

DEFAULT_THRESHOLD = 0.3

DEFAULT_MIN_PROMINENCE = 0.05
DEFAULT_MIN_EXCURSION = 0.04
DEFAULT_MIN_PEAK_PROB = 0.55
DEFAULT_MIN_FRAMES = 3
DEFAULT_MIN_DURATION_SEC = 1.0
DEFAULT_BASELINE_THRESHOLD = 0.1

DEFAULT_MIN_SPEECH_DURATION_MS = 1000
DEFAULT_MIN_SILENCE_DURATION_MS = 100

WaveState = Literal["below", "above"]

console = Console()

@dataclasses.dataclass
class WaveShapeConfig:
    """
    Tunable thresholds that decide whether a probability wave has a real
    mountain shape rather than being a flat plateau or a tiny ripple.

    Attributes:
        min_prominence: How much the peak must rise above the average of the
            two surrounding valley endpoints.
        min_excursion: The minimum difference between the highest and lowest
            probability inside the wave window.
        min_peak_prob: Absolute floor — the peak frame must reach at least
            this probability (guards against waves that never really fire).
        min_frames: Waves shorter than this many frames are discarded.
        min_duration_sec: Minimum wall-clock duration in seconds. Waves
            shorter than this are rejected even if they pass frame and shape
            checks. Derived independently of min_frames so both constraints
            must be satisfied.
        baseline_threshold: Probability threshold used to determine when a
            wave has truly fallen back to baseline/silence level. Used to
            detect wave boundaries and preroll adjustments.
    """

    min_prominence: float = DEFAULT_MIN_PROMINENCE
    min_excursion: float = DEFAULT_MIN_EXCURSION
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB
    min_frames: int = DEFAULT_MIN_FRAMES
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD


def is_prominent_wave(
    wave_probs: List[float],
    entry_prob: float,
    exit_prob: float,
    cfg: WaveShapeConfig,
) -> tuple[bool, dict]:
    """
    Decide whether a slice of probabilities forms a genuine mountain shape.

    The algorithm:
      1. Baseline = average of entry_prob and exit_prob (the "ground level").
      2. Peak     = maximum probability inside the slice.
      3. Prominence = peak - baseline.
      4. Excursion  = max - min inside the slice (vertical range).

    Returns:
        (passed: bool, diagnostics: dict)
    """
    if not wave_probs:
        return False, {}

    peak_prob = max(wave_probs)
    min_prob = min(wave_probs)
    baseline = (entry_prob + exit_prob) / 2.0
    prominence = peak_prob - baseline
    excursion = peak_prob - min_prob
    n_frames = len(wave_probs)

    passed = (
        prominence >= cfg.min_prominence
        and excursion >= cfg.min_excursion
        and peak_prob >= cfg.min_peak_prob
        and n_frames >= cfg.min_frames
    )

    diagnostics = {
        "baseline": round(baseline, 6),
        "peak_prob": round(peak_prob, 6),
        "prominence": round(prominence, 6),
        "excursion": round(excursion, 6),
        "n_frames": n_frames,
        "shape_passed": passed,
    }
    return passed, diagnostics


def check_speech_waves(
    speech_probs: List[float],
    threshold: float = DEFAULT_THRESHOLD,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> List[SpeechWave]:
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()

    if not speech_probs:
        return []

    waves: List[SpeechWave] = []
    current_wave: SpeechWave | None = None
    state: WaveState = "below"
    rise_frame_idx: int | None = None

    if speech_probs:
        if speech_probs[0] < shape_cfg.baseline_threshold:
            current_wave = SpeechWave(
                has_risen=False,
                has_multi_passed=False,
                has_fallen=False,
                is_valid=False,
                start_sec=0.0,
                end_sec=0.0,
                details={
                    "frame_start": 0,
                    "frame_end": 0,
                    "frame_len": 0,
                    "duration_sec": 0.0,
                    "min_prob": speech_probs[0],
                    "max_prob": speech_probs[0],
                    "avg_prob": speech_probs[0],
                    "std_prob": 0.0,
                    "composite_score": 0.0,
                },
            )
            state = "below"

        elif speech_probs[0] >= threshold:
            state = "above"

    for i, prob in enumerate(speech_probs):
        frame_time_sec = i * HOP_SIZE / sampling_rate

        if state == "below":
            if prob >= threshold:
                rise_frame_idx = i

                # ── Preroll: walk back from rise_frame_idx until we find a
                #    frame strictly below baseline_threshold (or hit index 0).
                preroll_start = rise_frame_idx
                while (
                    preroll_start > 0
                    and speech_probs[preroll_start - 1] >= shape_cfg.baseline_threshold
                ):
                    preroll_start -= 1
                preroll_start_sec = preroll_start * HOP_SIZE / sampling_rate

                current_wave = SpeechWave(
                    has_risen=current_wave["has_risen"] if current_wave else True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=preroll_start_sec,
                    end_sec=preroll_start_sec,
                    details={
                        "frame_start": preroll_start,
                        "frame_end": preroll_start,
                        "frame_len": 0,
                        "duration_sec": 0.0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "avg_prob": prob,
                        "std_prob": 0.0,
                        "composite_score": 0.0,
                    },
                )

                state = "above"
        else:
            if prob >= threshold:
                if current_wave is not None:
                    current_wave["has_multi_passed"] = True
            else:
                if current_wave is not None:
                    if prob <= shape_cfg.baseline_threshold:
                        current_wave["has_fallen"] = True

                    # frame_start uses the preroll-adjusted value stored in details
                    frame_start = current_wave["details"]["frame_start"]
                    frame_end = i
                    wave_probs = speech_probs[frame_start:frame_end]
                    frame_len = frame_end - frame_start

                    # entry_prob: the frame immediately before the preroll start
                    entry_prob = (
                        speech_probs[frame_start - 1] if frame_start > 0 else 0.0
                    )
                    exit_prob = prob

                    shape_ok, shape_diag = is_prominent_wave(
                        wave_probs, entry_prob, exit_prob, shape_cfg
                    )

                    duration_sec = frame_time_sec - current_wave["start_sec"]
                    duration_ok = duration_sec >= shape_cfg.min_duration_sec

                    current_wave["is_valid"] = (
                        current_wave["has_risen"]
                        and current_wave["has_multi_passed"]
                        and current_wave["has_fallen"]
                        and shape_ok
                        and duration_ok
                    )
                    current_wave["end_sec"] = frame_time_sec
                    current_wave["details"] = {
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "frame_len": frame_len,
                        "duration_sec": duration_sec,
                        "min_prob": min(wave_probs) if wave_probs else 0.0,
                        "max_prob": max(wave_probs) if wave_probs else 0.0,
                        "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                        "std_prob": statistics.stdev(wave_probs)
                        if frame_len > 1
                        else 0.0,
                        "duration_ok": duration_ok,
                        **shape_diag,
                        "composite_score": 0.0,
                    }
                    current_wave["details"]["composite_score"] = (
                        compute_composite_score(current_wave)
                    )

                # FIX: Only append if current_wave is not None
                if prob < shape_cfg.baseline_threshold:
                    if current_wave is not None:
                        waves.append(current_wave)
                    current_wave = None
                    rise_frame_idx = None
                    state = "below"

    # FIX: Handle a wave that never fell back below the threshold
    # Guard ensures we only append if current_wave exists
    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False
        current_wave["end_sec"] = len(speech_probs) * HOP_SIZE / sampling_rate

        if rise_frame_idx is not None:
            # frame_start is already preroll-adjusted in details
            frame_start = current_wave["details"]["frame_start"]
            frame_end = len(speech_probs)
            wave_probs = speech_probs[frame_start:frame_end]
            frame_len = frame_end - frame_start
            duration_sec = current_wave["end_sec"] - current_wave["start_sec"]
            entry_prob = speech_probs[frame_start - 1] if frame_start > 0 else 0.0
            exit_prob = threshold
            shape_ok, shape_diag = is_prominent_wave(
                wave_probs, entry_prob, exit_prob, shape_cfg
            )
            current_wave["details"] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_len": frame_len,
                "duration_sec": duration_sec,
                "min_prob": min(wave_probs) if wave_probs else 0.0,
                "max_prob": max(wave_probs) if wave_probs else 0.0,
                "avg_prob": statistics.mean(wave_probs) if wave_probs else 0.0,
                "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
                "duration_ok": False,
                **shape_diag,
                "composite_score": 0.0,
            }
            current_wave["details"]["composite_score"] = compute_composite_score(
                current_wave
            )

        waves.append(current_wave)

    return waves


def get_speech_waves(
    audio: AudioInput,
    speech_probs: List[float],
    threshold: float = DEFAULT_THRESHOLD,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
    with_audio: bool = False,
) -> List[SpeechWave] | List[Tuple[SpeechWave, np.ndarray]]:
    """
    Identify complete speech waves (rise → sustained high → fall) from FireRedVAD probabilities.

    Follows the same pipeline as _main_speech_waves.main():
      1. Runs shape analysis on pre-computed VAD scores
      2. Filters to valid waves
      3. Optionally loads audio and extracts segments

    Args:
        audio: Audio input (file path, bytes, numpy array, or torch tensor)
        speech_probs: Speech probability scores from VAD
        threshold: VAD probability threshold
        sampling_rate: Audio sample rate in Hz
        shape_cfg: Configuration for wave shape validation (defaults to WaveShapeConfig())
        with_audio: If True, returns list of tuples (SpeechWave, np.ndarray) with 
                   the audio data for each wave extracted from the loaded audio

    Returns:
        If with_audio=False: List[SpeechWave] containing valid speech waves
        If with_audio=True: List[Tuple[SpeechWave, np.ndarray]] containing valid 
                           speech waves paired with their audio segments
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"get_speech_waves called with with_audio={with_audio}, threshold={threshold}"
    )

    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()
        logger.debug(f"Using default WaveShapeConfig: {shape_cfg}")

    # Step 1: Shape analysis on existing scores (same as _main_speech_waves)
    all_waves = check_speech_waves(
        speech_probs=speech_probs,
        threshold=threshold,
        sampling_rate=sampling_rate,
        shape_cfg=shape_cfg,
    )
    logger.info(f"Total waves detected: {len(all_waves)}")

    # Step 2: Filter to valid waves only
    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave.get("is_valid", False):
            valid_waves.append(wave)

    logger.info(f"Valid waves (without audio): {len(valid_waves)}")

    # Step 3: Return early if audio extraction not requested
    if not with_audio:
        return valid_waves

    # Step 4: Load audio only when needed for extraction
    loaded_audio, loaded_sr = load_audio(audio, sr=sampling_rate, mono=True)
    logger.debug(f"Audio loaded for extraction: shape={loaded_audio.shape}, sr={loaded_sr}")

    # Step 5: Extract audio segments
    valid_waves_with_audio: List[Tuple[SpeechWave, np.ndarray]] = []
    for wave in valid_waves:
        frame_start = wave["details"]["frame_start"]
        frame_end = wave["details"]["frame_end"]

        start_sample = frame_start * HOP_SIZE
        end_sample = (frame_end + 1) * HOP_SIZE
        start_sample = max(0, start_sample)
        end_sample = min(len(loaded_audio), end_sample)

        if end_sample > start_sample:
            wave_audio = loaded_audio[start_sample:end_sample].copy()
            valid_waves_with_audio.append((wave, wave_audio))
            logger.debug(
                f"Wave audio extracted: frames [{frame_start}:{frame_end}], "
                f"samples [{start_sample}:{end_sample}], "
                f"duration={wave['details']['duration_sec']:.3f}s"
            )

    logger.info(f"Valid waves (with audio): {len(valid_waves_with_audio)}")
    return valid_waves_with_audio


def get_valid_speech_waves(
    audio: AudioInput,
    sampling_rate: int = SAMPLE_RATE,
    vad_threshold: float = DEFAULT_THRESHOLD,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_excursion: float = DEFAULT_MIN_EXCURSION,
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    with_audio: bool = False,
    with_scores: bool = False,
) -> Union[
    List[SpeechWave],
    List[Tuple[SpeechWave, np.ndarray]],
    Tuple[List[SpeechWave], List[float]],
    Tuple[List[Tuple[SpeechWave, np.ndarray]], List[float]],
]:
    """
    Identify valid speech waves from audio using VAD and shape analysis.

    This function follows the same pipeline as _main_speech_waves.main():
      1. Loads audio (accepts file path, bytes, numpy array, or torch tensor)
      2. Runs VAD (extract_speech_timestamps) to get probability scores
      3. Identifies speech waves via shape analysis (check_speech_waves)
      4. Filters to only valid (is_valid=True) waves
      5. Optionally extracts audio segments for each wave
      6. Optionally returns VAD probability scores

    All parameters default to the module-level DEFAULT_* constants,
    matching the all-defaults usage in _main_speech_waves.

    Args:
        audio: Audio input — file path (str/Path), bytes, numpy array, or torch tensor.
               Accepts the same types as load_audio() (AudioInput union).
        sampling_rate: Audio sampling rate in Hz (used when audio is not a file)
        vad_threshold: VAD probability threshold (above = speech)
        min_prominence: Minimum peak prominence above baseline
        min_excursion: Minimum peak-to-valley excursion
        min_peak_prob: Minimum peak probability
        min_frames: Minimum frames per wave
        min_duration_sec: Minimum wave duration in seconds
        baseline_threshold: Probability threshold for silence/baseline
        min_speech_duration_ms: Minimum speech segment for VAD
        min_silence_duration_ms: Minimum silence gap for VAD
        with_audio: If True, returns list of tuples (SpeechWave, np.ndarray)
                   with the audio data for each wave extracted from the input audio
        with_scores: If True, also returns the raw VAD probability scores as a
                    second element in a tuple

    Returns:
        If with_audio=False and with_scores=False:
            List[SpeechWave] containing valid speech waves.
        If with_audio=True and with_scores=False:
            List[Tuple[SpeechWave, np.ndarray]] containing valid speech waves
            paired with their audio segments.
        If with_audio=False and with_scores=True:
            Tuple[List[SpeechWave], List[float]] containing valid speech waves
            and VAD probability scores.
        If with_audio=True and with_scores=True:
            Tuple[List[Tuple[SpeechWave, np.ndarray]], List[float]] containing
            valid speech waves with audio and VAD probability scores.
        Returns empty list if no valid speech found or VAD fails.

    Example:
        >>> # Basic usage
        >>> waves = get_valid_speech_waves("recording.wav")
        >>>
        >>> # With scores
        >>> waves, scores = get_valid_speech_waves("recording.wav", with_scores=True)
        >>> print(f"Found {len(waves)} waves, {len(scores)} probability scores")
        >>>
        >>> # With audio extraction
        >>> waves_with_audio = get_valid_speech_waves("recording.wav", with_audio=True)
        >>> for wave, audio_chunk in waves_with_audio:
        ...     print(f"Duration: {wave['details']['duration_sec']:.2f}s")
        >>>
        >>> # With both audio and scores
        >>> waves_with_audio, scores = get_valid_speech_waves(
        ...     "recording.wav",
        ...     with_audio=True,
        ...     with_scores=True
        ... )
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"get_valid_speech_waves called with with_audio={with_audio}, "
        f"with_scores={with_scores}, vad_threshold={vad_threshold}, "
        f"min_duration={min_duration_sec}s"
    )

    shape_cfg = WaveShapeConfig(
        min_prominence=min_prominence,
        min_excursion=min_excursion,
        min_peak_prob=min_peak_prob,
        min_frames=min_frames,
        min_duration_sec=min_duration_sec,
        baseline_threshold=baseline_threshold,
    )
    logger.debug(f"WaveShapeConfig: {shape_cfg}")

    audio_np, sr = load_audio(audio, sr=sampling_rate, mono=True)
    logger.debug(
        f"Audio loaded: shape={audio_np.shape}, sr={sr}, dtype={audio_np.dtype}"
    )

    try:
        _, scores = extract_speech_timestamps(
            audio=audio_np,
            include_non_speech=False,
            threshold=vad_threshold,
            min_speech_duration_sec=min_speech_duration_ms / 1000.0,
            min_silence_duration_sec=min_silence_duration_ms / 1000.0,
            with_scores=True,
        )
    except Exception as e:
        logger.error(f"VAD extraction failed: {e}")
        console.print(f"[error]VAD extraction failed: {e}[/error]")
        if with_scores:
            return [], []
        return []

    if not scores:
        logger.warning("No speech scores returned from VAD")
        if with_scores:
            return [], []
        return []

    logger.info(f"VAD produced {len(scores)} probability scores")

    all_waves = check_speech_waves(
        speech_probs=scores,
        threshold=vad_threshold,
        sampling_rate=sr,
        shape_cfg=shape_cfg,
    )
    logger.info(f"Total waves detected by shape analysis: {len(all_waves)}")

    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave is None:
            continue
        if not isinstance(wave, dict):
            continue
        if not wave.get("is_valid", False):
            continue
        valid_waves.append(wave)

    logger.info(f"Valid waves after filtering: {len(valid_waves)}")

    # Handle all return type combinations
    if not with_audio and not with_scores:
        return valid_waves

    if with_audio:
        valid_waves_with_audio: List[Tuple[SpeechWave, np.ndarray]] = []
        for wave in valid_waves:
            frame_start = wave["details"]["frame_start"]
            frame_end = wave["details"]["frame_end"]
            start_sample = frame_start * HOP_SIZE
            end_sample = (frame_end + 1) * HOP_SIZE
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_np), end_sample)
            if end_sample > start_sample:
                wave_audio = audio_np[start_sample:end_sample].copy()
                valid_waves_with_audio.append((wave, wave_audio))
                logger.debug(
                    f"Wave audio extracted: frames [{frame_start}:{frame_end}], "
                    f"samples [{start_sample}:{end_sample}], "
                    f"duration={wave['details']['duration_sec']:.3f}s"
                )
            else:
                logger.warning(
                    f"Skipping wave with invalid sample range: "
                    f"frames [{frame_start}:{frame_end}] → "
                    f"samples [{start_sample}:{end_sample}] "
                    f"(audio_np length={len(audio_np)})"
                )
        logger.info(f"Valid waves (with audio): {len(valid_waves_with_audio)}")

        if with_scores:
            logger.info("Returning waves with audio AND scores")
            return valid_waves_with_audio, scores
        return valid_waves_with_audio

    # with_scores=True, with_audio=False
    logger.info("Returning waves with scores (no audio)")
    return valid_waves, scores


def extract_pure_speech_segments(
    audio: np.ndarray,
    sampling_rate: int = SAMPLE_RATE,
    hop_size: int = HOP_SIZE,
    vad_threshold: float = DEFAULT_THRESHOLD,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_excursion: float = DEFAULT_MIN_EXCURSION,
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
) -> List[np.ndarray]:
    """
    Extract high-confidence speech audio segments from a raw waveform.
    
    This function:
    1. Calls get_valid_speech_waves to identify valid speech waves
    2. Extracts audio samples for each wave based on frame indices
    
    Args:
        audio: Raw audio as numpy array (int16 or float32)
        sampling_rate: Audio sampling rate in Hz
        hop_size: Frame hop size for VAD processing
        vad_threshold: VAD probability threshold (above = speech)
        min_prominence: Minimum peak prominence above baseline
        min_excursion: Minimum peak-to-valley excursion
        min_peak_prob: Minimum peak probability
        min_frames: Minimum frames per wave
        min_duration_sec: Minimum wave duration in seconds
        baseline_threshold: Probability threshold for silence/baseline
        min_speech_duration_ms: Minimum speech segment for VAD
        min_silence_duration_ms: Minimum silence gap for VAD
    
    Returns:
        List[np.ndarray]: List of speech audio segments (same dtype as input).
        Returns empty list if no valid speech found.
    """
    # Get valid speech waves
    valid_waves = get_valid_speech_waves(
        audio=audio,
        sampling_rate=sampling_rate,
        vad_threshold=vad_threshold,
        min_prominence=min_prominence,
        min_excursion=min_excursion,
        min_peak_prob=min_peak_prob,
        min_frames=min_frames,
        min_duration_sec=min_duration_sec,
        baseline_threshold=baseline_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    
    if not valid_waves:
        return []
    
    # Extract audio segments from each valid wave
    speech_segments = []
    for wave in valid_waves:
        frame_start = wave["details"]["frame_start"]
        frame_end = wave["details"]["frame_end"]
        
        start_sample = frame_start * hop_size
        end_sample = (frame_end + 1) * hop_size
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if end_sample > start_sample:
            speech_segments.append(audio[start_sample:end_sample])
    
    return speech_segments


def extract_pure_speech_audio(
    audio: np.ndarray,
    sampling_rate: int = SAMPLE_RATE,
    hop_size: int = HOP_SIZE,
    vad_threshold: float = DEFAULT_THRESHOLD,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_excursion: float = DEFAULT_MIN_EXCURSION,
    min_peak_prob: float = DEFAULT_MIN_PEAK_PROB,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
) -> np.ndarray:
    """
    Extract high-confidence speech audio from a raw waveform and concatenate.
    
    This is a convenience wrapper that:
    1. Calls extract_pure_speech_segments to get individual speech segments
    2. Concatenates them into a single audio array
    
    Args:
        audio: Raw audio as numpy array (int16 or float32)
        sampling_rate: Audio sampling rate in Hz
        hop_size: Frame hop size for VAD processing
        vad_threshold: VAD probability threshold (above = speech)
        min_prominence: Minimum peak prominence above baseline
        min_excursion: Minimum peak-to-valley excursion
        min_peak_prob: Minimum peak probability
        min_frames: Minimum frames per wave
        min_duration_sec: Minimum wave duration in seconds
        baseline_threshold: Probability threshold for silence/baseline
        min_speech_duration_ms: Minimum speech segment for VAD
        min_silence_duration_ms: Minimum silence gap for VAD
    
    Returns:
        numpy.ndarray: Combined pure speech audio (same dtype as input).
        Returns empty array if no valid speech found.
    """
    # Convert dtype
    audio_int16 = convert_audio_dtype(audio, "int16")
    audio = audio_int16

    # Get individual speech segments
    speech_segments = extract_pure_speech_segments(
        audio=audio,
        sampling_rate=sampling_rate,
        hop_size=hop_size,
        vad_threshold=vad_threshold,
        min_prominence=min_prominence,
        min_excursion=min_excursion,
        min_peak_prob=min_peak_prob,
        min_frames=min_frames,
        min_duration_sec=min_duration_sec,
        baseline_threshold=baseline_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    
    # Concatenate and return
    if not speech_segments:
        return np.array([], dtype=audio.dtype)
    
    return np.concatenate(speech_segments)


def save_wave_audio(
    audio_np: np.ndarray,
    sampling_rate: int,
    frame_start: int,
    frame_end: int,
    output_path: Path,
    hop_size: int = HOP_SIZE,
) -> None:
    """Extract and save audio chunk for a wave based on frame indices."""
    start_sample = frame_start * hop_size
    end_sample = (frame_end + 1) * hop_size
    wave_audio = audio_np[start_sample:end_sample]
    wavfile.write(output_path, sampling_rate, wave_audio)


def compute_composite_score(wave: SpeechWave) -> float:
    """
    Composite quality score for ranking speech waves.

    Formula:
        score = avg_prob * prominence * log1p(duration_sec) * (1 + 0.3 * excursion)

    Rationale for each term:
    - avg_prob: rewards sustained confidence across the whole wave, not just
      a single spike; a wave hovering at 0.95 outranks one that spikes once
      and sits at 0.55.
    - prominence: the mountain height above the noise floor (peak minus
      baseline); guards against flat plateaus that happen to be above threshold.
    - log1p(duration_sec): duration reward with diminishing returns so long
      but featureless segments don't dominate short, sharp utterances.
      log1p(1 s) ≈ 0.69, log1p(3 s) ≈ 1.39, log1p(10 s) ≈ 2.40.
    - (1 + 0.3 * excursion): small multiplicative bonus for shape sharpness;
      high excursion means the wave truly rises and falls rather than
      lingering as a flat plateau. Coefficient 0.3 caps the bonus at ×1.3
      (when excursion = 1.0) so it modulates rather than dominates.
    """
    d = wave["details"]
    avg_prob = d.get("avg_prob", 0.0)
    prominence = d.get("prominence", d["max_prob"])
    duration_sec = d.get("duration_sec", 0.0)
    excursion = d.get("excursion", 0.0)
    return avg_prob * prominence * math.log1p(duration_sec) * (1.0 + 0.3 * excursion)


def save_wave_plot(
    probs: List[float],
    rms_values: List[float],
    output_path: Path,
    wave_num: int,
    seg_num: int,
    wave: Optional[SpeechWave] = None,
    threshold: float = DEFAULT_THRESHOLD,
    hop_size: int = HOP_SIZE,
    sampling_rate: int = SAMPLE_RATE,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> None:
    """
    Create a two-panel visualization for a single speech wave.

    Top panel — VAD probability:
    - X-axis in milliseconds (real time, not frame index)
    - Above-threshold region shaded in light blue
    - Vertical dashed markers at wave start and end
    - Baseline shown as a horizontal dashed line with label
    - Peak annotated with a dot and probability label
    - Metric text-box: peak, avg, prominence, excursion, baseline, composite,
      duration (drawn in the upper-right corner so it never overlaps the curve)

    Bottom panel — RMS energy:
    - Normalised to [0, 1] within the plot window for readability at any
      absolute amplitude; annotated with "(normalised)" on the y-axis
    - Same x-axis and time markers as the top panel
    """
    if shape_cfg is None:
        shape_cfg = WaveShapeConfig()

    baseline_threshold = shape_cfg.baseline_threshold

    # --- align arrays --------------------------------------------------------
    min_length = min(len(probs), len(rms_values))
    probs_aligned = probs[:min_length]
    rms_aligned = rms_values[:min_length]

    # Convert frame indices to milliseconds
    ms_per_frame = hop_size / sampling_rate * 1000.0
    frames = np.arange(min_length)
    times_ms = frames * ms_per_frame

    # --- pull wave metadata --------------------------------------------------
    d = wave["details"] if wave is not None else {}
    peak_prob = d.get("max_prob", max(probs_aligned) if probs_aligned else 0.0)
    avg_prob = d.get("avg_prob", 0.0)
    prominence = d.get("prominence", 0.0)
    excursion = d.get("excursion", 0.0)
    baseline = d.get("baseline", 0.0)
    duration_s = d.get("duration_sec", min_length * hop_size / sampling_rate)
    composite = compute_composite_score(wave) if wave is not None else 0.0

    # Wave start/end in milliseconds relative to the wave window origin
    # (frame_start is absolute; the slice already starts there, so t=0 in
    # the plot is the wave's own first frame)
    wave_start_ms = 0.0
    wave_end_ms = duration_s * 1000.0

    # --- normalise RMS -------------------------------------------------------
    rms_arr = np.array(rms_aligned, dtype=float)
    rms_max = rms_arr.max() if rms_arr.size and rms_arr.max() > 0 else 1.0
    rms_norm = rms_arr / rms_max

    # --- figure setup --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6]},
    )
    fig.subplots_adjust(hspace=0.08, left=0.09, right=0.97, top=0.92, bottom=0.11)

    # ── TOP PANEL: VAD probability ──────────────────────────────────────────
    # Above-threshold shading
    ax1.fill_between(
        times_ms,
        probs_aligned,
        threshold,
        where=[p >= threshold for p in probs_aligned],
        alpha=0.18,
        color="#2196F3",
        interpolate=True,
        label=None,
    )

    # Probability curve
    ax1.plot(times_ms, probs_aligned, color="#1565C0", linewidth=1.4, zorder=3)

    # Threshold line
    ax1.axhline(
        y=threshold,
        color="#E53935",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"Threshold ({threshold:.2f})",
    )

    # Baseline threshold line
    ax1.axhline(
        y=baseline_threshold,
        color="#6D4C41",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label=f"Baseline threshold ({baseline_threshold:.3f})",
    )

    # Wave start / end vertical markers
    ax1.axvline(
        wave_start_ms,
        color="#4CAF50",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Wave start",
    )
    ax1.axvline(
        wave_end_ms,
        color="#FF7043",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Wave end",
    )

    # Peak annotation
    if probs_aligned:
        peak_frame = int(np.argmax(probs_aligned))
        peak_ms = times_ms[peak_frame]
        ax1.plot(
            peak_ms,
            probs_aligned[peak_frame],
            "o",
            color="#E53935",
            markersize=5,
            zorder=5,
        )
        ax1.annotate(
            f"{probs_aligned[peak_frame]:.3f}",
            xy=(peak_ms, probs_aligned[peak_frame]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#E53935",
            zorder=6,
        )

    # Metric text-box (upper-right corner)
    metrics_text = (
        f"peak:        {peak_prob:.3f}\n"
        f"avg:         {avg_prob:.3f}\n"
        f"prominence:  {prominence:.3f}\n"
        f"excursion:   {excursion:.3f}\n"
        f"baseline:    {baseline:.3f}\n"
        f"duration:    {duration_s:.2f} s\n"
        f"composite:   {composite:.4f}"
    )
    ax1.text(
        0.985,
        0.97,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=7.5,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#BDBDBD",
            alpha=0.88,
            linewidth=0.6,
        ),
        zorder=7,
    )

    ax1.set_ylabel("VAD probability", fontsize=9)
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.85, edgecolor="#BDBDBD")
    ax1.set_title(
        f"Segment {seg_num:03d}  ·  Wave {wave_num:03d}  ·  {duration_s * 1000:.0f} ms",
        fontsize=10,
        pad=6,
    )

    # ── BOTTOM PANEL: normalised RMS energy ─────────────────────────────────
    ax2.fill_between(times_ms[: len(rms_norm)], rms_norm, alpha=0.25, color="#388E3C")
    ax2.plot(times_ms[: len(rms_norm)], rms_norm, color="#2E7D32", linewidth=1.2)

    ax2.axvline(
        wave_start_ms, color="#4CAF50", linestyle="--", linewidth=1.0, alpha=0.7
    )
    ax2.axvline(wave_end_ms, color="#FF7043", linestyle="--", linewidth=1.0, alpha=0.7)

    ax2.set_xlabel("Time (ms)", fontsize=9)
    ax2.set_ylabel("RMS energy\n(normalised)", fontsize=8)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_wave_data(
    wave: SpeechWave,
    audio_np: np.ndarray,
    speech_probs: List[float],
    sampling_rate: int,
    output_dir: Path,
    seg_num: int,
    wave_num: int,
    hop_size: int = HOP_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    shape_cfg: Optional[WaveShapeConfig] = None,
) -> None:
    """Save all wave-related data to the specified directory."""
    wave_dir = output_dir / f"segment_{seg_num:03d}_wave_{wave_num:03d}"
    wave_dir.mkdir(parents=True, exist_ok=True)

    # Extract frame info
    frame_start = wave["details"]["frame_start"]
    frame_end = wave["details"]["frame_end"]

    # Save wave audio
    wav_path = wave_dir / "sound.wav"
    save_wave_audio(audio_np, sampling_rate, frame_start, frame_end, wav_path, hop_size)

    # Save wave probabilities slice
    wave_probs = speech_probs[frame_start:frame_end]
    probs_path = wave_dir / "speech_probs.json"
    with open(probs_path, "w") as f:
        json.dump(wave_probs, f, indent=2)

    # Calculate and save RMS energies
    rms_values = compute_rms_per_frame(audio_np, hop_size, frame_start, frame_end)
    energies_path = wave_dir / "energies.json"
    with open(energies_path, "w") as f:
        json.dump(rms_values, f, indent=2)

    # Save wave metadata
    wave_json_path = wave_dir / "wave.json"
    wave_copy = wave.copy()
    wave_copy["segment_num"] = seg_num
    wave_copy["wave_num"] = wave_num
    with open(wave_json_path, "w") as f:
        json.dump(wave_copy, f, indent=2)

    # Create and save visualization (pass full wave context)
    plot_path = wave_dir / "wave_plot.png"
    save_wave_plot(
        probs=wave_probs,
        rms_values=rms_values,
        output_path=plot_path,
        wave_num=wave_num,
        seg_num=seg_num,
        wave=wave,
        threshold=threshold,
        hop_size=hop_size,
        sampling_rate=sampling_rate,
        shape_cfg=shape_cfg,
    )


# ── Reporting helpers ──


def find_parent_segment(wave: SpeechWave, segments: list) -> int:
    """
    Find which segment a wave belongs to based on time overlap.
    Returns 1-based segment number.
    """
    wave_start = wave["start_sec"]
    wave_end = wave["end_sec"]

    for seg in segments:
        seg_start = seg.get("start_sec", 0.0)
        seg_end = seg.get("end_sec", 0.0)

        # Check for any time overlap between wave and segment
        if wave_start <= seg_end and wave_end >= seg_start:
            return seg.get("num", seg.get("segment_num", 1))

    # Fallback to first segment if no match found
    return 1


def build_wave_report(
    wave: SpeechWave,
    wave_idx: int,
    waves_dir: Path,
    segments: list,
) -> dict:
    """
    Flatten one SpeechWave into a clean, self-contained report dict.
    Used for both summary.json rows and top_5_waves.json entries.
    """
    parent_seg_num = find_parent_segment(wave, segments)

    dir_name = f"segment_{parent_seg_num:03d}_wave_{wave_idx:03d}"
    wav_abs = (waves_dir / dir_name / "sound.wav").resolve()
    plot_abs = (waves_dir / dir_name / "wave_plot.png").resolve()

    d = wave["details"]
    return {
        # ── identity ──────────────────────────────────────────────────
        "wave": wave_idx,
        "dir": dir_name,
        # ── timing ────────────────────────────────────────────────────
        "start_sec": round(wave["start_sec"], 4),
        "end_sec": round(wave["end_sec"], 4),
        "dur_sec": round(d["duration_sec"], 4),
        # ── Plot file ────────────────────────────────────────────────
        "plot_path": str(plot_abs),
        # ── audio file ────────────────────────────────────────────────
        "sound_path": str(wav_abs),
        # ── probability scores ────────────────────────────────────────
        "scores": {
            "min_prob": round(d["min_prob"], 6),
            "max_prob": round(d["max_prob"], 6),
            "avg_prob": round(d["avg_prob"], 6),
            "std_prob": round(d["std_prob"], 6),
            "baseline": round(d.get("baseline", 0.0), 6),
            "prominence": round(d.get("prominence", 0.0), 6),
            "excursion": round(d.get("excursion", 0.0), 6),
            "composite": round(compute_composite_score(wave), 6),
        },
    }


def top5_reports(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
) -> list[dict]:
    """
    Return the 5 waves with the highest composite score, already serialised
    as report dicts (not raw SpeechWave objects).

    Composite score (see compute_composite_score for full rationale):
        avg_prob * prominence * log1p(duration_sec) * (1 + 0.3 * excursion)

    - avg_prob rewards sustained confidence across the whole wave (not just
      a single spike).
    - prominence measures mountain height above the noise floor.
    - log1p(duration_sec) applies a duration bonus with diminishing returns.
    - (1 + 0.3 * excursion) gives a small multiplicative bonus for waves
      that genuinely rise and fall rather than sitting as flat plateaus.
    """
    indexed = list(enumerate(speech_waves, 1))  # [(1, wave), (2, wave), …]
    ranked = sorted(
        indexed, key=lambda iv: compute_composite_score(iv[1]), reverse=True
    )
    return [
        build_wave_report(wave, idx, waves_dir, segments) for idx, wave in ranked[:5]
    ]


def build_summary_rows(
    speech_waves: List[SpeechWave],
    waves_dir: Path,
    segments: list,
) -> list[dict]:
    """
    Build a flat list of report dicts — one per valid wave — used for both
    the rich summary table and summary.json.
    """
    return [
        build_wave_report(wave, idx, waves_dir, segments)
        for idx, wave in enumerate(speech_waves, 1)
    ]


if __name__ == "__main__":
    from main._main_speech_waves import main
    main()    
