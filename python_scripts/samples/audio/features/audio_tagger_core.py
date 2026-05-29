"""
audio_tagger_core.py
====================
Shared dataclasses, constants, and logging/console setup for sherpa-onnx audio tagging.
This module imports NOTHING from audio_tagger_base or audio_tagger_utils,
breaking circular dependencies.
Aligned with FireRed VAD:
    - Uses FRAME_SHIFT_SAMPLE (160 samples) as the fundamental unit
    - Windows are multiples of FireRed frames (100 frames = 1s window)
    - Supports per-segment tagging with absolute UTC timestamps
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, TypedDict
import numpy as np
from fireredvad.core.constants import (
    FRAME_SHIFT_SAMPLE,
)
from fireredvad.core.constants import (
    SAMPLE_RATE as FIRERED_SAMPLE_RATE,
)
from rich.console import Console

# Torch availability flag for type checking in tag_audio
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

console = Console()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    log.addHandler(handler)

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
SAMPLE_RATE = FIRERED_SAMPLE_RATE
HOP_LENGTH = FRAME_SHIFT_SAMPLE


# Speech label identifiers (case-insensitive matching)
SPEECH_LABEL_NAMES = {"speech", "speech music", "speech-music", "talking", "voice", "dialogue", "conversation"}
SPEECH_LABEL_INDEX = 0  # Common index for Speech in audio tagging models


class SpeechLabelStats(TypedDict, total=False):
    """
    Statistics for the Speech label across all processing chunks.
    
    Attributes:
        present: Whether the Speech label was found in results
        class_index: The class index of the Speech label
        label_name: The exact name of the Speech label
        num_chunks: Number of chunks where Speech was detected
        total_chunks: Total number of chunks processed
        min_prob: Minimum probability across all chunks
        max_prob: Maximum probability across all chunks
        mean_prob: Mean (average) probability across all chunks
        median_prob: Median probability across all chunks
        std_prob: Standard deviation of probabilities
        q25_prob: 25th percentile (first quartile)
        q75_prob: 75th percentile (third quartile)
        iqr_prob: Interquartile range (Q75 - Q25)
        prob_variance: Variance of probabilities
        prob_skewness: Skewness of the probability distribution
        prob_kurtosis: Kurtosis of the probability distribution
        prob_range: Range (max - min)
        coefficient_of_variation: Standard deviation / mean (relative variability)
        detection_rate: Percentage of chunks where Speech was detected
        prob_values: All individual probability values for Speech
        chunk_times: List of (chunk_start, chunk_end) tuples where Speech was detected
    """
    present: bool
    class_index: Optional[int]
    label_name: Optional[str]
    num_chunks: int
    total_chunks: int
    min_prob: float
    max_prob: float
    mean_prob: float
    median_prob: float
    std_prob: float
    q25_prob: float
    q75_prob: float
    iqr_prob: float
    prob_variance: float
    prob_skewness: float
    prob_kurtosis: float
    prob_range: float
    coefficient_of_variation: float
    detection_rate: float
    prob_values: List[float]
    chunk_times: List[tuple[float, float]]


@dataclass
class TaggingEvent:
    """Normalised event produced by either model backend.
    Now supports absolute UTC timestamps when processing speech segments.
    """
    name: Optional[str]
    class_index: Optional[int]
    prob: float
    max_prob: float = 0.0
    occurrences: int = 1
    chunk_start: float = 0.0
    chunk_end: float = 0.0
    chunk_index: int = 0
    time_utc_start: Optional[datetime] = None
    time_utc_end: Optional[datetime] = None


@dataclass
class TaggingResult:
    """Full result from one audio file or speech segment tagging run.
    Now supports both file-based and segment-based processing.
    
    Properties:
        real_time_factor: Processing speed relative to audio duration
        speech_label_stats: Comprehensive statistics for Speech label if present
    """
    audio_path: str
    sample_rate: int
    duration: float
    elapsed_time: float
    events: List[TaggingEvent]
    chunk_count: int
    backend_name: str
    model_variant: str
    top_k: int
    is_speech_segment: bool = False
    segment_start_utc: Optional[datetime] = None
    segment_end_utc: Optional[datetime] = None
    # Raw chunk events for detailed analysis (populated by tag_audio/tag_file)
    _chunk_events: Optional[List[dict]] = field(default=None, repr=False)

    @property
    def real_time_factor(self) -> float:
        """Processing speed relative to audio duration.
        Values < 1.0 mean faster than real-time."""
        return self.elapsed_time / self.duration if self.duration > 0 else 0.0

    @property
    def speech_label_stats(self) -> SpeechLabelStats:
        """
        Comprehensive statistics for the Speech label across all processing chunks.
        
        Searches for events matching common Speech label names (case-insensitive):
        'speech', 'speech music', 'talking', 'voice', 'dialogue', etc.
        
        If raw chunk events are available (via _chunk_events), statistics are computed
        from all per-chunk probabilities. Otherwise, falls back to aggregated events.
        
        Returns:
            SpeechLabelStats dict with comprehensive statistics, or a dict with
            present=False if no Speech label is found.
        
        Example:
            result = tagger.tag_audio("meeting.wav", Path("output"))
            speech_stats = result.speech_label_stats
            
            if speech_stats["present"]:
                print(f"Speech detected in {speech_stats['detection_rate']:.1%} of chunks")
                print(f"Mean speech probability: {speech_stats['mean_prob']:.3f}")
                print(f"Speech probability range: {speech_stats['prob_range']:.3f}")
                print(f"Coefficient of variation: {speech_stats['coefficient_of_variation']:.3f}")
        """
        # Find the Speech event in aggregated results
        speech_event = self._find_speech_event()
        
        # If raw chunk events are available, compute stats from them
        if self._chunk_events and speech_event:
            return self._compute_speech_stats_from_chunks(speech_event)
        
        # Fall back to aggregated event stats
        if speech_event:
            return self._compute_speech_stats_from_aggregated(speech_event)
        
        # No Speech label found
        return self._empty_speech_stats()

    def _find_speech_event(self) -> Optional[TaggingEvent]:
        """
        Find the Speech event in aggregated events.
        Searches by common name patterns first, then by class_index.
        """
        # Search by name (case-insensitive)
        for event in self.events:
            if event.name and event.name.lower() in SPEECH_LABEL_NAMES:
                return event
        
        # Search by common class index
        for event in self.events:
            if event.class_index == SPEECH_LABEL_INDEX:
                return event
        
        # Broader search: any event containing "speech" in name
        for event in self.events:
            if event.name and "speech" in event.name.lower():
                return event
        
        return None

    def _compute_speech_stats_from_chunks(
        self, speech_event: TaggingEvent
    ) -> SpeechLabelStats:
        """
        Compute comprehensive speech statistics from raw chunk events.
        This gives the most accurate per-chunk probability distribution.
        """
        speech_name = speech_event.name
        speech_index = speech_event.class_index
        
        # Extract Speech probabilities from each chunk
        probs = []
        chunk_times = []
        chunks_with_speech = set()
        
        for chunk_event in self._chunk_events:
            # Match by name first, then by index
            is_speech = False
            if chunk_event.get("name"):
                if chunk_event["name"].lower() in SPEECH_LABEL_NAMES:
                    is_speech = True
                elif "speech" in chunk_event["name"].lower():
                    is_speech = True
            elif chunk_event.get("index") == speech_index:
                is_speech = True
            
            if is_speech:
                prob = chunk_event.get("prob", 0.0)
                probs.append(prob)
                chunk_times.append((
                    chunk_event.get("chunk_start", 0.0),
                    chunk_event.get("chunk_end", 0.0)
                ))
                chunks_with_speech.add(chunk_event.get("chunk_index", -1))
        
        if not probs:
            return self._empty_speech_stats()
        
        return self._compute_statistics(
            probs=probs,
            chunk_times=chunk_times,
            num_chunks_with_speech=len(chunks_with_speech),
            total_chunks=self.chunk_count,
            label_name=speech_name,
            class_index=speech_index,
        )

    def _compute_speech_stats_from_aggregated(
        self, speech_event: TaggingEvent
    ) -> SpeechLabelStats:
        """
        Compute speech statistics from aggregated event only.
        Less detailed than chunk-level stats but works without raw data.
        """
        probs = [speech_event.prob, speech_event.max_prob]
        chunk_times = [(speech_event.chunk_start, speech_event.chunk_end)]
        
        return self._compute_statistics(
            probs=probs,
            chunk_times=chunk_times,
            num_chunks_with_speech=speech_event.occurrences,
            total_chunks=self.chunk_count,
            label_name=speech_event.name,
            class_index=speech_event.class_index,
        )

    def _compute_statistics(
        self,
        probs: List[float],
        chunk_times: List[tuple[float, float]],
        num_chunks_with_speech: int,
        total_chunks: int,
        label_name: Optional[str],
        class_index: Optional[int],
    ) -> SpeechLabelStats:
        """
        Compute full statistical summary from probability values.
        Handles edge cases: single value, all zeros, extreme distributions.
        """
        prob_array = np.array(probs, dtype=np.float64)
        n = len(prob_array)
        
        # Basic stats
        min_prob = float(np.min(prob_array))
        max_prob = float(np.max(prob_array))
        mean_prob = float(np.mean(prob_array))
        median_prob = float(np.median(prob_array))
        
        # Variance and standard deviation (handle single-value case)
        if n > 1:
            std_prob = float(np.std(prob_array, ddof=1))  # Sample std
            variance = float(np.var(prob_array, ddof=1))
        else:
            std_prob = 0.0
            variance = 0.0
        
        # Quartiles
        if n >= 4:
            q25 = float(np.percentile(prob_array, 25))
            q75 = float(np.percentile(prob_array, 75))
        elif n >= 2:
            # With 2-3 values, use linear interpolation manually
            sorted_probs = sorted(probs)
            if n == 2:
                q25 = sorted_probs[0] + 0.25 * (sorted_probs[1] - sorted_probs[0])
                q75 = sorted_probs[0] + 0.75 * (sorted_probs[1] - sorted_probs[0])
            else:  # n == 3
                q25 = sorted_probs[0] + 0.5 * (sorted_probs[1] - sorted_probs[0])
                q75 = sorted_probs[1] + 0.5 * (sorted_probs[2] - sorted_probs[1])
        else:
            q25 = min_prob
            q75 = max_prob
        
        iqr = q75 - q25
        
        # Range
        prob_range = max_prob - min_prob
        
        # Coefficient of variation (handle mean near zero)
        if mean_prob > 1e-10:
            cv = std_prob / mean_prob
        else:
            cv = float('inf') if std_prob > 0 else 0.0
        
        # Skewness and kurtosis (require at least 3 values)
        if n >= 3 and std_prob > 1e-10:
            z_scores = (prob_array - mean_prob) / std_prob
            skewness = float(np.mean(z_scores ** 3))
            kurtosis = float(np.mean(z_scores ** 4) - 3)  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Detection rate
        detection_rate = num_chunks_with_speech / total_chunks if total_chunks > 0 else 0.0
        
        return SpeechLabelStats(
            present=True,
            class_index=class_index,
            label_name=label_name,
            num_chunks=num_chunks_with_speech,
            total_chunks=total_chunks,
            min_prob=min_prob,
            max_prob=max_prob,
            mean_prob=mean_prob,
            median_prob=median_prob,
            std_prob=std_prob,
            q25_prob=q25,
            q75_prob=q75,
            iqr_prob=iqr,
            prob_variance=variance,
            prob_skewness=skewness,
            prob_kurtosis=kurtosis,
            prob_range=prob_range,
            coefficient_of_variation=cv,
            detection_rate=detection_rate,
            prob_values=[round(p, 6) for p in probs],
            chunk_times=[(round(t[0], 3), round(t[1], 3)) for t in chunk_times],
        )

    @staticmethod
    def _empty_speech_stats() -> SpeechLabelStats:
        """Return empty stats when no Speech label is present."""
        return SpeechLabelStats(
            present=False,
            class_index=None,
            label_name=None,
            num_chunks=0,
            total_chunks=0,
            min_prob=0.0,
            max_prob=0.0,
            mean_prob=0.0,
            median_prob=0.0,
            std_prob=0.0,
            q25_prob=0.0,
            q75_prob=0.0,
            iqr_prob=0.0,
            prob_variance=0.0,
            prob_skewness=0.0,
            prob_kurtosis=0.0,
            prob_range=0.0,
            coefficient_of_variation=0.0,
            detection_rate=0.0,
            prob_values=[],
            chunk_times=[],
        )
