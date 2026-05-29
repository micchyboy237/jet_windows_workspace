"""
speech_checker.py
================
Reusable class for checking if audio contains speech using Zipformer models.
Provides speech detection, chunk extraction, and comprehensive insights.

Speech Categories (from class_labels_indices.csv):
  index 0: "Speech" (general)
  index 1: "Male speech, man speaking"
  index 2: "Female speech, woman speaking"
  index 3: "Child speech, kid speaking"
  index 4: "Conversation"
  index 5: "Narration, monologue"
  index 6: "Babbling"
  index 7: "Speech synthesizer"

IMPORTANT: has_speech is determined by comparing the MAX probability
across ALL labels per chunk against the threshold - NOT by filtering
on specific speech indices. Speech indices are used for reporting only.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audio_tagger_core import (
    HOP_LENGTH,
    SAMPLE_RATE,
    SpeechLabelStats,
    TaggingEvent,
    TaggingResult,
    log,
)
from audio_tagger_zipformer import ZipformerAudioTagger

console = Console()

# Default output directory following project conventions
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

SPEECH_INDICES = {
    0: "Speech",
    1: "Male speech, man speaking",
    2: "Female speech, woman speaking",
    3: "Child speech, kid speaking",
    4: "Conversation",
    5: "Narration, monologue",
    6: "Babbling",
    7: "Speech synthesizer",
}

DEFAULT_SPEECH_INDICES = [0, 1, 2, 3, 4, 5]
DEFAULT_MIN_SPEECH_THRESHOLD = 0.6


@dataclass
class SpeechChunk:
    """Represents a chunk of audio that contains speech above threshold."""
    start_time: float
    end_time: float
    duration: float
    speech_probability: float
    speech_type: str
    chunk_index: int
    events: List[TaggingEvent] = field(default_factory=list)

    @property
    def time_range_str(self) -> str:
        """Human-readable time range."""
        return f"{self.start_time:.2f}s - {self.end_time:.2f}s"


@dataclass
class SpeechCheckResult:
    """Complete result of speech checking analysis."""
    has_speech: bool
    total_speech_probability: float
    speech_duration: float
    total_duration: float
    speech_percentage: float
    speech_chunks: List[SpeechChunk]
    speech_types_detected: Dict[str, float]
    threshold_used: float
    speech_indices_used: List[int]
    audio_path: str
    processing_time: float
    backend_name: str = "Zipformer"
    speech_stats: Optional[SpeechLabelStats] = None

    @property
    def speech_ratio(self) -> float:
        """Ratio of speech to total duration (0.0 to 1.0)."""
        return self.speech_duration / self.total_duration if self.total_duration > 0 else 0.0

    @property
    def confidence_level(self) -> str:
        """Qualitative confidence level based on max speech probability."""
        # Use max probability across all speech chunks for confidence
        max_prob = max((c.speech_probability for c in self.speech_chunks), default=0.0)
        if max_prob >= 0.8:
            return "High"
        elif max_prob >= 0.5:
            return "Medium"
        elif max_prob >= 0.3:
            return "Low"
        else:
            return "Very Low"


class SpeechChecker:
    """
    Reusable class for checking if audio contains speech using Zipformer models.
    
    DETECTION LOGIC (UPDATED):
    - Uses TaggingResult.speech_label_stats for robust speech analysis
    - has_speech = speech_label_stats.present AND speech_label_stats.max_prob >= threshold
    - If speech_label_stats not available, falls back to per-chunk max probability
    - Speech indices are used for FILTERING/REPORTING only, not for detection
    
    Features:
    - Configurable speech threshold
    - Extraction of speech chunks above threshold
    - Comprehensive speech insights and statistics
    - Visualization of speech probability over time
    - Support for filtering by speech type categories
    - Leverages TaggingResult.speech_label_stats for detailed analysis
    
    Usage:
        # Basic usage (auto-builds)
        checker = SpeechChecker(threshold=0.3)
        result = checker.check_speech("audio.wav")
        
        if result.has_speech:
            print(f"Speech detected: {result.speech_percentage:.1f}% of audio")
            print(f"Mean speech prob: {result.speech_stats['mean_prob']:.3f}")
            for chunk in result.speech_chunks:
                print(f"  {chunk.time_range_str}: {chunk.speech_type}")
        
        # Filter results to only show specific speech types
        checker = SpeechChecker(
            threshold=0.3,
            speech_indices=[1, 2]  # Only report male/female speech
        )
        
        # Get detailed insights
        insights = checker.get_speech_insights(result)
        checker.plot_speech_timeline(result, "speech_timeline.png")
    """

    def __init__(
        self,
        threshold: float = DEFAULT_MIN_SPEECH_THRESHOLD,
        speech_indices: Optional[List[int]] = None,
        variant: str = "standard",
        top_k: int = 10,
    ):
        """
        Initialize SpeechChecker with configurable parameters.
        
        Args:
            threshold: Minimum probability (0.0-1.0) for speech detection.
                      Compares against MAX probability across ALL labels per chunk.
            speech_indices: List of class indices to FILTER results by.
                           Does NOT affect has_speech detection.
                           Default: [0,1,2,3,4,5]
            variant: Zipformer model variant ("standard" or "small")
            top_k: Number of top predictions to consider
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.threshold = threshold
        self.speech_indices = speech_indices if speech_indices is not None else DEFAULT_SPEECH_INDICES
        self.variant = variant
        self.top_k = top_k
        
        invalid_indices = set(self.speech_indices) - set(SPEECH_INDICES.keys())
        if invalid_indices:
            raise ValueError(
                f"Invalid speech indices: {invalid_indices}. "
                f"Valid indices: {list(SPEECH_INDICES.keys())}"
            )
        
        # Auto-build: ZipformerAudioTagger builds in __init__
        self._tagger: Optional[ZipformerAudioTagger] = None
        self._is_built = False
        self.last_result: Optional[SpeechCheckResult] = None
        
        # Build immediately
        self.build()

    def build(self) -> "SpeechChecker":
        """
        Build and initialize the underlying Zipformer tagger.
        Returns self for method chaining.
        """
        if self._is_built:
            return self
        
        log.info(f"Building SpeechChecker with threshold={self.threshold}")
        log.info(f"Speech indices (for filtering): {[SPEECH_INDICES[i] for i in self.speech_indices]}")
        
        # ZipformerAudioTagger auto-builds in __init__
        self._tagger = ZipformerAudioTagger(
            variant=self.variant,
            top_k=self.top_k
        )
        self._is_built = True
        
        console.print(
            Panel.fit(
                f"[bold green]✓ SpeechChecker Ready[/bold green]\n"
                f"[dim]Threshold: {self.threshold:.0%} | "
                f"Filter Types: {len(self.speech_indices)} | "
                f"Model: Zipformer-{self.variant}[/dim]",
                border_style="green"
            )
        )
        return self

    def check_speech(
        self,
        audio_path: str,
        output_dir: Optional[Path] = None,
        save_visualizations: bool = True,
    ) -> SpeechCheckResult:
        """
        Check if audio contains speech above the configured threshold.
        
        Detection: compares MAX probability across ALL labels per chunk
        against threshold. Uses speech_label_stats for robust analysis.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save results. If None, uses OUTPUT_DIR
            save_visualizations: Whether to generate and save plots
            
        Returns:
            SpeechCheckResult with comprehensive speech analysis
        """
        if not self._is_built:
            raise RuntimeError("Call .build() before .check_speech()")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use default output directory if none provided
        if output_dir is None:
            output_dir = OUTPUT_DIR
            log.info(f"No output directory specified, using default: {output_dir}")
        
        output_dir = Path(output_dir)
        
        console.print(f"\n[bold cyan]🔍 Analyzing Speech in:[/bold cyan] {Path(audio_path).name}")
        
        start_time = time.time()
        
        # Use tag_audio (new API) with output_dir for saving
        raw_result = self._tagger.tag_audio(
            audio_input=audio_path,
            output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        
        # Get speech_label_stats from TaggingResult
        speech_stats = raw_result.speech_label_stats
        
        # Determine has_speech using speech_label_stats
        if speech_stats["present"] and speech_stats["max_prob"] >= self.threshold:
            has_speech = True
            log.info(
                f"Speech detected via speech_label_stats: "
                f"max_prob={speech_stats['max_prob']:.4f} >= threshold={self.threshold:.4f}"
            )
        else:
            has_speech = False
            if speech_stats["present"]:
                log.info(
                    f"Speech label found but below threshold: "
                    f"max_prob={speech_stats['max_prob']:.4f} < threshold={self.threshold:.4f}"
                )
            else:
                log.info("No speech label found in results")
        
        # Extract speech chunks using max probability across ALL labels
        speech_chunks = self._extract_speech_chunks(raw_result, speech_stats)
        speech_types = self._analyze_speech_types(speech_chunks)
        
        # Calculate overall statistics
        total_speech_prob = speech_stats["max_prob"] if speech_stats["present"] else 0.0
        total_speech_duration = sum(chunk.duration for chunk in speech_chunks)
        
        # If no chunks found but speech_label_stats says present, estimate duration
        if not speech_chunks and speech_stats["present"] and speech_stats["max_prob"] >= self.threshold:
            total_speech_duration = raw_result.duration * speech_stats["detection_rate"]
            log.info(
                f"No speech chunks extracted but speech detected: "
                f"estimated duration={total_speech_duration:.2f}s "
                f"(detection_rate={speech_stats['detection_rate']:.1%})"
            )
        
        result = SpeechCheckResult(
            has_speech=has_speech,
            total_speech_probability=float(total_speech_prob),
            speech_duration=total_speech_duration,
            total_duration=raw_result.duration,
            speech_percentage=(total_speech_duration / raw_result.duration * 100)
                            if raw_result.duration > 0 else 0.0,
            speech_chunks=speech_chunks,
            speech_types_detected=speech_types,
            threshold_used=self.threshold,
            speech_indices_used=self.speech_indices,
            audio_path=audio_path,
            processing_time=processing_time,
            speech_stats=speech_stats if speech_stats["present"] else None,
        )
        
        self.last_result = result
        
        # Print results to console
        self._print_speech_check_results(result)
        
        # Save analysis files if requested
        if save_visualizations:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_speech_analysis(result, output_dir, raw_result)
        
        return result

    def _get_chunk_max_probabilities(
        self, 
        raw_result: TaggingResult,
        speech_stats: SpeechLabelStats
    ) -> List[Dict]:
        """
        Extract per-chunk maximum probability across ALL labels.
        
        Prioritizes raw chunk events from speech_stats if available,
        then falls back to chunk_results in raw_result.
        
        Args:
            raw_result: TaggingResult from Zipformer tagger
            speech_stats: SpeechLabelStats from raw_result
            
        Returns:
            List of dicts with per-chunk max probability info
        """
        chunk_max_probs = []
        
        # Try to use speech_stats chunk_times with prob_values (most detailed)
        if speech_stats["present"] and speech_stats["prob_values"]:
            prob_values = speech_stats["prob_values"]
            chunk_times = speech_stats["chunk_times"]
            
            log.info(
                f"[DEBUG] Using speech_stats: {len(prob_values)} probability values, "
                f"{len(chunk_times)} chunk time ranges"
            )
            
            for i, (prob, (start_time, end_time)) in enumerate(
                zip(prob_values, chunk_times)
            ):
                chunk_max_probs.append({
                    'chunk_index': i,
                    'chunk_start': start_time,
                    'chunk_end': end_time,
                    'max_prob': prob,
                    'max_label': speech_stats.get("label_name", "Speech"),
                    'max_index': speech_stats.get("class_index", 0),
                    'all_events': [],  # Will be populated from raw_result if needed
                })
        
        # Fallback: try raw_result._chunk_events
        if not chunk_max_probs and hasattr(raw_result, '_chunk_events') and raw_result._chunk_events:
            log.info(f"[DEBUG] Using raw_result._chunk_events ({len(raw_result._chunk_events)} events)")
            
            chunks_by_index = defaultdict(list)
            for event in raw_result._chunk_events:
                chunk_idx = event.get("chunk_index", 0)
                chunks_by_index[chunk_idx].append(event)
            
            for chunk_idx in sorted(chunks_by_index.keys()):
                events = chunks_by_index[chunk_idx]
                # Find max probability across ALL events in this chunk
                max_event = max(events, key=lambda e: e.get("prob", 0.0))
                
                chunk_max_probs.append({
                    'chunk_index': chunk_idx,
                    'chunk_start': events[0].get("chunk_start", 0.0),
                    'chunk_end': events[0].get("chunk_end", 0.0),
                    'max_prob': max_event.get("prob", 0.0),
                    'max_label': max_event.get("name", "Unknown"),
                    'max_index': max_event.get("index", -1),
                    'all_events': events,
                })
        
        # Last fallback: use aggregated events
        if not chunk_max_probs:
            log.info(f"[DEBUG] Using raw_result.events (fallback, {len(raw_result.events)} events)")
            
            chunk_events = defaultdict(list)
            for event in raw_result.events:
                chunk_key = (event.chunk_start, event.chunk_end, event.chunk_index)
                chunk_events[chunk_key].append(event)
            
            for (chunk_start, chunk_end, chunk_index), events in chunk_events.items():
                max_event = max(events, key=lambda e: e.prob)
                chunk_max_probs.append({
                    'chunk_index': chunk_index,
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'max_prob': max_event.prob,
                    'max_label': max_event.name,
                    'max_index': max_event.class_index,
                    'all_events': events,
                })
        
        # Sort by chunk_index for consistent ordering
        chunk_max_probs.sort(key=lambda c: c['chunk_index'])
        
        return chunk_max_probs

    def _extract_speech_chunks(
        self, 
        raw_result: TaggingResult,
        speech_stats: SpeechLabelStats
    ) -> List[SpeechChunk]:
        """
        Extract speech chunks from raw tagging result.
        
        DETECTION: Uses MAX probability across ALL labels per chunk vs threshold.
        Now also leverages speech_label_stats for more accurate chunk detection.
        
        Args:
            raw_result: TaggingResult from Zipformer tagger
            speech_stats: SpeechLabelStats for enhanced detection
            
        Returns:
            List of SpeechChunk objects for chunks passing threshold
        """
        speech_chunks = []
        
        # Get per-chunk max probabilities
        chunk_max_probs = self._get_chunk_max_probabilities(raw_result, speech_stats)
        
        log.info(
            f"[DEBUG] _extract_speech_chunks: threshold={self.threshold:.4f}, "
            f"total chunks={len(chunk_max_probs)}"
        )
        
        for chunk_data in chunk_max_probs:
            max_prob = chunk_data['max_prob']
            max_label = chunk_data['max_label']
            max_index = chunk_data['max_index']
            
            passes_threshold = max_prob >= self.threshold
            
            log.debug(
                f"[DEBUG]   Chunk {chunk_data['chunk_index']}: "
                f"max={max_prob:.4f} ({max_label}, idx={max_index}), "
                f"threshold={self.threshold:.4f}, "
                f"passes={passes_threshold}"
            )
            
            if passes_threshold:
                # Determine speech type: use the best-matching speech-index label
                speech_label = self._get_best_speech_label(chunk_data)
                
                chunk = SpeechChunk(
                    start_time=chunk_data['chunk_start'],
                    end_time=chunk_data['chunk_end'],
                    duration=chunk_data['chunk_end'] - chunk_data['chunk_start'],
                    speech_probability=max_prob,
                    speech_type=speech_label,
                    chunk_index=chunk_data['chunk_index'],
                )
                speech_chunks.append(chunk)
        
        log.info(f"[DEBUG] Chunks passing threshold: {len(speech_chunks)}")
        
        # Merge overlapping chunks
        merged_chunks = self._merge_overlapping_chunks(speech_chunks)
        
        log.info(f"[DEBUG] After merging: {len(merged_chunks)} speech segments")
        for i, chunk in enumerate(merged_chunks):
            log.info(
                f"[DEBUG]   Segment {i+1}: {chunk.time_range_str}, "
                f"prob={chunk.speech_probability:.4f}, type={chunk.speech_type}"
            )
        
        return merged_chunks

    def _get_best_speech_label(self, chunk_data: Dict) -> str:
        """
        Get the best speech-type label for a chunk.
        
        Looks through all events in the chunk and finds the highest-probability
        event whose class_index is in speech_indices. Falls back to the max label.
        
        Args:
            chunk_data: Per-chunk data dict from _get_chunk_max_probabilities
            
        Returns:
            Best speech label string
        """
        all_events = chunk_data.get('all_events', [])
        
        # Handle both dict events and TaggingEvent objects
        best_speech_event = None
        best_speech_prob = 0.0
        
        for event in all_events:
            # Handle dict events (from raw chunk data)
            if isinstance(event, dict):
                event_index = event.get("index")
                event_prob = event.get("prob", 0.0)
                event_name = event.get("name", "Unknown")
            # Handle TaggingEvent objects
            else:
                event_index = event.class_index
                event_prob = event.prob
                event_name = event.name
            
            if event_index in self.speech_indices:
                if event_prob > best_speech_prob:
                    best_speech_prob = event_prob
                    best_speech_event = event_name
        
        if best_speech_event:
            return best_speech_event
        
        # Fallback: use the max label
        return chunk_data.get('max_label', 'Unknown')

    def _merge_overlapping_chunks(
        self,
        chunks: List[SpeechChunk],
        max_gap: float = 0.5
    ) -> List[SpeechChunk]:
        """
        Merge overlapping or adjacent speech chunks.
        
        Args:
            chunks: List of speech chunks
            max_gap: Maximum gap between chunks to consider them continuous
            
        Returns:
            Merged list of speech chunks
        """
        if not chunks:
            return []
        
        # Sort by start time
        sorted_chunks = sorted(chunks, key=lambda c: c.start_time)
        merged = [sorted_chunks[0]]
        
        for chunk in sorted_chunks[1:]:
            last = merged[-1]
            
            # Check if chunks overlap or are close enough
            if chunk.start_time <= last.end_time + max_gap:
                # Merge: extend end time, take max probability, combine events
                last.end_time = max(last.end_time, chunk.end_time)
                last.duration = last.end_time - last.start_time
                last.speech_probability = max(last.speech_probability, chunk.speech_probability)
                last.events.extend(chunk.events)
                
                # Update speech type if more specific than generic "Speech"
                if chunk.speech_type != "Speech" and last.speech_type == "Speech":
                    last.speech_type = chunk.speech_type
            else:
                merged.append(chunk)
        
        return merged

    def _analyze_speech_types(
        self,
        speech_chunks: List[SpeechChunk]
    ) -> Dict[str, float]:
        """
        Analyze which types of speech were detected.
        
        Args:
            speech_chunks: List of speech chunks
            
        Returns:
            Dictionary mapping speech types to their average probabilities
        """
        speech_types = {}
        
        for chunk in speech_chunks:
            speech_type = chunk.speech_type
            if speech_type not in speech_types:
                speech_types[speech_type] = []
            speech_types[speech_type].append(chunk.speech_probability)
        
        return {
            stype: float(np.mean(probs))
            for stype, probs in speech_types.items()
        }

    def get_speech_insights(self, result: Optional[SpeechCheckResult] = None) -> Dict:
        """
        Get comprehensive insights about speech in the audio.
        
        Args:
            result: SpeechCheckResult (uses last_result if None)
            
        Returns:
            Dictionary with detailed speech insights
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            raise ValueError("No result available. Run check_speech() first.")
        
        insights = {
            "has_speech": result.has_speech,
            "confidence": result.confidence_level,
            "max_speech_probability": f"{result.total_speech_probability:.1%}",
            "speech_duration_seconds": f"{result.speech_duration:.2f}",
            "total_duration_seconds": f"{result.total_duration:.2f}",
            "speech_percentage": f"{result.speech_percentage:.1f}%",
            "number_of_speech_segments": len(result.speech_chunks),
            "speech_types_detected": result.speech_types_detected,
            "threshold_used": f"{result.threshold_used:.0%}",
            "processing_time_seconds": f"{result.processing_time:.2f}",
            "speech_density": self._calculate_speech_density(result),
            "longest_speech_segment": self._get_longest_segment(result),
            "average_segment_duration": self._get_average_segment_duration(result),
        }
        
        # Add speech_label_stats insights if available
        if result.speech_stats:
            stats = result.speech_stats
            insights.update({
                "speech_label_present": stats["present"],
                "speech_mean_probability": f"{stats['mean_prob']:.1%}",
                "speech_median_probability": f"{stats['median_prob']:.1%}",
                "speech_std_probability": f"{stats['std_prob']:.3f}",
                "speech_min_probability": f"{stats['min_prob']:.1%}",
                "speech_probability_range": f"{stats['prob_range']:.3f}",
                "speech_coefficient_of_variation": f"{stats['coefficient_of_variation']:.3f}",
                "speech_detection_rate": f"{stats['detection_rate']:.1%}",
                "speech_chunks_with_detection": f"{stats['num_chunks']}/{stats['total_chunks']}",
            })
        
        return insights

    def _calculate_speech_density(self, result: SpeechCheckResult) -> str:
        """Calculate speech density (how concentrated the speech is)."""
        if not result.speech_chunks:
            return "No speech"
        
        total_gaps = result.total_duration - result.speech_duration
        
        if total_gaps <= 0:
            return "Continuous speech"
        
        avg_gap = total_gaps / (len(result.speech_chunks) + 1)
        
        if avg_gap < 1.0:
            return "Dense (frequent speech)"
        elif avg_gap < 3.0:
            return "Moderate"
        else:
            return "Sparse (infrequent speech)"

    def _get_longest_segment(self, result: SpeechCheckResult) -> Optional[str]:
        """Get the longest continuous speech segment."""
        if not result.speech_chunks:
            return None
        
        longest = max(result.speech_chunks, key=lambda c: c.duration)
        return f"{longest.duration:.2f}s ({longest.time_range_str})"

    def _get_average_segment_duration(self, result: SpeechCheckResult) -> Optional[str]:
        """Get average speech segment duration."""
        if not result.speech_chunks:
            return None
        
        avg = np.mean([chunk.duration for chunk in result.speech_chunks])
        return f"{avg:.2f}s"

    def plot_speech_timeline(
        self,
        result: Optional[SpeechCheckResult] = None,
        output_path: Optional[str] = None,
        show_plot: bool = False,
    ) -> Optional[str]:
        """
        Generate a timeline visualization of speech probability.
        
        Args:
            result: SpeechCheckResult (uses last_result if None)
            output_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if output_path provided
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            raise ValueError("No result available. Run check_speech() first.")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Speech probability scatter plot
        ax1 = axes[0]
        
        if result.speech_chunks:
            times = [chunk.start_time for chunk in result.speech_chunks]
            probs = [chunk.speech_probability for chunk in result.speech_chunks]
            durations = [chunk.duration for chunk in result.speech_chunks]
            
            scatter = ax1.scatter(
                times, probs,
                s=[d * 100 for d in durations],  # Bubble size proportional to duration
                c=probs,
                cmap='RdYlGn',
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax1, label='Max Probability')
        
        # Threshold line
        ax1.axhline(y=self.threshold, color='red', linestyle='--',
                   label=f'Threshold ({self.threshold:.0%})')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Max Probability per Chunk')
        ax1.set_title(f'Speech Detection Timeline - {Path(result.audio_path).name}')
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speech segments timeline
        ax2 = axes[1]
        
        if result.speech_chunks:
            colors = plt.cm.Set3(np.linspace(0, 1, len(result.speech_chunks)))
            
            for i, chunk in enumerate(result.speech_chunks):
                ax2.barh(
                    0,
                    chunk.duration,
                    left=chunk.start_time,
                    height=0.8,
                    color=colors[i],
                    alpha=0.7,
                    edgecolor='black',
                    label=f"Seg {i+1}: {chunk.speech_type}"
                )
                
                # Add probability label in middle of segment
                mid_point = chunk.start_time + chunk.duration / 2
                ax2.text(
                    mid_point, 0,
                    f"{chunk.speech_probability:.0%}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold'
                )
            
            # Only show legend if manageable number of segments
            if len(result.speech_chunks) <= 10:
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Speech Segments')
        ax2.set_title(f'Speech Segments ({len(result.speech_chunks)} segments)')
        ax2.set_xlim(0, result.total_duration)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Speech probability distribution (box plot)
        ax3 = axes[2]
        
        if result.speech_stats and result.speech_stats["prob_values"]:
            stats = result.speech_stats
            prob_values = stats["prob_values"]
            
            # Box plot
            bp = ax3.boxplot(
                [prob_values],
                vert=False,
                patch_artist=True,
                widths=0.5,
            )
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][0].set_alpha(0.7)
            
            # Add statistics annotations
            ax3.axvline(x=self.threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.threshold:.0%})')
            ax3.axvline(x=stats["mean_prob"], color='blue', linestyle=':', 
                       label=f'Mean ({stats["mean_prob"]:.1%})')
            ax3.axvline(x=stats["median_prob"], color='green', linestyle=':', 
                       label=f'Median ({stats["median_prob"]:.1%})')
            
            ax3.legend(loc='upper right')
            
            ax3.set_title(
                f'Speech Probability Distribution\n'
                f'Min={stats["min_prob"]:.1%}, Max={stats["max_prob"]:.1%}, '
                f'Std={stats["std_prob"]:.3f}, CV={stats["coefficient_of_variation"]:.3f}'
            )
        else:
            ax3.text(0.5, 0.5, 'No speech probability data available',
                    ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_xlabel('Probability')
        ax3.set_xlim(0, 1.05)
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            log.info(f"Speech timeline saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path if output_path else None

    def _save_speech_analysis(
        self,
        result: SpeechCheckResult,
        output_dir: Path,
        raw_result: TaggingResult,
    ) -> None:
        """
        Save comprehensive speech analysis results.
        
        Args:
            result: SpeechCheckResult
            output_dir: Directory to save files
            raw_result: Original TaggingResult for additional data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save speech check results JSON
        speech_data = {
            "has_speech": result.has_speech,
            "confidence_level": result.confidence_level,
            "max_speech_probability": result.total_speech_probability,
            "speech_duration_seconds": result.speech_duration,
            "total_duration_seconds": result.total_duration,
            "speech_percentage": result.speech_percentage,
            "threshold_used": result.threshold_used,
            "speech_indices_used": result.speech_indices_used,
            "speech_types_detected": result.speech_types_detected,
            "number_of_speech_chunks": len(result.speech_chunks),
            "processing_time_seconds": result.processing_time,
            "audio_path": result.audio_path,
            "speech_chunks": [
                {
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration,
                    "probability": chunk.speech_probability,
                    "speech_type": chunk.speech_type,
                }
                for chunk in result.speech_chunks
            ],
        }
        
        # Add speech_label_stats if available
        if result.speech_stats:
            speech_data["speech_label_stats"] = {
                "present": result.speech_stats["present"],
                "label_name": result.speech_stats["label_name"],
                "class_index": result.speech_stats["class_index"],
                "mean_prob": result.speech_stats["mean_prob"],
                "median_prob": result.speech_stats["median_prob"],
                "max_prob": result.speech_stats["max_prob"],
                "min_prob": result.speech_stats["min_prob"],
                "std_prob": result.speech_stats["std_prob"],
                "prob_range": result.speech_stats["prob_range"],
                "iqr_prob": result.speech_stats["iqr_prob"],
                "coefficient_of_variation": result.speech_stats["coefficient_of_variation"],
                "detection_rate": result.speech_stats["detection_rate"],
                "num_chunks": result.speech_stats["num_chunks"],
                "total_chunks": result.speech_stats["total_chunks"],
            }
        
        speech_json_path = output_dir / "speech_check_results.json"
        with open(speech_json_path, "w", encoding="utf-8") as f:
            json.dump(speech_data, f, indent=2, ensure_ascii=False)
        
        # Save insights JSON
        insights = self.get_speech_insights(result)
        insights_path = output_dir / "speech_insights.json"
        with open(insights_path, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        # Save timeline visualization
        timeline_path = output_dir / "speech_timeline.png"
        self.plot_speech_timeline(result, str(timeline_path))
        
        console.print(
            Panel(
                f"[cyan]Speech analysis saved to:[/cyan]\n"
                f"  • {speech_json_path.name}\n"
                f"  • {insights_path.name}\n"
                f"  • {timeline_path.name}",
                title="💾 Saved Files",
                border_style="green"
            )
        )

    def _print_speech_check_results(self, result: SpeechCheckResult) -> None:
        """Print formatted speech check results."""
        if result.has_speech:
            status = f"[bold green]✓ SPEECH DETECTED[/bold green]"
            confidence_color = {
                "High": "green",
                "Medium": "yellow",
                "Low": "red",
                "Very Low": "red"
            }.get(result.confidence_level, "white")
        else:
            status = f"[bold red]✗ NO SPEECH DETECTED[/bold red]"
            confidence_color = "red"
        
        console.print(f"\n{status}")
        
        # Main results table
        table = Table(title="📊 Speech Analysis Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")
        
        table.add_row(
            "Speech Detected",
            "Yes" if result.has_speech else "No",
            ""
        )
        table.add_row(
            "Confidence",
            f"[{confidence_color}]{result.confidence_level}[/{confidence_color}]",
            f"Max probability: {result.total_speech_probability:.1%}"
        )
        table.add_row(
            "Speech Duration",
            f"{result.speech_duration:.2f}s",
            f"{result.speech_percentage:.1f}% of {result.total_duration:.2f}s total"
        )
        table.add_row(
            "Speech Segments",
            str(len(result.speech_chunks)),
            f"Longest: {self._get_longest_segment(result) or 'N/A'}"
        )
        table.add_row(
            "Speech Types",
            str(len(result.speech_types_detected)),
            ", ".join(result.speech_types_detected.keys()) if result.speech_types_detected else "None"
        )
        table.add_row(
            "Processing Time",
            f"{result.processing_time:.2f}s",
            f"{result.total_duration/result.processing_time:.1f}x real-time" if result.processing_time > 0 else ""
        )
        
        # Add speech_label_stats if available
        if result.speech_stats:
            stats = result.speech_stats
            table.add_row(
                "Speech Label Stats",
                f"Mean: {stats['mean_prob']:.1%}",
                f"Std: {stats['std_prob']:.3f}, CV: {stats['coefficient_of_variation']:.3f}"
            )
            table.add_row(
                "Detection Rate",
                f"{stats['detection_rate']:.1%}",
                f"{stats['num_chunks']}/{stats['total_chunks']} chunks"
            )
        
        console.print(table)
        
        # Detailed chunks table
        if result.speech_chunks:
            chunks_table = Table(
                title=f"🎤 Speech Segments (threshold: {result.threshold_used:.0%})",
                show_header=True
            )
            chunks_table.add_column("Segment", style="cyan", width=8)
            chunks_table.add_column("Time Range", style="yellow")
            chunks_table.add_column("Duration", style="green")
            chunks_table.add_column("Max Prob", style="magenta")
            chunks_table.add_column("Type", style="blue")
            
            for i, chunk in enumerate(result.speech_chunks, 1):
                chunks_table.add_row(
                    f"#{i}",
                    chunk.time_range_str,
                    f"{chunk.duration:.2f}s",
                    f"{chunk.speech_probability:.1%}",
                    chunk.speech_type,
                )
            
            console.print(chunks_table)


def main():
    """Example usage and CLI for SpeechChecker."""
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(
        description="Check audio for speech using Zipformer models"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=DEFAULT_MIN_SPEECH_THRESHOLD,
        help=f"Speech probability threshold (0.0-1.0, default: {DEFAULT_MIN_SPEECH_THRESHOLD})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "-v", "--variant",
        choices=["standard", "small"],
        default="standard",
        help="Zipformer model variant"
    )
    parser.add_argument(
        "--speech-indices",
        nargs="+",
        type=int,
        default=DEFAULT_SPEECH_INDICES,
        help=f"Speech class indices to FILTER by (default: {DEFAULT_SPEECH_INDICES})"
    )
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-builds in __init__
    checker = SpeechChecker(
        threshold=args.threshold,
        speech_indices=args.speech_indices,
        variant=args.variant,
    )
    
    try:
        result = checker.check_speech(
            args.audio_path,
            output_dir=output_dir,
        )
        
        # Print insights
        insights = checker.get_speech_insights()
        console.print("\n[bold cyan]📈 Detailed Insights:[/bold cyan]")
        
        insights_table = Table(show_header=True)
        insights_table.add_column("Insight", style="cyan")
        insights_table.add_column("Value", style="green")
        
        for key, value in insights.items():
            insights_table.add_row(
                key.replace("_", " ").title(),
                str(value) if value is not None else "N/A"
            )
        
        console.print(insights_table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
