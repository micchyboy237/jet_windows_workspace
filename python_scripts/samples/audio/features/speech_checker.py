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

Default speech indices: [0, 1, 2, 3, 4, 5] (general speech + specific types)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from audio_tagger_zipformer import ZipformerAudioTagger
from audio_tagger_core import (
    TaggingEvent,
    TaggingResult,
    SAMPLE_RATE,
    HOP_LENGTH,
    log,
)

console = Console()

# Speech-related class indices from the Zipformer model
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

# Default speech indices to check (excluding babbling and speech synthesizer by default)
DEFAULT_SPEECH_INDICES = [0, 1, 2, 3, 4, 5]


@dataclass
class SpeechChunk:
    """Represents a chunk of audio that contains speech above threshold."""
    start_time: float  # seconds from audio start
    end_time: float  # seconds from audio start
    duration: float  # seconds
    speech_probability: float  # 0.0 to 1.0
    speech_type: str  # type of speech detected
    chunk_index: int  # original chunk index
    events: List[TaggingEvent] = field(default_factory=list)  # all speech events in chunk
    
    @property
    def time_range_str(self) -> str:
        """Human-readable time range."""
        return f"{self.start_time:.2f}s - {self.end_time:.2f}s"


@dataclass
class SpeechCheckResult:
    """Complete result of speech checking analysis."""
    has_speech: bool
    total_speech_probability: float  # 0.0 to 1.0
    speech_duration: float  # total seconds of speech
    total_duration: float  # total audio duration
    speech_percentage: float  # percentage of audio containing speech
    speech_chunks: List[SpeechChunk]
    speech_types_detected: Dict[str, float]  # speech type -> average probability
    threshold_used: float
    speech_indices_used: List[int]
    audio_path: str
    processing_time: float
    backend_name: str = "Zipformer"
    
    @property
    def speech_ratio(self) -> float:
        """Ratio of speech to total duration (0.0 to 1.0)."""
        return self.speech_duration / self.total_duration if self.total_duration > 0 else 0.0
    
    @property
    def confidence_level(self) -> str:
        """Qualitative confidence level based on speech probability."""
        if self.total_speech_probability >= 0.8:
            return "High"
        elif self.total_speech_probability >= 0.5:
            return "Medium"
        elif self.total_speech_probability >= 0.3:
            return "Low"
        else:
            return "Very Low"


class SpeechChecker:
    """
    Reusable class for checking if audio contains speech using Zipformer models.
    
    Features:
    - Configurable speech threshold
    - Extraction of speech chunks above threshold
    - Comprehensive speech insights and statistics
    - Visualization of speech probability over time
    - Support for different speech type categories
    
    Usage:
        # Basic usage
        checker = SpeechChecker(threshold=0.3)
        checker.build()
        result = checker.check_speech("audio.wav")
        
        if result.has_speech:
            print(f"Speech detected: {result.speech_percentage:.1f}% of audio")
            for chunk in result.speech_chunks:
                print(f"  {chunk.time_range_str}: {chunk.speech_type}")
        
        # Custom speech indices (e.g., only male/female speech)
        checker = SpeechChecker(
            threshold=0.3,
            speech_indices=[1, 2]  # Only male and female speech
        )
        
        # Get detailed insights
        insights = checker.get_speech_insights(result)
        checker.plot_speech_timeline(result, "speech_timeline.png")
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        speech_indices: Optional[List[int]] = None,
        variant: str = "standard",
        top_k: int = 10,
    ):
        """
        Initialize SpeechChecker with configurable parameters.
        
        Args:
            threshold: Minimum probability (0.0-1.0) for speech detection
            speech_indices: List of class indices considered as speech.
                           Default: [0,1,2,3,4,5] (excludes babbling & synthesizer)
            variant: Zipformer model variant ("standard" or "small")
            top_k: Number of top predictions to consider
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.threshold = threshold
        self.speech_indices = speech_indices if speech_indices is not None else DEFAULT_SPEECH_INDICES
        self.variant = variant
        self.top_k = top_k
        
        # Validate speech indices
        invalid_indices = set(self.speech_indices) - set(SPEECH_INDICES.keys())
        if invalid_indices:
            raise ValueError(
                f"Invalid speech indices: {invalid_indices}. "
                f"Valid indices: {list(SPEECH_INDICES.keys())}"
            )
        
        self._tagger: Optional[ZipformerAudioTagger] = None
        self._is_built = False
        
        # Store last result for reference
        self.last_result: Optional[SpeechCheckResult] = None
    
    def build(self) -> "SpeechChecker":
        """
        Build and initialize the underlying Zipformer tagger.
        Returns self for method chaining.
        """
        if self._is_built:
            return self
        
        log.info(f"Building SpeechChecker with threshold={self.threshold}")
        log.info(f"Speech indices: {[SPEECH_INDICES[i] for i in self.speech_indices]}")
        
        self._tagger = ZipformerAudioTagger(
            variant=self.variant,
            top_k=self.top_k
        )
        self._tagger.build()
        self._is_built = True
        
        console.print(
            Panel.fit(
                f"[bold green]✓ SpeechChecker Ready[/bold green]\n"
                f"[dim]Threshold: {self.threshold:.0%} | "
                f"Speech Types: {len(self.speech_indices)} | "
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
        
        Args:
            audio_path: Path to audio file
            output_dir: Optional directory to save results
            save_visualizations: Whether to generate and save plots
            
        Returns:
            SpeechCheckResult with comprehensive speech analysis
        """
        if not self._is_built:
            raise RuntimeError("Call .build() before .check_speech()")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        console.print(f"\n[bold cyan]🔍 Analyzing Speech in:[/bold cyan] {Path(audio_path).name}")
        
        # Get raw tagging result from Zipformer
        import time
        start_time = time.time()
        
        raw_result = self._tagger.tag_file(
            audio_path=audio_path,
            output_dir=output_dir or Path("speech_check_output")
        )
        
        processing_time = time.time() - start_time
        
        # Analyze speech events
        speech_chunks = self._extract_speech_chunks(raw_result)
        speech_types = self._analyze_speech_types(speech_chunks)
        
        # Calculate speech metrics
        total_speech_prob = (
            np.mean([chunk.speech_probability for chunk in speech_chunks])
            if speech_chunks else 0.0
        )
        
        total_speech_duration = sum(chunk.duration for chunk in speech_chunks)
        
        result = SpeechCheckResult(
            has_speech=len(speech_chunks) > 0,
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
        )
        
        self.last_result = result
        
        # Print results
        self._print_speech_check_results(result)
        
        # Save visualizations if requested
        if save_visualizations and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_speech_analysis(result, output_dir, raw_result)
        
        return result
    
    def _extract_speech_chunks(self, raw_result: TaggingResult) -> List[SpeechChunk]:
        """
        Extract speech chunks from raw tagging result.
        Only includes chunks where speech probability exceeds threshold.
        
        Args:
            raw_result: TaggingResult from Zipformer tagger
            
        Returns:
            List of SpeechChunk objects for speech segments
        """
        speech_chunks = []
        
        for event in raw_result.events:
            # Check if event is speech-related and above threshold
            if (event.class_index in self.speech_indices and 
                event.prob >= self.threshold):
                
                chunk = SpeechChunk(
                    start_time=event.chunk_start,
                    end_time=event.chunk_end,
                    duration=event.chunk_end - event.chunk_start,
                    speech_probability=event.prob,
                    speech_type=SPEECH_INDICES.get(event.class_index, "Unknown speech"),
                    chunk_index=event.chunk_index,
                    events=[event],
                )
                speech_chunks.append(chunk)
        
        # Merge overlapping chunks
        merged_chunks = self._merge_overlapping_chunks(speech_chunks)
        
        log.info(
            f"Extracted {len(merged_chunks)} speech chunks "
            f"(from {len(speech_chunks)} raw chunks)"
        )
        
        return merged_chunks
    
    def _merge_overlapping_chunks(
        self, 
        chunks: List[SpeechChunk],
        max_gap: float = 0.5  # seconds
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
                # Merge chunks
                last.end_time = max(last.end_time, chunk.end_time)
                last.duration = last.end_time - last.start_time
                last.speech_probability = max(last.speech_probability, chunk.speech_probability)
                last.events.extend(chunk.events)
                
                # Use the more specific speech type if available
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
        
        # Calculate average probabilities
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
            "speech_probability": f"{result.total_speech_probability:.1%}",
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
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Speech probability timeline
        ax1 = axes[0]
        
        if result.speech_chunks:
            times = [chunk.start_time for chunk in result.speech_chunks]
            probs = [chunk.speech_probability for chunk in result.speech_chunks]
            durations = [chunk.duration for chunk in result.speech_chunks]
            
            # Create scatter plot with size representing duration
            scatter = ax1.scatter(
                times, probs, 
                s=[d * 100 for d in durations],  # Size proportional to duration
                c=probs, 
                cmap='RdYlGn', 
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            plt.colorbar(scatter, ax=ax1, label='Speech Probability')
        
        ax1.axhline(y=self.threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.threshold:.0%})')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Speech Probability')
        ax1.set_title(f'Speech Detection Timeline - {Path(result.audio_path).name}')
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speech segments as a Gantt-like chart
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
                
                # Add probability label
                mid_point = chunk.start_time + chunk.duration / 2
                ax2.text(
                    mid_point, 0, 
                    f"{chunk.speech_probability:.0%}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold'
                )
            
            if len(result.speech_chunks) <= 10:  # Only show legend if manageable
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Speech Segments')
        ax2.set_title(f'Speech Segments ({len(result.speech_chunks)} segments)')
        ax2.set_xlim(0, result.total_duration)
        ax2.grid(True, alpha=0.3, axis='x')
        
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
        
        # Save speech check results
        speech_data = {
            "has_speech": result.has_speech,
            "confidence_level": result.confidence_level,
            "total_speech_probability": result.total_speech_probability,
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
        
        speech_json_path = output_dir / "speech_check_results.json"
        with open(speech_json_path, "w", encoding="utf-8") as f:
            json.dump(speech_data, f, indent=2, ensure_ascii=False)
        
        # Save insights
        insights = self.get_speech_insights(result)
        insights_path = output_dir / "speech_insights.json"
        with open(insights_path, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        # Generate timeline plot
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
        
        # Status indicator
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
            f"Average probability: {result.total_speech_probability:.1%}"
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
        
        console.print(table)
        
        # Speech chunks detail if any
        if result.speech_chunks:
            chunks_table = Table(
                title=f"🎤 Speech Segments (threshold: {result.threshold_used:.0%})",
                show_header=True
            )
            chunks_table.add_column("Segment", style="cyan", width=8)
            chunks_table.add_column("Time Range", style="yellow")
            chunks_table.add_column("Duration", style="green")
            chunks_table.add_column("Probability", style="magenta")
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
    
    parser = argparse.ArgumentParser(
        description="Check audio for speech using Zipformer models"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.3,
        help="Speech probability threshold (0.0-1.0, default: 0.3)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("speech_check_output"),
        help="Output directory for results"
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
        help=f"Speech class indices to check (default: {DEFAULT_SPEECH_INDICES})"
    )
    
    args = parser.parse_args()
    
    # Create and use SpeechChecker
    checker = SpeechChecker(
        threshold=args.threshold,
        speech_indices=args.speech_indices,
        variant=args.variant,
    )
    
    try:
        checker.build()
        result = checker.check_speech(
            args.audio_path,
            output_dir=args.output_dir,
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
