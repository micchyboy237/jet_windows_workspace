# speaker_labeler.py

"""
SpeakerLabeler: Dynamic speaker labeling with progressive reference building
using pyannote/segmentation-3.0 model.
"""

import os
import tempfile
from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.audio.utils.signal import Binarize
import soundfile as sf

AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

console = Console()


class SpeakerLabeler:
    """
    A reusable class that dynamically labels speaker segments with progressive
    reference building using pyannote/segmentation-3.0 model.
    
    Features:
    - Voice Activity Detection (VAD) via VoiceActivityDetection pipeline
    - Overlapped Speech Detection via Inference with custom binarization
    - Speaker segmentation within chunks
    - Progressive speaker reference building across chunks
    - Multiple audio input formats supported
    """
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        min_duration_on: float = 0.1,
        min_duration_off: float = 0.3,
        max_speakers_per_chunk: int = 3,
        max_speakers_per_frame: int = 2,
        chunk_duration: float = 10.0,
        overlap_threshold: float = 0.5,
    ):
        """
        Initialize the SpeakerLabeler.
        
        Args:
            hf_token: HuggingFace access token (required for model download)
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            min_duration_on: Minimum duration (seconds) for speech regions
            min_duration_off: Minimum duration (seconds) for non-speech gaps
            max_speakers_per_chunk: Maximum unique speakers per chunk
            max_speakers_per_frame: Maximum overlapping speakers per frame
            chunk_duration: Duration of each processing chunk in seconds
            overlap_threshold: Threshold for detecting overlapping speech (0-1)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required. Provide it or set HF_TOKEN environment variable.\n"
                "Get your token at: https://huggingface.co/settings/tokens\n"
                "Accept model terms at: https://huggingface.co/pyannote/segmentation-3.0"
            )
        
        # Convert device string to torch.device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.chunk_duration = chunk_duration
        self.overlap_threshold = overlap_threshold
        
        self.speaker_references: Dict[str, List[Segment]] = {}
        self.all_segments: List[Dict] = []
        
        self.model = None
        self.vad_pipeline = None
        self.segmentation_inference = None  # Will be Inference instance
        self._classes = []
        self._overlap_class_indices = []
        
        self._load_model()
        console.log("[green]✓ SpeakerLabeler initialized successfully[/green]")

    def _load_model(self):
        """Load the segmentation model and set up pipelines."""
        with console.status("[bold blue]Loading pyannote/segmentation-3.0 model...[/bold blue]"):
            try:
                # Load the model
                self.model = Model.from_pretrained(
                    "pyannote/segmentation-3.0",
                    use_auth_token=self.hf_token,
                ).to(self.device)
                
                self._classes = self.model.specifications.classes
                console.log(f"[dim]Model classes: {self._classes}[/dim]")
                
                # Identify overlap classes (classes with multiple speakers)
                self._overlap_class_indices = self._identify_overlap_classes()
                console.log(f"[dim]Overlap class indices: {self._overlap_class_indices}[/dim]")
                
                # Setup VAD pipeline
                self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
                vad_params = {
                    "min_duration_on": self.min_duration_on,
                    "min_duration_off": self.min_duration_off,
                }
                self.vad_pipeline.instantiate(vad_params)
                
                # Setup Inference for segmentation (using 'whole' window to get 2D output)
                # Pass device as torch.device object
                self.segmentation_inference = Inference(
                    self.model,
                    window="whole",  # 'whole' gives 2D output, 'sliding' gives 3D
                    duration=self.chunk_duration,
                    step=self.chunk_duration,  # No overlap between chunks
                    device=self.device,  # Now passing torch.device, not string
                )
                
                console.log("[green]✓ Model and pipelines loaded[/green]")
            except Exception as e:
                console.log(f"[red]✗ Failed to load model: {e}[/red]")
                raise
    
    def _identify_overlap_classes(self) -> List[int]:
        """
        Identify which class indices represent overlapping speech.
        
        For powerset encoding, overlap classes have multiple speaker labels
        (e.g., "SPEAKER_00_SPEAKER_01").
        
        Returns:
            List of class indices that represent overlapping speech.
        """
        overlap_indices = []
        for i, class_name in enumerate(self._classes):
            # Count occurrences of "SPEAKER" in the class name
            speaker_count = class_name.count("SPEAKER")
            if speaker_count > 1:
                overlap_indices.append(i)
        return overlap_indices
    
    def _get_overlap_class_names(self) -> List[str]:
        """Get the names of classes that represent overlapping speech."""
        return [self._classes[i] for i in self._overlap_class_indices]
    
    def _extract_overlap_regions(
        self,
        segmentation_output: SlidingWindowFeature,
        chunk_start_time: float = 0.0
    ) -> List[Tuple[float, float]]:
        """
        Extract overlapping speech regions from segmentation output.
        
        Args:
            segmentation_output: 2D SlidingWindowFeature from Inference
            chunk_start_time: Start time offset for this chunk
            
        Returns:
            List of (start, end) tuples for overlapping regions
        """
        overlap_regions = []
        overlap_class_names = self._get_overlap_class_names()
        
        if not overlap_class_names:
            console.log("[dim]  No overlap classes detected in model[/dim]")
            return overlap_regions
        
        console.log(f"[dim]  Processing {len(overlap_class_names)} overlap classes[/dim]")
        
        # For each overlap class, binarize and extract segments
        for class_name in overlap_class_names:
            class_idx = self._classes.index(class_name)
            
            # Extract scores for this class (shape: num_frames, 1)
            class_scores_data = segmentation_output.data[:, class_idx:class_idx+1]
            class_scores = SlidingWindowFeature(
                class_scores_data,
                segmentation_output.sliding_window
            )
            
            # Binarize using hysteresis thresholding
            binarizer = Binarize(
                onset=self.overlap_threshold,
                offset=self.overlap_threshold * 0.8,  # Lower offset for hysteresis
                min_duration_on=0.05,  # Small segments to catch overlaps
                min_duration_off=0.05,
            )
            
            annotation = binarizer(class_scores)
            
            # Add segments with time offset
            for segment, _, _ in annotation.itertracks(yield_label=True):
                overlap_regions.append((
                    segment.start + chunk_start_time,
                    segment.end + chunk_start_time
                ))
        
        # Merge overlapping regions
        if overlap_regions:
            overlap_regions.sort()
            merged = []
            current_start, current_end = overlap_regions[0]
            for start, end in overlap_regions[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            merged.append((current_start, current_end))
            return merged
        
        return overlap_regions
    
    def _prepare_audio(self, audio: AudioInput) -> Tuple[np.ndarray, int, Optional[str]]:
        """
        Prepare audio input into standardized format.
        
        Returns:
            Tuple of (waveform numpy array, sample_rate, temp_file_path or None)
        """
        temp_file = None
        try:
            if isinstance(audio, (str, os.PathLike)):
                audio_path = str(audio)
                console.log(f"[cyan]Loading audio file: {audio_path}[/cyan]")
                waveform, sample_rate = sf.read(audio_path, dtype='float32')
                # Convert to mono if stereo
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform, sample_rate, audio_path
            
            elif isinstance(audio, bytes):
                console.log("[cyan]Processing raw bytes audio input[/cyan]")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio)
                    temp_file = f.name
                waveform, sample_rate = sf.read(temp_file, dtype='float32')
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform, sample_rate, temp_file
            
            elif isinstance(audio, np.ndarray):
                console.log(f"[cyan]Processing NumPy array input (shape: {audio.shape})[/cyan]")
                if audio.dtype in [np.float32, np.float64]:
                    waveform = audio.astype(np.float32)
                else:
                    waveform = audio.astype(np.float32) / np.iinfo(audio.dtype).max
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform.flatten(), 16000, None
            
            elif isinstance(audio, torch.Tensor):
                console.log(f"[cyan]Processing PyTorch tensor input (shape: {audio.shape})[/cyan]")
                waveform = audio.cpu().numpy().astype(np.float32)
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform.flatten(), 16000, None
            
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio)}")
        
        except Exception as e:
            console.log(f"[red]✗ Error preparing audio: {e}[/red]")
            raise
    
    def _save_temp_audio(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Save numpy array to temporary WAV file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(temp_file, waveform, sample_rate)
        return temp_file
    
    def _get_speaker_label(self, speaker_num: int) -> str:
        """Generate a speaker label."""
        return f"SPEAKER_{speaker_num:02d}"
    
    def _run_segmentation_on_chunk(
        self,
        chunk_file: str,
        chunk_start_time: float
    ) -> Tuple[Annotation, List[Tuple[float, float]]]:
        """
        Run segmentation on a single chunk using Inference.
        
        Returns:
            Tuple of (vad_result, overlap_regions)
        """
        # Run VAD
        console.log("[yellow]  → Running VAD...[/yellow]")
        vad_result = self.vad_pipeline(chunk_file)
        
        # Run segmentation using Inference (returns 2D SlidingWindowFeature)
        console.log("[yellow]  → Running segmentation and detecting overlaps...[/yellow]")
        
        # Use inference on the chunk file
        segmentation_output = self.segmentation_inference(chunk_file)
        
        # Extract overlap regions
        overlap_regions = self._extract_overlap_regions(segmentation_output, chunk_start_time)
        
        return vad_result, overlap_regions
    
    def process_chunk(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        chunk_start_time: float = 0.0
    ) -> Dict:
        """
        Process a single chunk with the segmentation model.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            chunk_start_time: Start time of this chunk in the overall recording
            
        Returns:
            Dictionary with VAD, overlap detection, and speaker segmentation results
        """
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            console.log(f"[dim]Resampling chunk from {sample_rate}Hz to 16000Hz[/dim]")
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Ensure chunk has correct duration (pad or trim)
        expected_samples = int(self.chunk_duration * sample_rate)
        if len(waveform) < expected_samples:
            pad_length = expected_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length))
        elif len(waveform) > expected_samples:
            waveform = waveform[:expected_samples]
        
        # Save to temporary file
        temp_chunk_file = self._save_temp_audio(waveform, sample_rate)
        
        try:
            # Run VAD and segmentation
            vad_result, overlap_regions = self._run_segmentation_on_chunk(
                temp_chunk_file, chunk_start_time
            )
            
            # Extract speaker segments
            speaker_segments = self._extract_speaker_segments(
                vad_result,
                overlap_regions,
                chunk_start_time
            )
            
            return {
                "vad": vad_result,
                "osd_regions": overlap_regions,
                "speaker_segments": speaker_segments,
            }
            
        finally:
            if os.path.exists(temp_chunk_file):
                os.unlink(temp_chunk_file)
    
    def _extract_speaker_segments(
        self,
        vad_result: Annotation,
        overlap_regions: List[Tuple[float, float]],
        chunk_start_time: float = 0.0
    ) -> List[Dict]:
        """
        Extract and label speaker segments from VAD and overlap detection results.
        
        Args:
            vad_result: Voice activity detection Annotation
            overlap_regions: List of (start, end) tuples for overlapping speech
            chunk_start_time: Start time offset for this chunk
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        def is_overlapping(seg_start: float, seg_end: float) -> bool:
            """Check if a segment overlaps with any detected overlap region."""
            for o_start, o_end in overlap_regions:
                if seg_start < o_end and seg_end > o_start:
                    return True
            return False
        
        # Get all speech segments from VAD
        for segment, _, label in vad_result.itertracks(yield_label=True):
            start_time = segment.start + chunk_start_time
            end_time = segment.end + chunk_start_time
            duration = end_time - start_time
            
            is_overlapped = is_overlapping(start_time, end_time)
            
            # Simple speaker assignment - in production, you'd use clustering
            speaker_idx = (len(self.speaker_references) % self.max_speakers_per_chunk) + 1
            speaker_label = self._get_speaker_label(speaker_idx)
            
            # Track speaker references
            if speaker_label not in self.speaker_references:
                self.speaker_references[speaker_label] = []
            
            segment_info = {
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "speaker": speaker_label,
                "is_overlapped": is_overlapped,
                "chunk_id": f"chunk_{chunk_start_time:.1f}",
            }
            
            segments.append(segment_info)
            self.speaker_references[speaker_label].append(Segment(start_time, end_time))
            self.all_segments.append(segment_info)
        
        return segments
    
    def label_speakers(self, audio: AudioInput) -> Dict:
        """
        Main method to label speakers in an audio input.
        
        Args:
            audio: Audio input in any supported format
            
        Returns:
            Dictionary with complete labeling results
        """
        # Reset state
        self.speaker_references.clear()
        self.all_segments.clear()
        
        console.print(Panel.fit(
            "[bold blue]Starting Speaker Labeling Process[/bold blue]",
            border_style="blue"
        ))
        
        # Prepare audio
        waveform, sample_rate, file_path = self._prepare_audio(audio)
        duration = len(waveform) / sample_rate
        console.print(f"[cyan]Audio Duration: {duration:.2f}s | Sample Rate: {sample_rate}Hz[/cyan]")
        
        # Resample entire file to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            console.log(f"[dim]Resampling entire file from {sample_rate}Hz to 16000Hz[/dim]")
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            duration = len(waveform) / sample_rate
        
        # Process in chunks
        chunk_samples = int(self.chunk_duration * sample_rate)
        
        all_results = {
            "audio_info": {
                "duration": duration,
                "sample_rate": sample_rate,
                "num_channels": 1,
            },
            "speaker_segments": [],
            "vad_segments": [],
            "overlap_regions": [],
            "summary": {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[green]Processing audio chunks...", total=None)
            
            for i, start_sample in enumerate(range(0, len(waveform), chunk_samples)):
                end_sample = min(start_sample + chunk_samples, len(waveform))
                chunk_waveform = waveform[start_sample:end_sample]
                chunk_start_time = start_sample / sample_rate
                
                # Skip if chunk is too short (less than 0.5 seconds)
                if len(chunk_waveform) < sample_rate * 0.5:
                    console.log(f"[dim]Skipping short chunk {i+1} ({len(chunk_waveform)/sample_rate:.2f}s)[/dim]")
                    continue
                
                console.log(
                    f"[blue]Processing chunk {i+1}: "
                    f"{chunk_start_time:.1f}s - {end_sample/sample_rate:.1f}s[/blue]"
                )
                
                chunk_results = self.process_chunk(
                    chunk_waveform,
                    sample_rate,
                    chunk_start_time
                )
                
                # Collect VAD segments
                for segment, _, label in chunk_results["vad"].itertracks(yield_label=True):
                    all_results["vad_segments"].append({
                        "start": segment.start + chunk_start_time,
                        "end": segment.end + chunk_start_time,
                        "label": label
                    })
                
                # Collect overlap regions
                for start, end in chunk_results["osd_regions"]:
                    all_results["overlap_regions"].append({
                        "start": start,
                        "end": end,
                        "duration": end - start
                    })
                
                # Collect speaker segments
                all_results["speaker_segments"].extend(chunk_results["speaker_segments"])
            
            progress.update(task, completed=True)
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results)
        
        # Clean up temp file if we created one
        if file_path and isinstance(audio, bytes):
            try:
                os.unlink(file_path)
            except:
                pass
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a summary of the labeling results."""
        speaker_segments = results["speaker_segments"]
        
        summary = {
            "total_duration": results["audio_info"]["duration"],
            "total_speech_segments": len(speaker_segments),
            "total_speech_duration": sum(seg["duration"] for seg in speaker_segments),
            "total_overlapped_segments": sum(1 for seg in speaker_segments if seg["is_overlapped"]),
            "total_overlap_duration": sum(
                seg["duration"] for seg in speaker_segments if seg["is_overlapped"]
            ),
            "unique_speakers": len(set(seg["speaker"] for seg in speaker_segments)),
            "speaker_breakdown": {},
            "vad_segments_count": len(results["vad_segments"]),
            "overlap_regions_count": len(results["overlap_regions"]),
        }
        
        for seg in speaker_segments:
            speaker = seg["speaker"]
            if speaker not in summary["speaker_breakdown"]:
                summary["speaker_breakdown"][speaker] = {
                    "total_duration": 0.0,
                    "segment_count": 0,
                    "overlapped_segments": 0,
                    "overlap_duration": 0.0,
                }
            summary["speaker_breakdown"][speaker]["total_duration"] += seg["duration"]
            summary["speaker_breakdown"][speaker]["segment_count"] += 1
            if seg["is_overlapped"]:
                summary["speaker_breakdown"][speaker]["overlapped_segments"] += 1
                summary["speaker_breakdown"][speaker]["overlap_duration"] += seg["duration"]
        
        return summary
    
    def display_results(self, results: Dict):
        """Display labeling results in a rich formatted table."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]Speaker Labeling Results[/bold green]",
            border_style="green"
        ))
        
        console.print(f"[cyan]Audio Duration: {results['audio_info']['duration']:.2f}s[/cyan]")
        console.print(f"[cyan]Sample Rate: {results['audio_info']['sample_rate']}Hz[/cyan]\n")
        
        summary = results["summary"]
        
        table = Table(title="Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Speech Segments", str(summary["total_speech_segments"]))
        table.add_row("Total Speech Duration", f"{summary['total_speech_duration']:.2f}s")
        table.add_row("Overlapped Segments", str(summary["total_overlapped_segments"]))
        table.add_row("Overlap Duration", f"{summary['total_overlap_duration']:.2f}s")
        table.add_row("Unique Speakers Detected", str(summary["unique_speakers"]))
        table.add_row("VAD Raw Segments", str(summary["vad_segments_count"]))
        table.add_row("Overlap Regions", str(summary["overlap_regions_count"]))
        
        console.print(table)
        console.print("\n")
        
        if summary["speaker_breakdown"]:
            speaker_table = Table(
                title="Speaker Breakdown",
                show_header=True,
                header_style="bold blue"
            )
            speaker_table.add_column("Speaker", style="green")
            speaker_table.add_column("Duration", justify="right")
            speaker_table.add_column("Segments", justify="right")
            speaker_table.add_column("Overlapped", justify="right")
            speaker_table.add_column("Overlap %", justify="right")
            
            for speaker, info in summary["speaker_breakdown"].items():
                overlap_pct = (
                    (info["overlap_duration"] / info["total_duration"] * 100)
                    if info["total_duration"] > 0
                    else 0.0
                )
                speaker_table.add_row(
                    speaker,
                    f"{info['total_duration']:.2f}s",
                    str(info["segment_count"]),
                    str(info["overlapped_segments"]),
                    f"{overlap_pct:.1f}%",
                )
            
            console.print(speaker_table)
        
        if results["overlap_regions"]:
            console.print("\n[bold]Overlap Regions (first 10):[/bold]")
            for region in results["overlap_regions"][:10]:
                console.print(f"  {region['start']:.2f}s - {region['end']:.2f}s ({region['duration']:.2f}s)")
        
        if results["speaker_segments"]:
            console.print("\n[bold]Last 5 Speaker Segments:[/bold]")
            for seg in results["speaker_segments"][-5:]:
                overlap_marker = "[red]OVERLAP[/red]" if seg["is_overlapped"] else "[dim]clean[/dim]"
                console.print(
                    f"  {seg['start']:.2f}s - {seg['end']:.2f}s | "
                    f"{seg['speaker']} | {seg['duration']:.2f}s | {overlap_marker}"
                )

    def get_speakers_at_time(self, timestamp: float) -> List[Dict]:
        """
        Get all active speakers at a specific timestamp.
        
        Args:
            timestamp: Time in seconds from start of audio
            
        Returns:
            List of dicts with 'speaker' and 'confidence' keys
        """
        active_speakers = []
        for segment_info in self.all_segments:
            if segment_info["start"] <= timestamp <= segment_info["end"]:
                active_speakers.append({
                    "speaker": segment_info["speaker"],
                    "start": segment_info["start"],
                    "end": segment_info["end"],
                    "confidence": 0.8,  # Placeholder - can be improved with actual confidence
                    "is_overlapped": segment_info["is_overlapped"],
                })
        return active_speakers

    def get_speaker_timeline(self) -> List[Dict]:
        """
        Get complete timeline of speaker segments.
        
        Returns:
            List of all speaker segments with timing and speaker info
        """
        return self.all_segments.copy()

    def get_unique_speakers(self) -> List[str]:
        """
        Get list of all unique speakers detected.
        
        Returns:
            List of speaker labels
        """
        return list(self.speaker_references.keys())

    def get_speaker_stats(self) -> Dict:
        """
        Get comprehensive statistics about detected speakers.
        
        Returns:
            Dictionary with speaker statistics
        """
        stats = {}
        for speaker, segments in self.speaker_references.items():
            total_duration = sum(seg.duration for seg in segments)
            stats[speaker] = {
                "segment_count": len(segments),
                "total_duration_seconds": total_duration,
                "average_segment_duration": total_duration / len(segments) if segments else 0,
            }
        return stats

    def reset(self) -> None:
        """Reset all speaker labeling state."""
        self.speaker_references.clear()
        self.all_segments.clear()
        self.global_speaker_counter = 0
        console.log("[yellow]✓ Speaker labeler state reset[/yellow]")


if __name__ == "__main__":
    from _main_speaker_labeler import main
    main()
