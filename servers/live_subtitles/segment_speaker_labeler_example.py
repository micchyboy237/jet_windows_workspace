from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from scipy.spatial.distance import cdist
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from segment_speaker_labeler import SegmentSpeakerLabeler

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo / Usage Example
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Demo script for SegmentSpeakerLabeler.
    
    This demo simulates a conversation between two speakers with multiple
    audio segments arriving in sequence. It demonstrates:
    
    1. Initial speaker creation
    2. Progressive matching against known speakers
    3. Temporal smoothing to prevent label flickering
    4. Confidence scores and match types
    5. State serialization and restoration
    6. Speaker merging
    
    For real usage, replace the mock embedding model with:
    
        from pyannote.audio import Inference, Model
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")
    
    Then pass `embedding_model=inference` to SegmentSpeakerLabeler.
    """
    
    # ─── Mock Embedding Model for Demo ───────────────────────────────────────
    class MockEmbeddingModel:
        """Mock embedding model that returns synthetic embeddings.
        
        In a real application, replace this with pyannote.audio Inference.
        This mock takes a dict with optional 'speaker_id' key for demo purposes.
        """
        def __init__(self, dimension: int = 192, seed: int = 42):
            self.dimension = dimension
            self.rng = np.random.RandomState(seed)
            
            # Create fixed "voice prints" for two speakers
            self.speaker_0_voice = self.rng.randn(1, dimension)
            self.speaker_1_voice = self.rng.randn(1, dimension)
            
            # Normalize for cosine similarity
            self.speaker_0_voice = self.speaker_0_voice / np.linalg.norm(self.speaker_0_voice)
            self.speaker_1_voice = self.speaker_1_voice / np.linalg.norm(self.speaker_1_voice)
        
        def __call__(self, audio_dict: dict) -> np.ndarray:
            """Simulate embedding computation with noise.
            
            Uses a hidden "speaker_id" in the audio dict to generate
            consistent embeddings per speaker with some noise.
            
            Parameters
            ----------
            audio_dict : dict
                Must contain:
                    - "waveform": torch.Tensor
                    - "sample_rate": int
                May contain:
                    - "speaker_id": int (0 or 1) for demo speaker selection
            """
            speaker_id = audio_dict.get("speaker_id", 0)
            noise_level = 0.15
            
            if speaker_id == 0:
                base = self.speaker_0_voice.copy()
            else:
                base = self.speaker_1_voice.copy()
            
            # Add noise to simulate real-world variation
            noise = self.rng.randn(1, self.dimension) * noise_level
            embedding = base + noise
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    # ─── Custom SegmentSpeakerLabeler with Mock Support ──────────────────────
    class MockSegmentSpeakerLabeler(SegmentSpeakerLabeler):
        """Extended labeler that supports mock speaker_id in waveforms."""
        
        def compute_embedding(
            self,
            waveform,
            sample_rate: int,
            speaker_id: int = 0,
        ) -> np.ndarray:
            """Override to pass speaker_id to mock model.
            
            Parameters
            ----------
            waveform : torch.Tensor
                Audio waveform.
            sample_rate : int
                Sample rate.
            speaker_id : int
                Mock speaker ID for demo.
            
            Returns
            -------
            np.ndarray
                Speaker embedding.
            """
            # Pass speaker_id through the audio dict to the mock model
            return self.embedding_model({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "speaker_id": speaker_id,
            })
        
        def label_segment_with_speaker_id(
            self,
            waveform: torch.Tensor,
            sample_rate: int,
            timestamp: float,
            speaker_id: int = 0,
            context: Optional[Dict] = None,
        ) -> Tuple[str, float, Dict]:
            """Label a segment with a known mock speaker ID for demo purposes.
            
            Parameters
            ----------
            waveform : torch.Tensor
                Audio waveform.
            sample_rate : int
                Sample rate.
            timestamp : float
                Segment timestamp.
            speaker_id : int
                Mock speaker ID (0 or 1).
            context : dict, optional
                Additional context.
            
            Returns
            -------
            Tuple[str, float, Dict]
                Label, confidence, metadata.
            """
            self.total_segments_processed += 1
            
            # Compute embedding with speaker_id for mock
            embedding = self.compute_embedding(
                waveform, sample_rate, speaker_id=speaker_id
            )
            
            # Find best match among known speakers
            best_label, best_score, all_scores = self.find_best_match(embedding)
            
            metadata = {
                "timestamp": timestamp,
                "all_scores": all_scores,
                "is_new_speaker": False,
                "match_type": "none",
            }
            
            assigned_label = None
            confidence = 0.0
            
            # Decision logic
            if best_label is None:
                assigned_label = self.create_new_speaker(embedding, timestamp)
                metadata["is_new_speaker"] = True
                metadata["match_type"] = "first_speaker"
                confidence = 1.0
                
            elif best_score >= self.threshold_same:
                ref = self._speakers[best_label]
                if ref.segment_count >= self.min_segments_for_reference:
                    assigned_label = best_label
                    confidence = best_score
                    metadata["match_type"] = "strong_match"
                else:
                    assigned_label = best_label
                    confidence = best_score * 0.9
                    metadata["match_type"] = "early_match"
                
            elif best_score >= self.threshold_possible:
                smoothed_label = self.apply_temporal_smoothing(
                    best_label, timestamp, best_score
                )
                
                if context and "previous_speaker" in context:
                    prev_speaker = context["previous_speaker"]
                    if prev_speaker in self._speakers:
                        prev_similarity = all_scores.get(prev_speaker, 0.0)
                        if prev_similarity >= self.threshold_same - 0.05:
                            assigned_label = prev_speaker
                            confidence = prev_similarity
                            metadata["match_type"] = "context_match"
                        else:
                            assigned_label = smoothed_label
                            confidence = best_score
                            metadata["match_type"] = "possible_match"
                    else:
                        assigned_label = smoothed_label
                        confidence = best_score
                        metadata["match_type"] = "possible_match"
                else:
                    assigned_label = smoothed_label
                    confidence = best_score
                    metadata["match_type"] = "possible_match"
                
            else:
                assigned_label = self.create_new_speaker(embedding, timestamp)
                metadata["is_new_speaker"] = True
                metadata["match_type"] = "new_speaker"
                confidence = 1.0 - best_score
            
            self.update_reference(assigned_label, embedding, timestamp)
            
            if self.debug:
                console.print(
                    f"[dim]Segment {self.total_segments_processed}: "
                    f"t={timestamp:.2f}s → {assigned_label} "
                    f"(confidence: {confidence:.3f}, type: {metadata['match_type']})[/]"
                )
            
            return assigned_label, confidence, metadata
    
    # ─── Demo Setup ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Segment Speaker Labeler Demo[/bold cyan]")
    console.print()
    
    console.print(
        Panel.fit(
            "[bold]Scenario:[/bold] Simulated conversation between 2 speakers.\n"
            "Audio segments arrive in sequence.\n"
            "The labeler progressively learns and tracks speakers.",
            title="Demo Overview",
            border_style="cyan",
        )
    )
    console.print()
    
    # Initialize with mock model
    mock_model = MockEmbeddingModel(dimension=192)
    labeler = MockSegmentSpeakerLabeler(
        embedding_model=mock_model,
        threshold_same=0.15,
        threshold_possible=0.075,
        min_segments_for_reference=2,
        max_embeddings_per_speaker=50,
        temporal_smoothing_window=3.0,
        debug=True,
    )
    
    # ─── Simulated Audio Segments ────────────────────────────────────────────
    # Each segment has:
    #   - timestamp: when the segment occurs (seconds)
    #   - speaker_id: actual speaker (0 or 1) for mock embedding generation
    #   - duration: length of segment in seconds
    
    segments = [
        # Speaker 0 starts talking (segments 1-4)
        {"timestamp": 0.5,  "speaker_id": 0, "duration": 2.0},
        {"timestamp": 3.0,  "speaker_id": 0, "duration": 1.5},
        {"timestamp": 5.0,  "speaker_id": 0, "duration": 2.5},
        {"timestamp": 8.0,  "speaker_id": 0, "duration": 1.0},
        
        # Speaker 1 interjects (segments 5-7)
        {"timestamp": 10.0, "speaker_id": 1, "duration": 1.5},
        {"timestamp": 12.0, "speaker_id": 1, "duration": 2.0},
        {"timestamp": 14.5, "speaker_id": 1, "duration": 1.0},
        
        # Back to Speaker 0 (segments 8-10)
        {"timestamp": 16.0, "speaker_id": 0, "duration": 2.0},
        {"timestamp": 18.5, "speaker_id": 0, "duration": 1.5},
        {"timestamp": 20.5, "speaker_id": 0, "duration": 2.0},
        
        # Speaker 1 again (segments 11-12)
        {"timestamp": 23.0, "speaker_id": 1, "duration": 1.5},
        {"timestamp": 25.0, "speaker_id": 1, "duration": 2.0},
        
        # Brief Speaker 0 (segment 13)
        {"timestamp": 27.5, "speaker_id": 0, "duration": 0.8},
    ]
    
    # ─── Process Segments ────────────────────────────────────────────────────
    console.print("[bold]Processing segments...[/bold]\n")
    
    results = []
    current_speaker = None
    
    for i, seg in enumerate(segments, 1):
        # Create a dummy waveform tensor (silence) for the demo
        num_samples = int(seg["duration"] * 16000)
        waveform = torch.zeros(1, num_samples)
        
        # Build context for temporal smoothing
        context = {
            "previous_speaker": current_speaker,
            "segment_duration": seg["duration"],
            "segment_index": i,
        }
        
        # Label the segment with mock speaker_id
        label, confidence, metadata = labeler.label_segment_with_speaker_id(
            waveform=waveform,
            sample_rate=16000,
            timestamp=seg["timestamp"],
            speaker_id=seg["speaker_id"],
            context=context,
        )
        
        # Update current speaker tracking
        if label != current_speaker:
            current_speaker = label
        
        results.append({
            "segment": i,
            "timestamp": seg["timestamp"],
            "actual_speaker": f"Person_{seg['speaker_id']}",
            "labeled_as": label,
            "confidence": confidence,
            "match_type": metadata["match_type"],
            "all_scores": metadata.get("all_scores", {}),
        })
    
    # ─── Display Results Table ───────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Results[/bold green]")
    console.print()
    
    table = Table(
        title="Segment Labeling Results",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    
    table.add_column("Seg", justify="right", style="dim")
    table.add_column("Time", justify="right", style="magenta")
    table.add_column("Actual", style="yellow")
    table.add_column("Labeled As", style="green bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Match Type")
    
    for r in results:
        # Color code confidence
        conf = r["confidence"]
        if conf >= 0.85:
            conf_str = f"[green]{conf:.3f}[/green]"
        elif conf >= 0.65:
            conf_str = f"[yellow]{conf:.3f}[/yellow]"
        else:
            conf_str = f"[red]{conf:.3f}[/red]"
        
        # Color code match type
        mt = r["match_type"]
        if mt == "strong_match":
            mt_str = f"[green]{mt}[/green]"
        elif mt in ("early_match", "context_match"):
            mt_str = f"[yellow]{mt}[/yellow]"
        elif mt == "possible_match":
            mt_str = f"[dim yellow]{mt}[/dim yellow]"
        else:
            mt_str = f"[cyan]{mt}[/cyan]"
        
        table.add_row(
            str(r["segment"]),
            f"{r['timestamp']:.1f}s",
            r["actual_speaker"],
            r["labeled_as"],
            conf_str,
            mt_str,
        )
    
    console.print(table)
    
    # ─── Similarity Scores Detail ────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Similarity Scores Detail[/bold green]")
    console.print()
    
    score_table = Table(
        title="Per-Segment Similarity Scores",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    
    score_table.add_column("Seg", justify="right", style="dim")
    score_table.add_column("Time", justify="right", style="magenta")
    score_table.add_column("Chosen", style="green bold")
    
    # Dynamic columns for each speaker
    max_speakers = max(len(r["all_scores"]) for r in results) if results else 0
    for si in range(max_speakers):
        score_table.add_column(f"Spk_{si+1}", justify="right")
    
    for r in results:
        row = [
            str(r["segment"]),
            f"{r['timestamp']:.1f}s",
            r["labeled_as"],
        ]
        
        for label in sorted(r["all_scores"].keys()):
            score = r["all_scores"][label]
            if score >= 0.85:
                row.append(f"[green]{score:.3f}[/green]")
            elif score >= 0.65:
                row.append(f"[yellow]{score:.3f}[/yellow]")
            else:
                row.append(f"[red]{score:.3f}[/red]")
        
        # Pad remaining columns if fewer speakers
        row.extend(["[dim]—[/dim]"] * (max_speakers - len(r["all_scores"])))
        
        score_table.add_row(*row)
    
    console.print(score_table)
    
    # ─── Speaker Summary ─────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Speaker Summary[/bold green]")
    console.print()
    
    speaker_table = Table(
        title="Known Speakers",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    
    speaker_table.add_column("Label", style="green bold")
    speaker_table.add_column("Segments", justify="right", style="bright_white")
    speaker_table.add_column("First Seen", justify="right", style="magenta")
    speaker_table.add_column("Last Seen", justify="right", style="magenta")
    speaker_table.add_column("Duration", justify="right", style="yellow")
    
    for label in labeler.known_speakers:
        info = labeler.get_speaker_info(label)
        speaker_table.add_row(
            info["label"],
            str(info["segment_count"]),
            f"{info['first_seen']:.1f}s",
            f"{info['last_seen']:.1f}s",
            f"{info['active_duration']:.1f}s",
        )
    
    console.print(speaker_table)
    console.print()
    
    console.print(
        f"[bold]Total segments processed:[/bold] "
        f"[bright_white]{labeler.total_segments_processed}[/bright_white]"
    )
    console.print(
        f"[bold]Total speakers created:[/bold] "
        f"[bright_white]{labeler.total_speakers_created}[/bright_white]"
    )
    
    # ─── Accuracy Summary ────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Accuracy Check[/bold green]")
    console.print()
    
    # Check if Person_0 consistently maps to one label and Person_1 to another
    person0_labels = set()
    person1_labels = set()
    
    for r in results:
        if r["actual_speaker"] == "Person_0":
            person0_labels.add(r["labeled_as"])
        else:
            person1_labels.add(r["labeled_as"])
    
    if len(person0_labels) == 1 and len(person1_labels) == 1:
        if person0_labels != person1_labels:
            console.print(
                f"[green]✓ Perfect! Person_0 → {person0_labels.pop()}, "
                f"Person_1 → {person1_labels.pop()}[/green]"
            )
        else:
            console.print(
                f"[yellow]⚠ Both speakers got the same label: {person0_labels.pop()}[/yellow]"
            )
    else:
        console.print(
            f"[yellow]⚠ Label mixing detected: "
            f"Person_0 → {person0_labels}, Person_1 → {person1_labels}[/yellow]"
        )
    
    # ─── Serialization Demo ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Serialization Demo[/bold green]")
    console.print()
    
    # Serialize state
    state = labeler.to_dict()
    console.print("[bold]State serialized successfully[/bold]")
    console.print(f"  Speakers saved: [bright_white]{len(state['speakers'])}[/bright_white]")
    console.print(f"  Next speaker ID: [bright_white]{state['next_speaker_id']}[/bright_white]")
    console.print(f"  Total segments: [bright_white]{state['total_segments_processed']}[/bright_white]")
    
    # Check state structure
    for label, spk_data in state["speakers"].items():
        console.print(
            f"  [dim]{label}: {spk_data['segment_count']} embeddings, "
            f"{len(spk_data['embeddings'])} stored[/dim]"
        )
    
    # Create a new labeler and restore state
    console.print()
    console.print("[bold]Creating new labeler from saved state...[/bold]")
    
    labeler2 = MockSegmentSpeakerLabeler.from_dict(
        state,
        embedding_model=mock_model,
    )
    
    console.print(f"  Restored speakers: [bright_white]{labeler2.speaker_count}[/bright_white]")
    console.print(f"  Known speakers: [bright_white]{labeler2.known_speakers}[/bright_white]")
    
    # Verify restored state by processing a new segment
    console.print()
    console.print("[bold]Verifying restored labeler with new segment...[/bold]")
    
    new_waveform = torch.zeros(1, 16000)  # 1 second of silence
    
    label, confidence, metadata = labeler2.label_segment_with_speaker_id(
        waveform=new_waveform,
        sample_rate=16000,
        timestamp=30.0,
        speaker_id=0,
        context={"previous_speaker": labeler2.known_speakers[0] if labeler2.known_speakers else None},
    )
    
    console.print(
        f"  New segment (Person_0) → [green]{label}[/green] "
        f"(confidence: {confidence:.3f}, type: {metadata['match_type']})"
    )
    
    # ─── Speaker Merge Demo ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Speaker Merge Demo[/bold green]")
    console.print()
    
    # Create a scenario where we mistakenly have 3 speakers
    console.print("[dim]Simulating accidental speaker split...[/dim]")
    console.print()
    
    # Add some segments for a "third" speaker (actually same as speaker 0 but
    # with different mock_id for demonstration)
    labeler.reset()
    console.print("[yellow]Labeler reset for merge demo[/yellow]")
    
    # First, build up two speakers normally
    for i, seg in enumerate(segments[:8], 1):
        waveform = torch.zeros(1, int(seg["duration"] * 16000))
        label, _, _ = labeler.label_segment_with_speaker_id(
            waveform=waveform,
            sample_rate=16000,
            timestamp=seg["timestamp"],
            speaker_id=seg["speaker_id"],
            context={"previous_speaker": current_speaker},
        )
        current_speaker = label
    
    console.print()
    console.print(f"[bold]Speakers before merge:[/bold] [bright_white]{labeler.known_speakers}[/bright_white]")
    
    if labeler.speaker_count >= 2:
        speakers = labeler.known_speakers
        console.print()
        console.print(
            f"[yellow]Merging [bold]{speakers[0]}[/bold] and [bold]{speakers[1]}[/bold]...[/yellow]"
        )
        merged = labeler.merge_speakers(speakers[0], speakers[1])
        
        if merged:
            console.print(f"[green]✓ Merged into: [bold]{merged}[/bold][/green]")
            console.print(f"  Remaining speakers: [bright_white]{labeler.known_speakers}[/bright_white]")
            console.print(f"  Speaker count: [bright_white]{labeler.speaker_count}[/bright_white]")
            
            # Show merged speaker info
            info = labeler.get_speaker_info(merged)
            console.print(f"  Total segments for {merged}: [bright_white]{info['segment_count']}[/bright_white]")
        else:
            console.print("[red]✗ Merge failed[/red]")
    
    # ─── Reset Demo ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Reset Demo[/bold green]")
    console.print()
    
    labeler.reset()
    console.print("[yellow]✓ Labeler reset[/yellow]")
    console.print(f"  Speakers: [bright_white]{labeler.known_speakers or '[]'}[/bright_white]")
    console.print(f"  Speaker count: [bright_white]{labeler.speaker_count}[/bright_white]")
    console.print(f"  Total segments: [bright_white]{labeler.total_segments_processed}[/bright_white]")
    console.print(f"  Next speaker ID: [bright_white]{labeler._next_speaker_id}[/bright_white]")
    
    # Verify reset by processing first segment again
    console.print()
    console.print("[bold]Verifying reset with fresh segment...[/bold]")
    
    waveform = torch.zeros(1, int(segments[0]["duration"] * 16000))
    label, confidence, metadata = labeler.label_segment_with_speaker_id(
        waveform=waveform,
        sample_rate=16000,
        timestamp=segments[0]["timestamp"],
        speaker_id=segments[0]["speaker_id"],
    )
    
    console.print(
        f"  First segment after reset → [green]{label}[/green] "
        f"(confidence: {confidence:.3f}, type: {metadata['match_type']})"
    )
    console.print(f"  Speaker count: [bright_white]{labeler.speaker_count}[/bright_white]")
    
    # ─── Final Notes ─────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Demo Complete[/bold cyan]")
    console.print()
    
    console.print(
        Panel.fit(
            "[bold]Key Takeaways:[/bold]\n\n"
            "1. [green]Progressive Learning[/green]\n"
            "   The labeler builds speaker references incrementally\n"
            "   as segments arrive — no need for full audio upfront.\n\n"
            "2. [green]Automatic Detection[/green]\n"
            "   New speakers are automatically detected and labeled\n"
            "   when embeddings don't match existing references.\n\n"
            "3. [green]Temporal Smoothing[/green]\n"
            "   Prevents rapid label flickering by considering\n"
            "   recent label history before switching.\n\n"
            "4. [green]State Serialization[/green]\n"
            "   Full state can be saved/restored for persistence\n"
            "   across server restarts.\n\n"
            "5. [green]Speaker Merging[/green]\n"
            "   Accidentally split speakers can be merged post-hoc\n"
            "   without reprocessing audio.\n\n"
            "[dim]Replace MockEmbeddingModel with pyannote.audio Inference for real use.[/dim]",
            title="Summary",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()
