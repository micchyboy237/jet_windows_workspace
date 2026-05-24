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
        
        Generates consistent, separable embeddings for two speakers with
        controlled noise to simulate real-world variation.
        
        Key design:
        - Two orthogonal "voice prints" ensure speakers are distinguishable
        - Per-speaker seed gives consistent but slightly varied embeddings
        - Low noise level (0.03) simulates realistic within-speaker variation
        """
        def __init__(self, dimension: int = 192, seed: int = 42):
            self.dimension = dimension
            self.base_rng = np.random.RandomState(seed)
            
            # Create orthogonal "voice prints" for two speakers
            # Orthogonal vectors maximize cosine distance between speakers
            v0 = self.base_rng.randn(dimension)
            v1 = self.base_rng.randn(dimension)
            # Make orthogonal via Gram-Schmidt
            v1 = v1 - np.dot(v1, v0) * v0 / np.dot(v0, v0)
            
            self.speaker_0_voice = v0 / np.linalg.norm(v0)
            self.speaker_1_voice = v1 / np.linalg.norm(v1)
            
            # Per-speaker RNGs for consistent variation
            self.spk0_rng = np.random.RandomState(seed + 100)
            self.spk1_rng = np.random.RandomState(seed + 200)
            
            # Verify separation
            cosine_sim = np.dot(self.speaker_0_voice, self.speaker_1_voice)
            console.print(
                f"[dim]Mock voice prints cosine similarity: {cosine_sim:.6f} "
                f"(near 0 = well-separated)[/dim]"
            )
        
        def __call__(self, audio_dict: dict) -> np.ndarray:
            """Simulate embedding computation with realistic noise.
            
            Parameters
            ----------
            audio_dict : dict
                Must contain:
                    - "waveform": torch.Tensor
                    - "sample_rate": int
                May contain:
                    - "speaker_id": int (0 or 1)
                    - "noise_level": float (default 0.03)
            
            Returns
            -------
            np.ndarray
                Normalized speaker embedding of shape (1, dimension).
            """
            speaker_id = audio_dict.get("speaker_id", 0)
            noise_level = audio_dict.get("noise_level", 0.03)
            
            if speaker_id == 0:
                base = self.speaker_0_voice.copy()
                noise = self.spk0_rng.randn(self.dimension) * noise_level
            else:
                base = self.speaker_1_voice.copy()
                noise = self.spk1_rng.randn(self.dimension) * noise_level
            
            # Add noise to simulate real-world channel/session variation
            embedding = base + noise
            
            # L2 normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.reshape(1, -1)
    
    # ─── Custom SegmentSpeakerLabeler with Mock Support ──────────────────────
    class MockSegmentSpeakerLabeler(SegmentSpeakerLabeler):
        """Extended labeler that supports mock speaker_id in waveforms."""
        
        def compute_embedding(
            self,
            waveform,
            sample_rate: int,
            speaker_id: int = 0,
            noise_level: float = 0.03,
        ) -> np.ndarray:
            """Override to pass speaker_id and noise_level to mock model."""
            return self.embedding_model({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "speaker_id": speaker_id,
                "noise_level": noise_level,
            })
        
        def label_segment_with_speaker_id(
            self,
            waveform: torch.Tensor,
            sample_rate: int,
            timestamp: float,
            speaker_id: int = 0,
            noise_level: float = 0.03,
            context: Optional[Dict] = None,
        ) -> Tuple[str, float, Dict]:
            """Label a segment with a known mock speaker ID for demo purposes."""
            self.total_segments_processed += 1
            
            # Compute embedding with speaker_id for mock
            embedding = self.compute_embedding(
                waveform, sample_rate, speaker_id=speaker_id, noise_level=noise_level
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
            "The labeler progressively learns and tracks speakers.\n\n"
            "[bold]Expected:[/bold] All Person_0 segments → same label, "
            "all Person_1 segments → same (different) label.",
            title="Demo Overview",
            border_style="cyan",
        )
    )
    console.print()
    
    # Initialize with mock model
    mock_model = MockEmbeddingModel(dimension=192)
    labeler = MockSegmentSpeakerLabeler(
        embedding_model=mock_model,
        threshold_same=0.75,
        threshold_possible=0.60,
        min_segments_for_reference=2,
        max_embeddings_per_speaker=50,
        temporal_smoothing_window=3.0,
        debug=True,
    )
    
    # ─── Simulated Audio Segments ────────────────────────────────────────────
    segments = [
        # Speaker 0 starts talking (segments 1-4)
        {"timestamp": 0.5,  "speaker_id": 0, "duration": 2.0, "noise_level": 0.03},
        {"timestamp": 3.0,  "speaker_id": 0, "duration": 1.5, "noise_level": 0.04},
        {"timestamp": 5.0,  "speaker_id": 0, "duration": 2.5, "noise_level": 0.03},
        {"timestamp": 8.0,  "speaker_id": 0, "duration": 1.0, "noise_level": 0.05},
        
        # Speaker 1 interjects (segments 5-7)
        {"timestamp": 10.0, "speaker_id": 1, "duration": 1.5, "noise_level": 0.03},
        {"timestamp": 12.0, "speaker_id": 1, "duration": 2.0, "noise_level": 0.04},
        {"timestamp": 14.5, "speaker_id": 1, "duration": 1.0, "noise_level": 0.03},
        
        # Back to Speaker 0 (segments 8-10)
        {"timestamp": 16.0, "speaker_id": 0, "duration": 2.0, "noise_level": 0.04},
        {"timestamp": 18.5, "speaker_id": 0, "duration": 1.5, "noise_level": 0.03},
        {"timestamp": 20.5, "speaker_id": 0, "duration": 2.0, "noise_level": 0.05},
        
        # Speaker 1 again (segments 11-12)
        {"timestamp": 23.0, "speaker_id": 1, "duration": 1.5, "noise_level": 0.03},
        {"timestamp": 25.0, "speaker_id": 1, "duration": 2.0, "noise_level": 0.04},
        
        # Brief Speaker 0 (segment 13)
        {"timestamp": 27.5, "speaker_id": 0, "duration": 0.8, "noise_level": 0.03},
    ]
    
    # ─── Process Segments ────────────────────────────────────────────────────
    console.print("[bold]Processing segments...[/bold]\n")
    
    results = []
    current_speaker = None
    
    for i, seg in enumerate(segments, 1):
        # Create a dummy waveform tensor for the demo
        num_samples = int(seg["duration"] * 16000)
        waveform = torch.zeros(1, num_samples)
        
        # Build context
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
            noise_level=seg.get("noise_level", 0.03),
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
        conf = r["confidence"]
        if conf >= 0.85:
            conf_str = f"[green]{conf:.3f}[/green]"
        elif conf >= 0.65:
            conf_str = f"[yellow]{conf:.3f}[/yellow]"
        else:
            conf_str = f"[red]{conf:.3f}[/red]"
        
        mt = r["match_type"]
        if mt == "strong_match":
            mt_str = f"[green]{mt}[/green]"
        elif mt in ("early_match", "context_match"):
            mt_str = f"[yellow]{mt}[/yellow]"
        elif mt == "possible_match":
            mt_str = f"[dim yellow]{mt}[/dim yellow]"
        else:
            mt_str = f"[cyan]{mt}[/cyan]"
        
        # Highlight correct/incorrect assignments
        expected_prefix = "SPEAKER_0" if r["actual_speaker"] == "Person_0" else "SPEAKER_0"
        is_correct = (
            (r["actual_speaker"] == "Person_0" and r["labeled_as"].startswith("SPEAKER_0")) or
            False  # Will be checked after we know which labels map to which person
        )
        
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
    
    # Determine max speakers for dynamic columns
    all_labels = set()
    for r in results:
        all_labels.update(r["all_scores"].keys())
    sorted_labels = sorted(all_labels)
    
    # Truncate to first 5 speakers for readability
    display_labels = sorted_labels[:5]
    has_more = len(sorted_labels) > 5
    
    score_table = Table(
        title="Per-Segment Similarity Scores (top 5 speakers)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    
    score_table.add_column("Seg", justify="right", style="dim")
    score_table.add_column("Time", justify="right", style="magenta")
    score_table.add_column("Actual", style="yellow")
    score_table.add_column("Chosen", style="green bold")
    
    for label in display_labels:
        score_table.add_column(label, justify="right")
    
    if has_more:
        score_table.add_column("...", justify="right", style="dim")
    
    for r in results:
        row = [
            str(r["segment"]),
            f"{r['timestamp']:.1f}s",
            r["actual_speaker"],
            r["labeled_as"],
        ]
        
        for label in display_labels:
            score = r["all_scores"].get(label, None)
            if score is not None:
                if score >= 0.85:
                    row.append(f"[green]{score:.3f}[/green]")
                elif score >= 0.65:
                    row.append(f"[yellow]{score:.3f}[/yellow]")
                else:
                    row.append(f"[red]{score:.3f}[/red]")
            else:
                row.append("[dim]—[/dim]")
        
        if has_more:
            remaining = len(r["all_scores"]) - len(display_labels)
            row.append(f"[dim]+{remaining}[/dim]")
        
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
    speaker_table.add_column("Embeddings", justify="right", style="dim")
    
    for label in labeler.known_speakers:
        info = labeler.get_speaker_info(label)
        speaker_table.add_row(
            info["label"],
            str(info["segment_count"]),
            f"{info['first_seen']:.1f}s",
            f"{info['last_seen']:.1f}s",
            f"{info['active_duration']:.1f}s",
            str(len(labeler._speakers[label].embeddings)),
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
    
    person0_labels = set()
    person1_labels = set()
    
    for r in results:
        if r["actual_speaker"] == "Person_0":
            person0_labels.add(r["labeled_as"])
        else:
            person1_labels.add(r["labeled_as"])
    
    if len(person0_labels) == 1 and len(person1_labels) == 1:
        p0_label = person0_labels.pop()
        p1_label = person1_labels.pop()
        if p0_label != p1_label:
            # Count correct assignments
            correct = sum(
                1 for r in results
                if (r["actual_speaker"] == "Person_0" and r["labeled_as"] == p0_label) or
                   (r["actual_speaker"] == "Person_1" and r["labeled_as"] == p1_label)
            )
            accuracy = correct / len(results) * 100
            console.print(
                f"[green]✓ Perfect clustering![/green]\n"
                f"  Person_0 → [bold green]{p0_label}[/bold green] "
                f"({len([r for r in results if r['actual_speaker'] == 'Person_0'])} segments)\n"
                f"  Person_1 → [bold green]{p1_label}[/bold green] "
                f"({len([r for r in results if r['actual_speaker'] == 'Person_1'])} segments)\n"
                f"  Accuracy: [bold green]{accuracy:.0f}%[/bold green] ({correct}/{len(results)})"
            )
        else:
            console.print(
                f"[red]✗ Both speakers got the same label: {p0_label}[/red]\n"
                f"  Thresholds may need adjustment."
            )
    elif len(person0_labels) == 1 and len(person1_labels) <= 2:
        p0_label = person0_labels.pop()
        console.print(
            f"[yellow]⚠ Minor label mixing for Person_1[/yellow]\n"
            f"  Person_0 → [green]{p0_label}[/green] (consistent)\n"
            f"  Person_1 → [yellow]{person1_labels}[/yellow]\n"
            f"  Try lowering noise_level or adjusting thresholds."
        )
    elif len(person1_labels) == 1 and len(person0_labels) <= 2:
        p1_label = person1_labels.pop()
        console.print(
            f"[yellow]⚠ Minor label mixing for Person_0[/yellow]\n"
            f"  Person_1 → [green]{p1_label}[/green] (consistent)\n"
            f"  Person_0 → [yellow]{person0_labels}[/yellow]\n"
            f"  Try lowering noise_level or adjusting thresholds."
        )
    else:
        console.print(
            f"[red]✗ Significant label mixing detected[/red]\n"
            f"  Person_0 → [red]{person0_labels}[/red] ({len(person0_labels)} labels)\n"
            f"  Person_1 → [red]{person1_labels}[/red] ({len(person1_labels)} labels)\n"
            f"  [dim]Try: lower noise_level, increase threshold_same, or use more segments for reference.[/dim]"
        )
    
    # ─── Serialization Demo ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Serialization Demo[/bold green]")
    console.print()
    
    state = labeler.to_dict()
    console.print("[bold]State serialized successfully[/bold]")
    console.print(f"  Speakers saved: [bright_white]{len(state['speakers'])}[/bright_white]")
    console.print(f"  Next speaker ID: [bright_white]{state['next_speaker_id']}[/bright_white]")
    console.print(f"  Total segments: [bright_white]{state['total_segments_processed']}[/bright_white]")
    
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
    console.print("[bold]Verifying restored labeler with new segment (Person_0)...[/bold]")
    
    new_waveform = torch.zeros(1, 16000)
    
    label, confidence, metadata = labeler2.label_segment_with_speaker_id(
        waveform=new_waveform,
        sample_rate=16000,
        timestamp=30.0,
        speaker_id=0,
        noise_level=0.03,
        context={"previous_speaker": current_speaker},
    )
    
    # Check if it matched an existing speaker
    if metadata["match_type"] in ("strong_match", "early_match", "context_match"):
        console.print(
            f"  [green]✓ Correctly matched to existing speaker: {label}[/green] "
            f"(confidence: {confidence:.3f}, type: {metadata['match_type']})"
        )
    else:
        console.print(
            f"  [yellow]⚠ Created new speaker: {label}[/yellow] "
            f"(confidence: {confidence:.3f}, type: {metadata['match_type']})"
        )
    
    # ─── Speaker Merge Demo ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Speaker Merge Demo[/bold green]")
    console.print()
    
    # Reset and build up speakers for merge demo
    labeler.reset()
    console.print("[dim]Building speaker references for merge demo...[/dim]")
    
    current_speaker = None
    for i, seg in enumerate(segments[:8], 1):
        waveform = torch.zeros(1, int(seg["duration"] * 16000))
        label, _, _ = labeler.label_segment_with_speaker_id(
            waveform=waveform,
            sample_rate=16000,
            timestamp=seg["timestamp"],
            speaker_id=seg["speaker_id"],
            noise_level=seg.get("noise_level", 0.03),
            context={"previous_speaker": current_speaker},
        )
        current_speaker = label
    
    console.print()
    console.print(
        f"[bold]Speakers before merge:[/bold] "
        f"[bright_white]{labeler.known_speakers}[/bright_white]"
        f" ([bright_white]{labeler.speaker_count}[/bright_white] total)"
    )
    
    if labeler.speaker_count >= 2:
        speakers = labeler.known_speakers
        spk_a, spk_b = speakers[0], speakers[1]
        
        console.print()
        console.print(
            f"[yellow]Merging [bold]{spk_a}[/bold] and [bold]{spk_b}[/bold]...[/yellow]"
        )
        
        info_a = labeler.get_speaker_info(spk_a)
        info_b = labeler.get_speaker_info(spk_b)
        console.print(f"  {spk_a}: {info_a['segment_count']} segments")
        console.print(f"  {spk_b}: {info_b['segment_count']} segments")
        
        merged = labeler.merge_speakers(spk_a, spk_b)
        
        if merged:
            info_merged = labeler.get_speaker_info(merged)
            console.print()
            console.print(f"[green]✓ Merged into: [bold]{merged}[/bold][/green]")
            console.print(f"  Combined segments: [bright_white]{info_merged['segment_count']}[/bright_white]")
            console.print(f"  Remaining speakers: [bright_white]{labeler.known_speakers}[/bright_white]")
            console.print(f"  Speaker count: [bright_white]{labeler.speaker_count}[/bright_white]")
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
        noise_level=segments[0].get("noise_level", 0.03),
    )
    
    console.print(
        f"  First segment after reset → [green]{label}[/green] "
        f"(confidence: {confidence:.3f}, type: {metadata['match_type']})"
    )
    console.print(f"  Speaker count: [bright_white]{labeler.speaker_count}[/bright_white]")
    
    # ─── Threshold Guidance ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Threshold Tuning Guide[/bold cyan]")
    console.print()
    
    guide = Table(show_header=True, header_style="bold cyan", border_style="dim")
    guide.add_column("Issue", style="yellow")
    guide.add_column("Fix", style="green")
    
    guide.add_row(
        "Too many speakers created",
        "Lower [bold]noise_level[/bold] (e.g., 0.01–0.02)\nor raise [bold]threshold_same[/bold] (e.g., 0.80–0.85)"
    )
    guide.add_row(
        "Different speakers merged together",
        "Increase [bold]noise_level[/bold] (e.g., 0.05–0.08)\nor lower [bold]threshold_same[/bold] (e.g., 0.65–0.70)"
    )
    guide.add_row(
        "Label flickering between speakers",
        "Increase [bold]temporal_smoothing_window[/bold]\n(e.g., 5.0s instead of 3.0s)"
    )
    guide.add_row(
        "Late detection of new speakers",
        "Decrease [bold]min_segments_for_reference[/bold]\n(e.g., 1 instead of 2)"
    )
    
    console.print(guide)
    
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
            "[dim]Replace MockEmbeddingModel with pyannote.audio Inference for real use.\n"
            "Real embeddings are much more consistent than mock noise![/dim]",
            title="Summary",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()
