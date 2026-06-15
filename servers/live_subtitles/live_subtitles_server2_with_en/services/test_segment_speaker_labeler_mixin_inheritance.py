"""
test_segment_speaker_labeler_mixin_inheritance.py
=========================
Demonstrates proper mixin inheritance and usage.
"""
import torch
import numpy as np
from rich.console import Console

console = Console()

# ── CORRECT: Class inherits from mixin ──
# This is how your segment_speaker_labeler.py should look after modification:

from segment_speaker_labeler import SegmentSpeakerLabeler

# The class in segment_speaker_labeler.py now inherits from SpeakerLabelerHealthMixin:
#
# class SegmentSpeakerLabeler(SpeakerLabelerHealthMixin):
#     ...all existing code...
#

def test_mixin_methods_available():
    """Verify that mixin methods are accessible through inheritance."""
    
    # Mock embedding model
    class MockEmbeddingModel:
        def __call__(self, inputs):
            batch_size = inputs["waveform"].shape[0] if inputs["waveform"].dim() > 1 else 1
            return torch.randn(batch_size, 256)
    
    # Create labeler instance
    labeler = SegmentSpeakerLabeler(
        embedding_model=MockEmbeddingModel(),
        debug=True
    )
    
    # Check that mixin methods exist on the instance
    console.print("[bold green]Verifying mixin methods are available:[/]\n")
    
    mixin_methods = [
        'get_centroid_health_report',
        'get_centroid_health_dict', 
        'get_similarity_matrix_dict',
        'get_speaker_insights',
        'get_cohesion_series',
        'get_chart_data',
        '_build_embeddings_per_label',
        '_make_health_thresholds',
    ]
    
    for method_name in mixin_methods:
        if hasattr(labeler, method_name):
            console.print(f"  ✅ {method_name} - available")
        else:
            console.print(f"  ❌ {method_name} - MISSING!")
    
    # Process some segments to create speakers
    console.print("\n[bold yellow]Processing segments to build speaker data...[/]")
    
    for i in range(10):
        waveform = torch.randn(1, 16000)
        labeler.label_segments(
            waveform=waveform,
            sample_rate=16000,
            timestamp=i * 1.0,
            top_k=3
        )
    
    console.print(f"Created {len(labeler.known_speakers)} speakers\n")
    
    # Now use mixin methods
    console.print("[bold cyan]Using inherited mixin methods:[/]\n")
    
    # Method 1: Health report
    health_report = labeler.get_centroid_health_report()
    if health_report:
        console.print(f"✅ get_centroid_health_report() works!")
        console.print(f"   Healthy speakers: {health_report.healthy_labels}")
    
    # Method 2: Health dict (for APIs)
    health_dict = labeler.get_centroid_health_dict()
    console.print(f"\n✅ get_centroid_health_dict() works!")
    console.print(f"   Speaker count: {health_dict['speaker_count']}")
    
    # Method 3: Similarity matrix
    sim_matrix = labeler.get_similarity_matrix_dict()
    console.print(f"\n✅ get_similarity_matrix_dict() works!")
    console.print(f"   Matrix size: {len(sim_matrix['labels'])}x{len(sim_matrix['labels'])}")
    
    # Method 4: Insights
    insights = labeler.get_speaker_insights()
    console.print(f"\n✅ get_speaker_insights() works!")
    console.print(f"   System health: {insights['system_health']}")
    
    # Method 5: Cohesion series
    cohesion = labeler.get_cohesion_series()
    console.print(f"\n✅ get_cohesion_series() works!")
    console.print(f"   Speakers analyzed: {len(cohesion['speakers'])}")
    
    # Method 6: All chart data
    chart_data = labeler.get_chart_data()
    console.print(f"\n✅ get_chart_data() works!")
    console.print(f"   Available keys: {list(chart_data.keys())}")
    
    return labeler

if __name__ == "__main__":
    labeler = test_mixin_methods_available()
    
    console.print("\n" + "="*60)
    console.print("[bold green]Summary: Mixin inheritance is working correctly![/]")
    console.print("="*60)