#!/usr/bin/env python3
"""
Audio tagging with temporal segmentation using sliding window approach.

This script performs audio tagging on overlapping windows to provide
temporal localization of sound events, including speech and gender detection.

Model: sherpa-onnx-zipformer-audio-tagging-2024-04-09
Documentation: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sherpa_onnx


OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Specific indices for gender derivation as requested
GENDER_INDICES = {
    'male_speech': 1,
    'female_speech': 2,
    'male_singing': 32,
    'female_singing': 33
}

@dataclass
class SegmentInfo:
    """Information about a single audio segment (window)."""
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    top_predictions: List[Dict[str, Any]]
    speech_detected: bool
    male_prob: float
    female_prob: float
    dominant_gender: str
    all_probs: Dict[int, float]


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform audio tagging with temporal segmentation using sliding window."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio file (WAV format recommended)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(
            Path.home()
            / ".cache"
            / "pretrained_models"
            / "sherpa-onnx"
            / "sherpa-onnx-zipformer-audio-tagging-2024-04-09"
        ),
        help="Path to the model directory containing model.onnx and class_labels_indices.csv",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=1.0,
        help="Window size in seconds for each segment (default: 1.0)",
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        default=0.5,
        help="Hop size in seconds between windows (default: 0.5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,  # Increased default to ensure gender labels are captured
        help="Number of top predictions per segment (default: 50)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for inference (default: 1)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution provider: cpu or cuda (default: cpu)",
    )
    return parser.parse_args()


def load_class_labels(labels_path: Path) -> Dict[int, str]:
    """Load class labels from CSV file."""
    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 2)
            if len(parts) >= 3:
                idx = int(parts[0])
                display_name = parts[2].strip('"')
                labels[idx] = display_name
    return labels


def create_windows(audio: np.ndarray, sample_rate: int, 
                   window_size: float, hop_size: float) -> List[Tuple[int, int]]:
    """Create sliding windows over audio signal."""
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    windows = []
    start = 0
    
    while start + window_samples <= len(audio):
        end = start + window_samples
        windows.append((start, end))
        start += hop_samples
    
    # Handle last partial window if needed
    if len(audio) - start > window_samples * 0.5:
        windows.append((start, len(audio)))
    
    return windows


def analyze_segment(predictions: List, all_labels: Dict[int, str]) -> Dict[str, Any]:
    """Analyze predictions for a single segment with focused gender derivation."""
    
    # Build probability map for ALL labels returned in top-k
    label_probs = {}
    for event in predictions:
        label_probs[int(event.index)] = float(event.prob)
    
    # 1. Extract specific gender probabilities (default to 0.0 if not in top-k)
    p_male_speech = label_probs.get(GENDER_INDICES['male_speech'], 0.0)
    p_female_speech = label_probs.get(GENDER_INDICES['female_speech'], 0.0)
    p_male_singing = label_probs.get(GENDER_INDICES['male_singing'], 0.0)
    p_female_singing = label_probs.get(GENDER_INDICES['female_singing'], 0.0)
    
    # 2. Derive total gender scores
    male_score = max(p_male_speech, p_male_singing)
    female_score = max(p_female_speech, p_female_singing)
    
    # 3. Determine dominant gender based on relative strength
    # We use a small epsilon to avoid 'unknown' if one is clearly higher than the other
    if male_score > female_score and male_score > 1e-6:
        dominant_gender = "male"
    elif female_score > male_score and female_score > 1e-6:
        dominant_gender = "female"
    else:
        dominant_gender = "unknown"
        
    # 4. Speech detection (using general speech label 0)
    speech_prob = label_probs.get(0, 0.0)
    speech_detected = speech_prob > 0.3
    
    # 5. Prepare top predictions, ensuring gender labels are included if relevant
    top_preds = []
    for event in predictions:
        idx = int(event.index)
        pred_info = {
            "index": idx,
            "name": event.name,
            "probability": float(event.prob),
            "is_speech": idx in [0, 1, 2, 3],
            "is_gender_specific": idx in GENDER_INDICES.values(),
            "gender": None
        }
        
        if idx == GENDER_INDICES['male_speech'] or idx == GENDER_INDICES['male_singing']:
            pred_info["gender"] = "male"
        elif idx == GENDER_INDICES['female_speech'] or idx == GENDER_INDICES['female_singing']:
            pred_info["gender"] = "female"
            
        top_preds.append(pred_info)
        
    return {
        "speech_detected": speech_detected,
        "male_prob": male_score,
        "female_prob": female_score,
        "dominant_gender": dominant_gender,
        "top_predictions": top_preds,
        "all_probs": label_probs
    }


def process_audio_with_segmentation(
    audio: np.ndarray,
    sample_rate: int,
    tagger: sherpa_onnx.AudioTagging,
    window_size: float,
    hop_size: float,
    top_k: int,
    all_labels: Dict[int, str]
) -> List[SegmentInfo]:
    """Process audio with sliding window segmentation."""
    
    windows = create_windows(audio, sample_rate, window_size, hop_size)
    print(f"Created {len(windows)} windows (size={window_size}s, hop={hop_size}s)")
    
    segments = []
    
    for i, (start_idx, end_idx) in enumerate(windows):
        window_audio = audio[start_idx:end_idx]
        if len(window_audio) < sample_rate * 0.1:
            continue
        
        stream = tagger.create_stream()
        stream.accept_waveform(sample_rate, window_audio)
        predictions = tagger.compute(stream, top_k)
        
        analysis = analyze_segment(predictions, all_labels)
        
        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate
        duration = end_time - start_time
        
        segment = SegmentInfo(
            segment_id=i,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            top_predictions=analysis["top_predictions"],
            speech_detected=analysis["speech_detected"],
            male_prob=analysis["male_prob"],
            female_prob=analysis["female_prob"],
            dominant_gender=analysis["dominant_gender"],
            all_probs=analysis["all_probs"]
        )
        
        segments.append(segment)
        
        if (i + 1) % 10 == 0 or i == len(windows) - 1:
            print(f"Processed {i + 1}/{len(windows)} windows...")
    
    return segments


def save_segments_json(segments: List[SegmentInfo], all_labels: Dict[int, str], 
                       audio_info: Dict[str, Any], output_path: Path):
    """Save detailed segments to JSON."""
    
    segments_data = {
        "metadata": {
            "audio_file": audio_info['file'],
            "sample_rate": audio_info['sample_rate'],
            "duration_seconds": audio_info['duration'],
            "num_samples": audio_info['num_samples'],
            "analysis_type": "sliding_window_segmentation_raw",
            "total_segments": len(segments),
            "note": "Each segment represents a single sliding window. Gender is derived from indices 1, 2, 32, 33."
        },
        "segments": []
    }
    
    for seg in segments:
        segment_dict = {
            "segment_id": seg.segment_id,
            "start_time": round(seg.start_time, 3),
            "end_time": round(seg.end_time, 3),
            "duration": round(seg.duration, 3),
            "speech_detected": seg.speech_detected,
            "gender_analysis": {
                "dominant_gender": seg.dominant_gender,
                "male_probability": round(seg.male_prob, 6),
                "female_probability": round(seg.female_prob, 6),
            },
            "top_predictions": seg.top_predictions,
        }
        segments_data["segments"].append(segment_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed segments saved to: {output_path}")


def save_summary_json(segments: List[SegmentInfo], audio_info: Dict[str, Any], 
                      output_path: Path):
    """Save concise summary to JSON."""
    
    total_speech_segments = sum(1 for seg in segments if seg.speech_detected)
    speech_segments = [seg for seg in segments if seg.speech_detected]
    
    if speech_segments:
        male_count = sum(1 for seg in speech_segments if seg.dominant_gender == 'male')
        female_count = sum(1 for seg in speech_segments if seg.dominant_gender == 'female')
        unknown_count = sum(1 for seg in speech_segments if seg.dominant_gender == 'unknown')
        
        total_gender_detected = male_count + female_count
        
        gender_distribution = {}
        if total_gender_detected > 0:
            gender_distribution = {
                "male": {"count": male_count, "percentage": round(male_count / total_gender_detected * 100, 2)},
                "female": {"count": female_count, "percentage": round(female_count / total_gender_detected * 100, 2)},
            }
        
        dominant_overall = max(gender_distribution.items(), key=lambda x: x[1]['count'])[0] if gender_distribution else "unknown"
    else:
        gender_distribution = {}
        dominant_overall = "none"
        male_count = female_count = unknown_count = 0
    
    summary = {
        "audio_info": {
            "file": audio_info['file'],
            "duration_seconds": round(audio_info['duration'], 2),
            "sample_rate_hz": audio_info['sample_rate'],
        },
        "segmentation_info": {
            "total_windows": len(segments),
            "speech_windows": total_speech_segments,
            "non_speech_windows": len(segments) - total_speech_segments,
            "speech_coverage_percentage": round(total_speech_segments / len(segments) * 100, 2) if segments else 0,
        },
        "gender_analysis": {
            "dominant_gender_overall": dominant_overall,
            "gender_distribution": gender_distribution,
            "male_windows": male_count,
            "female_windows": female_count,
            "unknown_gender_windows": unknown_count,
        },
        "timeline_summary": [
            {
                "start": round(seg.start_time, 2),
                "end": round(seg.end_time, 2),
                "type": "speech" if seg.speech_detected else "non-speech",
                "gender": seg.dominant_gender if seg.speech_detected else None,
            }
            for seg in segments[:20]
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {output_path}")


def create_timeline_chart(segments: List[SegmentInfo], audio_info: Dict[str, Any], 
                          output_path: Path):
    """Create comprehensive timeline visualization."""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('Audio Tagging Timeline Analysis (Raw Windows)', fontsize=14, fontweight='bold')
    
    total_duration = audio_info['duration']
    
    # Plot 1: Main timeline
    ax1 = axes[0]
    colors_map = {
        'male': '#4A90E2',
        'female': '#E94B99',
        'unknown': '#95A5A6',
        'non-speech': '#BDC3C7'
    }
    
    for seg in segments:
        if seg.speech_detected:
            color = colors_map.get(seg.dominant_gender, colors_map['unknown'])
        else:
            color = colors_map['non-speech']
        
        ax1.barh(y=0, width=seg.duration, left=seg.start_time, height=0.6, 
                color=color, alpha=0.5, edgecolor=None)
    
    ax1.set_xlim(0, total_duration)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_title('Temporal Segmentation Timeline', fontsize=12, fontweight='bold')
    ax1.set_yticks([])
    ax1.grid(axis='x', alpha=0.3)
    
    legend_elements = [
        mpatches.Patch(color=colors_map['male'], label='Male Speech'),
        mpatches.Patch(color=colors_map['female'], label='Female Speech'),
        mpatches.Patch(color=colors_map['unknown'], label='Unknown Gender'),
        mpatches.Patch(color=colors_map['non-speech'], label='Non-Speech'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Plot 2: Speech probability
    ax2 = axes[1]
    times = [(seg.start_time + seg.end_time) / 2 for seg in segments]
    speech_probs = [seg.top_predictions[0]['probability'] if seg.speech_detected and seg.top_predictions else 0 
                    for seg in segments]
    
    ax2.plot(times, speech_probs, 'b-', linewidth=1, marker='.', markersize=8, alpha=0.7)
    ax2.fill_between(times, 0, speech_probs, alpha=0.2)
    ax2.set_xlim(0, total_duration)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Speech Probability', fontsize=11)
    ax2.set_title('Speech Detection Probability', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gender probabilities
    ax3 = axes[2]
    male_probs = [seg.male_prob for seg in segments]
    female_probs = [seg.female_prob for seg in segments]
    
    ax3.plot(times, male_probs, 'b-', linewidth=1, label='Male Score', alpha=0.7)
    ax3.plot(times, female_probs, 'r-', linewidth=1, label='Female Score', alpha=0.7)
    ax3.fill_between(times, 0, male_probs, alpha=0.1, color='blue')
    ax3.fill_between(times, 0, female_probs, alpha=0.1, color='red')
    
    ax3.set_xlim(0, total_duration)
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Probability Score', fontsize=11)
    ax3.set_title('Gender-Specific Scores (Indices 1, 2, 32, 33)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timeline chart saved to: {output_path}")


def main():
    """Main function to perform audio tagging with temporal segmentation."""
    args = get_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.int8.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"
    
    labels_path = model_dir / "class_labels_indices.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    print(f"Loading audio from: {audio_path}")
    print(f"Using model: {model_path}")
    print(f"Window size: {args.window_size}s, Hop size: {args.hop_size}s")

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    duration = len(audio) / sample_rate
    audio_info = {
        'file': str(audio_path),
        'sample_rate': sample_rate,
        'duration': duration,
        'num_samples': len(audio),
    }
    
    print(f"Audio info: sample_rate={sample_rate}Hz, duration={duration:.2f}s")

    all_labels = load_class_labels(labels_path)
    print(f"Loaded {len(all_labels)} class labels")

    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                model=str(model_path),
            ),
            num_threads=args.num_threads,
            debug=False,
            provider=args.provider,
        ),
        labels=str(labels_path),
        top_k=args.top_k,
    )

    print("\nInitializing audio tagger...")
    tagger = sherpa_onnx.AudioTagging(config)

    print(f"\nProcessing audio with sliding window segmentation...")
    segments = process_audio_with_segmentation(
        audio, sample_rate, tagger, 
        args.window_size, args.hop_size, 
        args.top_k, all_labels
    )
    
    print(f"\nTotal raw segments (windows): {len(segments)}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("SEGMENTATION SUMMARY")
    print("=" * 80)
    
    speech_segs = [seg for seg in segments if seg.speech_detected]
    print(f"Total windows: {len(segments)}")
    print(f"Speech windows: {len(speech_segs)}")
    
    if speech_segs:
        male_count = sum(1 for seg in speech_segs if seg.dominant_gender == 'male')
        female_count = sum(1 for seg in speech_segs if seg.dominant_gender == 'female')
        
        print(f"\nGender distribution in speech windows:")
        print(f"  Male:   {male_count} windows")
        print(f"  Female: {female_count} windows")
    
    print("\nFirst 10 windows:")
    print("-" * 80)
    for seg in segments[:10]:
        gender_str = f" [{seg.dominant_gender}]" if seg.speech_detected else ""
        top_label = seg.top_predictions[0]['name'] if seg.top_predictions else "N/A"
        print(f"  [{seg.start_time:6.2f}s - {seg.end_time:6.2f}s] {top_label:30s}{gender_str}")
    
    print("=" * 80)
    
    # Save outputs
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    segments_path = OUTPUT_DIR / "segments.json"
    save_segments_json(segments, all_labels, audio_info, segments_path)
    
    summary_path = OUTPUT_DIR / "summary.json"
    save_summary_json(segments, audio_info, summary_path)
    
    chart_path = OUTPUT_DIR / "timeline_chart.png"
    create_timeline_chart(segments, audio_info, chart_path)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    return {
        'segments': segments,
        'output_dir': OUTPUT_DIR,
    }


if __name__ == "__main__":
    main()