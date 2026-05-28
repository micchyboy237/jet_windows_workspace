"""
simple_demo.py
Simple demo for GenericSpeakerLabeler with audio files
Enhanced with rich logging, file source links, plots, and insights.
"""
import os
import wave
import shutil
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from speaker_labeling_system import GenericSpeakerLabeler

# Output structure
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sub-directories for organized output
AUDIO_DIR = OUTPUT_DIR / "audio"
PLOTS_DIR = OUTPUT_DIR / "plots"
INSIGHTS_DIR = OUTPUT_DIR / "insights"
AUDIO_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
INSIGHTS_DIR.mkdir(exist_ok=True)

# Setup rich logging
from speaker_labeling_system import setup_logger
logger = setup_logger(OUTPUT_DIR)

def terminal_link(filepath):
    """Create clickable terminal link (works in modern terminals)"""
    abs_path = Path(filepath).resolve()
    return f"\033]8;;file:///{abs_path}\033\\{abs_path.name}\033]8;;\033\\"

def create_sample_audio_files(messages, output_dir):
    """Create minimal sample audio files for demo"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = []
    
    logger.info(f"Creating {len(messages)} sample audio files in: {terminal_link(output_dir)}")
    
    for i, msg in enumerate(messages):
        speaker = msg['speaker']
        filename = output_dir / f"{speaker}_msg_{i}.wav"
        sample_rate = 16000
        duration = len(msg['text']) * 0.05
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        if 'speaker_0' in speaker:
            frequency = 180
        else:
            frequency = 120
        
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(filename), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        audio_files.append(str(filename))
        msg['audio_file'] = str(filename)
        logger.debug(f"  Created: {terminal_link(filename)} ({frequency}Hz, {duration:.2f}s)")
    
    return audio_files

def generate_visualizations(results, messages, audio_features, plots_dir):
    """Generate visualization plots for speaker analysis"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        logger.info(f"Generating visualization plots in: {terminal_link(plots_dir)}")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Label Progression Timeline
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Speaker Labeling Analysis', fontsize=16, fontweight='bold')
        
        # 1.1 Confidence progression
        ax = axes[0, 0]
        speakers = list(results.keys())
        confidences = [identity.confidence for identity in results.values()]
        qualities = [identity.quality.name for identity in results.values()]
        
        bars = ax.bar(speakers, confidences, color=['#2ecc71', '#e74c3c'])
        ax.set_title('Final Confidence Scores', fontweight='bold')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        
        for bar, quality in zip(bars, qualities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    quality, ha='center', va='bottom', fontsize=9)
        
        # 1.2 Label quality progression
        ax = axes[0, 1]
        quality_levels = {'GENERIC': 1, 'DESCRIPTIVE': 2, 'RELATIONAL': 3, 'PROPER_NAME': 4}
        
        for speaker, identity in results.items():
            progress = [quality_levels.get(step['from_quality'], 1) for step in identity.label_history]
            progress.append(quality_levels.get(identity.quality.name, 1))
            ax.plot(range(len(progress)), progress, marker='o', linewidth=2, label=speaker, markersize=8)
        
        ax.set_title('Label Quality Progression', fontweight='bold')
        ax.set_xlabel('Stage')
        ax.set_ylabel('Quality Level')
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Generic', 'Descriptive', 'Relational', 'Proper Name'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1.3 Linguistic patterns comparison
        ax = axes[1, 0]
        if any(identity.linguistic_profile for identity in results.values()):
            metrics = ['avg_words_per_turn', 'question_ratio', 'politeness_ratio', 'urgency_ratio']
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, (speaker, identity) in enumerate(results.items()):
                values = [identity.linguistic_profile.get(m, 0) for m in metrics]
                ax.bar(x + i * width, values, width, label=speaker, alpha=0.8)
            
            ax.set_title('Linguistic Pattern Comparison', fontweight='bold')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(['Avg Words', 'Questions', 'Politeness', 'Urgency'], rotation=45)
            ax.legend()
        
        # 1.4 Audio signature comparison
        ax = axes[1, 1]
        if audio_features:
            speakers_with_audio = [s for s in results.keys() if s in audio_features]
            if speakers_with_audio:
                audio_metrics = ['rms_energy', 'zero_crossing_rate', 'spectral_centroid']
                x = np.arange(len(audio_metrics))
                width = 0.35
                
                for i, speaker in enumerate(speakers_with_audio):
                    values = [audio_features[speaker].get(m, 0) for m in audio_metrics]
                    normalized = np.array(values) / max(values) if max(values) > 0 else values
                    ax.bar(x + i * width, normalized, width, label=speaker, alpha=0.8)
                
                ax.set_title('Audio Signature Comparison', fontweight='bold')
                ax.set_xticks(x + width / 2)
                ax.set_xticklabels(['RMS Energy', 'Zero Cross', 'Spectral Cent.'], rotation=45)
                ax.legend()
        
        plt.tight_layout()
        plot_path = plots_dir / "speaker_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved plot: {terminal_link(plot_path)}")
        
        # Plot 2: Name candidate confidence
        fig, ax = plt.subplots(figsize=(10, 5))
        has_names = False
        
        for speaker, identity in results.items():
            if identity.name_candidates:
                has_names = True
                names = list(identity.name_candidates.keys())
                scores = list(identity.name_candidates.values())
                ax.barh([f"{speaker}\n{name}" for name in names], scores, alpha=0.7)
        
        if has_names:
            ax.set_title('Name Candidate Confidence', fontweight='bold')
            ax.set_xlabel('Confidence Score')
            ax.set_xlim(0, 1)
            
            name_plot_path = plots_dir / "name_candidates.png"
            plt.tight_layout()
            plt.savefig(name_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✅ Saved name analysis: {terminal_link(name_plot_path)}")
        else:
            plt.close()
            logger.info("  ℹ️  No name candidates to plot")
        
        return plots_dir
        
    except ImportError as e:
        logger.warning(f"Matplotlib/seaborn not available for plotting: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        return None

def save_insights_report(results, messages, audio_files, insights_dir):
    """Save comprehensive JSON insights report"""
    logger.info(f"Saving insights report to: {terminal_link(insights_dir)}")
    
    # Build insights report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_messages': len(messages),
            'total_speakers': len(results),
            'audio_files': [Path(f).name for f in audio_files]
        },
        'speaker_analysis': {},
        'conversation_flow': [],
        'label_progression_summary': {}
    }
    
    # Speaker analysis
    for speaker, identity in results.items():
        speaker_data = identity.to_dict()
        
        # Count messages per speaker
        message_count = sum(1 for m in messages if m.get('speaker') == speaker)
        speaker_data['message_count'] = message_count
        speaker_data['total_words'] = sum(
            len(m.get('text', '').split()) 
            for m in messages 
            if m.get('speaker') == speaker
        )
        
        report['speaker_analysis'][speaker] = speaker_data
    
    # Conversation flow
    for i, msg in enumerate(messages):
        report['conversation_flow'].append({
            'turn': i,
            'speaker': msg['speaker'],
            'text_preview': msg['text'][:100] + ('...' if len(msg['text']) > 100 else ''),
            'audio_file': Path(msg.get('audio_file', '')).name if msg.get('audio_file') else None
        })
    
    # Label progression summary
    report['label_progression_summary'] = {
        speaker: {
            'initial': identity.label_history[0]['from_label'] if identity.label_history else speaker,
            'final': identity.current_label,
            'quality_achieved': identity.quality.name,
            'stages_completed': len(identity.label_history),
            'confidence': identity.confidence
        }
        for speaker, identity in results.items()
    }
    
    # Save report
    report_path = insights_dir / "speaker_insights.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"  ✅ Saved insights: {terminal_link(report_path)}")
    
    # Save human-readable summary
    summary_path = insights_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SPEAKER LABELING SYSTEM - ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total messages: {len(messages)}\n")
        f.write(f"Total speakers: {len(results)}\n\n")
        
        for speaker, identity in results.items():
            f.write(f"\n{speaker}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Final Label: {identity.current_label}\n")
            f.write(f"  Quality: {identity.quality.name}\n")
            f.write(f"  Confidence: {identity.confidence:.2f}\n")
            
            if identity.label_history:
                f.write(f"  Label Progression:\n")
                for step in identity.label_history:
                    f.write(f"    {step['from_quality']} → {step['to_quality']}: "
                           f"'{step['from_label']}' → '{step['to_label']}'\n")
    
    logger.info(f"  ✅ Saved summary: {terminal_link(summary_path)}")
    
    return insights_dir

def main():
    """Simple demo with audio files"""
    logger.info("=" * 60)
    logger.info("GENERIC SPEAKER LABELING SYSTEM DEMO")
    logger.info("=" * 60)
    
    messages = [
        {
            "speaker": "speaker_0",
            "text": "Thank you for calling TechSupport. My name is Sarah. How can I help you today?",
        },
        {
            "speaker": "speaker_1",
            "text": "Hi, I'm John. I've been having trouble with my laptop. It won't turn on.",
        },
        {
            "speaker": "speaker_0",
            "text": "I understand, John. Let me check your warranty status. Can you provide your serial number?",
        },
        {
            "speaker": "speaker_1",
            "text": "Sure, it's SN-2024-XYZ789. I really need this fixed quickly.",
        },
        {
            "speaker": "speaker_0",
            "text": "I see your warranty is active. Let me schedule a repair for tomorrow.",
        },
        {
            "speaker": "speaker_1",
            "text": "That would be perfect! Thank you so much, Sarah.",
        }
    ]
    
    logger.info(f"Creating sample audio files in: {terminal_link(AUDIO_DIR)}")
    audio_files = create_sample_audio_files(messages, AUDIO_DIR)
    
    logger.info("\n📋 Input Messages:")
    for m in messages:
        logger.info(f"  [{m['speaker']}]: {m['text'][:60]}...")
        logger.info(f"         📁 {terminal_link(m['audio_file'])}")
    
    # Process conversation
    labeler = GenericSpeakerLabeler(log_dir=OUTPUT_DIR)
    results = labeler.process_conversation(messages, audio_files)
    
    # Extract audio features per speaker for visualization
    audio_features = {}
    if hasattr(labeler, '_stage1_audio_features'):
        for i, (speaker, features) in enumerate(labeler._stage1_audio_features.items()):
            # Map index to speaker
            for msg in messages:
                if msg.get('speaker') == speaker:
                    audio_features[speaker] = features
                    break
    
    # Generate visualizations
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING OUTPUTS")
    logger.info("=" * 60)
    
    generate_visualizations(results, messages, audio_features, PLOTS_DIR)
    save_insights_report(results, messages, audio_files, INSIGHTS_DIR)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL LABEL PROGRESSION")
    logger.info("=" * 60)
    
    for speaker, identity in results.items():
        logger.info(f"\n{speaker}:")
        for step in identity.label_history:
            logger.info(f"  {step['from_quality']} → {step['to_quality']}: "
                       f"'{step['from_label']}' → '{step['to_label']}'")
        logger.info(f"  Final: '{identity.current_label}' "
                   f"({identity.quality.name}, confidence: {identity.confidence:.2f})")
    
    logger.info(f"\n📁 Output directory: {terminal_link(OUTPUT_DIR)}")
    logger.info(f"   ├── {terminal_link(AUDIO_DIR)}  - Audio files")
    logger.info(f"   ├── {terminal_link(PLOTS_DIR)}  - Visualization plots")
    logger.info(f"   ├── {terminal_link(INSIGHTS_DIR)}  - Analysis reports")
    logger.info(f"   └── speaker_labeling.log  - Detailed logs")
    
    return results

if __name__ == "__main__":
    main()
