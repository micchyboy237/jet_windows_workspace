"""
simple_demo.py
Simple demo for GenericSpeakerLabeler with audio files
"""

import os
import wave
import shutil
import numpy as np
from pathlib import Path
from speaker_labeling_system import GenericSpeakerLabeler

# Base output directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
# Clean and recreate output directory
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_audio_files(messages, output_dir):
    """Create minimal sample audio files for demo"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    
    for i, msg in enumerate(messages):
        speaker = msg['speaker']
        filename = output_dir / f"{speaker}_msg_{i}.wav"
        
        # Create a simple sine wave (different frequency per speaker)
        sample_rate = 16000
        duration = len(msg['text']) * 0.05  # Rough duration based on text length
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Different frequency for each speaker
        if 'speaker_0' in speaker:
            frequency = 180  # Higher pitch
        else:
            frequency = 120  # Lower pitch
        
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save WAV file
        with wave.open(str(filename), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        audio_files.append(str(filename))
        msg['audio_file'] = str(filename)
    
    return audio_files


def main():
    """Simple demo with audio files"""
    
    # Messages start with generic labels: speaker_0, speaker_1
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
    
    # Create sample audio files in output directory
    print(f"Creating sample audio files in: {OUTPUT_DIR}")
    audio_files = create_sample_audio_files(messages, OUTPUT_DIR / "audio")
    
    print("=" * 60)
    print("GENERIC SPEAKER LABELING DEMO")
    print("=" * 60)
    
    print("\nInput messages (with generic labels & audio):")
    for m in messages:
        print(f"  [{m['speaker']}]: {m['text'][:60]}...")
        print(f"         📁 {m['audio_file']}")
    
    # Run the labeler
    labeler = GenericSpeakerLabeler()
    results = labeler.process_conversation(messages, audio_files)
    
    # Show progression table
    print("\n" + "=" * 60)
    print("LABEL PROGRESSION")
    print("=" * 60)
    
    for speaker, identity in results.items():
        print(f"\n{speaker}:")
        for step in identity.label_history:
            print(f"  {step['from_quality']} → {step['to_quality']}: "
                  f"'{step['from_label']}' → '{step['to_label']}'")
        print(f"  Final: '{identity.current_label}' "
              f"({identity.quality.name}, confidence: {identity.confidence:.2f})")
    
    print(f"\n📁 Output files remain in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()