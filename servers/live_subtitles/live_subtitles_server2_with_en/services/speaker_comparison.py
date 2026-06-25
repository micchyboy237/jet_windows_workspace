import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

# Default audio paths
DEFAULT_AUDIO_PATHS = [
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves_spyx_3\waves\segment_001_wave_002\sound.wav",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_speech_waves_spyx_3\waves\segment_001_wave_004\sound.wav",
]


def preprocess_audio(input_path, output_dir=None, sr=16000):
    """Resample audio to target sample rate and downmix to mono if needed.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save processed audio (if None, uses temp directory)
        sr: Target sample rate (default: 16000)
    
    Returns:
        Path to processed audio file (or original path if already correct format)
    """
    # Get original audio info without loading full audio
    info = sf.info(input_path)
    orig_sr = info.samplerate
    orig_channels = info.channels
    orig_dtype = info.subtype
    orig_duration = info.duration
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(input_path).name}")
    print(f"  Original sample rate: {orig_sr} Hz")
    print(f"  Original channels: {orig_channels} {'(mono)' if orig_channels == 1 else '(stereo)' if orig_channels == 2 else f'({orig_channels} channels)'}")
    print(f"  Original dtype/subtype: {orig_dtype}")
    print(f"  Original duration: {orig_duration:.2f} seconds")
    
    # If already correct format, return original path
    if orig_sr == sr and orig_channels == 1:
        print(f"  ✓ Audio already in target format (16kHz mono) - using original file")
        return input_path
    
    if output_dir is None:
        output_dir = OUTPUT_DIR / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio with librosa (automatically converts to mono)
    print(f"  → Converting to {sr}Hz mono...")
    audio, _ = librosa.load(input_path, sr=sr, mono=True)
    
    # Get audio stats after conversion
    print(f"  Converted shape: {audio.shape}")
    print(f"  Converted dtype: {audio.dtype}")
    print(f"  Converted duration: {len(audio)/sr:.2f} seconds")
    print(f"  Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"  RMS energy: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # Save processed audio
    output_path = output_dir / Path(input_path).name
    sf.write(output_path, audio, sr)
    print(f"  ✓ Saved processed file to: {output_path}")
    
    return str(output_path)


def save_results(result, audio_paths, output_dir):
    """Save verification results to JSON file.
    
    Args:
        result: Verification result from pipeline
        audio_paths: List of audio file paths that were compared
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results data
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "audio_files": [str(Path(p).resolve()) for p in audio_paths],
        "num_files_compared": len(audio_paths),
        "verification_result": {
            "text": result.get("text", "unknown"),
            "score": float(result.get("score", 0.0)),
            "same_speaker": result.get("text", "").lower() == "yes"
        },
        "model": "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    }
    
    # Save as JSON
    json_path = output_dir / f"verification_result_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {json_path}")
    
    # Also save a human-readable summary
    summary_path = output_dir / f"verification_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SPEAKER VERIFICATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {results_data['timestamp']}\n")
        f.write(f"Files compared: {results_data['num_files_compared']}\n\n")
        f.write("Audio files:\n")
        for i, path in enumerate(results_data['audio_files'], 1):
            f.write(f"  {i}. {Path(path).name}\n")
            f.write(f"     {path}\n")
        f.write(f"\nResult: {results_data['verification_result']['text'].upper()}\n")
        f.write(f"Confidence score: {results_data['verification_result']['score']:.4f}\n")
        f.write(f"Same speaker: {'Yes ✓' if results_data['verification_result']['same_speaker'] else 'No ✗'}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    
    return json_path


def verify_speakers(audio_paths, model_id='iic/speech_eres2netv2_sv_zh-cn_16k-common', output_dir=None):
    """Verify if multiple audio files contain the same speaker.
    
    Args:
        audio_paths: List of audio file paths to compare (minimum 2)
        model_id: ModelScope model ID for speaker verification
        output_dir: Directory to save results (if None, uses OUTPUT_DIR)
    
    Returns:
        Pipeline result dictionary
    """
    if len(audio_paths) < 2:
        raise ValueError(f"Need at least 2 audio files for comparison, got {len(audio_paths)}")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    
    # Initialize pipeline
    sv_pipeline = pipeline(
        task=Tasks.speaker_verification,
        model=model_id,
    )
    
    # Preprocess all audio files
    print("="*60)
    print("AUDIO PREPROCESSING STAGE")
    print("="*60)
    processed_paths = [preprocess_audio(path) for path in audio_paths]
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    # Run speaker verification
    print("\n" + "="*60)
    print("SPEAKER VERIFICATION STAGE")
    print("="*60)
    result = sv_pipeline(processed_paths)
    print("\nVerification result:")
    print(result)
    
    # Save results to files
    save_results(result, audio_paths, output_dir / "results")
    
    return result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speaker verification - compare multiple audio files to check if they're from the same speaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s audio1.wav audio2.wav\n"
               "  %(prog)s audio1.wav audio2.wav audio3.wav\n"
               "  %(prog)s  # Uses default audio paths"
    )
    parser.add_argument(
        'audio_paths',
        nargs='*',
        help='Audio file paths to compare (minimum 2). If not provided, uses default paths.'
    )
    parser.add_argument(
        '--model',
        default='iic/speech_eres2netv2_sv_zh-cn_16k-common',
        help='ModelScope speaker verification model ID (default: iic/speech_eres2netv2_sv_zh-cn_16k-common)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for output files (default: generated/<script_name>)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Use provided paths or fall back to defaults
    audio_paths = args.audio_paths if args.audio_paths else DEFAULT_AUDIO_PATHS
    
    if len(audio_paths) < 2:
        print(f"Error: Need at least 2 audio files for comparison, but got {len(audio_paths)}")
        print(f"Default paths: {DEFAULT_AUDIO_PATHS}")
        return 1
    
    print(f"Comparing {len(audio_paths)} audio files:")
    for i, path in enumerate(audio_paths, 1):
        print(f"  {i}. {path}")
    
    # Clean output directory
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run verification
    result = verify_speakers(audio_paths, model_id=args.model, output_dir=args.output_dir or OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
