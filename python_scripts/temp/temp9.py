import shutil
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

speaker_1_path1 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_lloyd\sound1.wav"
speaker_1_path2 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_lloyd\sound2.wav"
speaker_2_path1 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_anya\sound1.wav"
speaker_2_path2 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_anya\sound2.wav"
speakers_overlap_path1 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_outliers\sound_lloyd_anya_speech1.wav"
speakers_overlap_path2 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_outliers\sound_lloyd_anya_speech2.wav"

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

sv_pipeline = pipeline(
    task=Tasks.speaker_verification,
    model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
    # model_revision='v1.0.2'
)

# Preprocess all audio files
print("="*60)
print("AUDIO PREPROCESSING STAGE")
print("="*60)
speaker_1_path1 = preprocess_audio(speaker_1_path1)
speaker_1_path2 = preprocess_audio(speaker_1_path2)
speaker_2_path1 = preprocess_audio(speaker_2_path1)
speaker_2_path2 = preprocess_audio(speaker_2_path2)
speakers_overlap_path1 = preprocess_audio(speakers_overlap_path1)
speakers_overlap_path2 = preprocess_audio(speakers_overlap_path2)
print(f"\n{'='*60}")
print("PREPROCESSING COMPLETE")
print("="*60)

# Same speakers
same_result = sv_pipeline([speaker_1_path1, speaker_1_path2])
print("\nSame result 1:")
print(same_result)
# assert same_result['text'] == 'yes'
# assert same_result['score'] > 0.5

same_result = sv_pipeline([speaker_2_path1, speaker_2_path2])
print("\nSame result 2:")
print(same_result)
# assert same_result['text'] == 'yes'
# assert same_result['score'] > 0.5

# Different speakers
diff_result = sv_pipeline([speaker_1_path1, speaker_2_path1])
print("\nDifference result 1:")
print(diff_result)
# assert diff_result['text'] == 'no'
# assert diff_result['score'] < 0.5

diff_result = sv_pipeline([speaker_1_path2, speaker_2_path2])
print("\nDifference result 2:")
print(diff_result)
# assert diff_result['text'] == 'no'
# assert diff_result['score'] < 0.5

# Overlapping speakers
overlap_result = sv_pipeline([speaker_1_path1, speakers_overlap_path1])
print("\nOverlap result 1:")
print(overlap_result)
# assert overlap_result['text'] == 'yes'
# assert overlap_result['score'] > 0.5

overlap_result = sv_pipeline([speaker_2_path1, speakers_overlap_path2])
print("\nOverlap result 2:")
print(overlap_result)
# assert overlap_result['text'] == 'yes'
# assert overlap_result['score'] > 0.5

# Get embeddings + score
result = sv_pipeline(
    [speaker_1_path1, speaker_2_path1],
    output_emb=True, # Return embeddings too
    save_dir=str(OUTPUT_DIR / "embeddings"), # Save .npy files
)
print("\nEmbeddings result:")
print(result['outputs'])
print(type(result['embs'])) # <class 'numpy.ndarray'>
print(result['embs'].shape) # (2, 192)
print(result['embs'].dtype) # float32
# assert result['outputs']['text'] == 'no'
# assert result['outputs']['score'] < 0.5

# Custom threshold
result3 = sv_pipeline([speaker_1_path1, speaker_2_path1], thr=0.25)
print("\nCustom threshold result:")
print(result3)
# assert result3['text'] == 'yes'
# assert result3['score'] > 0.25