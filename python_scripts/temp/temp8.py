from pyannote.audio.core.io import get_audio_metadata, Audio

# Test with string path (get_audio_metadata now handles this)
DEFAULT_AUDIO = "C:/Users/druiv/.cache/files/audio/recording_3_speakers.wav"

print("=" * 60)
print("Test 1: Getting metadata from string path")
print("=" * 60)
try:
    metadata = get_audio_metadata(DEFAULT_AUDIO)
    print(f"✓ Success! Metadata:\n{metadata}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Test 2: Loading full audio")
print("=" * 60)
try:
    audio = Audio(sample_rate=16000, mono='downmix')
    waveform, sample_rate = audio({"audio": DEFAULT_AUDIO})
    print(f"✓ Success! Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Test 3: Getting duration")
print("=" * 60)
try:
    duration = audio.get_duration({"audio": DEFAULT_AUDIO})
    print(f"✓ Success! Duration: {duration:.2f} seconds")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Test 4: Cropping a segment")
print("=" * 60)
try:
    from pyannote.core import Segment
    segment = Segment(0.0, 5.0)  # First 5 seconds
    crop_waveform, crop_sr = audio.crop({"audio": DEFAULT_AUDIO}, segment)
    print(f"✓ Success! Crop shape: {crop_waveform.shape}, Sample rate: {crop_sr}")
except Exception as e:
    print(f"✗ Failed: {e}")