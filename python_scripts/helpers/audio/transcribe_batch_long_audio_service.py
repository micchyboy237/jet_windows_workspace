from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio  # Optional helper for loading

# Load the model (adjust size, device, and compute_type for your hardware)
model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",       # Or "medium", "small", etc.
    device="cpu",       # "cpu" if no GPU; your GTX 1660 supports CUDA
    compute_type="int8"  # Faster on GPU; use "int8" for more savings if needed
)

# Wrap for batched inference (supports VAD for better chunking)
batched_model = BatchedInferencePipeline(
    model=model,
)

# Load audio (numpy array expected; supports file paths or arrays)
audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"
audio = decode_audio(audio_path)  # Or use your own loader (e.g., torchaudio)

# Transcribe with batching
segments, info = batched_model.transcribe(
    audio,
    batch_size=16,       # Key parameter: try 8-32; higher = faster throughput
    beam_size=5,         # Default in faster-whisper (better quality than greedy)
    # word_timestamps=True,  # Optional: for word-level timings
    language="ja"        # Optional: specify or None for auto-detect
)

# Iterate over results (generator for low memory)
full_text = ""
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text += segment.text

print("\nFull transcription:", full_text)
print("Detected language:", info.language, "Probability:", info.language_probability)