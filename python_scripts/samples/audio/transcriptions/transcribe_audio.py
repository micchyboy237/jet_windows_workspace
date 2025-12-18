from faster_whisper import WhisperModel

model_path = r"C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-tiny-ct2-int8_float16"
model_path = r"C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-base-ct2-int8_float16"
model_path = r"C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-small-ct2-int8_float16"
audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\1.wav"

# Run on GPU with FP16 (faster, lower precision)
model = WhisperModel(model_path, device="cuda")

# Transcribe audio file
segments_iter, info = model.transcribe(audio_path, beam_size=5, language="ja", task="translate")  # beam_size for better accuracy

print(f"Detected language: {info.language} (probability: {info.language_probability})")
print(f"Duration: {info.duration} seconds")

segments = []
for segment in segments_iter:
    segments.append(segment)
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

print(f"Segments: {len(segments)}")