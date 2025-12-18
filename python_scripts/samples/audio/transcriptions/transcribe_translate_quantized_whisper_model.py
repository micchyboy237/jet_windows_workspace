from faster_whisper import WhisperModel

model_name_or_path = r"C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-small-ct2"
# model_name_or_path = "kotoba-tech/kotoba-whisper-v2.0-faster"
model = WhisperModel(model_name_or_path, device="cpu", compute_type="int8")

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"

segments, info = model.transcribe(
    audio_path,
    language="ja",
    word_timestamps=True,  # Enables word-level timestamps
    log_progress=True,
    task="translate",
)

segment_texts = []
segment_words = []
for seg_num, segment in enumerate(segments, start=1):
    segment_texts.append(segment.text)
    segment_words.append(segment.words)

    print(f"\nSegment {seg_num} - Japanese Words:")
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word} (p={word.probability:.3f})")
