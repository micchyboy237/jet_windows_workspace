import json
from dataclasses import asdict
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"

segments, info = model.transcribe(
    audio_path,
    language="ja",
    word_timestamps=True,  # Enables word-level timestamps
    log_progress=True,
)

segment_texts = []
segment_words = []
for seg_num, segment in enumerate(segments, start=1):
    segment_texts.append(segment.text)
    segment_words.append(segment.words)

    print(f"\nSegment {seg_num} - Japanese Text:")
    print(segment.text)

    print(f"Words ({len(segment.words)}):")
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")

    segment_dict = asdict(segment)
    segment_dict.pop("text")
    segment_dict.pop("tokens")
    print(json.dumps(segment_dict, indent=2, ensure_ascii=False))
