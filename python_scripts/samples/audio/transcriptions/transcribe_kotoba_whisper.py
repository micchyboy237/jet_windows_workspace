import json
from dataclasses import asdict
from faster_whisper import WhisperModel
from translators.translate_jp_en_opus import translate_japanese_to_english

# model_name_or_path = r"C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-small-ct2"
model_name_or_path = "kotoba-tech/kotoba-whisper-v2.0-faster"
model = WhisperModel(model_name_or_path, device="cuda", compute_type="float32")

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

segments, info = model.transcribe(
    audio_path,
    language="ja",
    word_timestamps=False,  # Enables word-level timestamps
    log_progress=True,
)

segment_texts = []
segment_words = []
for seg_num, segment in enumerate(segments, start=1):
    segment_texts.append(segment.text)
    segment_words.append(segment.words)

    ja_text = segment.text
    en_text = translate_japanese_to_english(ja_text)

    print(f"\nSegment {seg_num}")
    print(f"JA: {ja_text}")
    print(f"EN: {en_text}")

    segment_dict = asdict(segment)
    segment_dict.pop("text")
    segment_dict.pop("tokens")
    print(json.dumps(segment_dict, indent=2, ensure_ascii=False))
