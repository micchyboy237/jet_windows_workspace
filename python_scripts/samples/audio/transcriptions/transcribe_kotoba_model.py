import json
from dataclasses import asdict
from faster_whisper import WhisperModel

def main(device: str, compute_type: str):
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device=device,
        compute_type=compute_type,
    )

    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"

    segments, info = model.transcribe(
        audio_path,
        language="ja",
        log_progress=True,
    )

    segment_texts = []
    for seg_num, segment in enumerate(segments, start=1):
        segment_texts.append(segment.text)

        print(f"\nSegment {seg_num} - Japanese Text:")
        print(segment.text)

        segment_dict = asdict(segment)
        segment_dict.pop("text")
        segment_dict.pop("tokens")
        print(json.dumps(segment_dict, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # main(
    #     device="cpu",
    #     compute_type="int8",
    # )

    # main(
    #     device="cuda",
    #     compute_type="float16",
    # )

    main(
        device="cuda",
        compute_type="float32",
    )