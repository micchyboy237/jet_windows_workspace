from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, List, Literal

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class TimestampSegment(TypedDict):
    """A single timestamp segment in milliseconds [start, end]."""
    start: int
    end: int

class TranscriptionResult(TypedDict):
    """TypedDict representing one transcription result item from SenseVoiceSmall."""
    key: str
    text: str
    timestamp: List[TimestampSegment]
    words: List[str]

# Optional: if you expect multiple results (even though current example has one)
TranscriptionResults = List[TranscriptionResult]


audio_path = Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav")
model_dir = "FunAudioLLM/SenseVoiceSmall"

print("Loading SenseVoiceSmall model (with VAD)...")
model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

print(f"Transcribing audio: {audio_path.name}")
res: TranscriptionResults = model.generate(
    input=str(audio_path),
    cache={},
    language="ja",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
    output_timestamp=True,
)

print("Transcription completed")
print("Raw result:")
print(json.dumps(res, indent=2, ensure_ascii=False))

if res:
    cleaned_text = rich_transcription_postprocess(res[0]["text"])
    print("🎼 Cleaned transcription:")
    print(cleaned_text)
else:
    print("No transcription result returned")