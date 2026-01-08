from typing import List, Tuple, Dict
import torch
import torchaudio
import soundfile as sf
from silero_vad import load_silero_vad, VADIterator
from transformers import pipeline
from rich.console import Console
from rich.table import Table
import numpy as np

# =========================
# Config
# =========================
REPO_ID = "litagin/anime_speech_emotion_classification"
AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512
MAX_MERGE_GAP_SEC = 0.3
MIN_SEGMENT_SEC = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
console = Console()

# =========================
# Audio loader (stable)
# =========================
def load_audio(path: str, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wave = torch.from_numpy(audio)
    if sr != target_sr:
        wave = torchaudio.functional.resample(wave, sr, target_sr)
    return wave

# =========================
# VAD segmentation (raw)
# =========================
def extract_vad_segments(audio_path: str) -> List[Tuple[float, float, torch.Tensor]]:
    waveform = load_audio(audio_path, SAMPLE_RATE)

    vad = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
    segments = []

    start = None
    buffer = []

    for i, chunk in enumerate(waveform.split(VAD_CHUNK_SIZE)):
        t = i * (VAD_CHUNK_SIZE / SAMPLE_RATE)
        if vad(chunk):
            if start is None:
                start = t
            buffer.append(chunk)
        elif start is not None:
            end = t
            segments.append((start, end, torch.cat(buffer)))
            start = None
            buffer = []

    if start is not None:
        segments.append(
            (start, len(waveform) / SAMPLE_RATE, torch.cat(buffer))
        )

    return segments

# =========================
# Merge + filter segments
# =========================
def merge_segments(
    segments: List[Tuple[float, float, torch.Tensor]],
    max_gap: float,
    min_duration: float,
) -> List[Tuple[float, float, torch.Tensor]]:

    if not segments:
        return []

    merged = []
    cur_start, cur_end, cur_wave = segments[0]

    for start, end, wave in segments[1:]:
        if start - cur_end <= max_gap:
            cur_end = end
            cur_wave = torch.cat([cur_wave, wave])
        else:
            if cur_end - cur_start >= min_duration:
                merged.append((cur_start, cur_end, cur_wave))
            cur_start, cur_end, cur_wave = start, end, wave

    if cur_end - cur_start >= min_duration:
        merged.append((cur_start, cur_end, cur_wave))

    return merged

# =========================
# Batch emotion inference
# =========================
def classify_segments(
    segments: List[Tuple[float, float, torch.Tensor]],
) -> List[List[Dict[str, float]]]:

    audio_batch = [
        seg[2].cpu().numpy() for seg in segments
    ]
    return classifier(audio_batch)

# =========================
# MAIN EXECUTION
# =========================
print(f"Loading VAD and classifier on {DEVICE}...")
vad_model = load_silero_vad()
classifier = pipeline(
    "audio-classification",
    model=REPO_ID,
    feature_extractor=REPO_ID,
    trust_remote_code=True,
    device=DEVICE,
)

raw_segments = extract_vad_segments(AUDIO_PATH)
merged_segments = merge_segments(
    raw_segments,
    max_gap=MAX_MERGE_GAP_SEC,
    min_duration=MIN_SEGMENT_SEC,
)

console.print(
    f"[bold green]Merged into {len(merged_segments)} emotion-safe segments[/bold green]"
)

results = classify_segments(merged_segments)

for i, ((start, end, _), preds) in enumerate(zip(merged_segments, results), 1):
    table = Table(title=f"Segment {i}: {start:.2f}s – {end:.2f}s")
    table.add_column("Rank")
    table.add_column("Emotion")
    table.add_column("Score")

    for rank, item in enumerate(preds, 1):
        table.add_row(str(rank), item["label"], f"{item['score']:.3f}")

    console.print(table)