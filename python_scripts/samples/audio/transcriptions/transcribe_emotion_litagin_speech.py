from typing import List, Dict, Tuple
import torch
import torchaudio
from silero_vad import load_silero_vad, VADIterator
from transformers import pipeline
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

# Config
REPO_ID = "litagin/anime_speech_emotion_classification"
AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000  # Model expects 16kHz
VAD_CHUNK_SIZE = 512  # required by Silero for 16kHz

# Load models
print(f"Loading VAD and classifier on {DEVICE}...")
vad_model = load_silero_vad()
classifier = pipeline(
    "audio-classification",
    model=REPO_ID,
    feature_extractor=REPO_ID,
    trust_remote_code=True,
    device=DEVICE,
)

console = Console()


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    wave = torch.from_numpy(audio)

    if sr != target_sr:
        wave = torchaudio.functional.resample(wave, sr, target_sr)

    return wave


def extract_segments(audio_path: str) -> List[Tuple[float, float, torch.Tensor]]:
    waveform = load_audio(audio_path, SAMPLE_RATE)

    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
    )

    VAD_CHUNK_SIZE = 512  # REQUIRED for 16kHz
    segments = []
    start_time = None
    buffer = []

    for i, chunk in enumerate(waveform.split(VAD_CHUNK_SIZE)):
        chunk_time = i * (VAD_CHUNK_SIZE / SAMPLE_RATE)

        if vad_iter(chunk, return_seconds=True):
            if start_time is None:
                start_time = chunk_time
            buffer.append(chunk)

        elif start_time is not None:
            end_time = chunk_time
            segment_wave = torch.cat(buffer)
            segments.append((start_time, end_time, segment_wave))
            start_time = None
            buffer = []

    if start_time is not None:
        segment_wave = torch.cat(buffer)
        segments.append(
            (start_time, len(waveform) / SAMPLE_RATE, segment_wave)
        )

    return segments


def classify_segment(segment_wave: torch.Tensor) -> List[Dict[str, float]]:
    """Run classifier on a segment."""
    audio_np = segment_wave.cpu().numpy()
    results = classifier(audio_np)
    return results  # list of {'label': str, 'score': float}

# Run
segments = extract_segments(AUDIO_PATH)
console.print(f"[bold green]Found {len(segments)} speech segments[/bold green]")

results_per_segment: List[List[Dict[str, float]]] = []
for i, (start, end, wave) in enumerate(segments, 1):
    result = classify_segment(wave)
    results_per_segment.append(result)
    
    table = Table(title=f"Segment {i}: {start:.1f}s – {end:.1f}s")
    table.add_column("Rank", justify="right")
    table.add_column("Emotion", style="cyan")
    table.add_column("Score", justify="right")
    for rank, item in enumerate(result, 1):
        table.add_row(str(rank), item['label'], f"{item['score']:.3f}")
    console.print(table)

# Visualization: Bar chart of top emotions per segment
emotions = [r[0]['label'] for r in results_per_segment]  # Top emotion per segment
scores = [r[0]['score'] for r in results_per_segment]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(segments))
ax.bar(x, scores, color='skyblue')
ax.set_xticks(x)
ax.set_xticklabels([f"{s[0]:.1f}-{s[1]:.1f}s" for s in segments], rotation=45)
ax.set_ylabel("Confidence Score")
ax.set_title("Top Emotion per Segment (Highest Score)")
ax.set_ylim(0, 1)

for i, emotion in enumerate(emotions):
    ax.text(i, scores[i] + 0.02, emotion, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()