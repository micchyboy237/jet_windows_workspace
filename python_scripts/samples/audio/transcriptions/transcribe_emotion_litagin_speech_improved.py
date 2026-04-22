from typing import List, Tuple, Dict
import torch
import torchaudio
import soundfile as sf
from silero_vad import load_silero_vad, VADIterator
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from rich.console import Console
from rich.table import Table
import numpy as np
import argparse

DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"

parser = argparse.ArgumentParser(description="Run speech separation model.")
parser.add_argument("audio_path",
                    nargs="?",
                    default=DEFAULT_AUDIO,
                    help="Path to input .wav audio file")
args = parser.parse_args()

AUDIO_PATH = args.audio_path

# =========================
# Config
# =========================
REPO_ID = "litagin/anime_speech_emotion_classification"

SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512
MAX_MERGE_GAP_SEC = 0.5
MIN_SEGMENT_SEC = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
console = Console()

# =========================
# Emotion Classifier (batch-optimized)
# =========================
class EmotionClassifier:
    def __init__(self, model_name: str, device: str):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.id2label = self.model.config.id2label

    def classify_batch(self, audio_list: List[np.ndarray]) -> List[List[Dict[str, float]]]:
        # FIX: handle empty input safely
        if not audio_list:
            return []
        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        results = []
        for row in probs:
            values, indices = torch.topk(row, k=row.shape[-1])
            results.append([
                {"label": self.id2label[i.item()], "score": v.item()}
                for v, i in zip(values, indices)
            ])
        return results

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
    segments: List[Tuple[float, float, torch.Tensor]] = []
    start = None
    buffer = []

    for i, chunk in enumerate(waveform.split(VAD_CHUNK_SIZE)):
        # Pad short chunks to VAD_CHUNK_SIZE to avoid Silero crash
        if chunk.shape[0] < VAD_CHUNK_SIZE:
            pad_size = VAD_CHUNK_SIZE - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))

        t = i * (VAD_CHUNK_SIZE / SAMPLE_RATE)
        if vad(chunk, return_seconds=True):
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
    return classifier.classify_batch(audio_batch)

# =========================
# MAIN EXECUTION
# =========================
print(f"Loading VAD and classifier on {DEVICE}...")
vad_model = load_silero_vad()
classifier = EmotionClassifier(REPO_ID, DEVICE)

raw_segments = extract_vad_segments(AUDIO_PATH)
merged_segments = merge_segments(
    raw_segments,
    max_gap=MAX_MERGE_GAP_SEC,
    min_duration=MIN_SEGMENT_SEC,
)

# FIX: fallback if everything filtered out
if not merged_segments and raw_segments:
    console.print("[yellow]No merged segments passed filter, using raw VAD segments[/yellow]")
    merged_segments = raw_segments

console.print(
    f"[bold green]Merged into {len(merged_segments)} emotion-safe segments[/bold green]"
)

results = classify_segments(merged_segments)

# FIX: handle no results case
if not results:
    console.print("[red]No segments to classify[/red]")
    exit(0)

for i, ((start, end, _), preds) in enumerate(zip(merged_segments, results), 1):
    table = Table(title=f"Segment {i}: {start:.2f}s – {end:.2f}s")
    table.add_column("Rank")
    table.add_column("Emotion")
    table.add_column("Score")

    for rank, item in enumerate(preds, 1):
        table.add_row(str(rank), item["label"], f"{item['score']:.3f}")

    console.print(table)