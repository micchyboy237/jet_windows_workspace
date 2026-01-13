from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table
from transformers import pipeline


def classify_emotion_sliding_window(
    audio_path: str | Path,
    window_sec: float = 6.0,
    step_sec: float = 2.0,
    target_sr: int = 16000,
    model_name: str = "litagin/anime_speech_emotion_classification",
    device: str = "cuda",  # "cpu" if no GPU
) -> List[Dict]:
    """
    Perform sliding-window emotion classification on an audio file.

    Returns a list of dictionaries with time, top label and score.
    """
    audio_path = Path(audio_path)

    # Load and resample audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    duration = len(audio) / target_sr

    # Initialise HF pipeline (handles feature extraction internally)
    pipe = pipeline(
        "audio-classification",
        model=model_name,
        trust_remote_code=True,
        device=device,
    )

    window_samples = int(window_sec * target_sr)
    step_samples = int(step_sec * target_sr)

    results = []
    times = []

    start_sample = 0
    with tqdm(total=int((duration - window_sec) / step_sec) + 1, desc="Classifying windows", unit="window") as pbar:
        while start_sample + window_samples <= len(audio):
            end_sample = start_sample + window_samples
            chunk = audio[start_sample:end_sample]

            # Pipeline accepts raw np.ndarray directly
            pred = pipe(chunk)[0]  # top prediction

            mid_time = (start_sample + end_sample) / 2 / target_sr
            results.append(pred)
            times.append(mid_time)

            start_sample += step_samples
            pbar.update(1)

    # Build and display rich table
    table = Table(title="Sliding Window Emotion Predictions")
    table.add_column("Time (middle)", justify="right")
    table.add_column("Top Emotion")
    table.add_column("Confidence", justify="right")

    for t, r in zip(times, results):
        table.add_row(f"{t:5.1f}s", r["label"], f"{r['score']:.1%}")

    rprint(table)

    return [{"time": t, "label": r["label"], "score": r["score"]} for t, r in zip(times, results)]


# Example usage
if __name__ == "__main__":
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    classify_emotion_sliding_window(audio_path)