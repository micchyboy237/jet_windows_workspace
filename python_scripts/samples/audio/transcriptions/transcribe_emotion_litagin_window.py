from transformers import pipeline
import numpy as np
from scipy.io import wavfile
import torch
import os
from pathlib import Path
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
import logging

# Better logging instead of just prints
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

# ──── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "window_sec": 3.0,
    "step_sec": 1.0,
    "model_name": "litagin/anime_speech_emotion_classification",
    "top_k": 5,
    "sample_rate_target": 16000,   # most speech emotion models like 16kHz
}

AUDIO_PATH = Path(r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav")


def load_audio(path: Path) -> Tuple[int, np.ndarray]:
    """Load audio, convert to mono float32, normalize peak"""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sr, data = wavfile.read(path)
    
    # stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    # to float32 & normalize peak
    data = data.astype(np.float32)
    if np.max(np.abs(data)) > 1e-6:
        data /= np.max(np.abs(data))  # simple peak normalization
    
    return sr, data


def create_pipeline(model_name: str, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    logger.info(f"Loading model on device: {device}")
    return pipeline(
        "audio-classification",
        model=model_name,
        feature_extractor=model_name,
        trust_remote_code=True,
        device=device,
    )


def sliding_window_emotion_analysis(
    data: np.ndarray,
    sr: int,
    pipe,
    window_sec: float,
    step_sec: float,
    top_k: int = 3
) -> List[Dict[str, any]]:
    """Process audio in overlapping windows without temporary files"""
    duration = len(data) / sr
    results = []
    
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)

    for start_idx in range(0, len(data) - window_samples + 1, step_samples):
        chunk = data[start_idx : start_idx + window_samples]
        
        # Most pipelines accept raw numpy float32 array (especially recent transformers)
        try:
            predictions = pipe(chunk, sampling_rate=sr, top_k=top_k)
        except Exception as e:
            logger.warning(f"Pipeline failed at {start_idx/sr:.1f}s → {e}")
            continue

        mid_time = (start_idx + window_samples // 2) / sr

        results.append({
            "mid_time": mid_time,
            "top_k": predictions
        })

    return results


def print_results(results: List[Dict]):
    table = Table(title="Anime Speech Emotion - Sliding Window Analysis")
    table.add_column("Time window", justify="right")
    table.add_column("Top Emotion")
    table.add_column("Confidence", justify="right")
    table.add_column("2nd", justify="left")
    table.add_column("3rd", justify="left")

    for i, item in enumerate(results):
        # Calculate actual window range
        start_time = i * CONFIG["step_sec"]
        end_time = start_time + CONFIG["window_sec"]

        top = item["top_k"][0]
        second = item["top_k"][1] if len(item["top_k"]) >= 2 else {"label":"-", "score":0.0}
        third  = item["top_k"][2] if len(item["top_k"]) >= 3 else {"label":"-", "score":0.0}

        table.add_row(
            f"{start_time:.1f}–{end_time:.1f}s",
            top["label"],
            f"{top['score']:.1%}",
            f"{second['label']} ({second['score']:.0%})",
            f"{third['label']}  ({third['score']:.0%})"
        )

    console.print(table)


def main():
    try:
        sr, audio = load_audio(AUDIO_PATH)

        # Optional: resample if model really wants different sr (most 16kHz models can handle it)
        # if sr != CONFIG["sample_rate_target"]:
        #     from scipy.signal import resample
        #     audio = resample(audio, int(len(audio) * CONFIG["sample_rate_target"] / sr))
        #     sr = CONFIG["sample_rate_target"]

        pipe = create_pipeline(CONFIG["model_name"])

        results = sliding_window_emotion_analysis(
            audio,
            sr,
            pipe,
            CONFIG["window_sec"],
            CONFIG["step_sec"],
            CONFIG["top_k"]
        )

        print_results(results)

    except Exception as e:
        logger.error(f"Failed to process audio: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()