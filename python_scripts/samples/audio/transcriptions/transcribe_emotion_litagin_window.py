import numpy as np
from scipy.io import wavfile
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any
from rich.console import Console
from rich.table import Table
import logging
import argparse

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"

parser = argparse.ArgumentParser(description="Run speech emotion analysis.")
parser.add_argument(
    "audio_path",
    nargs="?",
    type=Path,
    default=DEFAULT_AUDIO,
    help="Path to input .wav audio file",
)
args = parser.parse_args()

AUDIO_PATH = args.audio_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

# ──── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "window_sec": 6.0,
    "step_sec": 2.0,
    "model_name": "litagin/anime_speech_emotion_classification",
    "top_k": 5,
    "sample_rate_target": 16000,
}


# ──── Audio Loading ─────────────────────────────────────────────────────────
def load_audio(path: Path) -> Tuple[int, np.ndarray]:
    """Load audio, convert to mono float32, normalize peak"""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sr, data = wavfile.read(path)

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    if np.max(np.abs(data)) > 1e-6:
        data /= np.max(np.abs(data))

    return sr, data


# ──── Model Wrapper (NO TorchCodec) ─────────────────────────────────────────
class EmotionClassifier:
    def __init__(self, model_name: str, device: torch.device):
        logger.info(f"Loading model: {model_name} on {device}")

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

    def classify(self, audio: np.ndarray, sr: int, top_k: int) -> List[Dict[str, Any]]:
        """Run inference on a single audio chunk"""

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        values, indices = torch.topk(probs, k=top_k)

        return [
            {
                "label": self.id2label[idx.item()],
                "score": val.item(),
            }
            for val, idx in zip(values[0], indices[0])
        ]


# ──── Sliding Window ────────────────────────────────────────────────────────
def sliding_window_emotion_analysis(
    data: np.ndarray,
    sr: int,
    classifier: EmotionClassifier,
    window_sec: float,
    step_sec: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Process audio in overlapping windows"""

    results = []

    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)

    for start_idx in range(0, len(data) - window_samples + 1, step_samples):
        chunk = data[start_idx : start_idx + window_samples]

        try:
            predictions = classifier.classify(chunk, sr, top_k)
        except Exception as e:
            logger.warning(f"Failed at {start_idx/sr:.1f}s → {e}")
            continue

        mid_time = (start_idx + window_samples // 2) / sr

        results.append(
            {
                "mid_time": mid_time,
                "top_k": predictions,
            }
        )

    return results


# ──── Output ────────────────────────────────────────────────────────────────
def print_results(results: List[Dict[str, Any]]):
    table = Table(title="Anime Speech Emotion - Sliding Window Analysis")
    table.add_column("Time window", justify="right")

    for rank in range(1, CONFIG["top_k"] + 1):
        suffix = ["st", "nd", "rd"][rank - 1] if rank <= 3 else "th"
        table.add_column(f"{rank}{suffix}", justify="left")

    for i, item in enumerate(results):
        start_time = i * CONFIG["step_sec"]
        end_time = start_time + CONFIG["window_sec"]

        preds = item["top_k"]
        row = [f"{start_time:.1f}–{end_time:.1f}s"]

        for rank in range(CONFIG["top_k"]):
            if rank < len(preds):
                p = preds[rank]
                row.append(f"{p['label']} ({p['score']:.0%})")
            else:
                row.append("-")

        table.add_row(*row)

    console.print(table)


# ──── Main ──────────────────────────────────────────────────────────────────
def main():
    try:
        sr, audio = load_audio(AUDIO_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classifier = EmotionClassifier(
            CONFIG["model_name"],
            device,
        )

        results = sliding_window_emotion_analysis(
            audio,
            sr,
            classifier,
            CONFIG["window_sec"],
            CONFIG["step_sec"],
            CONFIG["top_k"],
        )

        print_results(results)

    except Exception as e:
        logger.error(f"Failed to process audio: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
