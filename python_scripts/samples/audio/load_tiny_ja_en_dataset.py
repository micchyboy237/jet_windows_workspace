# load_covost_tiny_fixed.py
from __future__ import annotations

from pathlib import Path
from typing import List
import json
from datasets import load_dataset, Audio


def load_tiny_ja_en_dataset(
    num_samples: int = 20,
    split: str = "test"
) -> List[dict]:
    """
    Load tiny JA→EN CoVoST 2 subset – NOW WORKS WITHOUT MANUAL DOWNLOAD.
    Uses the new fully hosted dataset: https://huggingface.co/datasets/covost2
    """
    # This works instantly – no manual download needed!
    dataset = load_dataset("covost2", "ja_en", split=split)

    # Optional: resample to 16kHz (Whisper standard)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # Take first N samples
    tiny = dataset.select(range(min(num_samples, len(dataset))))

    samples = []
    for item in tiny:
        # item["audio"] is dict with "path" (remote URL or cached path) and "array"
        samples.append({
            "audio_path": item["audio"]["path"],        # Works with Whisper directly
            "reference_en": item["translation"].strip(),
        })

    return samples


def save_to_jsonl(samples: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ─── Run directly ───
if __name__ == "__main__":
    samples = load_tiny_ja_en_dataset(num_samples=20, split="test")
    save_to_jsonl(samples, Path("data/tiny_covost_ja_en.jsonl"))

    print(f"Loaded {len(samples)} samples")
    print("First reference:", samples[0]["reference_en"])
    print("First audio path:", samples[0]["audio_path"])