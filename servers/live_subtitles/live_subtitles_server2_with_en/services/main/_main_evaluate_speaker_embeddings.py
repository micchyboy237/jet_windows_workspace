import argparse
import shutil
import torch

from pathlib import Path
from evaluate_speaker_embeddings import EmbeddingModelType, run_evaluation

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET = Path(r"C:\Users\druiv\.cache\files\audio\speakers")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate speaker embedding models on a custom dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset", type=Path, nargs="?",
        default=DEFAULT_DATASET,
        help="Root directory. Each subdirectory = one speaker.",
    )
    parser.add_argument(
        "-m", "--models", nargs="+",
        choices=[e.value for e in EmbeddingModelType],
        default=None,
        help="Models to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_DIR / "eval_results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "-c", "--cache", type=Path, default=OUTPUT_DIR / ".embedding_cache",
        help="Directory to cache extracted embeddings.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="Torch device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "-p", "--max-pos", type=int, default=10,
        help="Max positive trial pairs per speaker.",
    )
    parser.add_argument(
        "-r", "--neg-ratio", type=float, default=1.0,
        help="Negative-to-positive trial ratio.",
    )
    parser.add_argument(
        "-u", "--min-utts", type=int, default=2,
        help="Minimum utterances required per speaker.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device) if args.device else None

    run_evaluation(
        dataset_root=args.dataset,
        model_types=args.models,
        output_dir=args.output,
        cache_dir=args.cache,
        device=device,
        max_positive_per_speaker=args.max_pos,
        neg_pos_ratio=args.neg_ratio,
        min_utterances=args.min_utts,
        seed=args.seed,
    )


if __name__ == "__main__":
  main()
