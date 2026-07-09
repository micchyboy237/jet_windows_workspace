"""
_main_evaluate_speaker_cluster.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI entry point for evaluate_speaker_cluster.py.

Mirrors the pattern of _main_evaluate_speaker_embeddings.py —
scans a speaker dataset, runs clustering, and reports metrics.

Usage:
    python -m services.main._main_evaluate_speaker_cluster \\
        /path/to/speakers \\
        -m pyannote speechbrain_ecapa modelscope_eres2netv2
"""

import argparse
import shutil
from pathlib import Path

from evaluate_speaker_cluster import (
    EmbeddingModelType,
    run_evaluation,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET = Path(r"C:\Users\druiv\.cache\files\audio\speakers")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate speaker clustering on a multi-speaker dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Positional ----
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET,
        help="Root directory. Each subdirectory = one speaker.",
    )

    # ---- Model selection ----
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        choices=[e.value for e in EmbeddingModelType],
        default=None,
        help="Models to evaluate. Defaults to all.",
    )

    # ---- Output ----
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_DIR / "cluster_eval_results",
        help="Directory to save results.",
    )

    # ---- Device ----
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cuda' or 'cpu'.",
    )

    # ---- Clustering knobs ----
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help=(
            "Cosine-similarity threshold (0.0–1.0). "
            "Overrides per-model defaults. Higher = more clusters."
        ),
    )

    parser.add_argument(
        "-k", "--min-cluster-size",
        type=int,
        default=None,
        dest="min_cluster_size",
        help="Minimum cluster size (default: per-model best).",
    )

    parser.add_argument(
        "-l", "--linkage",
        dest="linkage_method",
        default=None,
        choices=["average", "ward", "complete", "single"],
        help="Hierarchical linkage method (default: per-model best).",
    )

    # ---- Threshold strategies ----
    threshold_group = parser.add_mutually_exclusive_group()

    threshold_group.add_argument(
        "--auto-threshold",
        action="store_true",
        default=False,
        help="Auto-estimate the best threshold from the data per model.",
    )

    threshold_group.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        dest="threshold_sweep",
        help=(
            "Run a full threshold sweep (0.30–0.90, 13 steps) per model "
            "and report the best.  Embeddings are extracted only once."
        ),
    )

    parser.add_argument(
        "--sweep-values",
        type=float,
        nargs="+",
        default=None,
        dest="sweep_values",
        help="Custom threshold values for --sweep (e.g. 0.4 0.5 0.6 0.7).",
    )

    # ---- Dataset ----
    parser.add_argument(
        "-u", "--min-utts",
        type=int,
        default=1,
        dest="min_utterances",
        help="Minimum utterances required per speaker.",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress bars and tables.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    run_evaluation(
        dataset_root=args.dataset,
        model_types=args.models,
        output_dir=args.output,
        device=args.device,
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
        linkage_method=args.linkage_method,
        auto_threshold=args.auto_threshold,
        threshold_sweep=args.threshold_sweep,
        sweep_values=args.sweep_values,
        min_utterances=args.min_utterances,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
