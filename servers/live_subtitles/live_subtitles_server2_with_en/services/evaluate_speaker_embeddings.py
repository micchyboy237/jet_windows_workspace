"""
evaluate_speaker_embeddings.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluate and compare speaker embedding models on a custom speaker dataset.

Dataset layout expected:
    dataset_root/
        speaker_A/
            utt_001.wav
            utt_002.wav
        speaker_B/
            ...

Usage:
    python evaluate_speaker_embeddings.py --dataset /path/to/speakers \
        --models pyannote speechbrain_ecapa \
        --output results/

Metrics computed per model:
    • EER            — Equal Error Rate (speaker verification)
    • minDCF         — Minimum Detection Cost Function
    • Intra-sim      — Avg cosine similarity, same speaker
    • Inter-sim      — Avg cosine similarity, different speakers
    • Separation     — Intra-sim minus Inter-sim (discrimination power)
    • Avg embed time — ms per audio file

All embeddings are cached to disk to avoid re-computation across runs.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

# ── Local import: your factory ───────────────────────────────────────────────
from embedding_model_factory import (
    BaseEmbeddingModel,
    EmbeddingModelType,
    create_embedding_model,
    list_available_models,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("SpeakerEval")
console = Console()

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trial:
    """A single verification trial: two audio paths + ground truth label."""
    path_a: Path
    path_b: Path
    label: int          # 1 = same speaker, 0 = different

    def __repr__(self) -> str:
        return f"Trial({self.path_a.name} vs {self.path_b.name}, label={self.label})"


@dataclass
class ModelMetrics:
    """Evaluation metrics for one embedding model."""
    model_type: str
    embedding_dim: int
    eer: float                      # Equal Error Rate [0, 1]
    eer_threshold: float            # Cosine threshold at EER
    min_dcf: float                  # Minimum Detection Cost
    intra_speaker_sim: float        # Avg cosine sim — same speaker
    inter_speaker_sim: float        # Avg cosine sim — different speakers
    separation: float               # intra_sim - inter_sim
    avg_embed_time_ms: float        # Avg embedding extraction time
    n_speakers: int
    n_trials: int
    n_positive_trials: int
    n_negative_trials: int
    per_speaker_intra_sim: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.model_type}] "
            f"EER={self.eer*100:.2f}% | "
            f"minDCF={self.min_dcf:.4f} | "
            f"Intra={self.intra_speaker_sim:.4f} | "
            f"Inter={self.inter_speaker_sim:.4f} | "
            f"Sep={self.separation:.4f} | "
            f"Latency={self.avg_embed_time_ms:.1f}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanning
# ─────────────────────────────────────────────────────────────────────────────

def scan_dataset(dataset_root: Path, min_utterances: int = 2) -> Dict[str, List[Path]]:
    """
    Walk dataset_root, treating each subdirectory as a speaker.
    Returns {speaker_id: [audio_path, ...]} for speakers with >= min_utterances files.

    Args:
        dataset_root:     Root directory. Each sub-folder is one speaker.
        min_utterances:   Drop speakers with fewer clips (can't form a pair).

    Returns:
        Dict mapping speaker name → list of audio Paths.
    """
    log.info(f"Scanning dataset at: [bold]{dataset_root}[/bold]")
    speakers: Dict[str, List[Path]] = defaultdict(list)

    for speaker_dir in sorted(dataset_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        files = [
            f for f in sorted(speaker_dir.rglob("*"))
            if f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        if len(files) >= min_utterances:
            speakers[speaker_dir.name] = files
        else:
            log.warning(
                f"Skipping speaker '{speaker_dir.name}': "
                f"only {len(files)} file(s), need >= {min_utterances}"
            )

    log.info(
        f"Found [green]{len(speakers)}[/green] speakers, "
        f"[green]{sum(len(v) for v in speakers.values())}[/green] total utterances"
    )
    return dict(speakers)


# ─────────────────────────────────────────────────────────────────────────────
# Trial generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_trials(
    speakers: Dict[str, List[Path]],
    max_positive_per_speaker: int = 10,
    neg_pos_ratio: float = 1.0,
    seed: int = 42,
) -> List[Trial]:
    """
    Build a balanced list of (path_a, path_b, label) trial pairs.

    Positive trials: all within-speaker pairs, capped at max_positive_per_speaker.
    Negative trials: random cross-speaker pairs, scaled by neg_pos_ratio.

    Args:
        speakers:                   Speaker → files mapping.
        max_positive_per_speaker:   Cap on same-speaker pairs per speaker.
        neg_pos_ratio:              Negatives per positive trial.
        seed:                       RNG seed for reproducibility.

    Returns:
        Shuffled list of Trial objects.
    """
    rng = random.Random(seed)
    trials: List[Trial] = []

    speaker_ids = list(speakers.keys())
    all_files = [(spk, p) for spk, paths in speakers.items() for p in paths]

    # ── Positive trials (same speaker) ───────────────────────────────────────
    positive: List[Trial] = []
    for spk, files in speakers.items():
        pairs = [
            (files[i], files[j])
            for i in range(len(files))
            for j in range(i + 1, len(files))
        ]
        rng.shuffle(pairs)
        for a, b in pairs[:max_positive_per_speaker]:
            positive.append(Trial(path_a=a, path_b=b, label=1))

    log.info(f"Generated [green]{len(positive)}[/green] positive trials")

    # ── Negative trials (different speakers) ─────────────────────────────────
    n_negative = int(len(positive) * neg_pos_ratio)
    negative: List[Trial] = []
    attempts = 0
    max_attempts = n_negative * 10

    while len(negative) < n_negative and attempts < max_attempts:
        spk_a, spk_b = rng.sample(speaker_ids, 2)
        a = rng.choice(speakers[spk_a])
        b = rng.choice(speakers[spk_b])
        negative.append(Trial(path_a=a, path_b=b, label=0))
        attempts += 1

    log.info(f"Generated [green]{len(negative)}[/green] negative trials")

    trials = positive + negative
    rng.shuffle(trials)
    log.info(f"Total trials: [bold]{len(trials)}[/bold]")
    return trials


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction with caching
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    model: BaseEmbeddingModel,
    audio_paths: List[Path],
    cache_dir: Optional[Path] = None,
) -> Tuple[Dict[Path, np.ndarray], float]:
    """
    Extract embeddings for all unique audio files.
    Caches results to {cache_dir}/{model_type}/{stem}.npy to avoid recomputation.

    Args:
        model:          An instance of BaseEmbeddingModel.
        audio_paths:    List of audio file paths (may contain duplicates).
        cache_dir:      Directory to read/write .npy cache files.

    Returns:
        (embeddings_dict, avg_time_ms)
        embeddings_dict: Path → np.ndarray of shape (1, dim)
        avg_time_ms:     Average time to compute one embedding (excluding cache hits)
    """
    unique_paths = list(dict.fromkeys(audio_paths))   # preserve order, deduplicate
    model_name = model.model_type.value
    embeddings: Dict[Path, np.ndarray] = {}
    timings: List[float] = []

    if cache_dir is not None:
        model_cache = cache_dir / model_name
        model_cache.mkdir(parents=True, exist_ok=True)

    log.info(
        f"[{model_name}] Extracting embeddings for "
        f"{len(unique_paths)} unique files..."
    )

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Embedding [{model_name}]", total=len(unique_paths)
        )

        for path in unique_paths:
            # ── Cache read ────────────────────────────────────────────────
            if cache_dir is not None:
                cache_file = model_cache / f"{path.stem}_{path.parent.name}.npy"
                if cache_file.exists():
                    embeddings[path] = np.load(str(cache_file))
                    progress.advance(task)
                    continue

            # ── Compute embedding ─────────────────────────────────────────
            try:
                t0 = time.perf_counter()
                emb = model(str(path))          # calls BaseEmbeddingModel.__call__
                elapsed_ms = (time.perf_counter() - t0) * 1000
                timings.append(elapsed_ms)

                # Normalise to (1, dim) float32
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                emb = emb.astype(np.float32)

                embeddings[path] = emb

                # ── Cache write ───────────────────────────────────────────
                if cache_dir is not None:
                    np.save(str(cache_file), emb)

            except Exception as exc:
                log.error(f"[{model_name}] Failed on {path.name}: {exc}")
                embeddings[path] = None   # mark as failed; handled in scoring

            progress.advance(task)

    avg_ms = float(np.mean(timings)) if timings else 0.0
    cache_hits = len(unique_paths) - len(timings)
    log.info(
        f"[{model_name}] Done. "
        f"Cache hits: {cache_hits}, Computed: {len(timings)}, "
        f"Avg time: {avg_ms:.1f} ms"
    )
    return embeddings, avg_ms


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity scoring
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors.
    Handles shape (1, dim) or (dim,) inputs.
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_trials(
    trials: List[Trial],
    embeddings: Dict[Path, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarity scores for all trials.

    Args:
        trials:     List of Trial objects.
        embeddings: Path → embedding array mapping.

    Returns:
        (scores, labels) as float32 numpy arrays.
    """
    scores: List[float] = []
    labels: List[int] = []

    skipped = 0
    for trial in trials:
        emb_a = embeddings.get(trial.path_a)
        emb_b = embeddings.get(trial.path_b)

        if emb_a is None or emb_b is None:
            skipped += 1
            continue

        scores.append(cosine_similarity(emb_a, emb_b))
        labels.append(trial.label)

    if skipped:
        log.warning(f"Skipped {skipped} trials due to missing embeddings")

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate and the threshold at which it occurs.

    EER is the point on the ROC curve where FAR == FRR.
    Lower EER → better discrimination.

    Returns:
        (eer, threshold) both as floats in [0, 1].
    """
    # Sort thresholds by score
    thresholds = np.unique(scores)
    min_diff = float("inf")
    eer = 0.5
    best_threshold = 0.0

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    for thresh in thresholds:
        far = float(np.mean(negative_scores >= thresh))   # False Accept Rate
        frr = float(np.mean(positive_scores < thresh))    # False Reject Rate
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            best_threshold = float(thresh)

    return eer, best_threshold


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """
    Compute minimum Detection Cost Function.

    minDCF = min over thresholds of:
        C_miss * P_target * FRR(t) + C_FA * (1 - P_target) * FAR(t)

    Lower minDCF → better. Standard NIST parameters: p_target=0.01.

    Args:
        scores:   Trial scores (cosine similarity).
        labels:   Binary labels (1=target, 0=non-target).
        p_target: Prior probability of target trial.
        c_miss:   Cost of a miss.
        c_fa:     Cost of a false alarm.

    Returns:
        Normalized minDCF (divided by the cost of a trivial system).
    """
    thresholds = np.unique(scores)
    positive = scores[labels == 1]
    negative = scores[labels == 0]

    min_cost = float("inf")
    p_non_target = 1.0 - p_target

    for thresh in thresholds:
        frr = float(np.mean(positive < thresh))
        far = float(np.mean(negative >= thresh))
        cost = c_miss * p_target * frr + c_fa * p_non_target * far
        if cost < min_cost:
            min_cost = cost

    # Normalise by the trivial system cost (always reject or always accept)
    default_cost = min(c_miss * p_target, c_fa * p_non_target)
    return min_cost / default_cost if default_cost > 0 else 0.0


def compute_speaker_similarities(
    speakers: Dict[str, List[Path]],
    embeddings: Dict[Path, np.ndarray],
) -> Tuple[Dict[str, float], float, float]:
    """
    Compute per-speaker intra similarity and overall inter-speaker similarity.

    Args:
        speakers:   Speaker → files mapping.
        embeddings: Path → embedding array mapping.

    Returns:
        (per_speaker_intra, mean_intra, mean_inter)
    """
    # ── Intra-speaker: all pairs within each speaker ──────────────────────────
    per_speaker_intra: Dict[str, float] = {}
    all_intra: List[float] = []

    for spk, files in speakers.items():
        valid_embs = [embeddings[f] for f in files if embeddings.get(f) is not None]
        if len(valid_embs) < 2:
            continue
        pair_sims = [
            cosine_similarity(valid_embs[i], valid_embs[j])
            for i in range(len(valid_embs))
            for j in range(i + 1, len(valid_embs))
        ]
        spk_mean = float(np.mean(pair_sims))
        per_speaker_intra[spk] = spk_mean
        all_intra.extend(pair_sims)

    # ── Inter-speaker: sample cross-speaker pairs ─────────────────────────────
    speaker_ids = list(speakers.keys())
    all_inter: List[float] = []
    rng = random.Random(0)

    n_inter_samples = min(5000, len(all_intra) * 2)
    for _ in range(n_inter_samples):
        spk_a, spk_b = rng.sample(speaker_ids, 2)
        file_a = rng.choice(speakers[spk_a])
        file_b = rng.choice(speakers[spk_b])
        emb_a = embeddings.get(file_a)
        emb_b = embeddings.get(file_b)
        if emb_a is not None and emb_b is not None:
            all_inter.append(cosine_similarity(emb_a, emb_b))

    mean_intra = float(np.mean(all_intra)) if all_intra else 0.0
    mean_inter = float(np.mean(all_inter)) if all_inter else 0.0

    return per_speaker_intra, mean_intra, mean_inter


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_type: str,
    speakers: Dict[str, List[Path]],
    trials: List[Trial],
    cache_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    model_kwargs: Optional[dict] = None,
) -> ModelMetrics:
    """
    Full evaluation pipeline for one embedding model.

    Steps:
        1. Load model via factory.
        2. Extract (cached) embeddings for all unique audio files.
        3. Score all trials using cosine similarity.
        4. Compute EER, minDCF.
        5. Compute intra/inter speaker similarity.

    Args:
        model_type:    One of EmbeddingModelType values.
        speakers:      Speaker → files mapping.
        trials:        Pre-generated trial pairs.
        cache_dir:     Embedding cache root.
        device:        Torch device override.
        model_kwargs:  Extra kwargs forwarded to create_embedding_model().

    Returns:
        Populated ModelMetrics object.
    """
    log.info(f"\n{'─'*60}")
    log.info(f"Evaluating model: [bold yellow]{model_type}[/bold yellow]")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model = create_embedding_model(
        model_type=model_type,
        device=device,
        **(model_kwargs or {}),
    )
    log.info(f"Model loaded: {model}")

    # ── 2. Extract embeddings ─────────────────────────────────────────────────
    all_paths = list({p for t in trials for p in (t.path_a, t.path_b)})
    embeddings, avg_time_ms = extract_embeddings(model, all_paths, cache_dir)

    # ── 3. Score trials ───────────────────────────────────────────────────────
    scores, labels = score_trials(trials, embeddings)
    log.info(f"[{model_type}] Scored {len(scores)} trials")

    # ── 4. EER + minDCF ───────────────────────────────────────────────────────
    eer, eer_thresh = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)
    log.info(f"[{model_type}] EER={eer*100:.2f}% @ threshold={eer_thresh:.4f}")
    log.info(f"[{model_type}] minDCF={min_dcf:.4f}")

    # ── 5. Intra / inter speaker similarity ───────────────────────────────────
    per_spk_intra, intra_sim, inter_sim = compute_speaker_similarities(
        speakers, embeddings
    )
    separation = intra_sim - inter_sim
    log.info(
        f"[{model_type}] Intra={intra_sim:.4f} | "
        f"Inter={inter_sim:.4f} | Sep={separation:.4f}"
    )

    return ModelMetrics(
        model_type=model_type,
        embedding_dim=model.embedding_dim,
        eer=eer,
        eer_threshold=eer_thresh,
        min_dcf=min_dcf,
        intra_speaker_sim=intra_sim,
        inter_speaker_sim=inter_sim,
        separation=separation,
        avg_embed_time_ms=avg_time_ms,
        n_speakers=len(speakers),
        n_trials=len(scores),
        n_positive_trials=int(np.sum(labels == 1)),
        n_negative_trials=int(np.sum(labels == 0)),
        per_speaker_intra_sim=per_spk_intra,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: List[ModelMetrics]) -> None:
    """
    Print a ranked comparison table for all evaluated models.
    Rank order: EER ascending (lower is better).

    Args:
        results: List of ModelMetrics, one per model.
    """
    ranked = sorted(results, key=lambda m: m.eer)

    table = Table(
        title="[bold]Speaker Embedding Model Comparison[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", min_width=20)
    table.add_column("Dim", justify="right")
    table.add_column("EER ↓", justify="right")
    table.add_column("minDCF ↓", justify="right")
    table.add_column("Intra ↑", justify="right")
    table.add_column("Inter ↓", justify="right")
    table.add_column("Sep ↑", justify="right")
    table.add_column("ms/file ↓", justify="right")

    for rank, m in enumerate(ranked, start=1):
        style = "bold green" if rank == 1 else ""
        table.add_row(
            str(rank),
            m.model_type,
            str(m.embedding_dim),
            f"{m.eer*100:.2f}%",
            f"{m.min_dcf:.4f}",
            f"{m.intra_speaker_sim:.4f}",
            f"{m.inter_speaker_sim:.4f}",
            f"{m.separation:.4f}",
            f"{m.avg_embed_time_ms:.1f}",
            style=style,
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]↓ = lower is better   ↑ = higher is better   "
        "Sep = Intra − Inter (discrimination power)[/dim]\n"
    )


def save_results(results: List[ModelMetrics], output_dir: Path) -> None:
    """
    Save all ModelMetrics to JSON and a markdown summary.

    Args:
        results:    List of evaluated model metrics.
        output_dir: Directory to write results into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = output_dir / "results.json"
    data = [asdict(m) for m in results]
    # Convert Path keys in per_speaker_intra_sim to strings (already strings from scan)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Results saved to [green]{json_path}[/green]")

    # ── Markdown summary ──────────────────────────────────────────────────────
    md_path = output_dir / "summary.md"
    ranked = sorted(results, key=lambda m: m.eer)

    lines = [
        "# Speaker Embedding Model Evaluation\n",
        "| Rank | Model | Dim | EER ↓ | minDCF ↓ | Intra ↑ | Inter ↓ | Sep ↑ | ms/file ↓ |",
        "|------|-------|-----|-------|----------|---------|---------|-------|-----------|",
    ]
    for rank, m in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {m.model_type} | {m.embedding_dim} "
            f"| {m.eer*100:.2f}% | {m.min_dcf:.4f} "
            f"| {m.intra_speaker_sim:.4f} | {m.inter_speaker_sim:.4f} "
            f"| {m.separation:.4f} | {m.avg_embed_time_ms:.1f} |"
        )

    lines += [
        "",
        "> ↓ = lower is better | ↑ = higher is better | Sep = Intra − Inter",
        f"\n**Dataset:** {ranked[0].n_speakers} speakers, {ranked[0].n_trials} trials",
    ]

    md_path.write_text("\n".join(lines))
    log.info(f"Summary saved to [green]{md_path}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    dataset_root: Path,
    model_types: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    max_positive_per_speaker: int = 10,
    neg_pos_ratio: float = 1.0,
    min_utterances: int = 2,
    seed: int = 42,
    model_kwargs: Optional[Dict[str, dict]] = None,
) -> List[ModelMetrics]:
    """
    Run the full evaluation pipeline across one or more embedding models.

    Args:
        dataset_root:              Root of the speaker dataset directory.
        model_types:               List of model type strings to evaluate.
                                   Defaults to all registered models.
        output_dir:                Directory to save results JSON + markdown.
        cache_dir:                 Directory to cache embedding .npy files.
        device:                    Torch device (auto-detected if None).
        max_positive_per_speaker:  Max same-speaker trial pairs per speaker.
        neg_pos_ratio:             Ratio of negative to positive trials.
        min_utterances:            Minimum audio files required per speaker.
        seed:                      Random seed for reproducibility.
        model_kwargs:              Per-model kwargs: {"pyannote": {"window": "whole"}, ...}

    Returns:
        List of ModelMetrics, one per model.
    """
    log.info("[bold green]Speaker Embedding Evaluation[/bold green]")
    log.info(f"Dataset: {dataset_root}")

    # ── Resolve models to evaluate ────────────────────────────────────────────
    if model_types is None:
        model_types = [e.value for e in EmbeddingModelType]
    log.info(f"Models to evaluate: {model_types}")

    available = list_available_models()
    for m in model_types:
        if m not in available:
            raise ValueError(
                f"Unknown model '{m}'. Available: {list(available.keys())}"
            )

    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: [bold]{device}[/bold]")

    # ── Scan dataset ──────────────────────────────────────────────────────────
    speakers = scan_dataset(dataset_root, min_utterances=min_utterances)
    if len(speakers) < 2:
        raise ValueError(
            f"Need at least 2 speakers. Found: {len(speakers)}"
        )

    # ── Generate trials once (shared across all models) ───────────────────────
    trials = generate_trials(
        speakers,
        max_positive_per_speaker=max_positive_per_speaker,
        neg_pos_ratio=neg_pos_ratio,
        seed=seed,
    )

    # ── Evaluate each model ───────────────────────────────────────────────────
    results: List[ModelMetrics] = []
    model_kwargs = model_kwargs or {}

    for model_type in model_types:
        try:
            metrics = evaluate_model(
                model_type=model_type,
                speakers=speakers,
                trials=trials,
                cache_dir=cache_dir,
                device=device,
                model_kwargs=model_kwargs.get(model_type),
            )
            results.append(metrics)
            log.info(metrics.summary())
        except Exception as exc:
            log.error(f"[red]Failed to evaluate '{model_type}': {exc}[/red]")
            import traceback
            traceback.print_exc()

    # ── Print comparison table ────────────────────────────────────────────────
    if results:
        compare_models(results)

    # ── Save results ──────────────────────────────────────────────────────────
    if output_dir and results:
        save_results(results, output_dir)

    return results


if __name__ == "__main__":
    from main._main_evaluate_speaker_embeddings import _parse_args

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
