"""
example_speechbrain_embedding.py
─────────────────────────────────
Speaker embedding extraction using SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb).
Outputs saved under OUTPUT_DIR
───────────────────────────────
  speakers.json          – per-file metadata + cluster assignment
  clusters.json          – per-cluster summary (members, centroid info)
  embeddings.npy         – raw (N, D) float32 embedding matrix
  similarity_matrix.npy  – raw (N, N) float32 cosine similarity matrix
  similarity_matrix.csv  – human-readable similarity matrix
  report.txt             – full similarity report (matches above threshold)
  run_metadata.json      – model, threshold, timestamp, file count
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
import logging
from custom_speaker_embedding_classes import SpeechBrainPretrainedSpeakerEmbedding

console = Console()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    handler = RichHandler(console=console, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)


def _shorten(path: Path, max_parts: int = 3) -> str:
    """Return a short display name: last *max_parts* components of *path*."""
    parts = path.parts
    if len(parts) <= max_parts:
        return str(path)
    return "…/" + "/".join(parts[-max_parts:])


def link(path: Path, label: Optional[str] = None) -> str:
    """
    Return an OSC 8 terminal hyperlink markup for Rich.
    ``file://`` URIs open the file in the default app on most modern terminals
    (iTerm2, Windows Terminal, Warp, Kitty, VS Code terminal, …).
    Falls back to the short name when the terminal does not support links.
    """
    uri = path.absolute().as_uri()
    display = label or _shorten(path)
    return f"[link={uri}]{display}[/link]"


def load_audio(audio_path: str | Path, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Load an audio file and return a (1, 1, T) mono tensor at *target_sample_rate*.

    Parameters
    ----------
    audio_path:
        Path to a WAV / FLAC / MP3 / OGG file.
    target_sample_rate:
        Target sample rate in Hz (SpeechBrain ECAPA expects 16 000).

    Returns
    -------
    torch.Tensor
        Shape ``(1, 1, num_samples)`` – batch × channels × samples.
    """
    path = Path(audio_path)
    waveform, sample_rate = torchaudio.load(str(path))
    if sample_rate != target_sample_rate:
        log.debug(
            "Resampling %s from %d → %d Hz",
            link(path),
            sample_rate,
            target_sample_rate,
        )
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.unsqueeze(0)


def audio_duration_seconds(waveform: torch.Tensor, sample_rate: int = 16000) -> float:
    """Return duration in seconds from a (1, 1, T) waveform tensor."""
    return waveform.shape[-1] / sample_rate


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the (N, N) pairwise cosine similarity matrix.

    Parameters
    ----------
    embeddings:
        Shape ``(N, D)`` – one row per speaker file.

    Returns
    -------
    np.ndarray
        Shape ``(N, N)`` float32, values in [-1, 1].
    """
    t = torch.from_numpy(embeddings).float()
    normed = F.normalize(t, p=2, dim=1)
    sim = (normed @ normed.T).numpy()
    return sim.astype(np.float32)


def cluster_speakers(
    embeddings: np.ndarray,
    threshold: float = 0.75,
) -> np.ndarray:
    """
    Assign a cluster label to each embedding using agglomerative clustering
    with cosine distance and average linkage.

    Parameters
    ----------
    embeddings:
        Shape ``(N, D)``.
    threshold:
        Cosine-distance cutoff for merging clusters.
        distance = 1 - cosine_similarity, so 0.75 similarity → 0.25 distance.

    Returns
    -------
    np.ndarray
        Integer cluster labels, shape ``(N,)``.
    """
    if embeddings.shape[0] == 1:
        return np.array([0], dtype=int)
    try:
        from sklearn.cluster import AgglomerativeClustering

        distance_threshold = 1.0 - threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels: np.ndarray = clustering.fit_predict(embeddings)
        return labels.astype(int)
    except ImportError:
        log.warning(
            "scikit-learn not installed – skipping clustering. "
            "Install with: pip install scikit-learn"
        )
        return np.zeros(embeddings.shape[0], dtype=int)


def build_speakers_payload(
    paths: list[Path],
    embeddings: np.ndarray,
    labels: np.ndarray,
    durations: list[float],
) -> list[dict]:
    """
    Build a list of per-speaker dicts suitable for ``speakers.json``.

    Each entry contains:
    - ``file``       – absolute path string
    - ``duration_s`` – audio duration in seconds
    - ``cluster``    – integer cluster label
    - ``embedding_preview`` – first 8 dimensions (for quick inspection)
    """
    return [
        {
            "file": str(p.resolve()),
            "duration_s": round(dur, 4),
            "cluster": int(lbl),
            "embedding_preview": emb[:8].tolist(),
        }
        for p, dur, lbl, emb in zip(paths, durations, labels, embeddings)
    ]


def build_clusters_payload(
    paths: list[Path],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    """
    Build a list of per-cluster summary dicts suitable for ``clusters.json``.

    Each entry contains:
    - ``cluster``     – integer cluster label
    - ``num_members`` – how many files belong
    - ``members``     – list of absolute path strings
    - ``centroid_norm`` – L2 norm of the mean embedding (quality indicator)
    """
    unique_labels = sorted(set(labels.tolist()))
    clusters = []
    for lbl in unique_labels:
        member_indices = [i for i, l in enumerate(labels) if l == lbl]
        member_paths = [str(paths[i].resolve()) for i in member_indices]
        member_embs = embeddings[member_indices]
        centroid = member_embs.mean(axis=0)
        clusters.append(
            {
                "cluster": lbl,
                "num_members": len(member_indices),
                "members": member_paths,
                "centroid_norm": float(np.linalg.norm(centroid)),
            }
        )
    return clusters


def build_run_metadata(
    model_name: str,
    threshold: float,
    num_files: int,
    num_clusters: int,
    device: str,
) -> dict:
    return {
        "model": model_name,
        "threshold": threshold,
        "num_files": num_files,
        "num_clusters": num_clusters,
        "device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
    }


def save_results(
    output_dir: Path,
    paths: list[Path],
    embeddings: np.ndarray,
    labels: np.ndarray,
    sim_matrix: np.ndarray,
    durations: list[float],
    threshold: float,
    model_name: str,
    device: str,
) -> None:
    """
    Persist all output artefacts under *output_dir*.

    Files written
    ─────────────
    speakers.json         – per-file metadata
    clusters.json         – per-cluster summary
    embeddings.npy        – raw embedding matrix  (N, D)
    similarity_matrix.npy – raw similarity matrix (N, N)
    similarity_matrix.csv – human-readable similarity matrix
    report.txt            – similarity report (matches above threshold)
    run_metadata.json     – run configuration + library versions
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    num_clusters = len(set(labels.tolist()))

    speakers_path = output_dir / "speakers.json"
    speakers_payload = build_speakers_payload(paths, embeddings, labels, durations)
    speakers_path.write_text(json.dumps(speakers_payload, indent=2))
    log.info("Saved %s", link(speakers_path))

    clusters_path = output_dir / "clusters.json"
    clusters_payload = build_clusters_payload(paths, embeddings, labels)
    clusters_path.write_text(json.dumps(clusters_payload, indent=2))
    log.info("Saved %s", link(clusters_path))

    emb_path = output_dir / "embeddings.npy"
    np.save(str(emb_path), embeddings)
    log.info("Saved %s  shape=%s", link(emb_path), embeddings.shape)

    sim_npy_path = output_dir / "similarity_matrix.npy"
    np.save(str(sim_npy_path), sim_matrix)
    log.info("Saved %s  shape=%s", link(sim_npy_path), sim_matrix.shape)

    sim_csv_path = output_dir / "similarity_matrix.csv"
    short_names = [_shorten(p, max_parts=1) for p in paths]
    with sim_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + short_names)
        for i, row in enumerate(sim_matrix):
            writer.writerow([short_names[i]] + [f"{v:.4f}" for v in row])
    log.info("Saved %s", link(sim_csv_path))

    report_path = output_dir / "report.txt"
    report_lines = _build_report_lines(paths, sim_matrix, labels, threshold)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Saved %s", link(report_path))

    meta_path = output_dir / "run_metadata.json"
    metadata = build_run_metadata(
        model_name=model_name,
        threshold=threshold,
        num_files=len(paths),
        num_clusters=num_clusters,
        device=device,
    )
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s", link(meta_path))


def _build_report_lines(
    paths: list[Path],
    sim_matrix: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> list[str]:
    """Build text report lines (no Rich markup – plain text for file output)."""
    n = len(paths)
    header_width = 72
    lines: list[str] = []
    lines.append("=" * header_width)
    lines.append("  SpeechBrain ECAPA-TDNN Embedding – Similarity Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * header_width)
    lines.append("")
    lines.append("── Files ──────────────────────────────────────────────────")
    for i, p in enumerate(paths):
        lines.append(f"  [{i:>2}] {p}")
    lines.append("")
    lines.append("── Cluster assignments ─────────────────────────────────────")
    for i, (p, lbl) in enumerate(zip(paths, labels)):
        lines.append(f"  [{i:>2}] cluster={lbl}  {p.name}")
    lines.append("")
    lines.append("── Pairwise cosine similarity ──────────────────────────────")
    col_w = 7
    header = " " * 6 + "".join(f"  [{i:>2}] " for i in range(n))
    lines.append(header)
    for i in range(n):
        row = f"[{i:>2}] "
        for j in range(n):
            row += f"  {sim_matrix[i, j]:.3f}"
        row += f"   {paths[i].name}"
        lines.append(row)
    lines.append("")
    matches = [
        (i, j, sim_matrix[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if sim_matrix[i, j] >= threshold
    ]
    lines.append(
        f"── Matches (cosine ≥ {threshold:.2f}) ───────────────────────────────"
    )
    if matches:
        for i, j, score in sorted(matches, key=lambda x: -x[2]):
            lines.append(
                f"  [MATCH]  [{i}] {paths[i].name}"
                f"  ↔  [{j}] {paths[j].name}"
                f"  (score: {score:.4f})"
            )
    else:
        lines.append(f"  No pairs exceed the threshold of {threshold:.2f}")
    lines.append("")
    lines.append("=" * header_width)
    return lines


def print_similarity_table(
    paths: list[Path],
    sim_matrix: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> None:
    """Render a Rich table of pairwise cosine similarities to the console."""
    n = len(paths)
    table = Table(
        title="Pairwise cosine similarity",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("", style="dim", no_wrap=True)
    for i in range(n):
        table.add_column(f"[{i}]", justify="right", no_wrap=True)
    table.add_column("file", style="dim", no_wrap=True)
    for i in range(n):
        cells = []
        for j in range(n):
            score = sim_matrix[i, j]
            if i == j:
                cells.append(f"[dim]{score:.3f}[/dim]")
            elif score >= threshold:
                cells.append(f"[bold green]{score:.3f}[/bold green]")
            else:
                cells.append(f"{score:.3f}")
        table.add_row(f"[{i}]", *cells, _shorten(paths[i], max_parts=1))
    console.print(table)


def print_match_report(
    paths: list[Path],
    sim_matrix: np.ndarray,
    threshold: float,
) -> None:
    """Print matching speaker pairs above the threshold."""
    n = len(paths)
    matches = [
        (i, j, sim_matrix[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if sim_matrix[i, j] >= threshold
    ]
    if not matches:
        console.print(
            f"[dim]  No pairs exceed threshold {threshold:.2f}[/dim]"
        )
        return
    table = Table(
        title=f"Matching pairs  (cosine ≥ {threshold:.2f})",
        show_header=True,
        border_style="dim",
    )
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("File A")
    table.add_column("File B")
    for i, j, score in sorted(matches, key=lambda x: -x[2]):
        table.add_row(
            f"{score:.4f}",
            link(paths[i]),
            link(paths[j]),
        )
    console.print(table)


def print_cluster_summary(
    paths: list[Path],
    labels: np.ndarray,
) -> None:
    """Print a Rich table summarising cluster membership."""
    unique = sorted(set(labels.tolist()))
    table = Table(
        title="Speaker clusters",
        show_header=True,
        border_style="dim",
    )
    table.add_column("Cluster", justify="right")
    table.add_column("Members", justify="right")
    table.add_column("Files")
    for lbl in unique:
        member_paths = [paths[i] for i, l in enumerate(labels) if l == lbl]
        names = ", ".join(link(p) for p in member_paths)
        table.add_row(str(lbl), str(len(member_paths)), names)
    console.print(table)


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_CACHE_DIR = str(
    Path("~/.cache/pretrained_models/spkrec-ecapa-voxceleb").expanduser().resolve()
)
DEFAULT_AUDIO = str(
    Path(
        "~/Desktop/Jet_Files/Jet_Windows_Workspace/python_scripts/samples/audio"
        "/features/generated/speech_waves/waves/"
    )
    .expanduser()
    .resolve()
)
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker embedding extraction and similarity with SpeechBrain ECAPA-TDNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_paths",
        nargs="*",
        default=[DEFAULT_AUDIO],
        help="One or more paths to audio files or directories.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help="Directory where all output files are saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for speaker matching and clustering.",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Skip agglomerative clustering (assign all files to cluster 0).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save any output files (dry run).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Cache directory for downloaded SpeechBrain model files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir).expanduser().resolve()

    console.print(
        Panel.fit(
            f"[bold]SpeechBrain ECAPA-TDNN Embedding[/bold]\n"
            f"model      : [cyan]{MODEL_NAME}[/cyan]\n"
            f"threshold  : [yellow]{args.threshold}[/yellow]\n"
            f"cache dir  : [dim]{args.cache_dir}[/dim]\n"
            f"output dir : {link(output_dir)}",
            border_style="blue",
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: [cyan]%s[/cyan]", device)

    log.info("Loading embedding model…")
    t0 = time.perf_counter()
    embedding_model = SpeechBrainPretrainedSpeakerEmbedding(
        embedding=MODEL_NAME,
        device=device,
        cache_dir=args.cache_dir,
    )
    log.info("Model ready in [green]%.2fs[/green]", time.perf_counter() - t0)

    # Resolve audio paths
    try:
        from audio_utils import resolve_audio_paths
    except ImportError:
        log.warning(
            "audio_utils not found – using built-in path resolver. "
            "Pass explicit audio file paths for best results."
        )
        resolve_audio_paths = _builtin_resolve

    resolved: list[Path] = [
        Path(p) for p in resolve_audio_paths(args.audio_paths, recursive=True)
    ]
    if not resolved:
        log.error("No audio files found. Exiting.")
        sys.exit(1)

    log.info("Found [bold]%d[/bold] audio file(s)", len(resolved))

    # ── Single-file fast path ──────────────────────────────────────────────
    if len(resolved) == 1:
        p = resolved[0]
        log.info("Single-file mode: %s", link(p))
        audio = load_audio(p)
        embedding = embedding_model(audio)
        dur = audio_duration_seconds(audio)
        log.info(
            "Embedding shape: [cyan]%s[/cyan]  duration: [cyan]%.2fs[/cyan]",
            embedding.shape,
            dur,
        )
        console.print(
            f"  First 8 dims: {embedding[0, :8].tolist()}"
        )
        if not args.no_save:
            save_results(
                output_dir=output_dir,
                paths=resolved,
                embeddings=embedding,
                labels=np.array([0]),
                sim_matrix=np.array([[1.0]], dtype=np.float32),
                durations=[dur],
                threshold=args.threshold,
                model_name=MODEL_NAME,
                device=str(device),
            )
        return

    # ── Batch mode ─────────────────────────────────────────────────────────
    log.info("Batch mode: loading and embedding [bold]%d[/bold] files…", len(resolved))

    waveforms: list[torch.Tensor] = []
    durations: list[float] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading audio", total=len(resolved))
        for p in resolved:
            w = load_audio(p).squeeze(0)
            durations.append(audio_duration_seconds(w.unsqueeze(0)))
            waveforms.append(w)
            progress.advance(task)

    # Pad waveforms to equal length and create masks
    max_len = max(w.shape[-1] for w in waveforms)
    padded, masks = [], []
    for w in waveforms:
        pad_len = max_len - w.shape[-1]
        padded.append(F.pad(w, (0, pad_len)))
        mask = torch.zeros(max_len)
        mask[: w.shape[-1]] = 1.0
        masks.append(mask)

    batch_audio = torch.stack(padded)
    batch_masks = torch.stack(masks)

    log.info("Extracting embeddings…")
    t0 = time.perf_counter()
    raw = embedding_model(batch_audio, masks=batch_masks)
    log.info(
        "Embeddings ready in [green]%.2fs[/green]  shape=[cyan]%s[/cyan]",
        time.perf_counter() - t0,
        raw.shape,
    )

    embeddings: np.ndarray = (
        raw if isinstance(raw, np.ndarray) else raw.numpy()
    ).astype(np.float32)

    # Cosine similarity matrix
    log.info("Computing cosine similarity matrix…")
    sim_matrix = cosine_similarity_matrix(embeddings)

    # Clustering
    if args.no_cluster:
        labels = np.zeros(len(resolved), dtype=int)
        log.info("Clustering skipped (--no-cluster flag set)")
    else:
        log.info("Clustering speakers (threshold=%.2f)…", args.threshold)
        labels = cluster_speakers(embeddings, threshold=args.threshold)
        num_clusters = len(set(labels.tolist()))
        log.info("Found [bold]%d[/bold] speaker cluster(s)", num_clusters)

    # Console output
    print_similarity_table(resolved, sim_matrix, labels, args.threshold)
    print_cluster_summary(resolved, labels)
    print_match_report(resolved, sim_matrix, args.threshold)

    # Save results
    if not args.no_save:
        log.info("Saving results to %s", link(output_dir))
        save_results(
            output_dir=output_dir,
            paths=resolved,
            embeddings=embeddings,
            labels=labels,
            sim_matrix=sim_matrix,
            durations=durations,
            threshold=args.threshold,
            model_name=MODEL_NAME,
            device=str(device),
        )
        console.print(
            Panel.fit(
                f"[green bold]✓ All outputs saved[/green bold]\n{link(output_dir)}",
                border_style="green",
            )
        )
    else:
        log.info("--no-save flag set: outputs not written to disk.")


def _builtin_resolve(
    paths: list[str],
    recursive: bool = True,
) -> list[str]:
    """
    Resolve a list of file/directory paths to a flat list of audio file paths.
    Recognised extensions: .wav .flac .mp3 .ogg .m4a .aac .opus
    """
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}
    result: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in audio_exts:
            result.append(str(p))
        elif p.is_dir():
            glob = p.rglob("*") if recursive else p.glob("*")
            result.extend(
                str(f) for f in sorted(glob) if f.suffix.lower() in audio_exts
            )
        else:
            log.warning("Skipping unrecognised path: %s", p)
    return result


if __name__ == "__main__":
    main()
