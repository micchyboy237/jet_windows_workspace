# speaker_diarization_detailed.py
import logging
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Hook: captures intermediate pipeline artifacts
# ─────────────────────────────────────────────

class DiarizationHook:
    """
    Captures all intermediate artifacts emitted by the
    SpeakerDiarization pipeline via its hook mechanism.

    Captured steps:
      - segmentation        : frame-level speaker activity (SlidingWindowFeature)
      - embeddings          : per-segment speaker embeddings (np.ndarray)
      - embeddings/model    : raw embedding model output
      - clustering          : cluster assignments + centroids
      - diarization         : final Annotation
    """

    def __init__(self):
        self.artifacts = {}

    def __call__(self, step_name: str, step_artifact, file=None, completed=None, total=None):
        # Skip noisy progress-only calls
        if completed is not None and step_artifact is None:
            return
        logger.debug(f"[hook] step='{step_name}' artifact_type={type(step_artifact).__name__}")
        self.artifacts[step_name] = step_artifact


# ─────────────────────────────────────────────
# Pairwise metrics between embeddings/centroids
# ─────────────────────────────────────────────

def compute_pairwise_metrics(matrix: np.ndarray, labels: list[str]) -> dict:
    """
    Compute pairwise euclidean distance and cosine similarity for
    a set of vectors (embeddings or centroids).

    Parameters
    ----------
    matrix : (N, D) np.ndarray
        N vectors of dimension D.
    labels : list of str
        Human-readable label for each row (e.g. ["SPEAKER_00", "SPEAKER_01"]).

    Returns
    -------
    dict with:
        - distance_matrix   : (N, N) euclidean distances
        - similarity_matrix : (N, N) cosine similarities  [-1, 1]
        - pairs             : list of per-pair dicts
    """
    dist_matrix = cdist(matrix, matrix, metric="euclidean")
    sim_matrix  = 1 - cdist(matrix, matrix, metric="cosine")  # cosine similarity

    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "a":          labels[i],
                "b":          labels[j],
                "distance":   round(float(dist_matrix[i, j]), 4),
                "similarity": round(float(sim_matrix[i, j]),  4),
            })

    return {
        "labels":            labels,
        "distance_matrix":   dist_matrix,
        "similarity_matrix": sim_matrix,
        "pairs":             pairs,
    }


# ─────────────────────────────────────────────
# Parse clustering artifact
# ─────────────────────────────────────────────

def parse_clustering(clustering_artifact) -> dict | None:
    """
    Extract centroids and cluster assignments from the clustering artifact.

    The SpeakerDiarization pipeline emits a dict at the 'clustering' step:
        {
          "embeddings":   (N, D) array  — per-segment embeddings
          "hard_clusters": (N,)  array  — cluster index per segment
          "soft_clusters": (N, K) array — soft assignment scores (optional)
          "centroids":    (K, D) array  — cluster centroids
        }
    """
    if clustering_artifact is None:
        logger.warning("[clustering] No clustering artifact captured.")
        return None

    if not isinstance(clustering_artifact, dict):
        logger.warning(f"[clustering] Unexpected type: {type(clustering_artifact)}")
        return None

    embeddings    = clustering_artifact.get("embeddings")
    hard_clusters = clustering_artifact.get("hard_clusters")
    soft_clusters = clustering_artifact.get("soft_clusters")
    centroids     = clustering_artifact.get("centroids")

    n_segments  = len(embeddings)    if embeddings    is not None else 0
    n_clusters  = len(centroids)     if centroids     is not None else 0

    logger.info(f"[clustering] {n_segments} segments → {n_clusters} clusters")

    return {
        "embeddings":    embeddings,
        "hard_clusters": hard_clusters,
        "soft_clusters": soft_clusters,
        "centroids":     centroids,
        "n_segments":    n_segments,
        "n_clusters":    n_clusters,
    }


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def get_speakers_detailed(audio_path: str, device: str = None) -> dict:
    """
    Run speaker diarization and return detailed internal metrics.

    Returns
    -------
    dict with:
        - speakers          : list of speaker IDs
        - segments          : [{speaker, start, end, duration, embedding, cluster_id, soft_scores}]
        - speaker_times     : {speaker: total_seconds}
        - centroids         : {speaker: centroid_vector}
        - centroid_metrics  : pairwise distance/similarity between centroids
        - embedding_metrics : pairwise distance/similarity between ALL segment embeddings
        - clustering_raw    : raw clustering dict from pipeline
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load pipeline
    logger.info("Loading pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        # use_auth_token=hf_token,
    ).to(torch.device(device))

    # Attach hook
    hook = DiarizationHook()

    # Run
    logger.info(f"Running diarization on: {audio_path.name}")
    diarization = pipeline(str(audio_path), hook=hook)
    logger.info("Diarization done.")

    # ── Parse diarization segments ──────────────────────────────
    raw_segments = []
    speaker_times = {}
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        raw_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end, "duration": dur})
        speaker_times[speaker] = speaker_times.get(speaker, 0) + dur

    speakers = sorted(speaker_times.keys())
    logger.info(f"Speakers: {speakers}")

    # ── Parse clustering artifact ────────────────────────────────
    clustering_data = parse_clustering(hook.artifacts.get("clustering"))

    # ── Build per-segment enriched info ─────────────────────────
    segments = []
    if clustering_data and clustering_data["embeddings"] is not None:
        embeddings    = clustering_data["embeddings"]     # (N, D)
        hard_clusters = clustering_data["hard_clusters"]  # (N,)
        soft_clusters = clustering_data.get("soft_clusters")  # (N, K) or None

        for i, seg in enumerate(raw_segments):
            entry = {**seg}
            if i < len(embeddings):
                entry["embedding"]   = embeddings[i].tolist()
                entry["cluster_id"]  = int(hard_clusters[i])
                entry["soft_scores"] = soft_clusters[i].tolist() if soft_clusters is not None else None
            segments.append(entry)
    else:
        segments = [{**s} for s in raw_segments]

    # ── Centroid metrics ────────────────────────────────────────
    centroid_metrics   = None
    centroid_map       = {}
    embedding_metrics  = None

    if clustering_data and clustering_data["centroids"] is not None:
        centroids = clustering_data["centroids"]  # (K, D)
        centroid_labels = [f"SPEAKER_{i:02d}" for i in range(len(centroids))]
        centroid_map = {label: centroids[i].tolist() for i, label in enumerate(centroid_labels)}

        logger.info(f"Computing centroid pairwise metrics ({len(centroids)} centroids)...")
        centroid_metrics = compute_pairwise_metrics(centroids, centroid_labels)

    # ── Per-segment embedding pairwise metrics ───────────────────
    if clustering_data and clustering_data["embeddings"] is not None:
        embs = clustering_data["embeddings"]
        seg_labels = [f"seg_{i:03d}({s['speaker']})" for i, s in enumerate(raw_segments)]
        logger.info(f"Computing segment pairwise metrics ({len(embs)} segments)...")
        embedding_metrics = compute_pairwise_metrics(embs, seg_labels)

    return {
        "speakers":         speakers,
        "speaker_times":    {k: round(v, 3) for k, v in speaker_times.items()},
        "segments":         segments,
        "centroids":        centroid_map,
        "centroid_metrics": centroid_metrics,
        "embedding_metrics":embedding_metrics,
        "clustering_raw":   clustering_data,
        "hook_steps":       list(hook.artifacts.keys()),
    }


# ─────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────

def print_detailed_report(result: dict):
    sep = "─" * 55

    print(f"\n{'═'*55}")
    print(f"  SPEAKERS FOUND: {len(result['speakers'])}")
    print(f"{'═'*55}")
    for spk in result["speakers"]:
        print(f"  {spk}: {result['speaker_times'][spk]:.2f}s")

    print(f"\n{sep}\n  PIPELINE STEPS CAPTURED\n{sep}")
    for step in result["hook_steps"]:
        print(f"  • {step}")

    print(f"\n{sep}\n  SEGMENTS (with cluster info)\n{sep}")
    for i, seg in enumerate(result["segments"]):
        cid  = seg.get("cluster_id", "?")
        soft = seg.get("soft_scores")
        soft_str = f"  soft={[round(s,3) for s in soft]}" if soft else ""
        print(f"  [{i:03d}] {seg['speaker']}  cluster={cid}  "
              f"{seg['start']:.2f}s→{seg['end']:.2f}s  ({seg['duration']:.2f}s){soft_str}")

    if result["centroid_metrics"]:
        m = result["centroid_metrics"]
        print(f"\n{sep}\n  CENTROID PAIRWISE METRICS\n{sep}")
        print(f"  {'Pair':<30} {'Distance':>10} {'Similarity':>12}")
        print(f"  {'─'*28} {'─'*10} {'─'*12}")
        for p in m["pairs"]:
            print(f"  {p['a']} ↔ {p['b']:<16} {p['distance']:>10.4f} {p['similarity']:>12.4f}")

        print(f"\n  Distance matrix:")
        labels = m["labels"]
        header = "".join(f"{l:>14}" for l in labels)
        print(f"  {'':14}{header}")
        for i, row in enumerate(m["distance_matrix"]):
            vals = "".join(f"{v:>14.4f}" for v in row)
            print(f"  {labels[i]:14}{vals}")

    if result["embedding_metrics"]:
        m = result["embedding_metrics"]
        print(f"\n{sep}\n  SEGMENT EMBEDDING PAIRWISE METRICS (top 10 most similar)\n{sep}")
        top = sorted(m["pairs"], key=lambda x: x["similarity"], reverse=True)[:10]
        print(f"  {'Pair':<45} {'Distance':>10} {'Similarity':>12}")
        print(f"  {'─'*43} {'─'*10} {'─'*12}")
        for p in top:
            pair = f"{p['a']} ↔ {p['b']}"
            print(f"  {pair:<45} {p['distance']:>10.4f} {p['similarity']:>12.4f}")


if __name__ == "__main__":
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_50_segments\segment_018\sound.wav"

    result = get_speakers_detailed(
        audio_path=audio_path,
        # hf_token="hf_your_token_here",
    )
    print_detailed_report(result)
