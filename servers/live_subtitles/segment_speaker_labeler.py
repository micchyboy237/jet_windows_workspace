# segment_speaker_labeler.py
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import List, Literal, TypedDict, Union

import numpy as np
import torch
import torchaudio
from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.clustering import (
    AgglomerativeClustering as PyannoteAgglomerativeClustering,
)
from tqdm import tqdm

AudioInput = Union[np.ndarray, bytes, bytearray, io.BytesIO, str, Path]


class SegmentResult(TypedDict):
    path: str
    parent_dir: str
    speaker_label: int
    centroid_cosine_similarity: float
    nearest_neighbor_cosine_similarity: float


class SegmentSpeakerLabeler:
    """
    A reusable class for clustering short speech segments using pyannote speaker embeddings
    and pyannote's clustering implementations (Agglomerative).

    Designed for cases where each segment is assumed to contain a single speaker
    (e.g., extracted speech clips named 'sound.wav' in subdirectories).

    Features:
    - Configurable embedding model
    - Progress bars via tqdm
    - Normalized embeddings for cosine similarity
    - Returns structured results with speaker labels and similarities
    - Generic and reusable – no hardcoded paths or business logic
    """

    def __init__(
        self,
        embedding_model: str = "pyannote/embedding",
        hf_token: str | None = None,
        distance_threshold: float = 0.7,
        clustering_method: Literal["average", "complete", "single"] = "average",
        min_cluster_size: int = 1,
        use_accelerator: bool = True,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the clusterer.

        Parameters
        ----------
        embedding_model : str, optional
            Hugging Face model name for speaker embedding.
        hf_token : str | None, optional
            Hugging Face authentication token (required for gated models).
        distance_threshold : float, optional
            Distance threshold for agglomerative clustering (lower → more clusters).
        clustering_method : Literal["average", "complete", "single"], optional
            Linkage method for agglomerative clustering. Defaults to "average".
        min_cluster_size : int, optional
            Minimum number of segments in a cluster.
        use_accelerator : bool, optional
            Use MPS (Apple), CUDA or CPU as available.
        device : str | torch.device | None, optional
            Overrides auto-detection if provided.
        verbose : bool, optional
            If True, enables verbose output for debugging.
        """

        self.verbose = verbose

        # ── Prefer provided device; otherwise, auto-select MPS, CUDA, then CPU ──────
        if device is not None:
            self.device = torch.device(device)
            device_str = str(self.device)
        else:
            if torch.backends.mps.is_available():
                device_str = "mps"
            elif use_accelerator and torch.cuda.is_available():
                device_str = "cuda"
            else:
                device_str = "cpu"
            self.device = torch.device(device_str)

        if self.verbose:
            print(
                f"Using device: {self.device} ({'MPS acceleration' if device_str == 'mps' else 'CUDA' if device_str == 'cuda' else 'CPU'})"
            )

        # ── Use HF_TOKEN from environment if not supplied ─────────────────────────
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN")

        # ── Suppress safetensors "open file:" prints during model load ─────────
        @contextlib.contextmanager
        def suppress_safetensors_output():
            with contextlib.redirect_stdout(io.StringIO()):
                yield

        with suppress_safetensors_output():
            self.model = Model.from_pretrained(
                embedding_model,
                use_auth_token=hf_token,
                map_location=self.device,
                strict=False,
            )

        with suppress_safetensors_output():
            self.inference = Inference(
                model=self.model,
                duration=3.0,
                step=0.5,
                window="sliding",
            )

        self.inference.to(self.device)

        # ────────────── Clustering config ──────────────
        self.distance_threshold = distance_threshold
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size

        # Instantiate the agglomerative clusterer
        self._clusterer = PyannoteAgglomerativeClustering(metric="cosine").instantiate(
            {
                "threshold": self.distance_threshold,
                "method": self.clustering_method,
                "min_cluster_size": self.min_cluster_size,
            }
        )

    def _load_audio_dict(self, path: str):
        waveform, sample_rate = torchaudio.load(path)
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    def _extract_embeddings(self, segments: List[AudioInput]) -> np.ndarray:
        """Extract and L2-normalize speaker embeddings with progress bar."""
        embeddings: List[np.ndarray] = []

        segments_list = (
            tqdm(segments, desc="Extracting embeddings") if self.verbose else segments
        )
        for item in segments_list:
            if isinstance(item, (str, Path)):
                audio = self._load_audio_dict(str(item))
            else:
                audio = self._normalize_audio_input(item)

            result = self.inference(audio)

            # When using sliding window, result is SlidingWindowFeature
            if hasattr(result, "data"):
                emb_array = result.data
                if emb_array.shape[0] == 0:
                    raise ValueError(
                        "No windows extracted for very short/empty segment"
                    )
                emb = np.mean(emb_array, axis=0)
            else:
                emb = result
                if emb.ndim == 2:
                    emb = emb.squeeze(0)
                elif emb.ndim > 2:
                    raise ValueError(f"Unexpected embedding shape {emb.shape}")

            norm = np.linalg.norm(emb)
            if norm == 0:
                raise ValueError("Zero-norm embedding – silent or invalid audio?")
            emb = emb / norm
            embeddings.append(emb)

        return np.stack(embeddings)

    def _normalize_audio_input(self, item: AudioInput):
        """Convert any AudioInput to format accepted by pyannote.audio.Inference"""

        if isinstance(item, (str, Path)):
            return str(Path(item))

        if isinstance(item, io.BytesIO):
            item.seek(0)
            return item

        if isinstance(item, (bytes, bytearray)):
            return io.BytesIO(bytes(item))

        if isinstance(item, np.ndarray):
            if item.ndim not in (1, 2):
                raise ValueError(f"Expected 1D or 2D waveform, got shape {item.shape}")

            # channels-first expected by pyannote: (channels, time)
            if item.ndim == 1:
                waveform = torch.from_numpy(item).unsqueeze(0)
            else:
                waveform = torch.from_numpy(item.T)

            if waveform.dtype != torch.float32:
                waveform = waveform.float()

            return {
                "waveform": waveform,
                "sample_rate": 16000,
            }

        raise TypeError(f"Unsupported audio input type: {type(item).__name__}")

    def _nearest_neighbor_similarity(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute nearest-neighbor cosine similarity for each embedding.
        Assumes embeddings are L2-normalized.
        """
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, -1.0)
        return sim_matrix.max(axis=1)

    def similarity(
        self,
        audio_a: AudioInput,
        audio_b: AudioInput,
    ) -> float:
        """
        Compute cosine similarity between two audio inputs using speaker embeddings.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        embeddings = self._extract_embeddings([audio_a, audio_b])
        return float(np.dot(embeddings[0], embeddings[1]))

    def is_same_speaker(
        self,
        audio_a: AudioInput,
        audio_b: AudioInput,
    ) -> bool:
        """
        Determine whether two audio inputs are likely from the same speaker.

        This method intentionally reuses `cluster_segments` to stay consistent
        with clustering-time similarity semantics.

        Returns
        -------
        bool
            True if both inputs are assigned the same speaker label AND
            the similarity evidence is defined (non-singleton cluster).
        """
        results = self.cluster_segments([audio_a, audio_b])

        if len(results) != 2:
            return False

        a, b = results

        # Different cluster → definitely different speakers
        if a["speaker_label"] != b["speaker_label"]:
            return False

        # Singleton / undefined similarity is represented as 0.0 (nan becomes float nan)
        if (
            a["centroid_cosine_similarity"] <= 0.0
            or b["centroid_cosine_similarity"] <= 0.0
        ):
            return False

        return True

    def cluster_segments(
        self,
        segments: AudioInput | List[AudioInput],
    ) -> List[SegmentResult]:
        """
        Cluster speaker embeddings from provided audio segments.

        Parameters
        ----------
        segments : AudioInput | List[AudioInput]
            Single audio item or list of items. Each item can be:
            • np.ndarray     → mono float32 waveform @ 16 kHz
            • bytes          → raw encoded audio bytes
            • io.BytesIO     → buffer with encoded audio
            • str / Path     → path to audio file
        """
        if not segments:
            raise ValueError("No segment(s) provided to cluster_segments.")

        # Normalize to list
        segment_list: List[AudioInput] = (
            [segments] if not isinstance(segments, list) else segments
        )

        if not segment_list:
            raise ValueError("Empty segment list provided.")

        if self.verbose:
            print(f"Found {len(segment_list)} segment(s). Clustering embeddings...")

        embeddings = self._extract_embeddings(segment_list)

        # Agglomerative clustering (only strategy left)
        labels = self._clusterer.cluster(
            embeddings,
            min_clusters=1,
            max_clusters=9999,
        )

        # Majority-size remapping (largest cluster → label 0, etc.)
        unique_labels = np.unique(labels)
        cluster_sizes_list = [(l, np.sum(labels == l)) for l in unique_labels]
        cluster_sizes_list.sort(key=lambda x: -x[1])
        old_label_to_priority = {
            old: idx for idx, (old, _) in enumerate(cluster_sizes_list)
        }
        labels = np.array([old_label_to_priority[l] for l in labels])
        unique_labels = np.unique(labels)

        # ── Compute centroids and nearest-neighbor similarities ──
        cluster_centroids: dict[int, np.ndarray] = {}
        cluster_sizes: dict[int, int] = {}

        for l in np.unique(labels):
            idx = labels == l
            size = int(np.sum(idx))
            cluster_sizes[int(l)] = size
            if size == 0:
                continue
            centroid = embeddings[idx].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            cluster_centroids[int(l)] = centroid

        nearest_neighbor_sim = np.zeros(len(embeddings), dtype=np.float32)
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if len(idx) <= 1:
                nearest_neighbor_sim[idx] = np.nan
                continue
            cluster_embs = embeddings[idx]
            nn_sim = self._nearest_neighbor_similarity(cluster_embs)
            nearest_neighbor_sim[idx] = nn_sim

        results: List[SegmentResult] = []

        for i, (item, label) in enumerate(zip(segment_list, labels)):
            if isinstance(item, (str, Path)):
                path_obj = Path(item)
                path_str = str(path_obj)
                parent_dir = path_obj.parent.name
            else:
                path_str = ""
                parent_dir = ""

            cluster_size = cluster_sizes.get(int(label), 0)
            if cluster_size <= 1:
                centroid_sim = np.nan
            else:
                centroid_sim = float(
                    np.dot(embeddings[i], cluster_centroids[int(label)])
                )

            results.append(
                {
                    "path": path_str,
                    "parent_dir": parent_dir,
                    "speaker_label": int(label),
                    "centroid_cosine_similarity": centroid_sim,
                    "nearest_neighbor_cosine_similarity": float(
                        nearest_neighbor_sim[i]
                    ),
                }
            )

        if self.verbose:
            print(f"Processing complete → {len(np.unique(labels))} speakers detected.")
        return results


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Cluster speech segments and compare speaker similarity."
    )
    parser.add_argument(
        "audio_paths",
        nargs="+",
        help="Audio file paths to cluster and compare (at least two required for similarity checks).",
    )
    args = parser.parse_args()

    if len(args.audio_paths) < 2:
        print(
            "Please provide at least two audio paths as positional arguments.",
            file=sys.stderr,
        )
        sys.exit(1)

    labeler = SegmentSpeakerLabeler()

    cluster_results = labeler.cluster_segments(args.audio_paths)
    similarity = labeler.similarity(args.audio_paths[0], args.audio_paths[1])
    same_speaker = labeler.is_same_speaker(args.audio_paths[0], args.audio_paths[1])

    print(f"Clusters:\n{json.dumps(cluster_results, indent=2, ensure_ascii=False)}")
    print(f"Similarity between first two: {similarity:.4f}")
    print(f"Same speaker (first two): {same_speaker}")
