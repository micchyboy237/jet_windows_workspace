# segment_speaker_labeler.py
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import List, Literal, Tuple, TypedDict, Union

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.clustering import AgglomerativeClustering as PyannoteAgglomerativeClustering
from pyannote.audio.pipelines.clustering import KMeansClustering as PyannoteKMeansClustering
from tqdm import tqdm

from vad_firered import extract_speech_segments, SpeechSegment, generate_plot, frames_from_seconds

AudioInput = Union[np.ndarray, bytes, bytearray, io.BytesIO, str, Path]


class SegmentResult(TypedDict):
    start_sec: float
    end_sec: float
    path: str
    parent_dir: str
    speaker_label: int
    centroid_cosine_similarity: float
    nearest_neighbor_cosine_similarity: float


class SegmentSpeakerLabeler:
    """
    A reusable class for clustering short speech segments using pyannote speaker embeddings
    and pyannote's clustering implementations (Agglomerative or KMeans).

    Designed for cases where each segment is assumed to contain a single speaker
    (e.g., extracted speech clips named 'sound.wav' in subdirectories).

    Features:
    - Configurable embedding model and clustering strategy
    - Progress bars via tqdm
    - Normalized embeddings for cosine similarity
    - Returns structured results with speaker labels and min similarity to centroid
    - Generic and reusable – no hardcoded paths or business logic
    """

    def __init__(
        self,
        embedding_model: str = "pyannote/embedding",
        hf_token: str | None = None,
        distance_threshold: float = 0.7,
        clustering_strategy: Literal["agglomerative", "kmeans"] = "agglomerative",
        n_clusters: int | None = None,
        clustering_method: Literal["average", "complete", "single"] = "average",
        min_cluster_size: int = 1,
        # ── New parameters for reference-based assignment ─────────────────────
        reference_embeddings_by_speaker: dict[int | str, np.ndarray] | None = None,
        reference_paths_by_speaker: dict[int | str, list[str | Path]] | None = None,
        use_references: bool = False,
        assignment_threshold: float = 0.68,
        assignment_strategy: Literal["centroid", "max"] = "centroid",
        # ──────────────────────────────────────────────────────────────────────
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
        clustering_strategy : Literal["agglomerative", "kmeans"], optional
            Clustering algorithm to use. Defaults to "agglomerative".
        n_clusters : int | None, optional
            Required when using "kmeans". Ignored for "agglomerative".
        clustering_method : Literal["average", "complete", "single"], optional
            Linkage method for agglomerative clustering. Defaults to "average".
        min_cluster_size : int, optional
            Minimum number of segments in a cluster (agglomerative only).
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
            print(f"Using device: {self.device} ({'MPS acceleration' if device_str == 'mps' else 'CUDA' if device_str == 'cuda' else 'CPU'})")

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
                map_location=self.device,   # ← load directly to target device
                strict=False,
            )

        with suppress_safetensors_output():
            self.inference = Inference(
                model=self.model,
                duration=3.0,      # seconds
                step=0.5,          # seconds – controls overlap, smaller = more robust
                window="sliding",  # explicit for clarity
            )

        # No need for .to() anymore — model already on correct device
        # But keep this line harmless if Inference needs explicit move in future
        self.inference.to(self.device)

        self.reference_embeddings_by_speaker = reference_embeddings_by_speaker or {}
        self.reference_centroids: dict[int | str, np.ndarray] = {}
        self.use_references = use_references
        self.assignment_threshold = assignment_threshold
        self.assignment_strategy = assignment_strategy

        if reference_paths_by_speaker:
            self._load_references_from_paths(reference_paths_by_speaker)

        # ────────────── Clustering config: store and validate, instantiate clusterer ──────────────
        self.distance_threshold = distance_threshold
        self.clustering_strategy = clustering_strategy
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size

        # Validate early (raise if config invalid)
        if self.clustering_strategy == "kmeans" and self.n_clusters is None:
            raise ValueError("n_clusters must be provided for kmeans strategy.")

        if self.clustering_strategy == "agglomerative" and self.n_clusters is not None:
            raise ValueError("n_clusters cannot be used with agglomerative strategy.")

        # Instantiate the clusterer once per instance, based on config
        self._clusterer = None

        if self.clustering_strategy == "agglomerative":
            self._clusterer = (
                PyannoteAgglomerativeClustering(metric="cosine")
                .instantiate(
                    {
                        "threshold": self.distance_threshold,
                        "method": self.clustering_method,
                        "min_cluster_size": self.min_cluster_size,
                    }
                )
            )

        elif self.clustering_strategy == "kmeans":
            self._clusterer = (
                PyannoteKMeansClustering(metric="cosine")
                .instantiate({})
            )

    def _load_references_from_paths(
        self,
        reference_paths_by_speaker: dict[int | str, list[str | Path]],
    ) -> None:
        """Pre-compute embeddings and centroids from reference audio paths."""
        for speaker_id, paths in reference_paths_by_speaker.items():
            path_objs = [Path(p) for p in paths]
            embs = self._extract_embeddings(path_objs)
            if len(embs) == 0:
                continue
            if self.assignment_strategy == "centroid":
                centroid = embs.mean(axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-12
                self.reference_centroids[speaker_id] = centroid
            else:
                # For "max" strategy: store all embeddings
                self.reference_embeddings_by_speaker[speaker_id] = embs

    def _assign_to_references(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign each embedding to the closest reference speaker or new label."""
        labels = np.full(len(embeddings), -1, dtype=int)
        next_new_label = max(self.reference_centroids.keys(), default=-1)
        if isinstance(next_new_label, (str, np.generic)):
            # Only int ids can serve as a starting max for new labels.
            next_new_label = -1
        next_new_label = next_new_label + 1 if next_new_label >= 0 else 0

        for i, emb in enumerate(embeddings):
            best_sim = -1.0
            best_speaker = -1

            for spk_id, ref in self.reference_centroids.items():
                sim = float(np.dot(emb, ref))
                if sim > best_sim:
                    best_sim = sim
                    best_speaker = spk_id

            if best_sim >= self.assignment_threshold:
                labels[i] = best_speaker
            else:
                labels[i] = next_new_label
                next_new_label += 1

        return labels

    def _load_audio_dict(self, path: str):
        waveform, sample_rate = torchaudio.load(path)
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    def _extract_embeddings(
        self,
        segments: List[AudioInput]
    ) -> np.ndarray:
        """Extract and L2-normalize speaker embeddings with progress bar."""
        embeddings: List[np.ndarray] = []

        segments_list = tqdm(segments, desc="Extracting embeddings") if self.verbose else segments
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
                    raise ValueError("No windows extracted for very short/empty segment")
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
                "sample_rate": 16000,  # ⚠️ MUST be correct
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

        # Singleton / undefined similarity is represented as 0.0
        if (
            a["centroid_cosine_similarity"] <= 0.0
            or b["centroid_cosine_similarity"] <= 0.0
        ):
            return False

        return True

    def prepare_speech_segments(
        self,
        audio: AudioInput,
        min_duration_sec: float = 0.4,
    ) -> List[Tuple[SpeechSegment, np.ndarray]]:
        """
        Use FireRedVAD to split input audio into speech segments.
        Filters out very short segments that are usually useless for speaker ID.
        """
        segments_with_meta, _ = extract_speech_segments(audio=audio)

        # Filter very short / useless segments
        filtered = [
            (meta, waveform)
            for meta, waveform in segments_with_meta
            if meta["duration_sec"] >= min_duration_sec
        ]

        if self.verbose and len(filtered) < len(segments_with_meta):
            print(f"Filtered out {len(segments_with_meta)-len(filtered)} too-short segments")

        return filtered

    def cluster_segments(
        self,
        segments: AudioInput | List[AudioInput],
        use_vad: bool = True,
        min_segment_duration_sec: float = 0.45,
    ) -> List[SegmentResult]:
        """
        Main entry point — now supports two modes:
          1. use_vad=True  → run VAD first, then cluster short speech segments
          2. use_vad=False → old behavior (treat each input as one segment)
        """
        if not segments:
            raise ValueError("No segment(s) provided.")

        if use_vad:
            # ──────────────── VAD mode ────────────────
            if isinstance(segments, list) and len(segments) > 1:
                raise ValueError(
                    "When use_vad=True, only one long audio file/buffer is supported."
                )
            if self.verbose:
                print("Running Voice Activity Detection → extracting speech segments...")
            speech_segments = self.prepare_speech_segments(
                audio=segments,
                min_duration_sec=min_segment_duration_sec,
            )
            if not speech_segments:
                print("[yellow]No usable speech segments after VAD filtering.[/yellow]")
                return []
            # Prepare list of waveforms + remember metadata
            waveforms = [wav for _, wav in speech_segments]
            metadata_list = [meta for meta, _ in speech_segments]
            segment_list_for_embedding = waveforms   # np.ndarray waveforms
        else:
            # ──────────────── Legacy / manual segments mode ────────────────
            segment_list_for_embedding: List[AudioInput] = (
                [segments] if not isinstance(segments, list) else segments
            )
            metadata_list = [None] * len(segment_list_for_embedding)

        # ────────────────────────────────────────────────
        #          Extract embeddings (common part)
        # ────────────────────────────────────────────────

        # Remember source file path for display when using VAD
        source_file_path = str(Path(segments).resolve()) if isinstance(segments, (str, Path)) else "[memory]"

        if self.verbose:
            print(f"Extracting embeddings from {len(segment_list_for_embedding)} segment(s)...")

        embeddings = self._extract_embeddings(segment_list_for_embedding)

        has_references = bool(self.reference_centroids or self.reference_embeddings_by_speaker)

        if self.use_references and has_references:
            if self.verbose:
                print(f"Found {len(segment_list_for_embedding)} segment(s). Assigning using provided references...")
            labels = self._assign_to_references(embeddings)
            unique_labels = np.unique(labels)
            if self.verbose:
                print(f"Assignment complete → {len(unique_labels)} speakers detected (including possible new).")
        else:
            if self.use_references and not has_references and self.verbose:
                print("Warning: use_references=True but no references provided → falling back to unsupervised clustering")
            if self.verbose:
                print(f"Found {len(segment_list_for_embedding)} segment(s). Clustering embeddings...")
            # Reuse per-instance clusterer to avoid repeated object construction
            if self.clustering_strategy == "agglomerative":
                labels = self._clusterer.cluster(
                    embeddings,
                    min_clusters=1,
                    max_clusters=9999,
                )
            elif self.clustering_strategy == "kmeans":
                labels = self._clusterer.cluster(
                    embeddings,
                    num_clusters=self.n_clusters,
                )
            else:
                raise ValueError(f"Unsupported clustering_strategy: {self.clustering_strategy}")

            # majority-size remapping (only in unsupervised mode)
            unique_labels = np.unique(labels)
            cluster_sizes_list = [(l, np.sum(labels == l)) for l in unique_labels]
            cluster_sizes_list.sort(key=lambda x: -x[1])
            old_label_to_priority = {old: idx for idx, (old, _) in enumerate(cluster_sizes_list)}
            labels = np.array([old_label_to_priority[l] for l in labels])
            unique_labels = np.unique(labels)

        # ─── compute cluster centroids & similarities ───
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
        for i, (item, label) in enumerate(zip(segment_list_for_embedding, labels)):
            # ── Populate path for table display ──
            if use_vad:
                path_str = source_file_path
                parent_dir = ""
            else:
                if isinstance(item, (str, Path)):
                    path_obj = Path(item)
                    path_str = str(path_obj)
                    parent_dir = path_obj.parent.name
                else:
                    path_str = ""
                    parent_dir = ""

            # VAD mode: we have timing information
            if use_vad and metadata_list[i] is not None:
                meta = metadata_list[i]
                start_sec = meta["start_sec"]
                end_sec   = meta["end_sec"]
            else:
                start_sec = None
                end_sec   = None

            cluster_size = cluster_sizes.get(int(label), 0)
            if cluster_size <= 1:
                centroid_sim = np.nan
            else:
                centroid_sim = float(np.dot(embeddings[i], cluster_centroids[int(label)]))

            result_dict = {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "path": path_str,              # now contains filename when using VAD
                "parent_dir": parent_dir,
                "speaker_label": int(label),
                "centroid_cosine_similarity": centroid_sim,
                "nearest_neighbor_cosine_similarity": float(nearest_neighbor_sim[i]),
            }

            results.append(result_dict)

        if self.verbose:
            print(f"Processing complete → {len(np.unique(labels))} speakers detected.")
        return results

    def save_segments(
        self,
        audio: AudioInput,
        results: List[SegmentResult],
        output_base_dir: Path,
    ) -> List[dict]:
        """
        Save VAD segments + speaker labels with FULL parity to vad_firered:
        Includes:
        - sound.wav
        - meta.json (with speaker info)
        - speech_probs.json
        - speech_probs.png
        - all_speech_segments.json
        - all_speech_segments.png (timeline plot)
        """
        output_base_dir.mkdir(parents=True, exist_ok=True)
        segments_dir = output_base_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # 🔥 Reuse VAD outputs (critical improvement)
        segments_with_meta, full_probs = extract_speech_segments(audio=audio)

        if not segments_with_meta:
            return []

        saved_metadata: List[dict] = []

        for (meta, audio_np), res in zip(segments_with_meta, results):
            idx = meta["segment_index"]

            seg_dir = segments_dir / f"segment_{idx:03d}"
            seg_dir.mkdir(exist_ok=True)

            wav_path = seg_dir / "sound.wav"
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

            torchaudio.save(
                str(wav_path),
                audio_tensor,
                16000,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            # 🔥 enrich original meta instead of recreating
            enriched_meta = {
                **meta,
                "output_path": str(wav_path.relative_to(output_base_dir)),
                "speaker_label": res["speaker_label"],
                "centroid_cosine_similarity": res["centroid_cosine_similarity"],
                "nearest_neighbor_cosine_similarity": res["nearest_neighbor_cosine_similarity"],
            }

            with open(seg_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(enriched_meta, f, indent=2, ensure_ascii=False)

            # ── speech_probs (same as vad_firered) ─────────────────────
            num_frames = meta["probs_info"].get("num_frames", 0)

            if num_frames > 0:
                start_frame = frames_from_seconds(meta["start_sec"])
                end_frame = frames_from_seconds(meta["end_sec"])

                if (
                    full_probs is not None
                    and start_frame < len(full_probs)
                    and end_frame <= len(full_probs)
                ):
                    segment_probs = full_probs[start_frame:end_frame]
                    is_dummy = False
                else:
                    t = np.linspace(0, 1, num_frames)
                    base = 0.12 + 0.76 / (1 + np.exp(-14 * (t - 0.48)))
                    noise = np.random.normal(0, 0.035, num_frames)
                    segment_probs = np.clip(base + noise, 0.03, 0.99)
                    is_dummy = True

                with open(seg_dir / "speech_probs.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "probs": segment_probs.tolist(),
                            "frame_shift_sec": 0.010,
                            "start_frame_global": start_frame,
                            "summary": meta["probs_info"],
                            "is_dummy": is_dummy,
                        },
                        f,
                        indent=2,
                    )

                plot_path = seg_dir / "speech_probs.png"
                generate_plot(
                    probs=segment_probs,
                    segment_idx=idx,
                    duration_sec=meta["duration_sec"],
                    output_path=plot_path,
                    is_dummy=is_dummy,
                )

            saved_metadata.append(enriched_meta)

        # Save combined JSON
        all_json_path = output_base_dir / "all_speech_segments.json"
        with open(all_json_path, "w", encoding="utf-8") as f:
            json.dump(saved_metadata, f, indent=2, ensure_ascii=False)

        # Generate timeline plot
        if saved_metadata:
            fig, ax = plt.subplots(figsize=(10, 2.5), dpi=140)

            for meta in saved_metadata:
                start = meta["start_sec"]
                end = meta["end_sec"]
                speaker = meta["speaker_label"]

                ax.barh(
                    y=speaker,
                    width=(end - start),
                    left=start,
                    height=0.6,
                )

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Speaker")
            ax.set_title("Speech Segments by Speaker")
            ax.grid(True, linestyle="--", alpha=0.3)

            plot_path = output_base_dir / "all_speech_segments.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=140)
            plt.close(fig)

        return saved_metadata


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    import numpy as np

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    console = Console()
    default_audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

    parser = argparse.ArgumentParser(description="Cluster short speech segments and assign speaker labels.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=default_audio_path,
        help="Path to audio file or directory (default: %(default)s)",
    )
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD → treat input as single segment (old behavior)")
    parser.add_argument("--min-seg-sec", type=float, default=0.45, help="Minimum speech segment length after VAD")
    args = parser.parse_args()

    audio_path = args.audio_path
    use_vad = not args.no_vad

    console.rule("Speaker Labeler Demo", style="bold cyan")
    labeler = SegmentSpeakerLabeler(verbose=True)

    with console.status("[bold blue]Extracting embeddings & clustering...", spinner="arc"):
        cluster_results = labeler.cluster_segments(
            audio_path,
            use_vad=use_vad,
            min_segment_duration_sec=args.min_seg_sec,
        )

    # Save segments with speaker labels (only works with VAD)
    # Save segments (creates segment_XXX/sound.wav files)
    if use_vad and cluster_results:
        console.print("\n[bold cyan]Saving segments with speaker labels...[/bold cyan]")

        saved_metadata = labeler.save_segments(
            audio=audio_path,
            results=cluster_results,
            output_base_dir=OUTPUT_DIR,
        )
    else:
        saved_metadata = []

    # ── Results Table ────────────────────────────────────────────────────────────────
    table = Table(title="Clustering Results", show_header=True, header_style="bold magenta")

    table.add_column("Start–End (s)",   justify="right")
    table.add_column("Speaker",         justify="center")
    table.add_column("Centroid Sim",    justify="right")
    table.add_column("NN Sim",          justify="right")
    table.add_column("Segment WAV",     style="cyan", no_wrap=True)

    # We prefer using the actual saved path when available
    segment_dirs_by_index = {
        meta["segment_index"]: OUTPUT_DIR / "segments" / f"segment_{meta['segment_index']:03d}" / "sound.wav"
        for meta in saved_metadata
    }

    for i, res in enumerate(cluster_results):
        if res.get("start_sec") is not None:
            time_str = f"{res['start_sec']:5.1f} – {res['end_sec']:5.1f}"
        else:
            time_str = "—"

        cent_sim = (
            f"{res['centroid_cosine_similarity']:.3f}"
            if not np.isnan(res['centroid_cosine_similarity'])
            else "—"
        )
        nn_sim = (
            f"{res['nearest_neighbor_cosine_similarity']:.3f}"
            if not np.isnan(res['nearest_neighbor_cosine_similarity'])
            else "—"
        )

        wav_link = "[dim]—[/dim]"

        if use_vad and i < len(saved_metadata):
            meta = saved_metadata[i]
            seg_idx = meta["segment_index"]  # ✅ correct (1-based)

            wav_path = OUTPUT_DIR / "segments" / f"segment_{seg_idx:03d}" / "sound.wav"

            if wav_path.exists():
                uri = wav_path.as_uri()
                display = f"segment_{seg_idx:03d}/sound.wav"
                wav_link = f"[link={uri}]{display}[/link]"

        table.add_row(
            time_str,
            f"[bold]{res['speaker_label']}[/]",
            cent_sim,
            nn_sim,
            wav_link,
        )

    console.print("\n")
    console.print(table)
