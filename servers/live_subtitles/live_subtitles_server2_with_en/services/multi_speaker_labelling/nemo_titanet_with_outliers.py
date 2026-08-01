"""
Automatic multi-speaker labeling using NVIDIA NeMo TitaNet-Large embeddings.
Difference from the pyannote/embedding version (temp1.py):
  - Embeddings come from NeMo's `EncDecSpeakerLabelModel` (titanet_large) instead of
    pyannote's `Model` + `Inference`.
  - NeMo does NOT provide a built-in sliding-window extractor for a whole file, so
    windowing is done manually (load audio once, slice in memory, batch through the model).
  - Similarity thresholds are re-tuned for TitaNet's score range (NVIDIA's own default
    "same speaker" verification threshold is 0.70 cosine similarity).
Clustering / merging / timeline logic is unchanged from temp1.py - it only depends on
embedding vectors, not on which model produced them.
"""
import logging
import numpy as np
import torch
import torchaudio
from typing import Literal, TypedDict
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class SpeakerStats(TypedDict):
    """Per-speaker statistics computed from clustered embeddings."""
    n_frames: int
    duration: float
    avg_similarity: float
    std_similarity: float
    quality: str
    frames_percent: float

class TimelineSegment(TypedDict):
    """A contiguous speaker segment in the timeline."""
    start: float
    end: float
    duration: float
    speaker_id: int
    speaker_label: str

class MultiSpeakerResult(TypedDict):
    """Complete return type for detect_multi_speakers."""
    embeddings: np.ndarray
    timestamps: list[tuple[float, float]]
    labels: np.ndarray
    centroids: dict[int, np.ndarray]
    speaker_stats: dict[int, SpeakerStats]
    timeline: list[TimelineSegment]
    confidences: np.ndarray
    n_speakers: int
    outlier_mask: np.ndarray

# Outlier detection functions
def detect_outliers_mahalanobis(embeddings: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """
    Detect outliers using Mahalanobis distance.
    Args:
        embeddings: Normalized embeddings (shape: [n_samples, n_features]).
        threshold: Confidence threshold (e.g., 0.99 for 99% confidence).
    Returns:
        Boolean mask where `True` indicates an outlier.
    """
    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    inv_cov = np.linalg.pinv(cov)  # Use pseudo-inverse for numerical stability

    distances = np.array([
        mahalanobis(emb, mean, inv_cov) for emb in embeddings
    ])
    p_values = 1 - chi2.cdf(distances, df=embeddings.shape[1])
    return p_values < (1 - threshold)

def detect_outliers_zscore(embeddings: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Z-score.
    Args:
        embeddings: Normalized embeddings.
        threshold: Number of standard deviations to consider as outlier.
    Returns:
        Boolean mask where `True` indicates an outlier.
    """
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    z_scores = np.abs((embeddings - mean) / (std + 1e-8))
    return np.any(z_scores > threshold, axis=1)

def detect_outliers_isolation_forest(embeddings: np.ndarray, contamination: float = 0.01) -> np.ndarray:
    """
    Detect outliers using Isolation Forest.
    Args:
        embeddings: Normalized embeddings.
        contamination: Expected proportion of outliers.
    Returns:
        Boolean mask where `True` indicates an outlier.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    return clf.fit_predict(embeddings) == -1

def detect_outliers_dbscan(embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Detect outliers using DBSCAN.
    Args:
        embeddings: Normalized embeddings.
        eps: Maximum distance between two samples for one to be considered in the neighborhood of the other.
        min_samples: Number of samples in a neighborhood for a point to be considered a core point.
    Returns:
        Boolean mask where `True` indicates an outlier.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    return labels == -1

class SpeakerAutoLabelerTitaNet:
    """
    Automatic speaker labeling and centroid extraction using NeMo TitaNet-Large embeddings.
    Fully automatic with intelligent cluster merging (same pipeline as the pyannote version).
    """

    def __init__(
        self,
        model_name: str = "titanet_large",
        duration: float = 1.5,
        step: float = 0.75,
        batch_size: int = 16,
        min_energy_percentile: float = 15.0,
        device: str | None = None,
    ):
        """
        Initialize speaker labeler with a NeMo TitaNet model.
        Args:
            model_name: Pretrained NeMo speaker embedding model ("titanet_large").
            duration: Sliding window duration in seconds. TitaNet is trained/used on
                shorter windows than pyannote/embedding, so 1.5s is a good default
                (vs 3.0s for pyannote).
            step: Sliding window step in seconds (50% overlap by default).
            batch_size: How many windows to embed per forward pass. Kept modest (16)
                to fit comfortably in 6GB VRAM (e.g. GTX 1660).
            min_energy_percentile: Windows whose RMS energy falls below this percentile
                of the whole recording's window energies are treated as silence/pause
                and skipped entirely (never embedded, never clustered). This is relative
                to the file itself, so it adapts to quiet vs loud recordings. Set to 0
                to disable. Real conversational audio has pauses/breaths that otherwise
                get embedded and drag cluster quality down.
            device: "cuda" or "cpu". Auto-detected if not given.
        """
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading NeMo model: {model_name} (device={self.device})")
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.duration = duration
        self.step = step
        self.batch_size = batch_size
        self.min_energy_percentile = min_energy_percentile
        self.sample_rate = 16000
        logger.info(
            f"Model loaded. Window: {duration}s, Step: {step}s, Batch size: {batch_size}, "
            f"Silence filter: bottom {min_energy_percentile:.0f}% energy skipped"
        )

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio, resample to 16kHz, and force mono."""
        logger.info(f"Loading audio: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.info("Downmixed stereo/multi-channel audio to mono")
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            logger.info(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
        return waveform.squeeze(0)

    @torch.no_grad()
    def extract_embeddings(
        self,
        audio_path: str,
        outlier_method: str = None,
        outlier_threshold: float = 0.99,
    ) -> tuple[np.ndarray, list[tuple[float, float]], np.ndarray]:
        """
        Extract embeddings and optionally filter outliers.
        Args:
            audio_path: Path to audio file.
            outlier_method: One of ["mahalanobis", "zscore", "isolation_forest", "dbscan"].
            outlier_threshold: Threshold for outlier detection.
        Returns:
            embeddings: Filtered embeddings.
            timestamps: Corresponding timestamps.
            outlier_mask: Boolean mask indicating outliers.
        """
        waveform = self._load_audio(audio_path)
        total_samples = waveform.shape[0]
        window_samples = int(self.duration * self.sample_rate)
        step_samples = int(self.step * self.sample_rate)

        if total_samples < window_samples:
            raise ValueError(
                f"Audio is shorter ({total_samples / self.sample_rate:.2f}s) than one "
                f"window ({self.duration}s). Reduce --duration or use a longer clip."
            )

        all_starts = list(range(0, total_samples - window_samples + 1, step_samples))
        logger.info(f"Extracting embeddings from: {audio_path}")
        logger.info(f"Total windows (before silence filtering): {len(all_starts)}")

        if self.min_energy_percentile > 0:
            energies = np.array([
                torch.sqrt(torch.mean(waveform[s: s + window_samples] ** 2)).item()
                for s in all_starts
            ])
            energy_floor = np.percentile(energies, self.min_energy_percentile)
            starts = [s for s, e in zip(all_starts, energies) if e >= energy_floor]
            logger.info(
                f"Silence filter: dropped {len(all_starts) - len(starts)} low-energy windows "
                f"(RMS floor={energy_floor:.5f}, bottom {self.min_energy_percentile:.0f}%)"
            )
        else:
            starts = all_starts

        logger.info(f"Total windows (after silence filtering): {len(starts)}")

        embeddings = []
        timestamps = []
        for batch_start in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_start: batch_start + self.batch_size]
            batch_signals = torch.stack(
                [waveform[s: s + window_samples] for s in batch_starts]
            ).to(self.device)
            batch_lengths = torch.tensor(
                [window_samples] * len(batch_starts), device=self.device
            )
            _, batch_embs = self.model.forward(
                input_signal=batch_signals, input_signal_length=batch_lengths
            )
            embeddings.append(batch_embs.cpu().numpy())
            for s in batch_starts:
                start_sec = s / self.sample_rate
                end_sec = (s + window_samples) / self.sample_rate
                timestamps.append((start_sec, end_sec))
            logger.info(f"  Processed windows {batch_start + 1}-{batch_start + len(batch_starts)} / {len(starts)}")

        embeddings = np.concatenate(embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        # Outlier detection
        outlier_mask = np.zeros(len(embeddings), dtype=bool)
        if outlier_method:
            if outlier_method == "mahalanobis":
                outlier_mask = detect_outliers_mahalanobis(embeddings, outlier_threshold)
            elif outlier_method == "zscore":
                outlier_mask = detect_outliers_zscore(embeddings, outlier_threshold)
            elif outlier_method == "isolation_forest":
                outlier_mask = detect_outliers_isolation_forest(embeddings, outlier_threshold)
            elif outlier_method == "dbscan":
                outlier_mask = detect_outliers_dbscan(embeddings)
            else:
                logger.warning(f"Unknown outlier method: {outlier_method}. Skipping outlier detection.")

            # Filter embeddings and timestamps
            embeddings = embeddings[~outlier_mask]
            timestamps = [ts for ts, is_outlier in zip(timestamps, outlier_mask) if not is_outlier]
            logger.info(f"Filtered {np.sum(outlier_mask)} outlier windows. Remaining: {len(embeddings)}")

        logger.info(f"Extracted {embeddings.shape[0]} normalized embeddings of dimension {embeddings.shape[1]}")
        logger.info(f"Time range: {timestamps[0][0]:.1f}s to {timestamps[-1][1]:.1f}s")

        return embeddings, timestamps, outlier_mask

    def auto_detect_speakers(self, embeddings, max_speakers=8, min_speakers=1):
        """Automatically detect the optimal number of speakers using multiple metrics."""
        logger.info(f"Auto-detecting speakers (range: {min_speakers}-{max_speakers})")
        n_samples = len(embeddings)
        max_possible = min(max_speakers, n_samples - 1)
        if max_possible <= min_speakers:
            return min_speakers
        candidates = []
        for n_clusters in range(min_speakers, max_possible + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(embeddings)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(embeddings, labels)
                candidates.append((n_clusters, sil_score, 'silhouette'))
        if candidates:
            best_silhouette = max(candidates, key=lambda x: x[1])
            optimal_n = best_silhouette[0]
            logger.info(f"  Silhouette optimal: {optimal_n} speakers (score: {best_silhouette[1]:.3f})")
        else:
            optimal_n = 2
        if optimal_n >= 5:
            inertias = []
            for n in range(min_speakers, min(max_possible, 10)):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            if len(inertias) > 2:
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                elbow_idx = np.argmax(diffs2) + 1
                elbow_n = elbow_idx + min_speakers
                if elbow_n < optimal_n:
                    logger.info(f"  Elbow suggests {elbow_n} speakers (vs silhouette's {optimal_n}) - using elbow")
                    optimal_n = elbow_n
        return optimal_n

    def merge_similar_clusters(self, embeddings, labels, similarity_threshold=0.55):
        """
        Merge clusters that are too similar (likely same speaker).
        NOTE: 0.70-0.72 (NVIDIA's clean VoxCeleb verification threshold) was tried first
        and over-split real conversational audio into too many low-quality clusters -
        short in-the-wild windows just don't reach that similarity range. 0.55 is
        recalibrated from actually observed same-speaker similarity on real recordings.
        Use --diagnose (see cluster_speakers) to check where your own audio lands.
        """
        unique_labels = [l for l in set(labels) if l != -1]
        if len(unique_labels) <= 1:
            return labels, {}
        centroids = {}
        for label in unique_labels:
            cluster_embs = embeddings[labels == label]
            centroid = np.mean(cluster_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[label] = centroid
        n_clusters = len(unique_labels)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for i, l1 in enumerate(unique_labels):
            for j, l2 in enumerate(unique_labels):
                similarity_matrix[i, j] = np.dot(centroids[l1], centroids[l2])
        off_diag = similarity_matrix[~np.eye(n_clusters, dtype=bool)]
        if len(off_diag) > 0:
            logger.info(
                f"  Inter-cluster similarity spread: min={off_diag.min():.3f}, "
                f"median={np.median(off_diag):.3f}, max={off_diag.max():.3f} "
                f"(threshold={similarity_threshold})"
            )
        merged_groups = []
        used = set()
        for i, l1 in enumerate(unique_labels):
            if l1 in used:
                continue
            group = [l1]
            used.add(l1)
            for j, l2 in enumerate(unique_labels):
                if l2 not in used and similarity_matrix[i, j] > similarity_threshold:
                    group.append(l2)
                    used.add(l2)
            merged_groups.append(group)
        merge_map = {}
        for new_id, group in enumerate(merged_groups):
            for old_label in group:
                merge_map[old_label] = new_id
        merged_labels = np.array([merge_map.get(l, -1) if l != -1 else -1 for l in labels])
        logger.info(f"Merged {len(unique_labels)} clusters into {len(merged_groups)} speakers (threshold={similarity_threshold})")
        return merged_labels, merge_map

    def cluster_speakers(self, embeddings, method='agglomerative', merge_threshold=0.55):
        """Cluster embeddings into speaker groups with auto-detection and auto-merging."""
        initial_n = self.auto_detect_speakers(embeddings)
        logger.info(f"Initial auto-detection: {initial_n} speakers")
        logger.info(f"Initial clustering into {initial_n} groups using {method}")
        if method == 'spectral':
            from sklearn.cluster import SpectralClustering
            clustering = SpectralClustering(
                n_clusters=initial_n,
                affinity='nearest_neighbors',
                n_neighbors=min(10, len(embeddings)//2),
                random_state=42
            )
            labels = clustering.fit_predict(embeddings)
        else:
            clustering = AgglomerativeClustering(n_clusters=initial_n)
            labels = clustering.fit_predict(embeddings)
        merged_labels, merge_map = self.merge_similar_clusters(
            embeddings, labels, similarity_threshold=merge_threshold
        )
        n_speakers = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
        logger.info(f"After merging (threshold={merge_threshold}): {n_speakers} speakers")
        if n_speakers > 4:
            aggressive_threshold = merge_threshold - 0.05
            logger.info(f"Still have {n_speakers} speakers, trying more aggressive merging (threshold={aggressive_threshold})")
            merged_labels, merge_map = self.merge_similar_clusters(
                embeddings, labels, similarity_threshold=aggressive_threshold
            )
            n_speakers = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
            logger.info(f"After aggressive merging: {n_speakers} speakers")
        unique, counts = np.unique(merged_labels, return_counts=True)
        for speaker, count in zip(unique, counts):
            if speaker != -1:
                logger.info(f"  Final Speaker {speaker}: {count} frames ({count * self.step:.1f}s)")
        return merged_labels, n_speakers

    def compute_speaker_centroids(self, embeddings, labels):
        """Compute centroid for each speaker cluster."""
        unique_labels = [l for l in set(labels) if l != -1]
        centroids = {}
        speaker_stats = {}
        for label in unique_labels:
            speaker_embeddings = embeddings[labels == label]
            centroid = np.mean(speaker_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            similarities = []
            for emb in speaker_embeddings:
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = np.dot(emb_norm, centroid)
                similarities.append(sim)
            centroids[label] = centroid
            speaker_stats[label] = SpeakerStats(
                n_frames=len(speaker_embeddings),
                duration=len(speaker_embeddings) * self.step,
                avg_similarity=np.mean(similarities),
                std_similarity=np.std(similarities),
                quality='good' if np.mean(similarities) > 0.60 else 'poor',
                frames_percent=len(speaker_embeddings) / len(embeddings) * 100
            )
        return centroids, speaker_stats

    def assign_speaker_labels(self, embeddings, centroids, threshold=0.55):
        """
        Assign speaker labels based on nearest centroid with confidence.
        NOTE: 0.70 (NVIDIA's clean verification default) discarded ~77% of windows as
        "unassigned" on real conversational audio in testing. 0.55 is recalibrated from
        the actual similarity range observed for correctly-matched speaker windows.
        """
        speaker_ids = []
        confidences = []
        speaker_list = list(centroids.keys())
        centroid_matrix = np.array([centroids[s] for s in speaker_list])
        for emb in embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarities = np.dot(centroid_matrix, emb_norm)
            max_sim = np.max(similarities)
            best_idx = np.argmax(similarities)
            if max_sim >= threshold:
                speaker_ids.append(speaker_list[best_idx])
                confidences.append(max_sim)
            else:
                speaker_ids.append(-1)
                confidences.append(max_sim)
        return np.array(speaker_ids), np.array(confidences)

    def generate_timeline(
        self,
        timestamps: list[tuple[float, float]],
        labels: np.ndarray,
        min_segment_duration: float = 1.0,
    ) -> list[TimelineSegment]:
        """Generate speaker timeline with segments, merging consecutive same-speaker segments."""
        timeline: list[TimelineSegment] = []
        if len(timestamps) == 0 or len(labels) == 0:
            logger.warning("No timestamps or labels available for timeline generation")
            return timeline
        raw_segments: list[TimelineSegment] = []
        current_speaker = labels[0]
        segment_start = timestamps[0][0]
        for i, (timestamp, label) in enumerate(zip(timestamps, labels)):
            if label != current_speaker:
                segment_end = timestamps[i - 1][1] if i > 0 else timestamp[0]
                duration = segment_end - segment_start
                if duration >= min_segment_duration and current_speaker != -1:
                    raw_segments.append(TimelineSegment(
                        start=round(segment_start, 3),
                        end=round(segment_end, 3),
                        duration=round(duration, 3),
                        speaker_id=int(current_speaker),
                        speaker_label=f"Speaker {chr(65 + int(current_speaker))}",
                    ))
                segment_start = timestamp[0]
                current_speaker = label
        segment_end = timestamps[-1][1]
        duration = segment_end - segment_start
        if duration >= min_segment_duration and current_speaker != -1:
            raw_segments.append(TimelineSegment(
                start=round(segment_start, 3),
                end=round(segment_end, 3),
                duration=round(duration, 3),
                speaker_id=int(current_speaker),
                speaker_label=f"Speaker {chr(65 + int(current_speaker))}",
            ))
        logger.info(f"Raw segments before merging: {len(raw_segments)}")
        if not raw_segments:
            return timeline
        merged_segments: list[TimelineSegment] = [raw_segments[0]]
        for seg in raw_segments[1:]:
            prev = merged_segments[-1]
            if seg["speaker_id"] == prev["speaker_id"] and seg["start"] <= prev["end"] + self.step:
                prev["end"] = max(prev["end"], seg["end"])
                prev["duration"] = round(prev["end"] - prev["start"], 3)
                logger.debug(
                    f"  Merged: {seg['speaker_label']} "
                    f"({seg['start']:.1f}s→{seg['end']:.1f}s) "
                    f"into ({prev['start']:.1f}s→{prev['end']:.1f}s)"
                )
            else:
                merged_segments.append(seg)
        timeline = [
            seg for seg in merged_segments
            if seg["duration"] >= min_segment_duration
        ]
        merge_count = len(raw_segments) - len(timeline)
        if merge_count > 0:
            logger.info(
                f"Merged {merge_count} consecutive same-speaker segments. "
                f"Final timeline: {len(timeline)} segments"
            )
        else:
            logger.info(f"No segments to merge. Timeline: {len(timeline)} segments")
        return timeline

def detect_multi_speakers(
    audio_path: str,
    model_name: str = "titanet_large",
    duration: float = 2.0,
    step: float = 0.75,
    batch_size: int = 16,
    min_energy_percentile: float = 15.0,
    min_segment_duration: float = 1.0,
    method: Literal["agglomerative", "spectral"] = "agglomerative",
    merge_threshold: float = 0.55,
    assign_threshold: float = 0.55,
    outlier_method: str = None,
    outlier_threshold: float = 0.99,
) -> MultiSpeakerResult:
    """Main execution flow - FULLY AUTOMATIC WITH SMART MERGING (TitaNet-Large version)."""
    logger.info("=" * 60)
    logger.info("AUTO SPEAKER LABELING WITH TITANET-LARGE + OUTLIER DETECTION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Merge threshold: {merge_threshold}")
    logger.info(f"Assignment threshold: {assign_threshold}")
    logger.info(f"Outlier method: {outlier_method}")
    logger.info(f"Silence filter percentile: {min_energy_percentile}")

    labeler = SpeakerAutoLabelerTitaNet(
        model_name=model_name, duration=duration, step=step, batch_size=batch_size,
        min_energy_percentile=min_energy_percentile,
    )

    embeddings, timestamps, outlier_mask = labeler.extract_embeddings(
        audio_path, outlier_method=outlier_method, outlier_threshold=outlier_threshold
    )

    labels, n_speakers = labeler.cluster_speakers(
        embeddings, method=method, merge_threshold=merge_threshold
    )
    centroids, speaker_stats = labeler.compute_speaker_centroids(embeddings, labels)
    refined_labels, confidences = labeler.assign_speaker_labels(
        embeddings, centroids, threshold=assign_threshold
    )
    timeline = labeler.generate_timeline(timestamps, refined_labels, min_segment_duration=min_segment_duration)

    return MultiSpeakerResult(
        embeddings=embeddings,
        timestamps=timestamps,
        labels=refined_labels,
        centroids=centroids,
        speaker_stats=speaker_stats,
        timeline=timeline,
        confidences=confidences,
        n_speakers=n_speakers,
        outlier_mask=outlier_mask,
    )

if __name__ == "__main__":
    from main._main_nemo_titanet_with_outliers import main
    main()
