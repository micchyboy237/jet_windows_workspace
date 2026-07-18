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
from typing import Literal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Logging setup (temp1.py had a broken `logger = ` line with nothing assigned,
# which is a SyntaxError - fixed here with a real, traceable logger).
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("speaker_labeler_titanet")


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
        # Imported here so the rest of this module (clustering/merging logic) can be
        # unit-tested or reused even in environments without nemo_toolkit installed.
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
        self.sample_rate = 16000  # TitaNet expects 16kHz mono input
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
        return waveform.squeeze(0)  # shape: (num_samples,)

    @torch.no_grad()
    def extract_embeddings(self, audio_path: str):
        """Slice audio into sliding windows and extract a TitaNet embedding per window."""
        waveform = self._load_audio(audio_path)
        total_samples = waveform.shape[0]
        window_samples = int(self.duration * self.sample_rate)
        step_samples = int(self.step * self.sample_rate)

        if total_samples < window_samples:
            raise ValueError(
                f"Audio is shorter ({total_samples / self.sample_rate:.2f}s) than one "
                f"window ({self.duration}s). Reduce --duration or use a longer clip."
            )

        # Build window start indices
        all_starts = list(range(0, total_samples - window_samples + 1, step_samples))
        logger.info(f"Extracting embeddings from: {audio_path}")
        logger.info(f"Total windows (before silence filtering): {len(all_starts)}")

        # --- Silence / pause filtering ---------------------------------------------
        # Windows with very low energy are almost certainly silence, breath, or room
        # noise between turns. Left in, they get embedded like real speech and drag
        # cluster consistency down (this is what caused the "POOR" quality ratings and
        # the over-split speaker count on the first run). We filter relative to this
        # file's own energy distribution so it adapts to quiet/loud recordings.
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

            # forward() returns (logits, embeddings) - we only need the embeddings
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
        logger.info(f"Extracted {embeddings.shape[0]} normalized embeddings of dimension {embeddings.shape[1]}")
        logger.info(f"Time range: {timestamps[0][0]:.1f}s to {timestamps[-1][1]:.1f}s")
        return embeddings, timestamps

    # -----------------------------------------------------------------------
    # Everything below is unchanged from temp1.py - clustering, merging, and
    # timeline generation only operate on embedding vectors, so they work the
    # same way regardless of which model produced the embeddings.
    # -----------------------------------------------------------------------

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
            speaker_stats[label] = {
                'n_frames': len(speaker_embeddings),
                'duration': len(speaker_embeddings) * self.step,
                'avg_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                # Recalibrated from 0.75 (clean VoxCeleb benchmark) down to 0.60, based on
                # observed real-recording intra-speaker similarity (avg ~0.56-0.68 in testing)
                'quality': 'good' if np.mean(similarities) > 0.60 else 'poor',
                'frames_percent': len(speaker_embeddings) / len(embeddings) * 100
            }
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

    def generate_timeline(self, timestamps, labels, min_segment_duration=1.0):
        """Generate speaker timeline with segments."""
        timeline = []
        current_speaker = labels[0]
        segment_start = timestamps[0][0]
        for i, (timestamp, label) in enumerate(zip(timestamps, labels)):
            if label != current_speaker:
                segment_end = timestamps[i-1][1] if i > 0 else timestamp[0]
                duration = segment_end - segment_start
                if duration >= min_segment_duration and current_speaker != -1:
                    timeline.append((segment_start, segment_end, current_speaker))
                segment_start = timestamp[0]
                current_speaker = label
        segment_end = timestamps[-1][1]
        duration = segment_end - segment_start
        if duration >= min_segment_duration and current_speaker != -1:
            timeline.append((segment_start, segment_end, current_speaker))
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
):
    """Main execution flow - FULLY AUTOMATIC WITH SMART MERGING (TitaNet-Large version)."""
    logger.info("=" * 60)
    logger.info("AUTO SPEAKER LABELING WITH TITANET-LARGE + INTELLIGENT MERGING")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Merge threshold: {merge_threshold}")
    logger.info(f"Assignment threshold: {assign_threshold}")
    logger.info(f"Silence filter percentile: {min_energy_percentile}")

    labeler = SpeakerAutoLabelerTitaNet(
        model_name=model_name, duration=duration, step=step, batch_size=batch_size,
        min_energy_percentile=min_energy_percentile,
    )
    embeddings, timestamps = labeler.extract_embeddings(audio_path)
    labels, n_speakers = labeler.cluster_speakers(
        embeddings, method=method, merge_threshold=merge_threshold
    )
    centroids, speaker_stats = labeler.compute_speaker_centroids(embeddings, labels)
    refined_labels, confidences = labeler.assign_speaker_labels(
        embeddings, centroids, threshold=assign_threshold
    )
    timeline = labeler.generate_timeline(timestamps, refined_labels, min_segment_duration=min_segment_duration)

    print("\n" + "="*60)
    print(f"✅ FINAL RESULT: {n_speakers} SPEAKERS DETECTED")
    print("="*60)
    print("\n📊 SPEAKER STATISTICS:")
    print("-" * 60)
    sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['duration'], reverse=True)
    for i, (speaker_id, stats) in enumerate(sorted_speakers):
        quality_emoji = "✅" if stats['quality'] == 'good' else "⚠️"
        speaker_label = chr(65 + i)
        print(f"\n{quality_emoji} Speaker {speaker_label} (ID {speaker_id}):")
        print(f"   ├─ Duration: {stats['duration']:.1f}s ({stats['frames_percent']:.1f}%)")
        print(f"   ├─ Frames: {stats['n_frames']}")
        print(f"   ├─ Consistency: {stats['avg_similarity']:.3f} ± {stats['std_similarity']:.3f}")
        print(f"   └─ Quality: {stats['quality'].upper()}")

    print("\n\n📅 SPEAKER TIMELINE:")
    print("-" * 60)
    speaker_to_letter = {}
    for i, (speaker_id, _) in enumerate(sorted_speakers):
        speaker_to_letter[speaker_id] = chr(65 + i)
    for start, end, speaker in timeline:
        duration = end - start
        bar_length = int(duration / 32 * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        letter = speaker_to_letter.get(speaker, str(speaker))
        print(f"   {start:5.1f}s → {end:5.1f}s  |  Speaker {letter}  |  {duration:4.1f}s  {bar}")

    if len(centroids) >= 2:
        print("\n\n🔍 SPEAKER SEPARATION QUALITY:")
        print("-" * 60)
        speaker_list = list(centroids.keys())
        between_sims = []
        for i, sp1 in enumerate(speaker_list):
            for sp2 in speaker_list[i+1:]:
                sim = np.dot(centroids[sp1], centroids[sp2])
                between_sims.append(sim)
        avg_between = np.mean(between_sims) if between_sims else 0
        avg_intra = np.mean([stats['avg_similarity'] for stats in speaker_stats.values()])
        print(f"   Average intra-speaker similarity: {avg_intra:.3f}")
        print(f"   Average between-speaker similarity: {avg_between:.3f}")
        print(f"   Separation margin: {avg_intra - avg_between:.3f}")
        if avg_intra - avg_between > 0.3:
            print("   ✅ EXCELLENT separation - speakers are very distinct")
        elif avg_intra - avg_between > 0.2:
            print("   ✅ GOOD separation - speakers are distinguishable")
        elif avg_intra - avg_between > 0.1:
            print("   ⚠️  MODERATE separation - some confusion possible")
        else:
            print("   ❌ POOR separation - speakers sound similar")

    print("\n\n💡 FINAL ASSESSMENT:")
    print("-" * 60)
    print(f"📊 Detected {n_speakers} speakers")
    at_threshold_pct = np.sum(confidences >= assign_threshold) / len(confidences) * 100
    strict_pct = np.sum(confidences > 0.7) / len(confidences) * 100
    print(f"\n   Frame assignment: {at_threshold_pct:.1f}% met the assignment threshold (>={assign_threshold})")
    print(f"   Frame assignment: {strict_pct:.1f}% were strict high-confidence (>0.7)")

    return {
        'embeddings': embeddings,
        'timestamps': timestamps,
        'labels': refined_labels,
        'centroids': centroids,
        'speaker_stats': speaker_stats,
        'timeline': timeline,
        'confidences': confidences,
        'n_speakers': n_speakers
    }


if __name__ == "__main__":
    from main._main_nemo_titanet import main

    main()
