# python_scripts\temp\temp10.py
import numpy as np
from typing import Literal
from pyannote.audio import Model, Inference
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakerAutoLabeler:
    """
    Automatic speaker labeling and centroid extraction from audio embeddings
    Fully automatic with intelligent cluster merging
    """
    
    def __init__(self, model_name="pyannote/embedding", duration=3.0, step=1.0):
        """
        Initialize speaker labeler with pyannote model
        
        Args:
            model_name: Pretrained embedding model
            duration: Window duration in seconds
            step: Window step in seconds
        """
        logger.info(f"Loading model: {model_name}")
        self.model = Model.from_pretrained(model_name)
        self.inference = Inference(self.model, window="sliding", duration=duration, step=step)
        self.duration = duration
        self.step = step
        logger.info(f"Model loaded. Window: {duration}s, Step: {step}s")
    
    def extract_embeddings(self, audio_path):
        """Extract sliding window embeddings from audio"""
        logger.info(f"Extracting embeddings from: {audio_path}")
        result = self.inference(audio_path)
        
        embeddings = result.data
        window = result.sliding_window
        
        # Generate timestamps
        timestamps = []
        for i in range(embeddings.shape[0]):
            start = window.start + i * window.step
            end = start + window.duration
            timestamps.append((start, end))
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        logger.info(f"Extracted {embeddings.shape[0]} normalized embeddings of dimension {embeddings.shape[1]}")
        logger.info(f"Time range: {timestamps[0][0]:.1f}s to {timestamps[-1][1]:.1f}s")
        
        return embeddings, timestamps
    
    def auto_detect_speakers(self, embeddings, max_speakers=8, min_speakers=1):
        """Automatically detect the optimal number of speakers using multiple metrics"""
        logger.info(f"Auto-detecting speakers (range: {min_speakers}-{max_speakers})")
        
        n_samples = len(embeddings)
        max_possible = min(max_speakers, n_samples - 1)
        
        if max_possible <= min_speakers:
            return min_speakers
        
        candidates = []
        
        # Try different numbers of clusters
        for n_clusters in range(min_speakers, max_possible + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(embeddings)
            
            if len(set(labels)) > 1:
                # Silhouette score (higher is better)
                sil_score = silhouette_score(embeddings, labels)
                candidates.append((n_clusters, sil_score, 'silhouette'))
        
        # Get best silhouette score
        if candidates:
            best_silhouette = max(candidates, key=lambda x: x[1])
            optimal_n = best_silhouette[0]
            logger.info(f"  Silhouette optimal: {optimal_n} speakers (score: {best_silhouette[1]:.3f})")
        else:
            optimal_n = 2
        
        # If silhouette suggests high number, try elbow method on KMeans for validation
        if optimal_n >= 5:
            inertias = []
            for n in range(min_speakers, min(max_possible, 10)):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            
            # Find elbow
            if len(inertias) > 2:
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                elbow_idx = np.argmax(diffs2) + 1
                elbow_n = elbow_idx + min_speakers
                
                # If elbow suggests fewer speakers, use that
                if elbow_n < optimal_n:
                    logger.info(f"  Elbow suggests {elbow_n} speakers (vs silhouette's {optimal_n}) - using elbow")
                    optimal_n = elbow_n
        
        return optimal_n
    
    def merge_similar_clusters(self, embeddings, labels, similarity_threshold=0.40):
        """
        Merge clusters that are too similar (likely same speaker)
        Uses a lower threshold for merging (0.40 is typical for same speaker)
        """
        unique_labels = [l for l in set(labels) if l != -1]
        
        if len(unique_labels) <= 1:
            return labels, {}
        
        # Compute centroids for current clusters
        centroids = {}
        for label in unique_labels:
            cluster_embs = embeddings[labels == label]
            centroid = np.mean(cluster_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[label] = centroid
        
        # Build similarity matrix
        n_clusters = len(unique_labels)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for i, l1 in enumerate(unique_labels):
            for j, l2 in enumerate(unique_labels):
                similarity_matrix[i, j] = np.dot(centroids[l1], centroids[l2])
        
        # Greedy hierarchical merging
        merged_groups = []
        used = set()
        
        for i, l1 in enumerate(unique_labels):
            if l1 in used:
                continue
            
            # Start new group
            group = [l1]
            used.add(l1)
            
            # Find all clusters similar to this one (symmetric)
            for j, l2 in enumerate(unique_labels):
                if l2 not in used and similarity_matrix[i, j] > similarity_threshold:
                    group.append(l2)
                    used.add(l2)
            
            merged_groups.append(group)
        
        # Create mapping
        merge_map = {}
        for new_id, group in enumerate(merged_groups):
            for old_label in group:
                merge_map[old_label] = new_id
        
        # Apply merging
        merged_labels = np.array([merge_map.get(l, -1) if l != -1 else -1 for l in labels])
        
        logger.info(f"Merged {len(unique_labels)} clusters into {len(merged_groups)} speakers (threshold={similarity_threshold})")
        
        return merged_labels, merge_map
    
    def cluster_speakers(self, embeddings, method='agglomerative'):
        """
        Cluster embeddings into speaker groups with auto-detection and auto-merging
        
        Returns:
            labels: Array of cluster labels
            n_speakers: Number of speakers after merging
        """
        # Step 1: Auto-detect initial number (may over-cluster)
        initial_n = self.auto_detect_speakers(embeddings)
        logger.info(f"Initial auto-detection: {initial_n} speakers")
        
        # Step 2: Cluster with that number
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
        else:  # agglomerative (default)
            clustering = AgglomerativeClustering(n_clusters=initial_n)
            labels = clustering.fit_predict(embeddings)
        
        # Step 3: Merge similar clusters
        merged_labels, merge_map = self.merge_similar_clusters(embeddings, labels, similarity_threshold=0.40)
        n_speakers = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
        
        logger.info(f"After merging: {n_speakers} speakers")
        
        # Step 4: If we still have too many, try more aggressive merging
        if n_speakers > 4:  # Assume reasonable max for typical conversations
            logger.info(f"Still have {n_speakers} speakers, trying more aggressive merging (threshold=0.35)")
            merged_labels, merge_map = self.merge_similar_clusters(embeddings, labels, similarity_threshold=0.35)
            n_speakers = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
            logger.info(f"After aggressive merging: {n_speakers} speakers")
        
        # Log final cluster sizes
        unique, counts = np.unique(merged_labels, return_counts=True)
        for speaker, count in zip(unique, counts):
            if speaker != -1:
                logger.info(f"  Final Speaker {speaker}: {count} frames ({count * self.step:.1f}s)")
        
        return merged_labels, n_speakers
    
    def compute_speaker_centroids(self, embeddings, labels):
        """Compute centroid for each speaker cluster"""
        unique_labels = [l for l in set(labels) if l != -1]
        centroids = {}
        speaker_stats = {}
        
        for label in unique_labels:
            speaker_embeddings = embeddings[labels == label]
            centroid = np.mean(speaker_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            
            # Compute intra-cluster similarity
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
                'quality': 'good' if np.mean(similarities) > 0.65 else 'poor',
                'frames_percent': len(speaker_embeddings) / len(embeddings) * 100
            }
        
        return centroids, speaker_stats
    
    def assign_speaker_labels(self, embeddings, centroids, threshold=0.60):
        """Assign speaker labels based on nearest centroid with confidence"""
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
        """Generate speaker timeline with segments"""
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


def main(
    audio_path: str,
    duration: float = 3.0,
    step: float = 1.0,
    min_segment_duration: float = 1.0,
    method: Literal["agglomerative", "spectral"] = "agglomerative",
):
    """Main execution flow - FULLY AUTOMATIC WITH SMART MERGING"""
    logger.info("=" * 60)
    logger.info("AUTO SPEAKER LABELING WITH INTELLIGENT MERGING")
    logger.info("=" * 60)
    
    # Initialize labeler
    labeler = SpeakerAutoLabeler(duration=duration, step=step)
    
    # Extract embeddings
    embeddings, timestamps = labeler.extract_embeddings(audio_path)
    
    # Auto-detect AND auto-merge speakers
    labels, n_speakers = labeler.cluster_speakers(embeddings, method=method)
    
    # Compute centroids
    centroids, speaker_stats = labeler.compute_speaker_centroids(embeddings, labels)
    
    # Refine assignments
    refined_labels, confidences = labeler.assign_speaker_labels(embeddings, centroids, threshold=0.60)
    
    # Generate timeline
    timeline = labeler.generate_timeline(timestamps, refined_labels, min_segment_duration=min_segment_duration)
    
    # Display Results
    print("\n" + "="*60)
    print(f"✅ FINAL RESULT: {n_speakers} SPEAKERS DETECTED")
    print("="*60)
    
    # Speaker statistics
    print("\n📊 SPEAKER STATISTICS:")
    print("-" * 60)
    
    # Sort speakers by duration (most speaking first)
    sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['duration'], reverse=True)
    
    for i, (speaker_id, stats) in enumerate(sorted_speakers):
        quality_emoji = "✅" if stats['quality'] == 'good' else "⚠️"
        speaker_label = chr(65 + i)  # A, B, C, D, E...
        print(f"\n{quality_emoji} Speaker {speaker_label} (ID {speaker_id}):")
        print(f"   ├─ Duration: {stats['duration']:.1f}s ({stats['frames_percent']:.1f}%)")
        print(f"   ├─ Frames: {stats['n_frames']}")
        print(f"   ├─ Consistency: {stats['avg_similarity']:.3f} ± {stats['std_similarity']:.3f}")
        print(f"   └─ Quality: {stats['quality'].upper()}")
    
    # Timeline visualization
    print("\n\n📅 SPEAKER TIMELINE:")
    print("-" * 60)
    
    # Create speaker letter mapping
    speaker_to_letter = {}
    for i, (speaker_id, _) in enumerate(sorted_speakers):
        speaker_to_letter[speaker_id] = chr(65 + i)
    
    for start, end, speaker in timeline:
        duration = end - start
        bar_length = int(duration / 32 * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        letter = speaker_to_letter.get(speaker, str(speaker))
        print(f"   {start:5.1f}s → {end:5.1f}s  |  Speaker {letter}  |  {duration:4.1f}s  {bar}")
    
    # Separation quality
    if len(centroids) >= 2:
        print("\n\n🔍 SPEAKER SEPARATION QUALITY:")
        print("-" * 60)
        
        speaker_list = list(centroids.keys())
        # Calculate average between-speaker similarity
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
    
    # Final assessment
    print("\n\n💡 FINAL ASSESSMENT:")
    print("-" * 60)
    
    print(f"📊 Detected {n_speakers} speakers")
    
    # Confidence distribution
    high_conf = np.sum(confidences > 0.7) / len(confidences) * 100
    print(f"\n   Frame assignment confidence: {high_conf:.1f}% high-confidence (>0.7)")
    
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
    import argparse
    
    DEFAULT_AUDIO = r"C:\Users\druiv\.cache\files\audio\recording_1_speaker.wav"

    parser = argparse.ArgumentParser(
        description="Automatic speaker labeling with pyannote embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Audio path argument
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        default=DEFAULT_AUDIO,
        help="Path to input audio file"
    )
    
    # Duration argument with shorthand
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=3.0,
        help="Window duration in seconds for embedding extraction"
    )
    
    # Step argument with shorthand
    parser.add_argument(
        "-s", "--step",
        type=float,
        default=1.0,
        help="Window step in seconds for sliding window"
    )
    
    # Min segment duration with shorthand
    parser.add_argument(
        "-m", "--min-segment-duration",
        type=float,
        default=1.0,
        help="Minimum duration in seconds for a speaker segment to be included"
    )
    
    # Clustering method with shorthand
    parser.add_argument(
        "-c", "--clustering-method",
        type=str,
        choices=["agglomerative", "spectral"],
        default="agglomerative",
        help="Clustering method to use for speaker grouping"
    )

    args = parser.parse_args()

    results = main(
        audio_path=args.audio_path,
        duration=args.duration,
        step=args.step,
        min_segment_duration=args.min_segment_duration,
        method=args.clustering_method
    )
