import torch
import numpy as np
import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pyannote.audio.pipelines.speaker_verification import PyannoteAudioPretrainedSpeakerEmbedding
from pyannote.audio import Inference, Model
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

class SpeakerLabeler:
    def __init__(self, token=None, output_dir=None, use_librosa=True):
        # Load embedding model
        self.embedding_model = PyannoteAudioPretrainedSpeakerEmbedding(
            embedding="pyannote/embedding",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            token=token
        )
        
        # Optional: Load segmentation model for VAD
        from pyannote.audio.pipelines.utils import get_model
        segmentation_model = get_model("pyannote/segmentation", token=token)
        self.segmentation = Inference(
            segmentation_model,
            pre_aggregation_hook=lambda scores: np.max(scores, axis=-1, keepdims=True)
        )
        
        # Audio loading preference
        self.use_librosa = use_librosa
        
        # Output directory setup
        self.output_dir = Path(output_dir) if output_dir else Path("./speaker_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.segments = []
        self.labeled_chunks = []
        self.embeddings = None
        self.speaker_labels = None
        
        # Test audio loading
        self._check_audio_loader()
    
    def _check_audio_loader(self):
        """Check and configure audio loading backend"""
        try:
            import librosa
            print("✅ Librosa available for audio loading")
        except ImportError:
            print("⚠️ Librosa not installed. Install with: pip install librosa")
            print("   Will try to use PyAnnote's native loader instead")
            self.use_librosa = False
        
        try:
            import soundfile as sf
            print("✅ SoundFile available (required for PyAnnote)")
        except ImportError:
            print("⚠️ SoundFile not installed. Install with: pip install soundfile")
    
    def load_audio(self, audio_path):
        """Load audio with multiple fallback methods"""
        audio_path = str(audio_path)
        errors = []
        
        # Method 1: Librosa (most compatible)
        if self.use_librosa:
            try:
                import librosa
                print(f"   Loading with librosa: {audio_path}")
                waveform, sample_rate = librosa.load(
                    audio_path, 
                    sr=self.embedding_model.sample_rate,
                    mono=True
                )
                # Convert to torch tensor (shape: 1, num_samples)
                waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
                return waveform_tensor, sample_rate
            except Exception as e:
                errors.append(f"Librosa: {e}")
        
        # Method 2: SoundFile
        try:
            import soundfile as sf
            print(f"   Loading with soundfile: {audio_path}")
            waveform, sample_rate = sf.read(audio_path)
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)  # Convert to mono
            
            # Resample if needed
            if sample_rate != self.embedding_model.sample_rate:
                import librosa
                waveform = librosa.resample(
                    waveform, 
                    orig_sr=sample_rate, 
                    target_sr=self.embedding_model.sample_rate
                )
                sample_rate = self.embedding_model.sample_rate
            
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            return waveform_tensor, sample_rate
        except Exception as e:
            errors.append(f"SoundFile: {e}")
        
        # Method 3: PyAnnote native loader (AudioDecoder)
        try:
            from pyannote.audio.core.io import AudioFile, AudioDecoder
            print(f"   Loading with PyAnnote: {audio_path}")
            decoder = AudioDecoder(audio_path)
            waveform, sample_rate = decoder.decode()
            
            # Convert to torch tensor if needed
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            
            # Ensure mono
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != self.embedding_model.sample_rate:
                import librosa
                waveform_np = waveform.numpy().squeeze()
                waveform_np = librosa.resample(
                    waveform_np,
                    orig_sr=sample_rate,
                    target_sr=self.embedding_model.sample_rate
                )
                waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
                sample_rate = self.embedding_model.sample_rate
            
            return waveform, sample_rate
        except Exception as e:
            errors.append(f"PyAnnote: {e}")
        
        # If all methods fail
        raise RuntimeError(
            f"Failed to load audio file '{audio_path}'. All methods failed:\n" + 
            "\n".join(f"  - {e}" for e in errors)
        )
    
    def extract_speech_segments(self, audio_path):
        """Extract speech segments with VAD"""
        try:
            # Method 1: Try PyAnnote's native loading
            print("   Attempting VAD with PyAnnote...")
            vad_scores = self.segmentation(audio_path).data
        except Exception as e:
            print(f"   PyAnnote VAD failed: {e}")
            print("   Using librosa-loaded audio for VAD...")
            
            # Method 2: Load with librosa and use manual VAD
            waveform, sample_rate = self.load_audio(audio_path)
            
            # Use energy-based VAD as fallback
            vad_scores = self._energy_based_vad(waveform, sample_rate)
        
        # Find speech regions (simple threshold)
        from scipy.ndimage import label
        
        if isinstance(vad_scores, np.ndarray):
            vad_array = vad_scores.squeeze()
        else:
            vad_array = np.array(vad_scores).squeeze()
        
        speech_mask = vad_array > 0.5
        labeled, num_features = label(speech_mask)
        
        segments = []
        for i in range(1, num_features + 1):
            region = np.where(labeled == i)[0]
            start_frame = region[0]
            end_frame = region[-1]
            
            # Calculate time based on frame length (typical VAD frame: 10ms)
            # Adjust based on actual VAD output length
            frame_duration = len(vad_array) / len(waveform.squeeze()) if 'waveform' in locals() else 0.01
            
            segments.append({
                'start': start_frame * frame_duration,
                'end': end_frame * frame_duration,
                'duration': (end_frame - start_frame) * frame_duration,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'vad_confidence': float(np.mean(vad_array[start_frame:end_frame]))
            })
        
        self.segments = segments
        return segments, vad_array
    
    def _energy_based_vad(self, waveform, sample_rate, frame_length=0.025, threshold=0.1):
        """Simple energy-based VAD as fallback"""
        import librosa
        
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            audio = waveform.numpy().squeeze()
        else:
            audio = np.array(waveform).squeeze()
        
        # Compute RMS energy
        frame_length_samples = int(frame_length * sample_rate)
        hop_length = frame_length_samples // 2
        
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length_samples,
            hop_length=hop_length
        ).squeeze()
        
        # Normalize
        rms_normalized = rms / (np.max(rms) + 1e-10)
        
        # Apply threshold
        vad_scores = (rms_normalized > threshold).astype(float)
        
        # Smooth
        from scipy.ndimage import uniform_filter1d
        vad_scores = uniform_filter1d(vad_scores, size=5)
        
        return vad_scores
    
    def embed_audio_chunks(self, audio_path, chunks):
        """Extract embeddings for each audio chunk"""
        # Load audio using robust loader
        waveform, sample_rate = self.load_audio(audio_path)
        
        embeddings = []
        valid_chunks = []
        rejected_chunks = []
        
        for chunk in chunks:
            start_sample = int(chunk['start'] * sample_rate)
            end_sample = int(chunk['end'] * sample_rate)
            
            # Extract chunk
            chunk_waveform = waveform[:, start_sample:end_sample]
            
            # Skip very short chunks
            if chunk_waveform.shape[1] < self.embedding_model.min_num_samples:
                chunk['status'] = 'rejected'
                chunk['reason'] = f'Too short: {chunk_waveform.shape[1]} samples < {self.embedding_model.min_num_samples}'
                rejected_chunks.append(chunk)
                continue
            
            # Get embedding
            try:
                emb = self.embedding_model(chunk_waveform.unsqueeze(0) if chunk_waveform.dim() == 2 else chunk_waveform.unsqueeze(0).unsqueeze(0))
                embeddings.append(emb[0])
                chunk['embedding'] = emb[0]
                chunk['embedding_norm'] = float(np.linalg.norm(emb[0]))
                chunk['status'] = 'processed'
                valid_chunks.append(chunk)
            except Exception as e:
                chunk['status'] = 'error'
                chunk['reason'] = str(e)
                rejected_chunks.append(chunk)
                print(f"   ⚠️ Failed to embed chunk [{chunk['start']:.2f}s - {chunk['end']:.2f}s]: {e}")
        
        self.embeddings = np.array(embeddings) if embeddings else np.array([])
        self.labeled_chunks = valid_chunks
        
        return self.embeddings, valid_chunks, rejected_chunks
    
    def cluster_speakers(self, chunks, n_speakers=None, threshold=0.5):
        """Cluster audio chunks by speaker"""
        embeddings = np.array([chunk['embedding'] for chunk in chunks if 'embedding' in chunk])
        valid_chunks = [chunk for chunk in chunks if 'embedding' in chunk]
        
        if len(embeddings) == 0:
            return chunks
        
        # Distance matrix
        distances = cdist(embeddings, embeddings, metric='cosine')
        
        # Clustering
        if n_speakers is None:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='cosine',
                linkage='average'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_speakers,
                metric='cosine',
                linkage='average'
            )
        
        labels = clustering.fit_predict(embeddings)
        self.speaker_labels = labels
        
        # Assign labels to chunks
        for i, chunk in enumerate(valid_chunks):
            chunk['speaker_label'] = f"SPEAKER_{labels[i]:02d}"
            chunk['cluster_id'] = int(labels[i])
        
        # Calculate cluster statistics
        self.cluster_stats = self._calculate_cluster_stats(embeddings, labels)
        
        return valid_chunks
    
    def _calculate_cluster_stats(self, embeddings, labels):
        """Calculate statistics for each speaker cluster"""
        stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            cluster_embs = embeddings[mask]
            
            centroid = np.mean(cluster_embs, axis=0)
            # Average pairwise distance within cluster (cohesion)
            if len(cluster_embs) > 1:
                intra_distances = cdist(cluster_embs, cluster_embs, metric='cosine')
                cohesion = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])
            else:
                cohesion = 0.0
            
            stats[f"SPEAKER_{label:02d}"] = {
                'cluster_id': int(label),
                'num_segments': int(np.sum(mask)),
                'cohesion': float(cohesion),
                'centroid_norm': float(np.linalg.norm(centroid))
            }
        
        return stats
    
    def save_all_results(self, audio_path):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(audio_path).stem
        
        print(f"\n💾 Saving results to {self.output_dir}/")
        saved_files = {}
        
        # Create output subdirectory for this audio
        audio_output_dir = self.output_dir / audio_name
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save labeled segments as CSV
        segments_file = audio_output_dir / f"segments_{timestamp}.csv"
        with open(segments_file, 'w', newline='') as f:
            if self.labeled_chunks:
                fieldnames = ['start', 'end', 'duration', 'speaker_label', 'vad_confidence', 
                             'embedding_norm', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.labeled_chunks)
        saved_files['segments_csv'] = str(segments_file)
        print(f"✅ Segments CSV saved: {segments_file.name}")
        
        # 2. Save speaker timeline as JSON
        timeline = []
        for chunk in sorted(self.labeled_chunks, key=lambda x: x['start']):
            if 'speaker_label' in chunk:
                timeline.append({
                    'speaker': chunk['speaker_label'],
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'duration': chunk['duration'],
                    'confidence': chunk.get('vad_confidence', 1.0)
                })
        
        timeline_file = audio_output_dir / f"timeline_{timestamp}.json"
        with open(timeline_file, 'w') as f:
            json.dump(timeline, f, indent=2)
        saved_files['timeline_json'] = str(timeline_file)
        print(f"✅ Timeline JSON saved: {timeline_file.name}")
        
        # 3. Save embeddings as NumPy array
        if self.embeddings is not None and len(self.embeddings) > 0:
            embeddings_file = audio_output_dir / f"embeddings_{timestamp}.npz"
            np.savez(
                embeddings_file,
                embeddings=self.embeddings,
                labels=self.speaker_labels if self.speaker_labels is not None else np.array([]),
                sample_rate=self.embedding_model.sample_rate
            )
            saved_files['embeddings_npz'] = str(embeddings_file)
            print(f"✅ Embeddings NPZ saved: {embeddings_file.name}")
        
        # 4. Save summary report as text
        report_file = audio_output_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SPEAKER LABELING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Audio File: {audio_path}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample Rate: {self.embedding_model.sample_rate} Hz\n")
            f.write(f"Audio Loader: {'librosa' if self.use_librosa else 'pyannote'}\n\n")
            
            # Speaker statistics
            speaker_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0})
            for chunk in self.labeled_chunks:
                if 'speaker_label' in chunk:
                    speaker = chunk['speaker_label']
                    speaker_stats[speaker]['count'] += 1
                    speaker_stats[speaker]['total_duration'] += chunk['duration']
            
            f.write("SPEAKER STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for speaker in sorted(speaker_stats.keys()):
                stats = speaker_stats[speaker]
                f.write(f"\n{speaker}:\n")
                f.write(f"  Segments: {stats['count']}\n")
                f.write(f"  Total Duration: {stats['total_duration']:.2f}s\n")
                if stats['count'] > 0:
                    f.write(f"  Average Segment: {stats['total_duration']/stats['count']:.2f}s\n")
            
            f.write("\n\nDETAILED TIMELINE:\n")
            f.write("-" * 40 + "\n")
            for chunk in sorted(self.labeled_chunks, key=lambda x: x['start']):
                if 'speaker_label' in chunk:
                    f.write(f"[{chunk['start']:7.2f}s - {chunk['end']:7.2f}s] "
                           f"{chunk['speaker_label']:15s} "
                           f"(duration: {chunk['duration']:.2f}s)\n")
        
        saved_files['report_txt'] = str(report_file)
        print(f"✅ Report saved: {report_file.name}")
        
        # 5. Save VAD segments if available
        if self.segments:
            vad_file = audio_output_dir / f"vad_segments_{timestamp}.csv"
            with open(vad_file, 'w', newline='') as f:
                fieldnames = ['start', 'end', 'duration', 'vad_confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([{k: s[k] for k in fieldnames} for s in self.segments])
            saved_files['vad_csv'] = str(vad_file)
            print(f"✅ VAD segments saved: {vad_file.name}")
        
        return saved_files
    
    def process_audio(self, audio_path, n_speakers=None, threshold=0.5):
        """Complete pipeline: VAD -> Embed -> Cluster -> Save"""
        print(f"\n🎤 Processing: {audio_path}")
        print("=" * 60)
        
        # Step 1: Extract speech segments
        print("1/3 Extracting speech segments...")
        try:
            segments, vad_scores = self.extract_speech_segments(audio_path)
            print(f"   Found {len(segments)} speech segments")
        except Exception as e:
            print(f"❌ Failed to extract segments: {e}")
            return None, None
        
        if not segments:
            print("⚠️ No speech detected!")
            return None, None
        
        # Step 2: Embed chunks
        print("2/3 Extracting speaker embeddings...")
        embeddings, valid_chunks, rejected_chunks = self.embed_audio_chunks(audio_path, segments)
        print(f"   Processed {len(valid_chunks)} chunks ({len(rejected_chunks)} rejected)")
        
        if not valid_chunks:
            print("⚠️ No valid chunks for embedding!")
            return None, None
        
        # Step 3: Cluster speakers
        print("3/3 Clustering speakers...")
        labeled_chunks = self.cluster_speakers(valid_chunks, n_speakers=n_speakers, threshold=threshold)
        
        # Get unique speakers
        unique_speakers = set(chunk.get('speaker_label', 'UNKNOWN') for chunk in labeled_chunks)
        print(f"   Identified {len(unique_speakers)} speakers: {sorted(unique_speakers)}")
        
        # Save results
        saved_files = self.save_all_results(audio_path)
        
        return labeled_chunks, saved_files

# Usage Example
if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    DEFAULT_AUDIO = str(
        Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
    )
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Speaker labeling with librosa audio loading support"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        type=str,
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-n",
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (auto-detect if not specified)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Clustering threshold (default: 0.5)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for model access",
    )
    parser.add_argument(
        "--no-librosa",
        action="store_true",
        help="Force use of PyAnnote's native audio loader",
    )

    args = parser.parse_args()
    audio_path = Path(args.audio_path)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        print("Please provide a valid audio file path")
        exit(1)
    
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize labeler with librosa support
    print("🚀 Initializing Speaker Labeler...")
    labeler = SpeakerLabeler(
        token=args.token, 
        output_dir=output_dir,
        use_librosa=not args.no_librosa
    )

    # Process audio with automatic saving
    labeled_chunks, saved_files = labeler.process_audio(
        str(audio_path), 
        n_speakers=args.num_speakers,
        threshold=args.threshold
    )

    # Print summary
    if labeled_chunks:
        print("\n" + "=" * 60)
        print("📊 PROCESSING COMPLETE")
        print("=" * 60)
        
        # Display segments
        print("\n🎯 Labeled Segments:")
        for chunk in sorted(labeled_chunks, key=lambda x: x['start']):
            if 'speaker_label' in chunk:
                print(f"  [{chunk['start']:6.2f}s - {chunk['end']:6.2f}s] "
                      f"{chunk['speaker_label']:15s} "
                      f"({chunk['duration']:.2f}s)")
        
        # Display saved files
        print(f"\n📁 Results saved to: {output_dir}")
        if saved_files:
            print("\n📄 Generated Files:")
            for file_type, filepath in saved_files.items():
                print(f"  • {file_type}: {Path(filepath).name}")
    else:
        print("\n❌ Processing failed or no speech detected")
