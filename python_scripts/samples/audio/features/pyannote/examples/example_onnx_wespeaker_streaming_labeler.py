import numpy as np
import torch
import json
import pickle
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
from pyannote.audio.pipelines.speaker_verification import ONNXWeSpeakerPretrainedSpeakerEmbedding
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import threading
import queue
from pathlib import Path
import csv
from datetime import datetime

@dataclass
class SpeakerIdentity:
    """Represents a dynamically learned speaker"""
    label: str
    first_seen: float
    last_seen: float
    embeddings: deque  # Rolling window of recent embeddings
    total_occurrences: int
    audio_samples: deque  # Recent audio samples for quality
    
    def to_serializable(self):
        """Convert to serializable format"""
        return {
            'label': self.label,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'avg_embedding': np.mean([emb for emb in self.embeddings], axis=0).tolist() if self.embeddings else None,
            'total_occurrences': self.total_occurrences,
            'active_duration': self.last_seen - self.first_seen
        }

@dataclass
class SpeakerSegment:
    """Represents a continuous speech segment by a speaker"""
    speaker: str
    start_time: float
    end_time: float
    duration: float
    confidence: float = 1.0

class StreamingSpeakerLabeler:
    """
    Dynamic speaker labeling for streaming audio.
    Automatically accumulates and identifies speakers in real-time.
    """
    
    def __init__(self, 
                 embedding_model: str = "hbredin/wespeaker-voxceleb-resnet34-LM",
                 similarity_threshold: float = 0.75,
                 unknown_threshold: float = 0.65,
                 embedding_history: int = 10,
                 inactivity_timeout: float = 30.0,
                 device: str = "cuda",
                 output_dir: Optional[Path] = None):
        """
        Args:
            embedding_model: ONNX model for speed
            similarity_threshold: Cosine similarity for same speaker
            unknown_threshold: Below this = new speaker
            embedding_history: Number of past embeddings to keep per speaker
            inactivity_timeout: Seconds before a speaker is considered inactive
            output_dir: Directory to save results
        """
        self.embedding_model = ONNXWeSpeakerPretrainedSpeakerEmbedding(
            embedding=embedding_model,
            device=torch.device(device)
        )
        
        self.similarity_threshold = similarity_threshold
        self.unknown_threshold = unknown_threshold
        self.embedding_history = embedding_history
        self.inactivity_timeout = inactivity_timeout
        self.output_dir = output_dir or Path("./speaker_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic speaker database
        self.known_speakers: Dict[str, SpeakerIdentity] = {}
        self.speaker_counter = 0
        self.lock = threading.Lock()
        
        # Streaming buffer
        self.audio_buffer = deque(maxlen=int(3 * self.embedding_model.sample_rate))  # 3 second buffer
        self.buffer_lock = threading.Lock()
        
        # Tracking results
        self.speaker_segments: List[SpeakerSegment] = []
        self.timeline: List[Tuple[float, str]] = []  # (timestamp, speaker)
        self.processing_log: List[Dict] = []
        
    def add_audio_chunk(self, chunk: np.ndarray, timestamp: float) -> Optional[str]:
        """
        Add streaming audio chunk and identify speaker
        
        Args:
            chunk: Audio samples (1D array)
            timestamp: Current timestamp in seconds
        
        Returns:
            Speaker label or None if not enough audio
        """
        # Add to buffer
        with self.buffer_lock:
            self.audio_buffer.extend(chunk)
        
        # Process when we have enough audio
        if len(self.audio_buffer) >= self.embedding_model.min_num_samples:
            # Extract a window of audio
            audio_window = np.array(list(self.audio_buffer))[-self.embedding_model.min_num_samples:]
            
            # Get embedding
            waveform = torch.from_numpy(audio_window).float().unsqueeze(0).unsqueeze(0)
            embedding = self.embedding_model(waveform)[0]
            
            # Identify or register speaker
            speaker_label = self._identify_speaker(embedding, timestamp)
            
            # Track timeline
            self.timeline.append((timestamp, speaker_label))
            
            # Log processing
            self.processing_log.append({
                'timestamp': timestamp,
                'speaker': speaker_label,
                'buffer_size': len(self.audio_buffer),
                'known_speakers': len(self.known_speakers)
            })
            
            return speaker_label
        
        return None
    
    def _identify_speaker(self, embedding: np.ndarray, timestamp: float) -> str:
        """Identify speaker from embedding or create new one"""
        with self.lock:
            # Clean inactive speakers
            self._cleanup_inactive_speakers(timestamp)
            
            if not self.known_speakers:
                # First speaker ever
                return self._register_new_speaker(embedding, timestamp)
            
            # Compare with all known speakers
            best_similarity = -1
            best_speaker = None
            
            for label, speaker in self.known_speakers.items():
                # Use recent embeddings for comparison
                recent_embs = np.array(list(speaker.embeddings))
                if len(recent_embs) == 0:
                    continue
                
                # Compare with average of recent embeddings
                avg_embedding = recent_embs.mean(axis=0)
                similarity = 1 - cdist([embedding], [avg_embedding], metric='cosine')[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = label
            
            if best_similarity >= self.similarity_threshold and best_speaker:
                # Known speaker
                return self._update_speaker(best_speaker, embedding, timestamp)
            
            elif best_similarity >= self.unknown_threshold:
                # Uncertain - could be known or unknown
                return self._handle_uncertain(embedding, timestamp, best_similarity, best_speaker)
            
            else:
                # New speaker
                return self._register_new_speaker(embedding, timestamp)
    
    def _register_new_speaker(self, embedding: np.ndarray, timestamp: float) -> str:
        """Register a new speaker"""
        self.speaker_counter += 1
        label = f"SPEAKER_{self.speaker_counter:03d}"
        
        speaker = SpeakerIdentity(
            label=label,
            first_seen=timestamp,
            last_seen=timestamp,
            embeddings=deque(maxlen=self.embedding_history),
            total_occurrences=1,
            audio_samples=deque(maxlen=5)
        )
        
        speaker.embeddings.append(embedding)
        self.known_speakers[label] = speaker
        
        print(f"🆕 New speaker detected: {label} at {timestamp:.2f}s")
        return label
    
    def _update_speaker(self, label: str, embedding: np.ndarray, timestamp: float) -> str:
        """Update existing speaker information"""
        speaker = self.known_speakers[label]
        speaker.last_seen = timestamp
        speaker.total_occurrences += 1
        speaker.embeddings.append(embedding)
        
        return label
    
    def _handle_uncertain(self, embedding: np.ndarray, timestamp: float, 
                          similarity: float, best_match: str) -> str:
        """Handle uncertain speaker identification"""
        # Check embedding quality
        embedding_quality = self._check_embedding_quality(embedding)
        
        if embedding_quality < 0.3:
            # Low quality embedding - likely noise
            return "UNKNOWN"
        
        # Could be a new speaker or degraded known speaker
        # Use temporal proximity as additional signal
        if best_match:
            time_gap = timestamp - self.known_speakers[best_match].last_seen
            if time_gap < 5.0:  # Recent speaker
                # Likely same speaker with degradation
                return self._update_speaker(best_match, embedding, timestamp)
        
        # Register as new speaker but mark as uncertain
        self.speaker_counter += 1
        label = f"SPEAKER_{self.speaker_counter:03d}?"
        
        speaker = SpeakerIdentity(
            label=label,
            first_seen=timestamp,
            last_seen=timestamp,
            embeddings=deque(maxlen=self.embedding_history),
            total_occurrences=1,
            audio_samples=deque(maxlen=5)
        )
        
        speaker.embeddings.append(embedding)
        self.known_speakers[label] = speaker
        
        return label
    
    def _check_embedding_quality(self, embedding: np.ndarray) -> float:
        """Check if embedding is high quality"""
        # Simple quality metric based on embedding statistics
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 1e-6:
            return 0.0
        
        # Normalize and check variance
        normalized = embedding / embedding_norm
        variance = np.var(normalized)
        
        return min(variance * 10, 1.0)  # Scale to 0-1
    
    def _cleanup_inactive_speakers(self, current_time: float):
        """Remove speakers inactive for too long"""
        inactive = []
        for label, speaker in self.known_speakers.items():
            if current_time - speaker.last_seen > self.inactivity_timeout:
                inactive.append(label)
        
        for label in inactive:
            print(f"⏰ Speaker {label} timed out (last seen {current_time - self.known_speakers[label].last_seen:.1f}s ago)")
            del self.known_speakers[label]
    
    def get_speaker_statistics(self) -> Dict:
        """Get current speaker statistics"""
        with self.lock:
            stats = {}
            for label, speaker in self.known_speakers.items():
                stats[label] = {
                    'total_occurrences': speaker.total_occurrences,
                    'first_seen': speaker.first_seen,
                    'last_seen': speaker.last_seen,
                    'active_duration': speaker.last_seen - speaker.first_seen,
                    'embeddings_count': len(speaker.embeddings)
                }
            return stats
    
    def merge_speakers(self, label1: str, label2: str):
        """Manually merge two speakers (e.g., if system split one person)"""
        with self.lock:
            if label1 not in self.known_speakers or label2 not in self.known_speakers:
                return
            
            # Keep the older speaker
            speaker1 = self.known_speakers[label1]
            speaker2 = self.known_speakers[label2]
            
            # Merge embeddings
            speaker1.embeddings.extend(speaker2.embeddings)
            speaker1.total_occurrences += speaker2.total_occurrences
            speaker1.last_seen = max(speaker1.last_seen, speaker2.last_seen)
            speaker1.first_seen = min(speaker1.first_seen, speaker2.first_seen)
            
            # Remove second speaker
            del self.known_speakers[label2]
            print(f"🔗 Merged {label2} into {label1}")
    
    def build_speaker_segments(self):
        """Build continuous speaker segments from timeline"""
        if not self.timeline:
            return
        
        segments = []
        current_speaker = self.timeline[0][1]
        start_time = self.timeline[0][0]
        
        for timestamp, speaker in self.timeline[1:]:
            if speaker != current_speaker:
                # End current segment
                duration = timestamp - start_time
                if duration > 0.5:  # Ignore very short segments
                    segments.append(SpeakerSegment(
                        speaker=current_speaker,
                        start_time=start_time,
                        end_time=timestamp,
                        duration=duration
                    ))
                # Start new segment
                current_speaker = speaker
                start_time = timestamp
        
        # Add final segment
        if self.timeline:
            final_duration = self.timeline[-1][0] - start_time
            if final_duration > 0.5:
                segments.append(SpeakerSegment(
                    speaker=current_speaker,
                    start_time=start_time,
                    end_time=self.timeline[-1][0],
                    duration=final_duration
                ))
        
        self.speaker_segments = segments
        return segments
    
    def save_all_results(self, audio_filename: str = None):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n💾 Saving results to {self.output_dir}/")
        
        # 1. Save speaker statistics (JSON)
        stats = self.get_speaker_statistics()
        stats_file = self.output_dir / f"speaker_statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Speaker statistics saved to {stats_file}")
        
        # 2. Save speaker segments (CSV)
        if not self.speaker_segments:
            self.build_speaker_segments()
        
        segments_file = self.output_dir / f"speaker_segments_{timestamp}.csv"
        with open(segments_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Speaker', 'Start Time (s)', 'End Time (s)', 'Duration (s)'])
            for segment in self.speaker_segments:
                writer.writerow([
                    segment.speaker,
                    f"{segment.start_time:.3f}",
                    f"{segment.end_time:.3f}",
                    f"{segment.duration:.3f}"
                ])
        print(f"✅ Speaker segments saved to {segments_file}")
        
        # 3. Save timeline (CSV)
        timeline_file = self.output_dir / f"speaker_timeline_{timestamp}.csv"
        with open(timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp (s)', 'Speaker'])
            for ts, speaker in self.timeline:
                writer.writerow([f"{ts:.3f}", speaker])
        print(f"✅ Timeline saved to {timeline_file}")
        
        # 4. Save speaker embeddings (NumPy)
        embeddings_file = self.output_dir / f"speaker_embeddings_{timestamp}.npz"
        embeddings_data = {}
        for label, speaker in self.known_speakers.items():
            embeddings_data[label] = np.mean([emb for emb in speaker.embeddings], axis=0)
            embeddings_data[f"{label}_all"] = np.array([emb for emb in speaker.embeddings])
        np.savez(embeddings_file, **embeddings_data)
        print(f"✅ Embeddings saved to {embeddings_file}")
        
        # 5. Save full speaker database (Pickle for reloading)
        database_file = self.output_dir / f"speaker_database_{timestamp}.pkl"
        database = {}
        for label, speaker in self.known_speakers.items():
            database[label] = speaker.to_serializable()
        
        with open(database_file, 'wb') as f:
            pickle.dump(database, f)
        print(f"✅ Speaker database saved to {database_file}")
        
        # 6. Save processing log (JSON Lines)
        log_file = self.output_dir / f"processing_log_{timestamp}.jsonl"
        with open(log_file, 'w') as f:
            for entry in self.processing_log:
                f.write(json.dumps(entry) + '\n')
        print(f"✅ Processing log saved to {log_file}")
        
        # 7. Save summary report (Text)
        report_file = self.output_dir / f"summary_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SPEAKER LABELING SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            if audio_filename:
                f.write(f"Audio File: {audio_filename}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {self.timeline[-1][0] if self.timeline else 0:.2f}s\n\n")
            
            f.write("SPEAKER STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for speaker, info in stats.items():
                f.write(f"\nSpeaker: {speaker}\n")
                f.write(f"  Occurrences: {info['total_occurrences']}\n")
                f.write(f"  Active Duration: {info['active_duration']:.2f}s\n")
                f.write(f"  First Seen: {info['first_seen']:.2f}s\n")
                f.write(f"  Last Seen: {info['last_seen']:.2f}s\n")
            
            f.write("\n\nSPEAKER SEGMENTS:\n")
            f.write("-" * 40 + "\n")
            for segment in self.speaker_segments:
                f.write(f"[{segment.start_time:6.2f}s - {segment.end_time:6.2f}s] "
                       f"{segment.speaker:15s} ({segment.duration:.2f}s)\n")
        
        print(f"✅ Summary report saved to {report_file}")
        
        return {
            'statistics': str(stats_file),
            'segments': str(segments_file),
            'timeline': str(timeline_file),
            'embeddings': str(embeddings_file),
            'database': str(database_file),
            'log': str(log_file),
            'report': str(report_file)
        }

# Real-time streaming simulation
class StreamingSimulator:
    """Simulates streaming audio for testing"""
    
    def __init__(self, audio_file: str, chunk_duration: float = 0.5):
        import soundfile as sf
        
        self.audio, self.sample_rate = sf.read(audio_file)
        if len(self.audio.shape) > 1:
            self.audio = self.audio.mean(axis=1)  # Convert to mono
        
        self.chunk_size = int(chunk_duration * self.sample_rate)
        self.position = 0
        
    def get_next_chunk(self) -> tuple:
        """Get next audio chunk"""
        if self.position >= len(self.audio):
            return None, None
        
        end_pos = min(self.position + self.chunk_size, len(self.audio))
        chunk = self.audio[self.position:end_pos]
        timestamp = self.position / self.sample_rate
        
        self.position = end_pos
        return chunk, timestamp

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
        description="Streaming speaker labeling with automatic result saving"
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
        "--similarity",
        type=float,
        default=0.75,
        help="Similarity threshold for same speaker"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Duration of streaming chunks in seconds"
    )

    args = parser.parse_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize streaming labeler
    labeler = StreamingSpeakerLabeler(
        similarity_threshold=args.similarity,
        unknown_threshold=0.65,
        embedding_history=10,
        inactivity_timeout=30.0,
        output_dir=output_dir
    )
    
    # Simulate streaming from a meeting recording
    stream = StreamingSimulator(audio_path, chunk_duration=args.chunk_duration)
    
    print("🎤 Starting streaming speaker labeling...")
    print("-" * 60)
    
    current_speaker = None
    speaker_start_time = 0
    
    while True:
        chunk, timestamp = stream.get_next_chunk()
        if chunk is None:
            break
        
        # Process chunk
        speaker_label = labeler.add_audio_chunk(chunk, timestamp)
        
        if speaker_label:
            # Track speaker changes
            if speaker_label != current_speaker:
                if current_speaker:
                    duration = timestamp - speaker_start_time
                    print(f"⏱️  {current_speaker} spoke for {duration:.2f}s")
                
                current_speaker = speaker_label
                speaker_start_time = timestamp
                print(f"🎯 [{timestamp:.2f}s] Speaker: {speaker_label}")
    
    # Build segments
    labeler.build_speaker_segments()
    
    # Print final statistics
    print("\n📊 Final Speaker Statistics:")
    stats = labeler.get_speaker_statistics()
    for speaker, info in stats.items():
        print(f"  {speaker}: {info['total_occurrences']} occurrences, "
              f"active for {info['active_duration']:.1f}s")
    
    # Save all results
    print("\n" + "=" * 60)
    saved_files = labeler.save_all_results(audio_filename=Path(audio_path).name)
    
    print("\n📁 All results saved successfully!")
    print(f"📂 Output directory: {output_dir}")
    print("\nGenerated files:")
    for file_type, filepath in saved_files.items():
        print(f"  • {file_type}: {Path(filepath).name}")
