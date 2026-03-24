from __future__ import annotations
import numpy as np
import torch
import sounddevice as sd
import queue
import threading
import time
from pathlib import Path
from typing import List, Dict, Literal, Optional

from segment_speaker_labeler import SegmentSpeakerLabeler, AudioInput, SegmentResult


class LiveSpeakerLabeler:
    """
    Live / streaming speaker labeler that starts with zero known speakers
    and dynamically adds new speakers as they appear.

    Built on top of SegmentSpeakerLabeler for embedding extraction and similarity.
    Designed to work with sounddevice microphone streams or short audio chunks.
    """

    def __init__(
        self,
        embedding_model: str = "pyannote/embedding",
        hf_token: str | None = None,
        assignment_threshold: float = 0.68,
        new_speaker_threshold: float = 0.65,   # Slightly stricter than assignment for safety
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.assignment_threshold = assignment_threshold
        self.new_speaker_threshold = new_speaker_threshold

        # Initialize the base labeler in reference mode (we manage references ourselves)
        self.base = SegmentSpeakerLabeler(
            embedding_model=embedding_model,
            hf_token=hf_token,
            use_references=True,
            assignment_threshold=assignment_threshold,
            assignment_strategy="centroid",
            device=device,
            verbose=verbose,
        )

        # Dynamic memory: speaker_id -> normalized centroid
        self.speaker_centroids: Dict[int, np.ndarray] = {}
        self.next_speaker_id: int = 0

        if self.verbose:
            print("LiveSpeakerLabeler initialized – starting with 0 known speakers.")

    def _add_new_speaker(self, embedding: np.ndarray) -> int:
        """Add a new speaker and return its ID."""
        centroid = embedding.copy()  # already L2-normalized by base extractor
        speaker_id = self.next_speaker_id
        self.speaker_centroids[speaker_id] = centroid
        self.next_speaker_id += 1

        if self.verbose:
            print(f"→ New speaker detected: Speaker {speaker_id}")
        return speaker_id

    def _assign_or_add(self, embedding: np.ndarray) -> tuple[int, float]:
        """Assign to existing speaker or create new one. Returns (speaker_id, similarity)"""
        if not self.speaker_centroids:
            # First speaker ever
            sid = self._add_new_speaker(embedding)
            return sid, 1.0

        best_sim = -1.0
        best_id = -1

        for sid, cent in self.speaker_centroids.items():
            sim = float(np.dot(embedding, cent))
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        if best_sim >= self.assignment_threshold:
            return best_id, best_sim
        else:
            # New speaker
            sid = self._add_new_speaker(embedding)
            return sid, best_sim   # similarity to closest known (will be < threshold)

    def process_segment(self, audio: AudioInput) -> SegmentResult:
        """
        Process one speech segment (file path, waveform, bytes, etc.)
        Returns a SegmentResult dict with speaker_label.
        Dynamically updates internal speaker memory.
        """
        # Extract embedding (already L2-normalized)
        embeddings = self.base._extract_embeddings([audio])
        emb = embeddings[0]

        speaker_id, sim_to_closest = self._assign_or_add(emb)

        # For output compatibility with SegmentResult
        result: SegmentResult = {
            "path": str(audio) if isinstance(audio, (str, Path)) else "",
            "parent_dir": "",
            "speaker_label": speaker_id,
            "centroid_cosine_similarity": float(np.dot(emb, self.speaker_centroids[speaker_id])),
            "nearest_neighbor_cosine_similarity": np.nan,   # not meaningful in live mode
        }

        if self.verbose:
            print(f"Segment → Speaker {speaker_id} (sim={sim_to_closest:.3f})")

        return result

    def process_stream(
        self,
        duration_per_segment: float = 3.0,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        max_duration: float | None = None,
    ):
        """
        Simple blocking live stream processor using sounddevice.
        Records fixed-length segments continuously and labels them on-the-fly.
        Press Ctrl+C to stop.
        """
        print(f"Starting live microphone stream – segment length: {duration_per_segment}s")
        print("Speakers will be assigned dynamically starting from Speaker 0.")

        q: queue.Queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status)
            q.put(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='float32',
            blocksize=int(sample_rate * duration_per_segment),
            device=device,
            callback=audio_callback,
        )

        start_time = time.time()
        segment_count = 0

        with stream:
            try:
                while True:
                    if max_duration and (time.time() - start_time) > max_duration:
                        break

                    # Get one full segment
                    audio_chunk = q.get()          # shape: (frames, channels)
                    if channels == 1:
                        audio_chunk = audio_chunk.squeeze()

                    # Convert to torch format expected by pyannote
                    waveform = torch.from_numpy(audio_chunk).unsqueeze(0) if audio_chunk.ndim == 1 else torch.from_numpy(audio_chunk.T)
                    audio_input = {"waveform": waveform.float(), "sample_rate": sample_rate}

                    result = self.process_segment(audio_input)

                    segment_count += 1
                    print(f"[{segment_count:03d}] Speaker {result['speaker_label']} "
                          f"(centroid sim: {result['centroid_cosine_similarity']:.3f})")

            except KeyboardInterrupt:
                print("\nStopped by user.")
            finally:
                print(f"\nSession ended. Total speakers discovered: {len(self.speaker_centroids)}")
                self.print_known_speakers()

    def print_known_speakers(self):
        """Print summary of all discovered speakers."""
        print(f"\nKnown speakers ({len(self.speaker_centroids)} total):")
        for sid, cent in self.speaker_centroids.items():
            print(f"  Speaker {sid}: norm={np.linalg.norm(cent):.4f}")

    def get_known_speakers(self) -> Dict[int, np.ndarray]:
        """Return a copy of current speaker centroids."""
        return {k: v.copy() for k, v in self.speaker_centroids.items()}

    def save_known_speakers(self, folder: str | Path = "speaker_references"):
        """Optional: save each speaker's centroid as .npy for later reuse."""
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        for sid, cent in self.speaker_centroids.items():
            np.save(folder / f"speaker_{sid}.npy", cent)
        print(f"Saved {len(self.speaker_centroids)} speaker centroids to {folder}")


if __name__ == "__main__":
    # live_demo.py
    # from live_speaker_labeler import LiveSpeakerLabeler

    labeler = LiveSpeakerLabeler(
        verbose=True,
        assignment_threshold=0.68,
    )

    # Option 1: Process individual segments (files, numpy arrays, etc.)
    result = labeler.process_segment("some_speech.wav")
    print(result)

    # Option 2: Live microphone stream (recommended for sounddevice)
    # labeler.process_stream(
    #     duration_per_segment=3.0,   # seconds per chunk
    #     sample_rate=16000,
    #     max_duration=60,            # optional: stop after N seconds
    # )
