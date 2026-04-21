import shutil
from pathlib import Path
from typing import List

import librosa
import numpy as np
from rich.console import Console
from ten_vad import TenVad
from vad_tenvad import extract_speech_timestamps

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def extract_pauses_with_vad(audio, threshold=0.5, min_pause_duration_ms=200):
    """Extract pauses using VAD probabilities."""
    segments, probs = extract_speech_timestamps(
        audio,
        include_non_speech=True,
        threshold=threshold,
        min_silence_duration_ms=min_pause_duration_ms,
        with_scores=True,
    )

    # Filter only non-speech segments (pauses)
    pauses = [seg for seg in segments if seg["type"] == "non-speech"]

    # Add pause metrics
    for pause in pauses:
        pause["rms_energy"] = np.mean(
            [
                probs[frame]
                for frame in range(pause["frame_start"], pause["frame_end"] + 1)
            ]
        )

    return pauses


def detect_pauses_by_energy(
    audio: np.ndarray,
    sr: int = 16000,
    hop_size: int = 256,
    energy_threshold: float = 0.01,
    min_pause_duration_ms: int = 200,
) -> List[dict]:
    """
    Detect pauses using RMS energy threshold.
    More reliable for detecting absolute silence vs. VAD's speech/non-speech.
    """
    # Calculate RMS for each frame
    num_frames = len(audio) // hop_size
    rms_values = []

    for i in range(num_frames):
        frame = audio[i * hop_size : (i + 1) * hop_size]
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)

    # Convert to dB for better thresholding
    rms_db = 20 * np.log10(np.maximum(rms_values, 1e-10))
    silence_threshold_db = 20 * np.log10(energy_threshold)

    # Find silence regions
    frame_duration_ms = (hop_size / sr) * 1000
    min_pause_frames = int(min_pause_duration_ms / frame_duration_ms)

    pauses = []
    in_pause = False
    pause_start = 0

    for i, rms in enumerate(rms_values):
        is_silent = rms < energy_threshold

        if is_silent and not in_pause:
            in_pause = True
            pause_start = i
        elif not is_silent and in_pause:
            in_pause = False
            pause_duration_frames = i - pause_start
            if pause_duration_frames >= min_pause_frames:
                pauses.append(
                    {
                        "start": pause_start * frame_duration_ms / 1000,
                        "end": i * frame_duration_ms / 1000,
                        "duration": pause_duration_frames * frame_duration_ms / 1000,
                        "avg_energy": np.mean(rms_values[pause_start:i]),
                        "min_energy": np.min(rms_values[pause_start:i]),
                    }
                )

    return pauses


class PauseDetector:
    def __init__(
        self,
        vad_threshold: float = 0.5,
        energy_threshold: float = 0.01,
        min_pause_ms: int = 200,
    ):
        self.vad_threshold = vad_threshold
        self.energy_threshold = energy_threshold
        self.min_pause_ms = min_pause_ms

    def detect_pauses(self, audio: AudioInput) -> List[dict]:
        """Combine VAD and energy-based detection for robust pause detection."""

        # Get VAD-based non-speech segments
        segments, probs = extract_speech_timestamps(
            audio,
            include_non_speech=True,
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.min_pause_ms,
            with_scores=True,
        )

        audio_np, sr = load_audio(audio, sr=16000, mono=True)

        # For each non-speech segment, verify with energy
        pauses = []
        for seg in segments:
            if seg["type"] != "non-speech":
                continue

            # Extract audio for this segment
            start_sample = seg["frame_start"] * 256
            end_sample = (seg["frame_end"] + 1) * 256
            segment_audio = audio_np[start_sample:end_sample]

            # Calculate energy metrics
            rms = np.sqrt(np.mean(segment_audio**2))
            peak = np.max(np.abs(segment_audio))

            # Classify pause type
            if rms < 0.001:
                pause_type = "absolute_silence"
                confidence = 1.0
            elif rms < self.energy_threshold:
                pause_type = "quiet_pause"
                confidence = 0.9
            else:
                pause_type = "vad_detected_pause"
                confidence = seg["prob"]

            pauses.append(
                {
                    **seg,
                    "pause_type": pause_type,
                    "confidence": confidence,
                    "rms_energy": float(rms),
                    "peak_amplitude": float(peak),
                    "zero_crossing_rate": self._calculate_zcr(segment_audio),
                }
            )

        return pauses

    def _calculate_zcr(self, audio: np.ndarray) -> float:
        """Zero crossing rate - useful for distinguishing noise from speech."""
        return float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio)))


def detect_pauses_spectral(
    audio: np.ndarray, sr: int = 16000, n_fft: int = 512, hop_length: int = 256
) -> List[dict]:
    """
    Use spectral features to detect pauses vs. background noise.
    """
    # Compute spectral features
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Spectral flatness (high for noise, low for speech)
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]

    # Spectral centroid (higher for speech)
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]

    # RMS energy
    rms = librosa.feature.rms(S=S)[0]

    # Combined pause score
    pause_score = (
        spectral_flatness * 0.4  # High flatness = noise/pause
        + (1 - rms / rms.max()) * 0.4  # Low energy = pause
        + (1 - spectral_centroid / spectral_centroid.max())
        * 0.2  # Low centroid = pause
    )

    # Threshold to find pauses
    pause_threshold = 0.7
    frame_duration = hop_length / sr

    pauses = []
    in_pause = False
    pause_start = 0

    for i, score in enumerate(pause_score):
        if score > pause_threshold and not in_pause:
            in_pause = True
            pause_start = i
        elif score <= pause_threshold and in_pause:
            in_pause = False
            duration = (i - pause_start) * frame_duration
            if duration > 0.2:  # 200ms minimum
                pauses.append(
                    {
                        "start": pause_start * frame_duration,
                        "end": i * frame_duration,
                        "duration": duration,
                        "avg_pause_score": float(np.mean(pause_score[pause_start:i])),
                    }
                )

    return pauses


class RealTimePauseDetector:
    def __init__(self, sr: int = 16000, buffer_duration: float = 1.0):
        self.sr = sr
        self.buffer_size = int(sr * buffer_duration)
        self.buffer = np.zeros(self.buffer_size)
        self.vad = TenVad()
        self.consecutive_silence_frames = 0
        self.min_pause_frames = 10  # ~160ms at 16kHz

    def process_frame(self, audio_frame: np.ndarray) -> dict:
        """Process incoming audio frame and detect pauses."""
        # Update circular buffer
        self.buffer = np.roll(self.buffer, -len(audio_frame))
        self.buffer[-len(audio_frame) :] = audio_frame

        # VAD probability
        prob, flag = self.vad.process(audio_frame)

        # Energy check
        rms = np.sqrt(np.mean(audio_frame**2))

        is_pause = prob < 0.5 and rms < 0.01

        if is_pause:
            self.consecutive_silence_frames += 1
        else:
            self.consecutive_silence_frames = 0

        return {
            "is_pause": is_pause,
            "pause_duration": self.consecutive_silence_frames
            * len(audio_frame)
            / self.sr,
            "vad_prob": prob,
            "rms": rms,
            "in_extended_pause": self.consecutive_silence_frames
            >= self.min_pause_frames,
        }


if __name__ == "__main__":
    import argparse

    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", 
        nargs="?",                    # Makes the argument optional
        default=DEFAULT_AUDIO,        # Sets the default value
        help=f"Input audio file path (default: {DEFAULT_AUDIO})"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output results dir (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="VAD probability threshold"
    )
