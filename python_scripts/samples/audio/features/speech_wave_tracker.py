from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from speech_waves import WaveShapeConfig, WaveState, is_prominent_wave


@dataclass
class WaveStateTracker:
    """
    Tracks the lifecycle of a speech wave in streaming VAD scenarios.

    Maintains the same state machine as check_speech_waves() but operates
    incrementally as new probability values arrive.
    """

    # Configuration
    threshold: float = 0.5
    shape_cfg: Optional[WaveShapeConfig] = None

    # Internal state
    state: WaveState = "below"
    has_risen: bool = False
    has_multi_passed: bool = False
    has_fallen: bool = False
    is_valid: Optional[bool] = None  # None = still in progress, True/False = finalized

    # Wave tracking
    rise_frame_idx: Optional[int] = None
    fall_frame_idx: Optional[int] = None
    total_frames_seen: int = 0
    wave_probs: List[float] = field(default_factory=list)

    # Entry/exit probabilities for baseline calculation
    entry_prob: float = 0.0
    exit_prob: Optional[float] = None

    # Timing
    start_sec: float = 0.0
    end_sec: Optional[float] = None

    def __post_init__(self):
        if self.shape_cfg is None:
            self.shape_cfg = WaveShapeConfig()

    def process_prob(
        self, prob: float, sampling_rate: int = 16000, hop_size: int = 160
    ) -> dict:
        """
        Process a single probability value and update the wave state.

        Args:
            prob: Current frame's speech probability
            sampling_rate: Audio sampling rate (default 16000)
            hop_size: Hop size in samples (default 160 = 10ms)

        Returns:
            dict with current state information
        """
        frame_time_sec = self.total_frames_seen * hop_size / sampling_rate
        self.total_frames_seen += 1

        if self.state == "below":
            if prob >= self.threshold:
                # RISE DETECTED - Start new wave
                self.has_risen = True
                self.has_multi_passed = False
                self.has_fallen = False
                self.is_valid = None  # In progress

                self.rise_frame_idx = self.total_frames_seen - 1
                self.wave_probs = [prob]
                self.start_sec = frame_time_sec
                self.entry_prob = self._get_previous_prob()

                self.state = "above"

        elif self.state == "above":
            if prob >= self.threshold:
                # SUSTAINED SPEECH - Mark multi-pass
                if not self.has_multi_passed:
                    self.has_multi_passed = True
                self.wave_probs.append(prob)
            else:
                # FALL DETECTED - Finalize wave
                self.has_fallen = True
                self.fall_frame_idx = self.total_frames_seen - 1
                self.exit_prob = prob
                self.end_sec = frame_time_sec

                # Add final below-threshold prob to wave for completeness
                self.wave_probs.append(prob)

                # Validate the complete wave
                self._validate_wave()

                self.state = "below"

        return self.get_current_state(sampling_rate, hop_size)

    def process_probs(
        self, probs: List[float], sampling_rate: int = 16000, hop_size: int = 160
    ) -> List[dict]:
        """
        Process a batch of probability values.
        Returns list of state snapshots, one per processed probability.
        """
        states = []
        for prob in probs:
            state = self.process_prob(prob, sampling_rate, hop_size)
            states.append(state)
        return states

    def _get_previous_prob(self) -> float:
        """Get the probability just before the wave started."""
        # In streaming, you'd need to maintain a buffer of recent probs
        # For now, return 0.0 as default
        return 0.0

    def _validate_wave(self):
        """Apply shape and duration validation to the completed wave."""
        if not self.wave_probs or self.rise_frame_idx is None:
            self.is_valid = False
            return

        # Calculate wave metrics
        peak_prob = max(self.wave_probs)
        min_prob = min(self.wave_probs)
        entry_prob = self.entry_prob
        exit_prob = self.exit_prob if self.exit_prob is not None else 0.0
        baseline = (entry_prob + exit_prob) / 2.0
        prominence = peak_prob - baseline
        excursion = peak_prob - min_prob
        n_frames = len(self.wave_probs)

        # Duration check
        duration_sec = (self.end_sec or 0) - self.start_sec
        duration_ok = duration_sec >= self.shape_cfg.min_duration_sec

        # Shape validation
        shape_ok, _ = is_prominent_wave(
            self.wave_probs, entry_prob, exit_prob, self.shape_cfg
        )

        # Final validity
        self.is_valid = (
            self.has_risen and self.has_multi_passed and shape_ok and duration_ok
        )

    def get_current_state(
        self, sampling_rate: int = 16000, hop_size: int = 160
    ) -> dict:
        """
        Get comprehensive current state of the wave tracker.

        Returns a dict that's compatible with the SpeechWave format
        but also includes streaming-specific fields.
        """
        current_time = self.total_frames_seen * hop_size / sampling_rate

        state_info = {
            # Core flags
            "has_risen": self.has_risen,
            "has_multi_passed": self.has_multi_passed,
            "has_fallen": self.has_fallen,
            "is_valid": self.is_valid,
            # State machine
            "state": self.state,
            "is_active": self.state == "above",
            "is_complete": self.is_valid is not None,
            # Timing
            "start_sec": self.start_sec if self.has_risen else None,
            "end_sec": self.end_sec,
            "duration_sec": (
                (self.end_sec or current_time) - self.start_sec
                if self.has_risen
                else 0.0
            ),
            # Frame info
            "frame_start": self.rise_frame_idx,
            "frame_end": self.fall_frame_idx,
            "total_frames_seen": self.total_frames_seen,
            "wave_frames": len(self.wave_probs),
            # Probability stats (available during wave)
            "current_prob": self.wave_probs[-1] if self.wave_probs else None,
            "min_prob": min(self.wave_probs) if self.wave_probs else 0.0,
            "max_prob": max(self.wave_probs) if self.wave_probs else 0.0,
            "avg_prob": (statistics.mean(self.wave_probs) if self.wave_probs else 0.0),
        }

        # Add validation details if wave is complete
        if self.is_valid is not None and self.wave_probs:
            entry_prob = self.entry_prob
            exit_prob = self.exit_prob if self.exit_prob is not None else 0.0
            baseline = (entry_prob + exit_prob) / 2.0
            peak_prob = max(self.wave_probs)
            min_prob = min(self.wave_probs)

            state_info.update(
                {
                    "baseline": baseline,
                    "prominence": peak_prob - baseline,
                    "excursion": peak_prob - min_prob,
                    "peak_prob": peak_prob,
                    "duration_ok": (
                        (self.end_sec or 0) - self.start_sec
                        >= self.shape_cfg.min_duration_sec
                    ),
                }
            )

            # Compute composite score for valid waves
            if self.is_valid:
                state_info["composite_score"] = self._compute_composite_score()

        return state_info

    def _compute_composite_score(self) -> float:
        """Compute composite score (same as original _compute_composite_score)."""
        if not self.wave_probs:
            return 0.0

        avg_prob = statistics.mean(self.wave_probs)
        peak_prob = max(self.wave_probs)
        entry_prob = self.entry_prob
        exit_prob = self.exit_prob if self.exit_prob is not None else 0.0
        baseline = (entry_prob + exit_prob) / 2.0
        prominence = peak_prob - baseline
        excursion = peak_prob - min(self.wave_probs)
        duration_sec = (self.end_sec or 0) - self.start_sec

        return (
            avg_prob * prominence * math.log1p(duration_sec) * (1.0 + 0.3 * excursion)
        )

    def reset(self):
        """Reset tracker for a new audio stream."""
        self.state = "below"
        self.has_risen = False
        self.has_multi_passed = False
        self.has_fallen = False
        self.is_valid = None
        self.rise_frame_idx = None
        self.fall_frame_idx = None
        self.total_frames_seen = 0
        self.wave_probs = []
        self.entry_prob = 0.0
        self.exit_prob = None
        self.start_sec = 0.0
        self.end_sec = None


# Example usage for streaming VAD
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from file_utils import save_file

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract and analyse speech waves from audio using FireRedVAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument(
        "probs_path",
        nargs="?",
        default=None,
        help="Input probs json file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help="Output results directory.",
    )

    args = parser.parse_args()

    # Simulate streaming probabilities
    streaming_probs = [
        0.1,
        0.2,
        0.3,  # Below threshold
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.92,
        0.88,
        0.85,  # Rise and sustained speech
        0.4,
        0.2,  # Fall back below threshold
        0.1,
        0.1,
        0.1,  # Below again
        0.7,
        0.85,
        0.9,
        0.8,
        0.5,  # New short wave
        0.3,
        0.1,  # Quick fall
    ]

    tracker = WaveStateTracker(threshold=0.5)

    print("Streaming VAD Wave State Tracking")
    print("=" * 60)

    for i, prob in enumerate(streaming_probs):
        state = tracker.process_prob(prob)

        # Print state changes
        if state["has_risen"] and i > 0 and streaming_probs[i - 1] < 0.5:
            print(f"\n📈 WAVE START at frame {i}: prob={prob:.2f}")
        elif state["has_multi_passed"] and i > 0 and not streaming_probs[i - 1] < 0.5:
            if state["wave_frames"] == 2:  # Just got multi-pass
                print(f"✓ Multi-pass confirmed at frame {i}: prob={prob:.2f}")
        elif state["is_valid"] == True:
            print(
                f"✅ VALID WAVE COMPLETE: duration={state['duration_sec']:.2f}s, "
                f"frames={state['wave_frames']}, composite={state.get('composite_score', 0):.3f}"
            )
        elif state["is_valid"] == False:
            print(f"❌ INVALID WAVE: duration={state['duration_sec']:.2f}s")

        # Print every state
        print(
            f"Frame {i:3d}: prob={prob:.2f} | state={state['state']:5s} | "
            f"flags: R={state['has_risen']} M={state['has_multi_passed']} "
            f"F={state['has_fallen']} V={state['is_valid']}"
        )

    # Save final state
    save_file(tracker.get_current_state(), args.output_dir / "final_state.json")
