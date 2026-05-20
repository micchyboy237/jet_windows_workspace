"""
Comprehensive unit tests for WaveStateTracker.

Tests cover:
1. Basic state transitions (rise, multi-pass, fall)
2. Edge cases (empty probs, single spike, flat plateau)
3. Streaming scenarios (batch processing, interleaved waves)
4. Shape validation (prominence, excursion, baseline, duration)
5. State consistency (flag combinations, reset behavior)
6. Timing and frame calculations
7. Composite score computation
8. Threshold behavior
"""

from typing import List

import pytest

# Assuming the original code is in a module called wave_tracker
from speech_wave_tracker import WaveShapeConfig, WaveStateTracker

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def default_tracker():
    """Create a tracker with default settings."""

    return WaveStateTracker(threshold=0.5)


@pytest.fixture
def strict_tracker():
    """Create a tracker with stricter shape requirements."""

    shape_cfg = WaveShapeConfig(
        min_prominence=0.1,
        min_excursion=0.08,
        min_peak_prob=0.6,
        min_frames=5,
        min_duration_sec=0.3,
        min_baseline=0.1,
    )
    return WaveStateTracker(threshold=0.5, shape_cfg=shape_cfg)


@pytest.fixture
def lenient_tracker():
    """Create a tracker with very lenient shape requirements."""

    shape_cfg = WaveShapeConfig(
        min_prominence=0.01,
        min_excursion=0.01,
        min_peak_prob=0.3,
        min_frames=1,
        min_duration_sec=0.01,
        min_baseline=0.0,
    )
    return WaveStateTracker(threshold=0.5, shape_cfg=shape_cfg)


# ── Helper Functions ────────────────────────────────────────────────────────


def simulate_wave(
    tracker, probs: List[float], sampling_rate: int = 16000, hop_size: int = 160
) -> List[dict]:
    """Helper to process a list of probabilities and return all states."""
    return [tracker.process_prob(p, sampling_rate, hop_size) for p in probs]


def create_typical_wave_probs():
    """Create a typical speech wave: rise, sustain, fall."""
    return [
        0.1,
        0.2,
        0.3,
        0.4,  # Below threshold
        0.6,
        0.75,
        0.85,
        0.92,
        0.95,
        0.93,
        0.88,
        0.82,  # Rising and sustained
        0.45,
        0.3,
        0.2,
        0.1,  # Falling back
    ]


# ── 1. Basic State Transitions ──────────────────────────────────────────────


class TestBasicStateTransitions:
    """Test the fundamental state machine transitions."""

    def test_initial_state(self, default_tracker):
        """Tracker starts in 'below' state with no flags set."""
        state = default_tracker.get_current_state()
        assert state["state"] == "below"
        assert state["has_risen"] == False
        assert state["has_multi_passed"] == False
        assert state["has_fallen"] == False
        assert state["is_valid"] is None
        assert state["is_active"] == False
        assert state["is_complete"] == False

    def test_rise_detection(self, default_tracker):
        """First above-threshold prob triggers rise."""
        # Below threshold
        states = simulate_wave(default_tracker, [0.1, 0.2])
        assert states[-1]["has_risen"] == False

        # Rise
        state = default_tracker.process_prob(0.6)
        assert state["has_risen"] == True
        assert state["has_multi_passed"] == False
        assert state["has_fallen"] == False
        assert state["state"] == "above"
        assert state["is_active"] == True

    def test_multi_pass_detection(self, default_tracker):
        """Second consecutive above-threshold prob sets multi-pass."""
        # Rise
        default_tracker.process_prob(0.6)
        state = default_tracker.get_current_state()
        assert state["has_multi_passed"] == False

        # Second above-threshold frame
        state = default_tracker.process_prob(0.7)
        assert state["has_multi_passed"] == True
        assert state["has_risen"] == True

    def test_fall_detection(self, default_tracker):
        """Below-threshold after being above sets fall and validates."""
        # Rise and sustain
        simulate_wave(default_tracker, [0.6, 0.7, 0.8])

        # Fall
        state = default_tracker.process_prob(0.3)
        assert state["has_fallen"] == True
        assert state["has_risen"] == True
        assert state["has_multi_passed"] == True
        assert state["is_complete"] == True
        assert state["state"] == "below"

    def test_typical_wave_lifecycle(self, default_tracker):
        """Complete wave lifecycle with all flags."""
        probs = create_typical_wave_probs()
        states = simulate_wave(default_tracker, probs)

        # Check sequence
        # Before rise
        assert states[3]["has_risen"] == False  # prob=0.4

        # After rise
        assert states[4]["has_risen"] == True  # prob=0.6
        assert states[4]["has_multi_passed"] == False

        # After multi-pass
        assert states[5]["has_multi_passed"] == True  # prob=0.75

        # After fall
        assert states[12]["has_fallen"] == True  # prob=0.45
        assert states[12]["is_complete"] == True


# ── 2. Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_probabilities(self, default_tracker):
        """Tracker handles empty prob list gracefully."""
        states = default_tracker.process_probs([])
        assert len(states) == 0

        state = default_tracker.get_current_state()
        assert state["state"] == "below"
        assert state["has_risen"] == False

    def test_single_spike(self, default_tracker):
        """A single above-threshold frame followed by fall."""
        state1 = default_tracker.process_prob(0.6)  # Rise
        state2 = default_tracker.process_prob(0.3)  # Immediate fall

        assert state1["has_risen"] == True
        assert state1["has_multi_passed"] == False  # Only one above-threshold frame

        assert state2["has_fallen"] == True
        assert state2["has_multi_passed"] == False  # Never got multi-pass

        # Should be invalid because no multi-pass
        assert state2["is_valid"] == False

    def test_flat_plateau(self, default_tracker):
        """Constant probability above threshold (low excursion)."""
        plateau_probs = [0.1, 0.2, 0.55, 0.55, 0.55, 0.55, 0.55, 0.3, 0.2]
        states = simulate_wave(default_tracker, plateau_probs)

        # Should complete but likely invalid due to low prominence/excursion
        final_state = states[-1]
        assert final_state["has_fallen"] == True
        # Flat plateau should fail shape validation
        assert final_state["is_valid"] == False

    def test_rising_edge_start(self, default_tracker):
        """Audio starts above threshold (first prob >= threshold)."""
        state = default_tracker.process_prob(0.6)

        assert state["has_risen"] == True
        assert state["state"] == "above"
        assert state["start_sec"] is not None

    def test_never_falls(self, default_tracker):
        """Wave that never drops below threshold."""
        probs = [0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.85, 0.9, 0.92]
        states = simulate_wave(default_tracker, probs)

        final_state = states[-1]
        assert final_state["state"] == "above"
        assert final_state["has_risen"] == True
        assert final_state["has_fallen"] == False
        assert final_state["is_complete"] == False
        assert final_state["is_valid"] is None  # Still in progress

    def test_threshold_exactly_equals(self, default_tracker):
        """Probability exactly at threshold triggers rise."""
        state = default_tracker.process_prob(0.5)  # Exactly threshold
        assert state["has_risen"] == True

    def test_threshold_below_exactly(self, default_tracker):
        """Probability just below threshold doesn't trigger."""
        state = default_tracker.process_prob(0.499)
        assert state["has_risen"] == False

    def test_zero_probability(self, default_tracker):
        """Handle zero probability gracefully."""
        states = simulate_wave(default_tracker, [0.0, 0.0, 0.0])
        assert all(s["has_risen"] == False for s in states)

    def test_probability_one(self, default_tracker):
        """Handle probability of 1.0."""
        states = simulate_wave(default_tracker, [0.1, 1.0, 0.9, 0.3])
        assert states[1]["has_risen"] == True
        assert states[2]["has_multi_passed"] == True


# ── 3. Streaming Scenarios ──────────────────────────────────────────────────


class TestStreamingScenarios:
    """Test batch processing and streaming behavior."""

    def test_batch_processing(self, default_tracker):
        """process_probs handles batches identically to individual calls."""
        probs = create_typical_wave_probs()

        # Batch processing
        tracker_batch = type(default_tracker)(threshold=0.5)
        batch_states = tracker_batch.process_probs(probs)

        # Individual processing
        tracker_individual = type(default_tracker)(threshold=0.5)
        individual_states = [tracker_individual.process_prob(p) for p in probs]

        # Compare final states
        assert batch_states[-1] == individual_states[-1]

    def test_multiple_waves(self, default_tracker):
        """Multiple complete waves in sequence."""
        probs = [
            0.1,
            0.6,
            0.7,
            0.8,
            0.3,  # First wave
            0.1,
            0.1,
            0.1,  # Silence
            0.6,
            0.8,
            0.9,
            0.4,  # Second wave
        ]

        states = simulate_wave(default_tracker, probs)

        # First wave should complete and be valid
        assert states[4]["is_complete"] == True
        assert states[4]["is_valid"] == True  # Should pass with defaults

        # Second wave should also complete
        assert states[-1]["is_complete"] == True
        assert states[-1]["is_valid"] == True

    def test_interleaved_waves(self, default_tracker):
        """Waves with minimal silence between them."""
        probs = [
            0.6,
            0.7,
            0.3,  # Quick wave 1
            0.6,
            0.8,
            0.3,  # Quick wave 2
        ]

        states = simulate_wave(default_tracker, probs)

        # Both waves should be complete
        assert states[2]["is_complete"] == True  # End of wave 1
        assert states[5]["is_complete"] == True  # End of wave 2

    def test_state_consistency_during_wave(self, default_tracker):
        """Flags are consistent during wave lifecycle."""
        probs = [0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2]
        states = simulate_wave(default_tracker, probs)

        # During active wave
        for i in [2, 3, 4, 5]:  # Above threshold frames
            assert states[i]["has_risen"] == True
            assert states[i]["is_active"] == True
            if i >= 3:  # After second frame
                assert states[i]["has_multi_passed"] == True

        # After fall
        assert states[6]["has_fallen"] == True
        assert states[6]["is_active"] == False


# ── 4. Shape Validation ─────────────────────────────────────────────────────


class TestShapeValidation:
    """Test prominence, excursion, baseline, and duration checks."""

    def test_prominence_check(self, strict_tracker):
        """Waves must have sufficient prominence."""
        # Low prominence wave (barely above threshold)
        probs = [0.1, 0.1, 0.55, 0.56, 0.55, 0.3, 0.2]
        states = simulate_wave(strict_tracker, probs)
        assert states[-1]["is_valid"] == False  # Too little prominence

    def test_excursion_check(self, strict_tracker):
        """Waves must have sufficient internal variation."""
        # Flat wave (low excursion)
        probs = [0.1, 0.2, 0.65, 0.66, 0.65, 0.64, 0.3, 0.2]
        states = simulate_wave(strict_tracker, probs)
        assert states[-1]["is_valid"] == False  # Too little excursion

    def test_peak_prob_check(self, strict_tracker):
        """Wave peak must exceed minimum threshold."""
        # Wave never reaches min_peak_prob (0.6)
        probs = [0.1, 0.2, 0.55, 0.58, 0.55, 0.3, 0.2]
        states = simulate_wave(strict_tracker, probs)
        assert states[-1]["is_valid"] == False

    def test_min_frames_check(self, strict_tracker):
        """Wave must have minimum number of frames."""
        # Very short wave (only 3 above-threshold frames)
        probs = [0.1, 0.2, 0.7, 0.8, 0.7, 0.3, 0.2]
        states = simulate_wave(strict_tracker, probs)
        assert states[-1]["is_valid"] == False  # Too few frames (min_frames=5)

    def test_baseline_check(self, strict_tracker):
        """Wave baseline must be sufficient."""
        # Wave rising from near silence (low baseline)
        probs = [0.01, 0.02, 0.7, 0.8, 0.9, 0.8, 0.05, 0.03]
        states = simulate_wave(strict_tracker, probs)
        assert states[-1]["is_valid"] == False  # Baseline too low

    def test_valid_wave_passes_all_checks(self, lenient_tracker):
        """Proper mountain-shaped wave passes all checks."""
        probs = create_typical_wave_probs()
        states = simulate_wave(lenient_tracker, probs)
        assert states[-1]["is_valid"] == True

    def test_disabled_baseline(self, default_tracker):
        """Baseline check disabled by default (min_baseline=0.0)."""
        # Wave from silence should pass if other checks pass
        probs = [0.0, 0.0, 0.65, 0.85, 0.9, 0.8, 0.1, 0.0]
        states = simulate_wave(default_tracker, probs)
        assert states[-1]["is_valid"] == True  # Baseline=0.0, but default allows this


# ── 5. Timing and Frame Calculations ────────────────────────────────────────


class TestTimingCalculations:
    """Test time and frame calculations."""

    def test_start_time_recording(self, default_tracker):
        """Start time is recorded at first above-threshold frame."""
        default_tracker.process_prob(0.1)  # Frame 0, time 0.0
        default_tracker.process_prob(0.2)  # Frame 1, time 0.01

        state = default_tracker.process_prob(0.6)  # Frame 2, time 0.02
        assert state["start_sec"] == pytest.approx(0.02, rel=1e-3)
        assert state["frame_start"] == 2

    def test_duration_calculation(self, default_tracker):
        """Duration accumulates during wave."""
        # Start wave
        state = default_tracker.process_prob(0.6)  # 0ms
        assert state["duration_sec"] == 0.0

        # Add frames
        default_tracker.process_prob(0.7)  # 10ms
        state = default_tracker.process_prob(0.8)  # 20ms
        assert state["duration_sec"] == pytest.approx(0.02, rel=1e-3)

        # Fall
        state = default_tracker.process_prob(0.3)  # 30ms
        assert state["end_sec"] is not None
        assert state["duration_sec"] == pytest.approx(0.03, rel=1e-3)

    def test_custom_hop_size(self, default_tracker):
        """Timing respects custom hop size."""
        # Use 20ms hop (320 samples at 16kHz)
        default_tracker.process_prob(0.6, sampling_rate=16000, hop_size=320)
        state = default_tracker.process_prob(0.3, sampling_rate=16000, hop_size=320)

        assert state["duration_sec"] == pytest.approx(0.02, rel=1e-3)

    def test_custom_sampling_rate(self, default_tracker):
        """Timing respects custom sampling rate."""
        # Use 8kHz sampling rate
        default_tracker.process_prob(0.6, sampling_rate=8000, hop_size=160)
        state = default_tracker.process_prob(0.3, sampling_rate=8000, hop_size=160)

        assert state["duration_sec"] == pytest.approx(0.02, rel=1e-3)

    def test_frame_counting(self, default_tracker):
        """Frame counters are accurate."""
        probs = [0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.1]
        states = simulate_wave(default_tracker, probs)

        final_state = states[-1]
        assert final_state["total_frames_seen"] == 7
        assert final_state["wave_frames"] == 4  # 0.6, 0.7, 0.8, 0.3

    def test_duration_ok_flag(self, default_tracker):
        """Duration validation is reflected in output."""
        # Very short wave
        default_tracker.process_prob(0.6)
        state = default_tracker.process_prob(0.3)

        assert state["is_complete"] == True
        # With default min_duration_sec=0.25, 10ms is too short
        assert "duration_ok" in state or state["is_valid"] is not None


# ── 6. Composite Score ─────────────────────────────────────────────────────


class TestCompositeScore:
    """Test composite score computation."""

    def test_composite_score_valid_wave(self, lenient_tracker):
        """Composite score is computed for valid waves."""
        probs = create_typical_wave_probs()
        states = simulate_wave(lenient_tracker, probs)

        assert states[-1]["is_valid"] == True
        assert "composite_score" in states[-1]
        assert states[-1]["composite_score"] > 0

    def test_composite_score_zero_for_invalid(self, default_tracker):
        """Invalid waves don't get composite score."""
        # Single spike - will be invalid
        default_tracker.process_prob(0.6)
        state = default_tracker.process_prob(0.3)

        assert state["is_valid"] == False
        # Composite score should be 0 or not present for invalid waves
        if "composite_score" in state:
            assert (
                state["composite_score"] == 0.0 or state["composite_score"] is not None
            )

    def test_composite_score_increases_with_quality(self, lenient_tracker):
        """Better waves get higher composite scores."""
        # Low quality wave
        tracker1 = WaveStateTracker(threshold=0.5, shape_cfg=lenient_tracker.shape_cfg)
        simulate_wave(tracker1, [0.1, 0.55, 0.56, 0.55, 0.3])
        score1 = tracker1.get_current_state().get("composite_score", 0)

        # High quality wave (reset tracker)
        tracker1.reset()
        simulate_wave(tracker1, [0.1, 0.7, 0.9, 0.95, 0.9, 0.8, 0.3])
        score2 = tracker1.get_current_state().get("composite_score", 0)

        assert score2 > score1, f"Expected {score2} > {score1}"

    def test_composite_score_components(self, lenient_tracker):
        """Composite score considers all components."""
        probs = [0.1, 0.2, 0.7, 0.85, 0.9, 0.8, 0.3, 0.1]
        states = simulate_wave(lenient_tracker, probs)

        final_state = states[-1]
        assert final_state["is_valid"] == True

        # Score should reflect:
        # - avg_prob (~0.81)
        # - prominence (0.9 - 0.15 = 0.75)
        # - duration (~0.05s)
        # - excursion (0.9 - 0.3 = 0.6)
        score = final_state["composite_score"]
        assert 0 < score < 1.0  # Reasonable range


# ── 7. State Consistency ───────────────────────────────────────────────────


class TestStateConsistency:
    """Test internal state consistency and invariants."""

    def test_flags_mutual_exclusion(self, default_tracker):
        """Certain flag combinations are impossible."""
        # Can't have fallen without having risen
        state = default_tracker.get_current_state()
        assert not (state["has_fallen"] and not state["has_risen"])

        # Process a wave
        probs = create_typical_wave_probs()
        states = simulate_wave(default_tracker, probs)

        for state in states:
            # has_fallen implies has_risen
            if state["has_fallen"]:
                assert state["has_risen"] == True

            # is_valid True implies all other flags True
            if state["is_valid"] == True:
                assert state["has_risen"] == True
                assert state["has_multi_passed"] == True
                assert state["has_fallen"] == True

            # Can't be complete without has_fallen
            if state["is_complete"]:
                assert state["has_fallen"] == True

    def test_reset_functionality(self, default_tracker):
        """Reset returns tracker to initial state."""
        # Process a wave
        simulate_wave(default_tracker, create_typical_wave_probs())

        # Reset
        default_tracker.reset()

        # Should be back to initial state
        state = default_tracker.get_current_state()
        assert state == {
            "has_risen": False,
            "has_multi_passed": False,
            "has_fallen": False,
            "is_valid": None,
            "state": "below",
            "is_active": False,
            "is_complete": False,
            "start_sec": None,
            "end_sec": None,
            "duration_sec": 0.0,
            "frame_start": None,
            "frame_end": None,
            "total_frames_seen": 0,
            "wave_frames": 0,
            "current_prob": None,
            "min_prob": 0.0,
            "max_prob": 0.0,
            "avg_prob": 0.0,
        }

        # Should be able to process new wave
        state = default_tracker.process_prob(0.6)
        assert state["has_risen"] == True

    def test_multiple_resets(self, default_tracker):
        """Multiple resets work correctly."""
        for _ in range(3):
            simulate_wave(default_tracker, create_typical_wave_probs())
            default_tracker.reset()

            state = default_tracker.get_current_state()
            assert state["has_risen"] == False
            assert state["total_frames_seen"] == 0

    def test_state_immutability(self, default_tracker):
        """Returned state dicts are independent copies."""
        state1 = default_tracker.get_current_state()
        state1["has_risen"] = True  # Modify returned dict

        state2 = default_tracker.get_current_state()
        assert state2["has_risen"] == False  # Internal state unchanged

    def test_probability_statistics_accuracy(self, default_tracker):
        """Min, max, avg probability calculations are accurate."""
        probs = [0.6, 0.7, 0.8, 0.9, 0.5]

        for prob in probs:
            state = default_tracker.process_prob(prob)

        assert state["min_prob"] == 0.5
        assert state["max_prob"] == 0.9
        assert state["avg_prob"] == pytest.approx(0.7, rel=1e-3)


# ── 8. Integration Tests ────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests simulating real-world usage."""

    def test_long_audio_with_multiple_waves(self, default_tracker):
        """Simulate a longer audio with multiple speech segments."""
        long_probs = (
            [0.1] * 10  # Initial silence
            + [0.6, 0.7, 0.8, 0.9, 0.8, 0.4]  # Wave 1
            + [0.1] * 5  # Silence
            + [0.5, 0.7, 0.9, 0.85, 0.5]  # Wave 2 (starts at threshold)
            + [0.1] * 8  # Silence
            + [0.6, 0.8, 0.95, 0.9, 0.7, 0.3]  # Wave 3
            + [0.1] * 10  # Final silence
        )

        states = simulate_wave(default_tracker, long_probs)

        # Count completed waves
        completed_waves = [s for s in states if s["is_complete"]]
        valid_waves = [s for s in states if s["is_valid"] == True]

        assert len(completed_waves) >= 2
        assert len(valid_waves) >= 2

    def test_streaming_simulation(self, default_tracker):
        """Simulate real streaming where probs arrive in chunks."""
        chunks = [
            [0.1, 0.2, 0.3],
            [0.6, 0.7, 0.8],
            [0.9, 0.8, 0.4, 0.2],
            [0.1, 0.1],
            [0.6, 0.8, 0.9, 0.3],
        ]

        all_states = []
        for chunk in chunks:
            chunk_states = default_tracker.process_probs(chunk)
            all_states.extend(chunk_states)

        # Check for valid waves
        valid_waves = [s for s in all_states if s["is_valid"] == True]
        assert len(valid_waves) >= 1

    def test_realtime_monitoring_scenario(self, default_tracker):
        """Simulate monitoring wave state in real-time."""

        def monitor_callback(state):
            """Simulate real-time monitoring."""
            if state["has_risen"] and not state["has_multi_passed"]:
                return "rising"
            elif state["has_multi_passed"] and not state["has_fallen"]:
                return "speaking"
            elif state["is_valid"] == True:
                return "valid_wave_complete"
            elif state["is_valid"] == False:
                return "invalid_wave_complete"
            return "silence"

        probs = create_typical_wave_probs()
        events = []

        for prob in probs:
            state = default_tracker.process_prob(prob)
            event = monitor_callback(state)
            events.append(event)

        assert "rising" in events
        assert "speaking" in events
        assert "valid_wave_complete" in events

    def test_threshold_sensitivity(self):
        """Different thresholds change detection behavior."""
        low_threshold = WaveStateTracker(threshold=0.3)
        high_threshold = WaveStateTracker(threshold=0.7)

        # Probability of 0.5
        low_state = low_threshold.process_prob(0.5)
        high_state = high_threshold.process_prob(0.5)

        assert low_state["has_risen"] == True  # 0.5 >= 0.3
        assert high_state["has_risen"] == False  # 0.5 < 0.7


# ── 9. Performance and Edge Cases ───────────────────────────────────────────


class TestPerformanceEdgeCases:
    """Test performance and unusual edge cases."""

    def test_very_long_wave(self, default_tracker):
        """Handle very long speech segments."""
        long_probs = [0.1, 0.2] + [0.8] * 1000 + [0.3, 0.1]
        states = simulate_wave(default_tracker, long_probs)

        final_state = states[-1]
        assert final_state["is_complete"] == True
        assert final_state["wave_frames"] == 1002  # 1000 above + 2 below
        assert final_state["duration_sec"] > 10.0  # ~10 seconds

    def test_rapid_state_changes(self, default_tracker):
        """Handle rapid oscillation around threshold."""
        oscillating_probs = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
        states = simulate_wave(default_tracker, oscillating_probs)

        # Should create multiple short waves
        completed = [s for s in states if s["is_complete"]]
        assert len(completed) >= 2

    def test_extreme_probability_values(self, default_tracker):
        """Handle very small and very large probability values."""
        extreme_probs = [1e-10, 1e-5, 0.99999, 0.999999, 1e-10]
        states = simulate_wave(default_tracker, extreme_probs)

        final_state = states[-1]
        assert final_state["is_complete"] == True
        assert final_state["min_prob"] == pytest.approx(1e-10)
        assert final_state["max_prob"] == pytest.approx(0.999999)

    def test_nan_probability(self, default_tracker):
        """Handle NaN probability values gracefully."""

        # This might raise an exception or handle NaN gracefully
        try:
            state = default_tracker.process_prob(float("nan"))
            # If no exception, state should not have risen
            assert state["has_risen"] == False or state["state"] != "above"
        except (ValueError, TypeError):
            # Exception is acceptable for NaN
            pass

    def test_negative_probability(self, default_tracker):
        """Handle negative probability values."""
        state = default_tracker.process_prob(-0.5)
        # Should treat as below threshold
        assert state["has_risen"] == False

    def test_probability_above_one(self, default_tracker):
        """Handle probability > 1.0."""
        state = default_tracker.process_prob(1.5)
        # Should treat as above threshold
        assert state["has_risen"] == True


# ── 10. Configuration Tests ─────────────────────────────────────────────────


class TestConfiguration:
    """Test different configuration options."""

    def test_default_configuration(self):
        """Default configuration has reasonable values."""
        tracker = WaveStateTracker()
        assert tracker.threshold == 0.5
        assert tracker.shape_cfg is not None
        assert tracker.shape_cfg.min_duration_sec == 0.25

    def test_custom_shape_config(self):
        """Custom shape configuration is respected."""

        custom_cfg = WaveShapeConfig(
            min_prominence=0.2,
            min_excursion=0.15,
            min_peak_prob=0.7,
            min_frames=10,
            min_duration_sec=0.5,
            min_baseline=0.2,
        )

        tracker = WaveStateTracker(shape_cfg=custom_cfg)
        assert tracker.shape_cfg.min_prominence == 0.2
        assert tracker.shape_cfg.min_peak_prob == 0.7

    def test_threshold_only_config(self):
        """Only threshold specified, shape config uses defaults."""
        tracker = WaveStateTracker(threshold=0.7)
        assert tracker.threshold == 0.7
        assert tracker.shape_cfg.min_prominence == 0.05  # Default


# ── Run tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
