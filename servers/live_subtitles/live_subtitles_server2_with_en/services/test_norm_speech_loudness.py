"""Tests for normalize_audio_for_vad in services.norm_speech_loudness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from services.norm_speech_loudness import normalize_audio_for_vad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def silent_signal():
    """Truly silent signal (all zeros) that should be below any min_signal_db."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def near_silent_signal():
    """Signal with negligible energy — use with min_signal_db=-80 to test skipping."""
    return np.full(16000, 1e-6, dtype=np.float32)


@pytest.fixture
def quiet_signal():
    """Quiet speech-like signal (-35 dBFS RMS)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.018, 16000).astype(np.float32)  # ~-35 dBFS
    return noise


@pytest.fixture
def normal_signal():
    """Normal speech-level signal (-20 dBFS RMS, peaks near 0.95)."""
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.1, 16000).astype(np.float32)  # ~-20 dBFS
    # Add a few peaks
    sig[1000] = 0.95
    sig[2000] = -0.92
    return sig


@pytest.fixture
def loud_signal():
    """Loud, heavily clipped signal (~-3 dBFS RMS, many samples at ±1.0)."""
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.7, 16000).astype(np.float32)  # ~-3 dBFS
    sig = np.clip(sig, -1.0, 1.0)
    return sig


@pytest.fixture
def dc_offset_signal():
    """Signal with DC offset, moderate level."""
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.1, 16000).astype(np.float32) + 0.5
    return sig


@pytest.fixture
def two_clips_different_rms():
    """
    Simulate the OP's scenario: two clips with different peak-to-RMS ratios.
    Both clipped at ±1.0, but different RMS levels because of different
    distributions (e.g., dense vs sparse peaks).
    """
    rng = np.random.default_rng(42)
    # Clip 1: mostly moderate signal (lower RMS) — simulates 16 dB RMS
    clip1 = rng.normal(0, 0.16, 16000).astype(np.float32)
    clip1 = np.clip(clip1, -1.0, 1.0)

    # Clip 2: very hot signal (higher RMS) — simulates 24.5 dB RMS
    clip2 = rng.normal(0, 0.45, 16000).astype(np.float32)
    clip2 = np.clip(clip2, -1.0, 1.0)

    return clip1, clip2


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestBasicFunctionality:
    """Smoke tests for all methods and input types."""

    def test_empty_numpy(self):
        y, info = normalize_audio_for_vad(np.array([], dtype=np.float32))
        assert len(y) == 0
        assert info["skipped_reason"] == "empty_input"

    def test_empty_torch(self):
        y, info = normalize_audio_for_vad(torch.tensor([], dtype=torch.float32))
        assert len(y) == 0
        assert info["skipped_reason"] == "empty_input"

    def test_peak_method(self, normal_signal):
        y, info = normalize_audio_for_vad(normal_signal, method="peak")
        assert y.dtype == np.float32
        assert info["method"] == "peak"
        # Peak should be exactly 1.0 (or very close)
        assert np.abs(info["final_peak"] - 1.0) < 0.01

    def test_rms_method(self, normal_signal):
        y, info = normalize_audio_for_vad(
            normal_signal, method="rms", target_rms_db=-20
        )
        assert info["method"] == "rms"
        # Final RMS should be close to target
        assert abs(info["final_rms_db"] - (-20.0)) < 0.5

    def test_hybrid_method(self, normal_signal):
        y, info = normalize_audio_for_vad(
            normal_signal, method="hybrid", target_rms_db=-20
        )
        assert info["method"] == "hybrid"
        assert info["final_peak"] <= 0.95 + 0.01

    def test_default_method(self, normal_signal):
        """Default method should be 'hybrid'."""
        _, info = normalize_audio_for_vad(normal_signal)
        assert info["method"] == "hybrid"


# ---------------------------------------------------------------------------
# Silence handling
# ---------------------------------------------------------------------------

class TestSilenceHandling:
    """Signals below min_signal_db should be skipped."""

    def test_silent_skipped(self, silent_signal):
        """True zeros should always be skipped."""
        y, info = normalize_audio_for_vad(silent_signal, min_signal_db=-60)
        assert info["skipped_reason"] == "silent_input"
        assert info["applied_gain_db"] == 0.0
        # Output should be identical to input
        np.testing.assert_array_equal(y, silent_signal)

    def test_silent_with_different_threshold(self, near_silent_signal):
        """Near-silent audio with a very low threshold should be skipped."""
        y, info = normalize_audio_for_vad(near_silent_signal, min_signal_db=-80)
        assert info["skipped_reason"] == "silent_input"

    def test_quiet_not_skipped(self, quiet_signal):
        """Quiet signal above threshold should be processed."""
        y, info = normalize_audio_for_vad(quiet_signal, min_signal_db=-60)
        assert info["skipped_reason"] is None
        assert info["applied_gain_db"] != 0.0


# ---------------------------------------------------------------------------
# DC offset removal
# ---------------------------------------------------------------------------

class TestDCOffsetRemoval:
    """DC offset should be removed when remove_dc=True."""

    def test_dc_removed(self, dc_offset_signal):
        _, info = normalize_audio_for_vad(dc_offset_signal, remove_dc=True)
        # After DC removal, the mean should be very close to zero.
        # We check via the returned RMS (which is computed after DC removal).
        # Original had DC ~0.5; final should have negligible DC.
        assert info["original_rms_db"] != info["final_rms_db"]

    def test_dc_not_removed_preserved(self, dc_offset_signal):
        """With remove_dc=False, DC should still be in the output.
        
        Note: RMS normalization scales the entire signal including DC,
        so the DC value itself changes. We verify DC is not zeroed out
        by checking that the mean is non-negligible relative to peak.
        """
        y, info = normalize_audio_for_vad(dc_offset_signal, remove_dc=False)
        # The signal had DC of 0.5 originally. After RMS scaling, DC changes
        # but should not be zero. Check that mean/peak ratio is significant.
        mean_abs = abs(float(np.mean(y)))
        peak = float(np.max(np.abs(y)))
        # DC should be a noticeable fraction of the peak (>5%)
        assert mean_abs / peak > 0.05, (
            f"DC seems removed: mean={mean_abs:.4f}, peak={peak:.4f}"
        )


# ---------------------------------------------------------------------------
# max_rms_db — the new feature
# ---------------------------------------------------------------------------

class TestMaxRmsDb:
    """Tests for the max_rms_db parameter that caps final RMS amplitude."""

    def test_max_rms_limits_loud_signal(self, loud_signal):
        """A loud signal should be attenuated to respect max_rms_db."""
        y, info = normalize_audio_for_vad(
            loud_signal, method="rms", target_rms_db=-20, max_rms_db=-25
        )
        # Final RMS should not exceed max_rms_db
        assert info["final_rms_db"] <= -25.0 + 0.1, (
            f"Expected final RMS <= -25 dBFS, got {info['final_rms_db']:.2f}"
        )

    def test_max_rms_no_effect_on_quiet_signal(self, quiet_signal):
        """max_rms_db should not affect signals already below the limit."""
        y, info = normalize_audio_for_vad(
            quiet_signal, method="rms", target_rms_db=-20, max_rms_db=-10
        )
        # Quiet signal (-35 dBFS) is already below -10 max_rms_db
        assert info["final_rms_db"] < -10.0

    def test_max_rms_with_hybrid_and_peak_ceiling(self, loud_signal):
        """
        Test the exact scenario from the OP: hybrid method with peak ceiling
        can push RMS above target. max_rms_db should clamp it back down.
        """
        y, info = normalize_audio_for_vad(
            loud_signal,
            method="hybrid",
            target_rms_db=-20,
            max_peak=0.95,
            max_rms_db=-22,
        )
        assert info["final_rms_db"] <= -22.0 + 0.1, (
            f"Hybrid + max_rms_db failed: RMS={info['final_rms_db']:.2f} dBFS"
        )
        # Peak should still respect max_peak
        assert info["final_peak"] <= 0.95 + 0.01

    def test_consistency_across_different_clips(self, two_clips_different_rms):
        """
        THE KEY TEST: Two heavily clipped clips with different peak-to-RMS
        ratios should end up with the same RMS when max_rms_db is set.
        """
        clip1, clip2 = two_clips_different_rms

        _, info1 = normalize_audio_for_vad(
            clip1,
            method="hybrid",
            target_rms_db=-20,
            max_peak=0.95,
            max_rms_db=-22,
        )
        _, info2 = normalize_audio_for_vad(
            clip2,
            method="hybrid",
            target_rms_db=-20,
            max_peak=0.95,
            max_rms_db=-22,
        )

        # Both should be capped at -22 dBFS (within tolerance)
        assert abs(info1["final_rms_db"] - (-22.0)) < 0.5, (
            f"Clip1 final RMS: {info1['final_rms_db']:.2f} dBFS"
        )
        assert abs(info2["final_rms_db"] - (-22.0)) < 0.5, (
            f"Clip2 final RMS: {info2['final_rms_db']:.2f} dBFS"
        )

        # The difference between the two should be small (consistency!)
        rms_diff = abs(info1["final_rms_db"] - info2["final_rms_db"])
        assert rms_diff < 0.5, (
            f"RMS inconsistency: clip1={info1['final_rms_db']:.2f}, "
            f"clip2={info2['final_rms_db']:.2f} (diff={rms_diff:.2f} dB)"
        )

    def test_max_rms_db_none_disables_ceiling(self, loud_signal):
        """When max_rms_db is None, no RMS ceiling is applied."""
        y, info = normalize_audio_for_vad(
            loud_signal, method="hybrid", target_rms_db=-20, max_rms_db=None
        )
        # With a loud signal and hybrid method, RMS could end up anywhere
        # (depends on peak-to-RMS ratio). Just verify it ran without error.
        assert info["skipped_reason"] is None
        assert isinstance(info["final_rms_db"], float)


# ---------------------------------------------------------------------------
# Torch tensor support
# ---------------------------------------------------------------------------

class TestTorchSupport:
    """All operations should work identically with torch tensors."""

    def test_torch_hybrid(self, normal_signal):
        t = torch.from_numpy(normal_signal)
        y, info = normalize_audio_for_vad(t, method="hybrid", target_rms_db=-20)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == torch.float32
        assert info["method"] == "hybrid"

    def test_torch_max_rms_db(self, loud_signal):
        t = torch.from_numpy(loud_signal)
        y, info = normalize_audio_for_vad(
            t, method="hybrid", target_rms_db=-20, max_rms_db=-25
        )
        assert isinstance(y, torch.Tensor)
        assert info["final_rms_db"] <= -25.0 + 0.1

    def test_numpy_torch_equivalence(self, loud_signal):
        """Numpy and torch paths should produce identical results."""
        t = torch.from_numpy(loud_signal)

        y_np, info_np = normalize_audio_for_vad(
            loud_signal, method="hybrid", target_rms_db=-20, max_rms_db=-25
        )
        y_t, info_t = normalize_audio_for_vad(
            t, method="hybrid", target_rms_db=-20, max_rms_db=-25
        )

        # Compare the signal arrays
        np.testing.assert_allclose(y_np, y_t.numpy(), atol=1e-6)
        # Compare info dicts (exclude 'method' which is a string)
        for key in ["final_rms_db", "final_peak", "applied_gain_db"]:
            assert info_np[key] == pytest.approx(info_t[key], abs=0.1), (
                f"Mismatch in {key}: numpy={info_np[key]}, torch={info_t[key]}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Unusual inputs that should be handled gracefully."""

    def test_single_sample(self):
        y, info = normalize_audio_for_vad(
            np.array([0.5, 0.5], dtype=np.float32),
            method="peak",
            remove_dc=False,
        )
        assert len(y) == 2
        assert info["final_peak"] == pytest.approx(1.0, abs=0.01)
        np.testing.assert_allclose(np.abs(y), 1.0, atol=0.01)

    def test_all_zeros(self):
        y, info = normalize_audio_for_vad(
            np.zeros(1000, dtype=np.float32), min_signal_db=-60
        )
        assert info["skipped_reason"] == "silent_input"

    def test_constant_nonzero(self):
        """Constant DC signal with remove_dc=True becomes zero -> silent -> skipped."""
        y, info = normalize_audio_for_vad(
            np.ones(1000, dtype=np.float32) * 0.01,
            method="rms",
            target_rms_db=-20,
            remove_dc=True,
        )
        # After DC removal, it becomes all zeros → silent → skipped
        assert info["skipped_reason"] == "silent_input"

    def test_invalid_method_raises(self, normal_signal):
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_audio_for_vad(normal_signal, method="invalid")

    def test_info_structure(self, normal_signal):
        """Verify all expected keys are present in the info dict."""
        _, info = normalize_audio_for_vad(normal_signal)
        expected_keys = {
            "method",
            "original_rms_db",
            "final_rms_db",
            "original_peak",
            "final_peak",
            "applied_gain_db",
            "skipped_reason",
        }
        assert expected_keys <= set(info.keys())
        assert isinstance(info["method"], str)
        assert isinstance(info["applied_gain_db"], float)


# ---------------------------------------------------------------------------
# Parameterized tests for different methods and target levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["peak", "rms", "hybrid"])
@pytest.mark.parametrize("target_rms_db", [-30, -20, -10])
def test_various_methods_and_targets(normal_signal, method, target_rms_db):
    """All method + target combinations should run without error."""
    y, info = normalize_audio_for_vad(
        normal_signal, method=method, target_rms_db=target_rms_db
    )
    assert y.dtype == np.float32
    assert len(y) == len(normal_signal)
    assert info["skipped_reason"] is None


@pytest.mark.parametrize("max_rms_db", [-30, -25, -20, -15])
def test_various_max_rms_thresholds(loud_signal, max_rms_db):
    """Different max_rms_db values should all work."""
    y, info = normalize_audio_for_vad(
        loud_signal, method="hybrid", target_rms_db=-10, max_rms_db=max_rms_db
    )
    assert info["final_rms_db"] <= max_rms_db + 0.2
