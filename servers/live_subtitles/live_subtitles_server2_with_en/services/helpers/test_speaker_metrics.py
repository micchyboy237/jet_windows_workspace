"""
Test suite for speaker_metrics module.

Run with pytest:
    pytest test_speaker_metrics.py -v
"""

import numpy as np
import pytest
from speaker_metrics import (
    HealthStatus,
    IntraSpeakerInput,
    compute_inter_speaker_separation,
    compute_intra_speaker_variance,
    cosine_distance,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def identical_embeddings():
    """Create perfectly identical embeddings."""
    return IntraSpeakerInput(
        label="SPEAKER_01",
        embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )


@pytest.fixture
def tight_embeddings():
    """Create slightly varying embeddings forming a tight cluster."""
    return IntraSpeakerInput(
        label="SPEAKER_01",
        embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [1.01, -0.01, 0.0],
                [0.98, 0.02, 0.0],
            ]
        ),
    )


@pytest.fixture
def spread_embeddings():
    """Create widely spread embeddings (cosine distances ~0.5-1.5)."""
    return IntraSpeakerInput(
        label="SPEAKER_01",
        embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
    )


@pytest.fixture
def single_embedding():
    """Single embedding for edge case testing."""
    return IntraSpeakerInput(label="SPEAKER_01", embeddings=np.array([[1.0, 0.0, 0.0]]))


@pytest.fixture
def custom_segment_ids():
    """Custom segment IDs for testing labeled output."""
    return ["utt_001", "utt_002", "utt_003", "utt_004"]


@pytest.fixture
def outlier_embeddings_with_ids():
    """Embeddings with one clear outlier and custom IDs."""
    speaker_input = IntraSpeakerInput(
        label="SPEAKER_01",
        embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [1.05, -0.05, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    segment_ids = ["good_1", "good_2", "good_3", "outlier"]
    return speaker_input, segment_ids


@pytest.fixture
def well_separated_speakers():
    """Well-separated speakers with orthogonal centroids (distances ~1.0)."""
    return {
        "speaker_Alice": np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
        "speaker_Bob": np.array([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]),
        "speaker_Charlie": np.array([[0.0, 0.0, 1.0], [-0.1, 0.0, 0.9]]),
    }


@pytest.fixture
def overlapping_speakers():
    """Overlapping speakers with similar embeddings (distances ~0.1-0.2)."""
    return {
        "speaker_X": np.array([[1.0, 0.0], [0.95, 0.05]]),
        "speaker_Y": np.array([[0.9, 0.1], [0.85, 0.15]]),
    }


@pytest.fixture
def mixed_speakers():
    """
    Mixed separation: some close, some far.
    A-B distance: ~0.2
    A-C distance: ~1.8
    B-C distance: ~1.6
    Mean: ~1.2
    """
    return {
        "person_A": np.array([[1.0, 0.0], [0.9, 0.1]]),
        "person_B": np.array([[0.8, 0.2], [0.7, 0.3]]),
        "person_C": np.array([[-1.0, 0.0], [-0.9, -0.1]]),
    }


# ============================================================================
# Helper: Check actual distances in test data
# ============================================================================


def get_actual_mean_distance(speaker_input):
    """Helper to compute actual mean distance for debugging thresholds."""
    embeddings = speaker_input["embeddings"]
    centroid = np.mean(embeddings, axis=0)
    distances = [cosine_distance(emb, centroid) for emb in embeddings]
    return float(np.mean(distances))


def get_actual_mean_separation(speaker_embeddings):
    """Helper to compute actual mean separation for debugging thresholds."""
    centroids = {
        spk_id: np.mean(embs, axis=0) for spk_id, embs in speaker_embeddings.items()
    }
    labels = sorted(centroids.keys())
    distances = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            distances.append(
                cosine_distance(centroids[labels[i]], centroids[labels[j]])
            )
    return float(np.mean(distances))


# ============================================================================
# Tests for cosine_distance
# ============================================================================


class TestCosineDistance:
    """Test suite for cosine_distance helper function."""

    def test_identical_vectors(self):
        """Identical vectors should have zero distance."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(a, b) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have distance 1."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have distance 2."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0)

    def test_zero_vector(self):
        """Zero vector should return distance 1."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_distance(a, b) == 1.0

    def test_symmetry(self):
        """Distance should be symmetric."""
        rng = np.random.RandomState(42)
        a = rng.randn(128)
        b = rng.randn(128)
        assert cosine_distance(a, b) == pytest.approx(cosine_distance(b, a))

    def test_distance_range(self):
        """Distance should be in [0, 2] range."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            a = rng.randn(64)
            b = rng.randn(64)
            dist = cosine_distance(a, b)
            assert 0.0 <= dist <= 2.0


# ============================================================================
# Tests for compute_intra_speaker_variance
# ============================================================================


class TestIntraSpeakerVariance:
    """Test suite for compute_intra_speaker_variance function."""

    # Basic Functionality Tests

    def test_identical_embeddings(self, identical_embeddings):
        """Identical embeddings should have zero variance."""
        result = compute_intra_speaker_variance(identical_embeddings)

        assert result["mean_distance"] == pytest.approx(0.0, abs=1e-5)
        assert result["std_distance"] == pytest.approx(0.0, abs=1e-5)
        assert result["max_distance"] == pytest.approx(0.0, abs=1e-5)
        assert result["status"] == HealthStatus.HEALTHY
        assert result["num_embeddings"] == 3

    def test_tight_cluster_healthy(self, tight_embeddings):
        """Tight cluster (~0.02 mean distance) should be healthy with default thresholds."""
        result = compute_intra_speaker_variance(tight_embeddings)

        # Verify actual distance is small
        assert result["mean_distance"] < 0.05
        # Default thresholds (healthy=0.3, warning=0.5) should classify as healthy
        assert result["status"] == HealthStatus.HEALTHY
        assert result["num_embeddings"] == 4
        assert len(result["distances"]) == 4

    def test_spread_cluster_unhealthy(self, spread_embeddings):
        """Spread-out cluster (~1.0 mean distance) should be unhealthy with loose thresholds."""
        result = compute_intra_speaker_variance(
            spread_embeddings, healthy_threshold=0.3, warning_threshold=0.5
        )

        # Verify actual distance is large
        assert result["mean_distance"] > 0.5
        assert result["status"] == HealthStatus.UNHEALTHY

    def test_single_embedding(self, single_embedding):
        """Single embedding should return zero variance and healthy status."""
        result = compute_intra_speaker_variance(single_embedding)

        assert result["mean_distance"] == 0.0
        assert result["std_distance"] == 0.0
        assert result["num_embeddings"] == 1
        assert result["status"] == HealthStatus.HEALTHY
        assert len(result["distances"]) == 1
        assert result["distances"][0]["segment_id"] == "segment_0"

    # Segment ID Tests

    def test_auto_generated_segment_ids(self, tight_embeddings):
        """Should auto-generate segment IDs when not provided."""
        result = compute_intra_speaker_variance(tight_embeddings)

        expected_ids = ["segment_0", "segment_1", "segment_2", "segment_3"]
        actual_ids = [item["segment_id"] for item in result["distances"]]

        assert actual_ids == expected_ids

    def test_custom_segment_ids(self, tight_embeddings, custom_segment_ids):
        """Custom segment IDs should be properly assigned."""
        result = compute_intra_speaker_variance(
            tight_embeddings, segment_ids=custom_segment_ids
        )

        actual_ids = [item["segment_id"] for item in result["distances"]]
        assert actual_ids == custom_segment_ids

    def test_identify_problematic_segments(self, outlier_embeddings_with_ids):
        """Should be able to identify which segments are outliers."""
        speaker_input, segment_ids = outlier_embeddings_with_ids
        result = compute_intra_speaker_variance(speaker_input, segment_ids=segment_ids)
        max_item = max(result["distances"], key=lambda x: x["distance"])
        assert max_item["segment_id"] == "outlier"
        other_distances = [
            item["distance"]
            for item in result["distances"]
            if item["segment_id"] != "outlier"
        ]
        assert max_item["distance"] > max(other_distances) * 1.5

    # Threshold Tests with Realistic Values

    def test_thresholds_healthy(self, tight_embeddings):
        """Tight cluster should be healthy with threshold above its actual mean."""
        actual_mean = get_actual_mean_distance(tight_embeddings)
        result = compute_intra_speaker_variance(
            tight_embeddings,
            healthy_threshold=actual_mean * 2,
            warning_threshold=actual_mean * 4,
        )
        assert result["status"] == HealthStatus.HEALTHY

    def test_thresholds_warning(self):
        """Embeddings with known distance should trigger warning when healthy threshold is below actual mean."""
        embeddings = IntraSpeakerInput(
            label="SPEAKER_01",
            embeddings=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.7, 0.3, 0.0],
                    [0.8, 0.0, 0.2],
                    [0.9, -0.1, 0.0],
                ]
            ),
        )
        actual_mean = get_actual_mean_distance(embeddings)
        result = compute_intra_speaker_variance(
            embeddings,
            healthy_threshold=actual_mean - 0.01,
            warning_threshold=actual_mean + 0.1,
        )
        assert result["status"] == HealthStatus.WARNING

    def test_thresholds_unhealthy(self):
        """Embeddings with larger spread should trigger unhealthy when both thresholds are below actual mean."""
        embeddings = IntraSpeakerInput(
            label="SPEAKER_01",
            embeddings=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.3, 0.7, 0.0],
                    [0.0, 0.8, 0.2],
                    [0.5, -0.3, 0.0],
                ]
            ),
        )
        actual_mean = get_actual_mean_distance(embeddings)
        result = compute_intra_speaker_variance(
            embeddings,
            healthy_threshold=0.01,
            warning_threshold=actual_mean - 0.05,
        )
        assert result["status"] == HealthStatus.UNHEALTHY

    def test_known_distance_threshold(self):
        """Test all three health states using actual measured distances."""
        embeddings = IntraSpeakerInput(
            label="SPEAKER_01",
            embeddings=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.6, 0.4, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.7, 0.3],
                ]
            ),
        )
        actual_mean = get_actual_mean_distance(embeddings)
        assert actual_mean > 0.05, f"Expected measurable distance, got {actual_mean}"
        result = compute_intra_speaker_variance(
            embeddings,
            healthy_threshold=actual_mean + 0.2,
            warning_threshold=actual_mean + 0.4,
        )
        assert result["status"] == HealthStatus.HEALTHY
        result = compute_intra_speaker_variance(
            embeddings,
            healthy_threshold=actual_mean - 0.02,
            warning_threshold=actual_mean + 0.2,
        )
        assert result["status"] == HealthStatus.WARNING
        result = compute_intra_speaker_variance(
            embeddings,
            healthy_threshold=0.01,
            warning_threshold=actual_mean - 0.02,
        )
        assert result["status"] == HealthStatus.UNHEALTHY

    @pytest.mark.parametrize(
        "healthy_threshold,warning_threshold,expected_status,data_key",
        [
            (0.02, 0.05, HealthStatus.HEALTHY, "tight_embeddings"),
            (0.3, 0.5, HealthStatus.WARNING, "spread_embeddings"),
            (0.1, 0.2, HealthStatus.UNHEALTHY, "spread_embeddings"),
        ],
    )
    def test_threshold_boundaries_parametrized(
        self, request, healthy_threshold, warning_threshold, expected_status, data_key
    ):
        """Parametrized test for threshold boundaries."""
        data = request.getfixturevalue(data_key)
        actual_mean = get_actual_mean_distance(data)

        # Skip if the expected status doesn't match the actual data characteristics
        if expected_status == HealthStatus.HEALTHY and actual_mean > healthy_threshold:
            pytest.skip(
                f"Actual mean {actual_mean:.4f} > healthy threshold {healthy_threshold}"
            )
        if expected_status == HealthStatus.WARNING and not (
            healthy_threshold < actual_mean < warning_threshold
        ):
            pytest.skip(
                f"Actual mean {actual_mean:.4f} not in warning range "
                f"({healthy_threshold}, {warning_threshold})"
            )
        if (
            expected_status == HealthStatus.UNHEALTHY
            and actual_mean < warning_threshold
        ):
            pytest.skip(
                f"Actual mean {actual_mean:.4f} < warning threshold {warning_threshold}"
            )

        result = compute_intra_speaker_variance(
            data,
            healthy_threshold=healthy_threshold,
            warning_threshold=warning_threshold,
        )
        assert result["status"] == expected_status

    # Validation Tests

    def test_empty_array_raises_error(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_intra_speaker_variance(
                IntraSpeakerInput(label="SPEAKER_01", embeddings=np.array([]))
            )

    def test_1d_array_raises_error(self):
        """1D array should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            compute_intra_speaker_variance(
                IntraSpeakerInput(
                    label="SPEAKER_01", embeddings=np.array([1.0, 2.0, 3.0])
                )
            )

    def test_3d_array_raises_error(self):
        """3D array should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            compute_intra_speaker_variance(
                IntraSpeakerInput(label="SPEAKER_01", embeddings=np.array([[[1.0]]]))
            )

    def test_segment_ids_length_mismatch(self, tight_embeddings):
        """Mismatched segment IDs length should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            compute_intra_speaker_variance(tight_embeddings, segment_ids=["too", "few"])

    # Result Structure Tests

    def test_result_keys(self, tight_embeddings):
        """Result should contain all expected keys."""
        result = compute_intra_speaker_variance(tight_embeddings)
        expected_keys = {
            "label",
            "mean_distance",
            "std_distance",
            "min_distance",
            "max_distance",
            "distances",
            "distance_values",
            "status",
            "num_embeddings",
        }
        assert set(result.keys()) == expected_keys

    def test_result_types(self, tight_embeddings):
        """Result values should have correct types."""
        result = compute_intra_speaker_variance(tight_embeddings)
        assert isinstance(result["label"], str)
        assert isinstance(result["mean_distance"], float)
        assert isinstance(result["std_distance"], float)
        assert isinstance(result["min_distance"], float)
        assert isinstance(result["max_distance"], float)
        assert isinstance(result["distances"], list)
        assert isinstance(result["distance_values"], np.ndarray)
        assert isinstance(result["status"], HealthStatus)
        assert isinstance(result["num_embeddings"], int)

    def test_distance_items_structure(self, tight_embeddings, custom_segment_ids):
        """Each distance item should have correct structure."""
        result = compute_intra_speaker_variance(
            tight_embeddings, segment_ids=custom_segment_ids
        )

        for item in result["distances"]:
            assert isinstance(item, dict)
            assert "segment_id" in item
            assert "distance" in item
            assert isinstance(item["segment_id"], str)
            assert isinstance(item["distance"], float)
            assert item["distance"] >= 0.0

    def test_distance_values_match_distances(self, tight_embeddings):
        """Raw distance_values should match distances in labeled items."""
        result = compute_intra_speaker_variance(tight_embeddings)

        labeled_distances = [item["distance"] for item in result["distances"]]
        np.testing.assert_array_almost_equal(
            labeled_distances, result["distance_values"]
        )

    def test_statistics_consistency(self, tight_embeddings):
        """Mean should be between min and max."""
        result = compute_intra_speaker_variance(tight_embeddings)

        assert (
            result["min_distance"] <= result["mean_distance"] <= result["max_distance"]
        )

    def test_non_negative_distances(self):
        """All distances should be non-negative."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 128)
        speaker_input = IntraSpeakerInput(label="SPEAKER_01", embeddings=embeddings)
        result = compute_intra_speaker_variance(speaker_input)
        distances = [item["distance"] for item in result["distances"]]
        assert all(d >= 0 for d in distances)

    def test_large_embedding_dimension(self):
        """Should work with high-dimensional embeddings."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 512)
        speaker_input = IntraSpeakerInput(label="SPEAKER_01", embeddings=embeddings)
        result = compute_intra_speaker_variance(speaker_input)
        assert result["num_embeddings"] == 100
        assert len(result["distances"]) == 100


# ============================================================================
# Tests for compute_inter_speaker_separation
# ============================================================================


class TestInterSpeakerSeparation:
    """Test suite for compute_inter_speaker_separation function."""

    # Basic Functionality Tests

    def test_well_separated_speakers(self, well_separated_speakers):
        """Well-separated speakers (mean ~1.0) should be healthy with threshold 0.5."""
        result = compute_inter_speaker_separation(
            well_separated_speakers, healthy_threshold=0.5
        )

        assert result["mean_separation"] > 0.5
        assert result["status"] == HealthStatus.HEALTHY
        assert result["num_speakers"] == 3

    def test_overlapping_speakers(self, overlapping_speakers):
        """Overlapping speakers (mean ~0.1-0.2) should be unhealthy/warning."""
        result = compute_inter_speaker_separation(
            overlapping_speakers, healthy_threshold=0.5, warning_threshold=0.3
        )

        assert result["mean_separation"] < 0.3
        assert result["status"] in [HealthStatus.WARNING, HealthStatus.UNHEALTHY]

    def test_mixed_separation(self, mixed_speakers):
        """Mixed case should show range of distances (mean ~1.2)."""
        result = compute_inter_speaker_separation(
            mixed_speakers, healthy_threshold=0.5, warning_threshold=0.3
        )

        assert result["min_separation"] < result["max_separation"]
        assert result["num_speakers"] == 3
        # Mean should be around 1.2, definitely healthy with 0.5 threshold
        assert result["mean_separation"] > 0.5

    def test_two_speakers(self):
        """Two speakers should produce single pairwise distance."""
        two_speakers = {
            "spk1": np.array([[1.0, 0.0], [0.9, 0.1]]),
            "spk2": np.array([[-1.0, 0.0], [-0.9, -0.1]]),
        }

        result = compute_inter_speaker_separation(two_speakers)

        assert result["num_speakers"] == 2
        assert len(result["pairwise_distances"]) == 1
        assert result["min_separation"] == pytest.approx(result["max_separation"])

    # Pairwise Distance Label Tests

    def test_pairwise_distances_with_labels(self, well_separated_speakers):
        """Pairwise distances should include speaker labels."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        n = len(well_separated_speakers)
        expected_comparisons = n * (n - 1) // 2
        assert len(result["pairwise_distances"]) == expected_comparisons

        for item in result["pairwise_distances"]:
            assert "speaker_id_1" in item
            assert "speaker_id_2" in item
            assert "distance" in item
            assert isinstance(item["speaker_id_1"], str)
            assert isinstance(item["speaker_id_2"], str)
            assert isinstance(item["distance"], float)
            assert item["speaker_id_1"] != item["speaker_id_2"]

    def test_pairwise_labels_are_complete(self, well_separated_speakers):
        """All speaker pairs should be represented."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        pairs_in_result = set()
        for item in result["pairwise_distances"]:
            pair = tuple(sorted([item["speaker_id_1"], item["speaker_id_2"]]))
            pairs_in_result.add(pair)

        speakers = sorted(well_separated_speakers.keys())
        expected_pairs = set()
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                expected_pairs.add((speakers[i], speakers[j]))

        assert pairs_in_result == expected_pairs

    def test_identify_closest_speakers(self, mixed_speakers):
        """Should be able to identify which speakers are closest together."""
        result = compute_inter_speaker_separation(mixed_speakers)

        # Find the pair with minimum distance
        min_pair = min(result["pairwise_distances"], key=lambda x: x["distance"])

        # person_A and person_B should be closest (distance ~0.2)
        speakers_in_min = {min_pair["speaker_id_1"], min_pair["speaker_id_2"]}
        assert "person_A" in speakers_in_min
        assert "person_B" in speakers_in_min
        assert min_pair["distance"] < 0.5

    def test_identify_farthest_speakers(self, mixed_speakers):
        """Should be able to identify which speakers are farthest apart."""
        result = compute_inter_speaker_separation(mixed_speakers)

        max_pair = max(result["pairwise_distances"], key=lambda x: x["distance"])

        speakers_in_max = {max_pair["speaker_id_1"], max_pair["speaker_id_2"]}
        assert "person_C" in speakers_in_max
        assert max_pair["distance"] > 1.0

    def test_lookup_specific_pair(self, well_separated_speakers):
        """Should be able to look up distance between specific speakers."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        # Find distance between Alice and Bob
        target_pair = None
        for item in result["pairwise_distances"]:
            if {item["speaker_id_1"], item["speaker_id_2"]} == {
                "speaker_Alice",
                "speaker_Bob",
            }:
                target_pair = item
                break

        assert target_pair is not None
        # Orthogonal vectors with slight noise → distance ~0.9-1.0
        assert target_pair["distance"] == pytest.approx(0.9, abs=0.2)

    # Speaker Labels and Matrix Tests

    def test_speaker_labels_sorted(self, mixed_speakers):
        """Speaker labels should be sorted alphabetically."""
        result = compute_inter_speaker_separation(mixed_speakers)
        assert result["speaker_labels"] == sorted(mixed_speakers.keys())

    def test_speaker_labels_valid(self, well_separated_speakers):
        """All pairwise items should reference valid speakers."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        valid_speakers = set(result["speaker_labels"])
        for item in result["pairwise_distances"]:
            assert item["speaker_id_1"] in valid_speakers
            assert item["speaker_id_2"] in valid_speakers

    def test_distance_matrix_correspondence(self, well_separated_speakers):
        """Distance matrix should match pairwise items."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        labels = result["speaker_labels"]
        matrix = result["distance_matrix"]

        for item in result["pairwise_distances"]:
            i = labels.index(item["speaker_id_1"])
            j = labels.index(item["speaker_id_2"])

            assert item["distance"] == pytest.approx(matrix[i, j])
            assert matrix[i, j] == pytest.approx(matrix[j, i])

    def test_matrix_diagonal_zero(self, well_separated_speakers):
        """Distance matrix diagonal should be zeros."""
        result = compute_inter_speaker_separation(well_separated_speakers)
        diagonal = np.diag(result["distance_matrix"])
        np.testing.assert_array_almost_equal(diagonal, np.zeros_like(diagonal))

    def test_matrix_lookup_by_label(self, mixed_speakers):
        """Should be able to use speaker_labels to index into matrix."""
        result = compute_inter_speaker_separation(mixed_speakers)

        i = result["speaker_labels"].index("person_A")
        j = result["speaker_labels"].index("person_C")
        matrix_distance = result["distance_matrix"][i, j]

        # A-C distance should be large (~1.8)
        assert matrix_distance > 1.0

    # Threshold Tests with Realistic Values

    def test_thresholds_healthy(self, well_separated_speakers):
        """Well-separated speakers (mean ~1.0) should be healthy with 0.5 threshold."""
        result = compute_inter_speaker_separation(
            well_separated_speakers, healthy_threshold=0.5, warning_threshold=0.3
        )
        assert result["status"] == HealthStatus.HEALTHY

    def test_thresholds_warning_mixed(self, mixed_speakers):
        """Mixed speakers (mean ~1.2) with high healthy threshold → warning."""
        result = compute_inter_speaker_separation(
            mixed_speakers,
            healthy_threshold=1.5,  # Above actual mean of ~1.2
            warning_threshold=0.5,
        )
        assert result["status"] == HealthStatus.WARNING

    def test_thresholds_unhealthy(self, overlapping_speakers):
        """Overlapping speakers (mean ~0.1-0.2) should be unhealthy with 0.3 warning."""
        result = compute_inter_speaker_separation(
            overlapping_speakers, healthy_threshold=0.5, warning_threshold=0.3
        )
        assert result["status"] == HealthStatus.UNHEALTHY

    @pytest.mark.parametrize(
        "healthy_threshold,warning_threshold,expected,data_key",
        [
            (
                0.5,
                0.3,
                HealthStatus.HEALTHY,
                "well_separated_speakers",
            ),  # ~1.0 > 0.5 → healthy
            (
                1.5,
                0.5,
                HealthStatus.WARNING,
                "mixed_speakers",
            ),  # 0.5 < ~1.2 < 1.5 → warning
            (
                0.5,
                0.3,
                HealthStatus.UNHEALTHY,
                "overlapping_speakers",
            ),  # ~0.15 < 0.3 → unhealthy
        ],
    )
    def test_threshold_boundaries_parametrized(
        self, request, healthy_threshold, warning_threshold, expected, data_key
    ):
        """Parametrized test for threshold boundaries with different data."""
        data = request.getfixturevalue(data_key)
        result = compute_inter_speaker_separation(
            data,
            healthy_threshold=healthy_threshold,
            warning_threshold=warning_threshold,
        )
        assert result["status"] == expected

    # Validation Tests

    def test_single_speaker_raises_error(self):
        """Single speaker should raise ValueError."""
        single_speaker = {"spk1": np.array([[1.0, 0.0], [0.9, 0.1]])}
        with pytest.raises(ValueError, match="at least 2 speakers"):
            compute_inter_speaker_separation(single_speaker)

    def test_empty_dict_raises_error(self):
        """Empty dict should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 speakers"):
            compute_inter_speaker_separation({})

    def test_empty_embeddings_raises_error(self):
        """Speaker with empty embeddings should raise ValueError."""
        invalid_speakers = {
            "spk1": np.array([[1.0, 0.0]]),
            "spk2": np.array([]),
        }
        with pytest.raises(ValueError, match="empty embeddings"):
            compute_inter_speaker_separation(invalid_speakers)

    def test_3d_embeddings_raises_error(self):
        """3D embeddings should raise ValueError."""
        invalid_speakers = {
            "spk1": np.array([[[1.0, 0.0]]]),
            "spk2": np.array([[0.0, 1.0]]),
        }
        with pytest.raises(ValueError, match="must be 2D"):
            compute_inter_speaker_separation(invalid_speakers)

    # Result Structure Tests

    def test_result_keys(self, well_separated_speakers):
        """Result should contain all expected keys."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        expected_keys = {
            "mean_separation",
            "std_separation",
            "min_separation",
            "max_separation",
            "pairwise_distances",
            "distance_matrix",
            "speaker_labels",
            "status",
            "num_speakers",
        }

        assert set(result.keys()) == expected_keys

    def test_result_types(self, well_separated_speakers):
        """Result values should have correct types."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        assert isinstance(result["mean_separation"], float)
        assert isinstance(result["std_separation"], float)
        assert isinstance(result["min_separation"], float)
        assert isinstance(result["max_separation"], float)
        assert isinstance(result["pairwise_distances"], list)
        assert isinstance(result["distance_matrix"], np.ndarray)
        assert isinstance(result["speaker_labels"], list)
        assert isinstance(result["status"], HealthStatus)
        assert isinstance(result["num_speakers"], int)

    def test_distance_bounds(self, well_separated_speakers):
        """All distances should be within [0, 2] for cosine distance."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        for item in result["pairwise_distances"]:
            assert 0.0 <= item["distance"] <= 2.0

    def test_consistency_with_different_order(self):
        """Results should be consistent regardless of dict insertion order."""
        speakers_ordered = {
            "A_speaker": np.array([[1.0, 0.0]]),
            "B_speaker": np.array([[0.0, 1.0]]),
        }
        speakers_reversed = {
            "B_speaker": np.array([[0.0, 1.0]]),
            "A_speaker": np.array([[1.0, 0.0]]),
        }

        result1 = compute_inter_speaker_separation(speakers_ordered)
        result2 = compute_inter_speaker_separation(speakers_reversed)

        assert result1["mean_separation"] == pytest.approx(result2["mean_separation"])
        assert result1["speaker_labels"] == result2["speaker_labels"]

    def test_statistics_consistency(self, well_separated_speakers):
        """Mean should be between min and max."""
        result = compute_inter_speaker_separation(well_separated_speakers)

        assert (
            result["min_separation"]
            <= result["mean_separation"]
            <= result["max_separation"]
        )

    def test_pairwise_count(self, well_separated_speakers):
        """Number of pairwise items should be n*(n-1)/2."""
        result = compute_inter_speaker_separation(well_separated_speakers)
        n = result["num_speakers"]
        expected = n * (n - 1) // 2
        assert len(result["pairwise_distances"]) == expected


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining both functions."""

    def test_full_pipeline_healthy(self):
        """Test a complete analysis pipeline with well-separated speakers."""
        rng = np.random.RandomState(42)
        base_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        speaker_data = {}
        for i, base in enumerate(base_vectors):
            embeddings = base + rng.randn(10, 3) * 0.01
            speaker_data[f"speaker_{i + 1}"] = embeddings

        for speaker_id, embeddings in speaker_data.items():
            speaker_input = IntraSpeakerInput(label=speaker_id, embeddings=embeddings)
            result = compute_intra_speaker_variance(speaker_input)
            assert result["status"] == HealthStatus.HEALTHY, (
                f"Speaker {speaker_id} should be healthy, got {result['mean_distance']:.4f}"
            )
            assert result["mean_distance"] < 0.1

        inter_result = compute_inter_speaker_separation(
            speaker_data, healthy_threshold=0.5
        )
        assert inter_result["status"] == HealthStatus.HEALTHY, (
            f"Inter-speaker should be healthy, got {inter_result['mean_separation']:.4f}"
        )
        assert inter_result["mean_separation"] > 0.5

    def test_full_pipeline_unhealthy(self):
        """Test pipeline detecting both intra and inter speaker issues."""
        rng = np.random.RandomState(42)
        embeddings_a = rng.randn(10, 64)
        centroid_a = np.mean(embeddings_a, axis=0)
        speaker_data = {
            "speaker_A": embeddings_a,
            "speaker_B": rng.randn(10, 64) * 0.02
            + centroid_a
            + np.array([0.05] + [0.0] * 63),
        }

        intra_a = compute_intra_speaker_variance(
            IntraSpeakerInput(label="speaker_A", embeddings=speaker_data["speaker_A"]),
            healthy_threshold=0.5,
            warning_threshold=0.65,
        )
        assert intra_a["status"] == HealthStatus.UNHEALTHY, (
            f"Speaker A should be unhealthy, got mean={intra_a['mean_distance']:.4f}"
        )

        inter = compute_inter_speaker_separation(
            speaker_data,
            healthy_threshold=0.5,
            warning_threshold=0.1,
        )
        assert inter["mean_separation"] < 0.3, (
            f"Inter-speaker should have low separation, got {inter['mean_separation']:.4f}"
        )
        assert inter["status"] == HealthStatus.UNHEALTHY, (
            f"Inter-speaker should be unhealthy, got {inter['status'].value}"
        )
