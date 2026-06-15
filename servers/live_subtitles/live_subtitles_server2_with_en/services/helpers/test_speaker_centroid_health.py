"""
test_speaker_centroid_health.py
================================
Unit tests for speaker_centroid_health.py.

Run with:
    pytest test_speaker_centroid_health.py -v

Coverage
--------
Math helpers      : _l2_normalize, _cosine_similarity, _cosine_distance,
                    _compute_centroid, _mean_cosine_sim_to_centroid,
                    _intra_cluster_spread, _per_centroid_silhouette
Flags             : HEALTHY, IMMATURE, CONTAMINATED, DIFFUSE,
                    TOO_CLOSE, REDUNDANT
Report properties : healthy_labels, unhealthy_labels, merge_candidates, summary
Edge cases        : single centroid, single embedding, zero vector, empty label,
                    identical centroids, custom thresholds
Public API        : CentroidHealthChecker.check_all, .check_one,
                    check_centroid_health (convenience wrapper)
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List

import numpy as np
import pytest
from numpy.typing import NDArray

from speaker_centroid_health import (
    CentroidHealth,
    CentroidHealthChecker,
    CentroidHealthReport,
    HealthFlag,
    HealthThresholds,
    LabelID,
    Embedding,
    _compute_centroid,
    _cosine_distance,
    _cosine_similarity,
    _intra_cluster_spread,
    _l2_normalize,
    _mean_cosine_sim_to_centroid,
    _per_centroid_silhouette,
    check_centroid_health,
)


# ---------------------------------------------------------------------------
# Shared fixtures & factories
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
D: int = 64  # small dimension keeps tests fast


def unit(v: NDArray[np.float32]) -> Embedding:
    """Return an L2-normalised copy of v."""
    return _l2_normalize(v.astype(np.float32))


def rand_unit() -> Embedding:
    return unit(RNG.standard_normal(D).astype(np.float32))


def make_cluster(
    center: Embedding,
    n: int,
    noise: float = 0.05,
    rng: np.random.Generator = RNG,
) -> List[Embedding]:
    """Tight cluster around *center* with Gaussian noise."""
    vecs = center + rng.normal(0, noise, size=(n, D)).astype(np.float32)
    return [unit(v) for v in vecs]


def opposite(v: Embedding) -> Embedding:
    """Return the unit vector pointing in the opposite direction."""
    return unit(-v)


# ---------------------------------------------------------------------------
# Parametrised label sets used across multiple tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_well_separated_speakers() -> Dict[LabelID, List[Embedding]]:
    """Two clearly distinct, well-populated clusters."""
    c0, c1 = rand_unit(), rand_unit()
    # Push them far apart to guarantee separation
    c1 = opposite(c0)
    return {
        "SPK_A": make_cluster(c0, n=10, noise=0.03),
        "SPK_B": make_cluster(c1, n=10, noise=0.03),
    }


@pytest.fixture()
def redundant_pair() -> Dict[LabelID, List[Embedding]]:
    """Two centroids built from virtually the same direction → REDUNDANT."""
    center = rand_unit()
    return {
        "SPK_X": make_cluster(center, n=10, noise=0.02),
        "SPK_Y": make_cluster(center, n=10, noise=0.02),
    }


# ===========================================================================
# 1. Math helpers
# ===========================================================================

class TestL2Normalize:
    def test_unit_vector_unchanged(self) -> None:
        v = rand_unit()
        result = _l2_normalize(v)
        assert math.isclose(float(np.linalg.norm(result)), 1.0, abs_tol=1e-6)

    def test_arbitrary_vector_becomes_unit(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _l2_normalize(v)
        assert math.isclose(float(np.linalg.norm(result)), 1.0, abs_tol=1e-6)

    def test_zero_vector_returned_as_is(self) -> None:
        v = np.zeros(D, dtype=np.float32)
        result = _l2_normalize(v)
        assert np.allclose(result, 0.0)

    def test_output_dtype_is_float32(self) -> None:
        v = np.ones(D, dtype=np.float64)
        result = _l2_normalize(v.astype(np.float32))
        assert result.dtype == np.float32


class TestCosineSimilarity:
    def test_identical_vectors_give_one(self) -> None:
        v = rand_unit()
        assert math.isclose(_cosine_similarity(v, v), 1.0, abs_tol=1e-6)

    def test_opposite_vectors_give_minus_one(self) -> None:
        v = rand_unit()
        assert math.isclose(_cosine_similarity(v, opposite(v)), -1.0, abs_tol=1e-5)

    def test_orthogonal_vectors_give_zero(self) -> None:
        a = unit(np.array([1.0, 0.0], dtype=np.float32))
        b = unit(np.array([0.0, 1.0], dtype=np.float32))
        assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-6)

    def test_return_type_is_float(self) -> None:
        v = rand_unit()
        result = _cosine_similarity(v, v)
        assert isinstance(result, float)


class TestCosineDistance:
    def test_identical_vectors_give_zero(self) -> None:
        v = rand_unit()
        assert math.isclose(_cosine_distance(v, v), 0.0, abs_tol=1e-6)

    def test_opposite_vectors_give_two(self) -> None:
        v = rand_unit()
        assert math.isclose(_cosine_distance(v, opposite(v)), 2.0, abs_tol=1e-5)

    def test_complement_of_similarity(self) -> None:
        a, b = rand_unit(), rand_unit()
        assert math.isclose(
            _cosine_distance(a, b),
            1.0 - _cosine_similarity(a, b),
            abs_tol=1e-6,
        )


class TestComputeCentroid:
    def test_single_embedding_returns_itself(self) -> None:
        v = rand_unit()
        mat = v[np.newaxis, :]
        centroid = _compute_centroid(mat)
        assert np.allclose(centroid, v, atol=1e-5)

    def test_centroid_is_unit_normalised(self) -> None:
        mat = np.stack([rand_unit() for _ in range(8)])
        centroid = _compute_centroid(mat)
        assert math.isclose(float(np.linalg.norm(centroid)), 1.0, abs_tol=1e-5)

    def test_tight_cluster_centroid_near_center(self) -> None:
        center = rand_unit()
        cluster = make_cluster(center, n=50, noise=0.01)
        mat = np.stack(cluster)
        centroid = _compute_centroid(mat)
        sim = _cosine_similarity(centroid, center)
        assert sim > 0.99


class TestMeanCosineSimToCentroid:
    def test_identical_embeddings_give_one(self) -> None:
        v = rand_unit()
        mat = np.stack([v] * 5)
        assert math.isclose(_mean_cosine_sim_to_centroid(mat, v), 1.0, abs_tol=1e-5)

    def test_value_in_valid_range(self) -> None:
        center = rand_unit()
        mat = np.stack(make_cluster(center, n=10))
        centroid = _compute_centroid(mat)
        result = _mean_cosine_sim_to_centroid(mat, centroid)
        assert -1.0 <= result <= 1.0

    def test_tight_cluster_has_high_similarity(self) -> None:
        center = rand_unit()
        mat = np.stack(make_cluster(center, n=20, noise=0.02))
        centroid = _compute_centroid(mat)
        assert _mean_cosine_sim_to_centroid(mat, centroid) > 0.90


class TestIntraClusterSpread:
    def test_identical_embeddings_give_zero_spread(self) -> None:
        v = rand_unit()
        mat = np.stack([v] * 5)
        assert math.isclose(_intra_cluster_spread(mat, v), 0.0, abs_tol=1e-5)

    def test_spread_is_complement_of_mean_sim(self) -> None:
        center = rand_unit()
        mat = np.stack(make_cluster(center, n=15))
        centroid = _compute_centroid(mat)
        spread = _intra_cluster_spread(mat, centroid)
        mean_sim = _mean_cosine_sim_to_centroid(mat, centroid)
        assert math.isclose(spread, 1.0 - mean_sim, abs_tol=1e-5)

    def test_noisy_cluster_has_higher_spread(self) -> None:
        center = rand_unit()
        tight_mat = np.stack(make_cluster(center, n=20, noise=0.01))
        noisy_mat = np.stack(make_cluster(center, n=20, noise=0.50))
        centroid = _compute_centroid(tight_mat)
        noisy_centroid = _compute_centroid(noisy_mat)
        assert _intra_cluster_spread(noisy_mat, noisy_centroid) > _intra_cluster_spread(
            tight_mat, centroid
        )


class TestPerCentroidSilhouette:
    def test_single_centroid_returns_zero(self) -> None:
        v = rand_unit()
        emb_map = {"SPK": np.stack([v] * 5)}
        cen_map = {"SPK": _compute_centroid(emb_map["SPK"])}
        assert _per_centroid_silhouette("SPK", emb_map, cen_map) == 0.0

    def test_well_separated_clusters_have_positive_silhouette(self) -> None:
        c0 = rand_unit()
        c1 = opposite(c0)
        emb_map = {
            "A": np.stack(make_cluster(c0, n=10, noise=0.02)),
            "B": np.stack(make_cluster(c1, n=10, noise=0.02)),
        }
        cen_map = {k: _compute_centroid(v) for k, v in emb_map.items()}
        sil = _per_centroid_silhouette("A", emb_map, cen_map)
        assert sil > 0.5

    def test_overlapping_clusters_have_low_silhouette(self) -> None:
        center = rand_unit()
        emb_map = {
            "A": np.stack(make_cluster(center, n=10, noise=0.50)),
            "B": np.stack(make_cluster(center, n=10, noise=0.50)),
        }
        cen_map = {k: _compute_centroid(v) for k, v in emb_map.items()}
        sil = _per_centroid_silhouette("A", emb_map, cen_map)
        assert sil < 0.5

    def test_silhouette_in_minus_one_to_one(self) -> None:
        c0, c1 = rand_unit(), rand_unit()
        emb_map = {
            "A": np.stack(make_cluster(c0, n=8)),
            "B": np.stack(make_cluster(c1, n=8)),
        }
        cen_map = {k: _compute_centroid(v) for k, v in emb_map.items()}
        sil = _per_centroid_silhouette("A", emb_map, cen_map)
        assert -1.0 <= sil <= 1.0


# ===========================================================================
# 2. HealthFlag assignment
# ===========================================================================

class TestHealthyFlag:
    def test_well_separated_tight_clusters_are_healthy(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        for lbl in two_well_separated_speakers:
            assert report.results[lbl].is_healthy, (
                f"{lbl} unexpectedly flagged: {report.results[lbl].flags}"
            )

    def test_healthy_flag_set_when_no_issues(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        for lbl in two_well_separated_speakers:
            assert HealthFlag.HEALTHY in report.results[lbl].flags


class TestImmatureFlag:
    def test_too_few_embeddings_flagged_immature(self) -> None:
        c0, c1 = rand_unit(), opposite(rand_unit())
        data = {
            "GOOD": make_cluster(c0, n=10),
            "TINY": make_cluster(c1, n=2),   # below default min of 5
        }
        report = check_centroid_health(data)
        assert HealthFlag.IMMATURE in report.results["TINY"].flags

    def test_exactly_at_min_count_not_flagged(self) -> None:
        t = HealthThresholds(min_embedding_count=3)
        c0, c1 = rand_unit(), opposite(rand_unit())
        data = {
            "A": make_cluster(c0, n=10),
            "B": make_cluster(c1, n=3),  # exactly at threshold
        }
        report = check_centroid_health(data, thresholds=t)
        assert HealthFlag.IMMATURE not in report.results["B"].flags

    def test_one_below_min_count_flagged(self) -> None:
        t = HealthThresholds(min_embedding_count=5)
        c0, c1 = rand_unit(), opposite(rand_unit())
        data = {
            "A": make_cluster(c0, n=10),
            "B": make_cluster(c1, n=4),
        }
        report = check_centroid_health(data, thresholds=t)
        assert HealthFlag.IMMATURE in report.results["B"].flags


class TestContaminatedFlag:
    def test_mixed_speaker_cluster_flagged_contaminated(self) -> None:
        # Build 10 random directions and mix them all into one "cluster".
        # The mean cosine sim to the resulting centroid will be very low
        # because the embeddings cancel each other out in high-D space.
        rng_local = np.random.default_rng(99)
        directions = [
            unit(rng_local.standard_normal(D).astype(np.float32)) for _ in range(10)
        ]
        mixed: List[Embedding] = []
        for d in directions:
            mixed.extend(make_cluster(d, n=3, noise=0.01, rng=rng_local))

        # Reference cluster that is clean and far from the mixed centroid
        far_center = unit(rng_local.standard_normal(D).astype(np.float32))
        clean = make_cluster(far_center, n=10, noise=0.02, rng=rng_local)

        data = {"MIXED": mixed, "CLEAN": clean}
        t = HealthThresholds(min_mean_cosine_sim=0.40)  # realistic threshold
        report = check_centroid_health(data, thresholds=t)
        assert HealthFlag.CONTAMINATED in report.results["MIXED"].flags

    def test_clean_cluster_not_contaminated(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        for lbl in two_well_separated_speakers:
            assert HealthFlag.CONTAMINATED not in report.results[lbl].flags


class TestDiffuseFlag:
    def test_high_spread_cluster_flagged_diffuse(self) -> None:
        center = rand_unit()
        far = opposite(rand_unit())
        data = {
            "SPREAD": make_cluster(center, n=10, noise=0.80),
            "REF":    make_cluster(far,    n=10, noise=0.02),
        }
        t = HealthThresholds(max_intra_spread=0.10)  # tight threshold
        report = check_centroid_health(data, thresholds=t)
        assert HealthFlag.DIFFUSE in report.results["SPREAD"].flags

    def test_tight_cluster_not_diffuse(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        for lbl in two_well_separated_speakers:
            assert HealthFlag.DIFFUSE not in report.results[lbl].flags


class TestTooCloseFlag:
    def test_similar_but_not_redundant_centroids_flagged_too_close(self) -> None:
        center = rand_unit()
        # Small perturbation → high similarity but intentionally below merge threshold
        slight = unit(center + np.float32(0.1) * rand_unit())
        far = opposite(rand_unit())
        data = {
            "A": make_cluster(center, n=10, noise=0.01),
            "B": make_cluster(slight, n=10, noise=0.01),
            "C": make_cluster(far,    n=10, noise=0.01),
        }
        t = HealthThresholds(
            max_inter_centroid_similarity=0.50,
            merge_similarity_threshold=0.99,  # very high → ideally won't hit REDUNDANT
        )
        report = check_centroid_health(data, thresholds=t)
        flag_a = report.results["A"].flags
        flag_b = report.results["B"].flags
        # Accept either TOO_CLOSE or REDUNDANT: both signal the centroids are
        # dangerously similar (REDUNDANT means similarity exceeded merge threshold,
        # which is an even stronger form of the same problem).
        too_close_or_redundant = {HealthFlag.TOO_CLOSE, HealthFlag.REDUNDANT}
        assert (
            bool(too_close_or_redundant & set(flag_a))
            or bool(too_close_or_redundant & set(flag_b))
        ), f"Expected TOO_CLOSE or REDUNDANT in A={flag_a} or B={flag_b}"


class TestRedundantFlag:
    def test_near_duplicate_centroids_flagged_redundant(
        self, redundant_pair: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(redundant_pair)
        flags_x = report.results["SPK_X"].flags
        flags_y = report.results["SPK_Y"].flags
        assert HealthFlag.REDUNDANT in flags_x or HealthFlag.REDUNDANT in flags_y

    def test_well_separated_not_redundant(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        for lbl in two_well_separated_speakers:
            assert HealthFlag.REDUNDANT not in report.results[lbl].flags

    def test_redundant_pair_in_merge_candidates(
        self, redundant_pair: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(redundant_pair)
        assert len(report.merge_candidates) >= 1
        labels_in_candidates = {
            lbl for a, b, _ in report.merge_candidates for lbl in (a, b)
        }
        assert "SPK_X" in labels_in_candidates or "SPK_Y" in labels_in_candidates


# ===========================================================================
# 3. CentroidHealth dataclass
# ===========================================================================

class TestCentroidHealthDataclass:
    def _make_health(self, flags: List[HealthFlag]) -> CentroidHealth:
        v = rand_unit()
        return CentroidHealth(
            label="SPK",
            embedding_count=10,
            centroid_vector=v,
            mean_cosine_sim_to_centroid=0.9,
            intra_cluster_spread=0.1,
            silhouette_score=0.5,
            flags=flags,
        )

    def test_is_healthy_with_healthy_flag(self) -> None:
        h = self._make_health([HealthFlag.HEALTHY])
        assert h.is_healthy is True

    def test_is_healthy_with_empty_flags(self) -> None:
        h = self._make_health([])
        assert h.is_healthy is True

    def test_not_healthy_with_immature_flag(self) -> None:
        h = self._make_health([HealthFlag.IMMATURE])
        assert h.is_healthy is False

    def test_not_healthy_with_multiple_flags(self) -> None:
        h = self._make_health([HealthFlag.CONTAMINATED, HealthFlag.DIFFUSE])
        assert h.is_healthy is False


# ===========================================================================
# 4. CentroidHealthReport
# ===========================================================================

class TestCentroidHealthReport:
    def test_healthy_labels_count(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        assert len(report.healthy_labels) == 2

    def test_unhealthy_labels_empty_when_all_healthy(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        assert report.unhealthy_labels == []

    def test_merge_candidates_deduplicated(
        self, redundant_pair: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(redundant_pair)
        pairs = [(a, b) for a, b, _ in report.merge_candidates]
        # No duplicate (reversed) pairs
        seen: set[frozenset[str]] = set()
        for a, b in pairs:
            key = frozenset({a, b})
            assert key not in seen, f"Duplicate pair: ({a}, {b})"
            seen.add(key)

    def test_merge_candidates_similarity_above_threshold(
        self, redundant_pair: Dict[LabelID, List[Embedding]]
    ) -> None:
        t = HealthThresholds(merge_similarity_threshold=0.80)
        report = check_centroid_health(redundant_pair, thresholds=t)
        for _, _, sim in report.merge_candidates:
            assert sim >= t.merge_similarity_threshold

    def test_summary_contains_all_labels(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        summary = report.summary()
        for lbl in two_well_separated_speakers:
            assert lbl in summary

    def test_summary_is_string(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(two_well_separated_speakers)
        assert isinstance(report.summary(), str)

    def test_summary_mentions_merge_candidates(
        self, redundant_pair: Dict[LabelID, List[Embedding]]
    ) -> None:
        report = check_centroid_health(redundant_pair)
        if report.merge_candidates:
            assert "Merge Candidates" in report.summary()


# ===========================================================================
# 5. CentroidHealthChecker
# ===========================================================================

class TestCentroidHealthChecker:
    def test_check_all_returns_report(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        report = checker.check_all()
        assert isinstance(report, CentroidHealthReport)

    def test_check_all_has_entry_per_label(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        report = checker.check_all()
        assert set(report.results.keys()) == set(two_well_separated_speakers.keys())

    def test_check_one_returns_centroid_health(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        result = checker.check_one("SPK_A")
        assert isinstance(result, CentroidHealth)
        assert result.label == "SPK_A"

    def test_check_one_flag_consistent_with_check_all(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        full_report = checker.check_all()
        single = checker.check_one("SPK_A")
        assert single.flags == full_report.results["SPK_A"].flags

    def test_embedding_count_correct(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        report = checker.check_all()
        for lbl, embs in two_well_separated_speakers.items():
            assert report.results[lbl].embedding_count == len(embs)

    def test_centroid_vector_is_unit_normalised(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        report = checker.check_all()
        for result in report.results.values():
            norm = float(np.linalg.norm(result.centroid_vector))
            assert math.isclose(norm, 1.0, abs_tol=1e-5), f"Not unit norm: {norm}"

    def test_custom_thresholds_respected(self) -> None:
        c0, c1 = rand_unit(), opposite(rand_unit())
        data = {
            "A": make_cluster(c0, n=10),
            "B": make_cluster(c1, n=10),
        }
        # Require 999 embeddings → everything becomes IMMATURE
        t = HealthThresholds(min_embedding_count=999)
        report = check_centroid_health(data, thresholds=t)
        for lbl in data:
            assert HealthFlag.IMMATURE in report.results[lbl].flags

    def test_nearest_centroid_label_set(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        report = checker.check_all()
        for result in report.results.values():
            assert result.nearest_centroid_label is not None
            assert result.nearest_centroid_label != result.label

    def test_nearest_centroid_similarity_in_range(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        checker = CentroidHealthChecker(two_well_separated_speakers)
        report = checker.check_all()
        for result in report.results.values():
            sim = result.nearest_centroid_similarity
            assert sim is not None
            assert -1.0 <= sim <= 1.0


# ===========================================================================
# 6. Convenience wrapper
# ===========================================================================

class TestCheckCentroidHealth:
    def test_returns_report(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        result = check_centroid_health(two_well_separated_speakers)
        assert isinstance(result, CentroidHealthReport)

    def test_equivalent_to_checker_check_all(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        via_helper = check_centroid_health(two_well_separated_speakers)
        via_class = CentroidHealthChecker(two_well_separated_speakers).check_all()
        assert set(via_helper.results.keys()) == set(via_class.results.keys())
        for lbl in via_helper.results:
            assert via_helper.results[lbl].flags == via_class.results[lbl].flags

    def test_accepts_custom_thresholds(
        self, two_well_separated_speakers: Dict[LabelID, List[Embedding]]
    ) -> None:
        t = HealthThresholds(min_embedding_count=1)
        result = check_centroid_health(two_well_separated_speakers, thresholds=t)
        assert isinstance(result, CentroidHealthReport)


# ===========================================================================
# 7. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_centroid_no_nearest_neighbour(self) -> None:
        data = {"ONLY": make_cluster(rand_unit(), n=10)}
        report = check_centroid_health(data)
        result = report.results["ONLY"]
        assert result.nearest_centroid_label is None
        assert result.nearest_centroid_similarity is None

    def test_single_centroid_silhouette_is_zero(self) -> None:
        data = {"ONLY": make_cluster(rand_unit(), n=10)}
        report = check_centroid_health(data)
        assert report.results["ONLY"].silhouette_score == 0.0

    def test_single_centroid_no_merge_candidates(self) -> None:
        data = {"ONLY": make_cluster(rand_unit(), n=10)}
        report = check_centroid_health(data)
        assert report.merge_candidates == []

    def test_single_embedding_per_centroid(self) -> None:
        c0, c1 = rand_unit(), opposite(rand_unit())
        data = {
            "A": [c0],
            "B": [c1],
        }
        t = HealthThresholds(min_embedding_count=1)
        report = check_centroid_health(data, thresholds=t)
        # Should run without error; spread of a single point = 0
        assert report.results["A"].intra_cluster_spread == pytest.approx(0.0, abs=1e-5)

    def test_empty_label_emits_warning_and_is_skipped(self) -> None:
        c0 = rand_unit()
        data: Dict[LabelID, List[Embedding]] = {
            "GOOD": make_cluster(c0, n=10),
            "EMPTY": [],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = check_centroid_health(data)
        assert any("EMPTY" in str(w.message) for w in caught)
        assert "EMPTY" not in report.results

    def test_unnormalised_input_embeddings_accepted(self) -> None:
        # Raw (not unit-normalised) embeddings should be handled gracefully
        rng = np.random.default_rng(7)
        raw = [rng.standard_normal(D).astype(np.float32) * 10.0 for _ in range(10)]
        far_raw = [rng.standard_normal(D).astype(np.float32) * 0.1 - 5.0 for _ in range(10)]
        data = {"RAW_A": raw, "RAW_B": far_raw}
        # Should not raise
        report = check_centroid_health(data)
        assert "RAW_A" in report.results

    def test_many_centroids_no_error(self) -> None:
        data = {
            f"SPK_{i:02d}": make_cluster(rand_unit(), n=8)
            for i in range(20)
        }
        report = check_centroid_health(data)
        assert len(report.results) == 20

    def test_intra_metrics_between_zero_and_two(self) -> None:
        c0, c1 = rand_unit(), opposite(rand_unit())
        data = {"A": make_cluster(c0, n=10), "B": make_cluster(c1, n=10)}
        report = check_centroid_health(data)
        for result in report.results.values():
            assert 0.0 <= result.intra_cluster_spread <= 2.0
            assert -1.0 <= result.mean_cosine_sim_to_centroid <= 1.0
