import numpy as np

from verifygs.core import (
    GaussianMap, Similarity, bidirectional_support, calibrate_affinity_floor,
    failure_conformal_pvalue, null_support_distribution, verify,
)


def synthetic():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(160, 3)) * np.array([1.0, 0.6, 0.25])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = Similarity(1.2, rotation, np.array([0.4, -0.2, 0.1]))
    target = transform.apply(source)
    covariance = np.repeat(np.eye(3)[None] * 0.01, len(source), axis=0)
    colors = np.clip(0.5 + 0.1 * source, 0, 1)
    target_covariance = transform.scale**2 * np.einsum(
        "ij,njk,lk->nil", transform.rotation, covariance, transform.rotation
    )
    return (
        GaussianMap(source, covariance, colors),
        GaussianMap(target, target_covariance, colors),
        transform,
    )


def test_correct_candidate_has_complete_support():
    source, target, transform = synthetic()
    result = verify(source, target, transform, "synthetic", null_count=8)
    assert result["support_04"] == 1.0
    assert result["ratio_q95"] > 1.0


def test_support_is_bidirectional_and_bounded():
    source, target, transform = synthetic()
    moved = transform.apply(source.means)
    value = bidirectional_support(moved, target.means, radius=1e-8)
    assert 0.99 <= value <= 1.0


def test_null_sampling_is_deterministic():
    source, target, _ = synthetic()
    first = null_support_distribution(source.means, target.means, "same-pair", 8)
    second = null_support_distribution(source.means, target.means, "same-pair", 8)
    np.testing.assert_allclose(first, second)


def test_final_gate_uses_conformal_rank_and_gaussian_affinity():
    source, target, transform = synthetic()
    failures = np.zeros(19, dtype=float)
    result = verify(
        source, target, transform, "synthetic", null_count=8,
        calibration_failures=failures, alpha=0.05, affinity_floor=0.0,
    )
    assert result["failure_conformal_pvalue"] == 0.05
    assert result["gaussian_affinity"] is not None
    assert result["accepted"]


def test_calibration_helpers_are_conservative():
    assert failure_conformal_pvalue(3.0, np.array([1.0, 2.0, 3.0])) == 0.5
    assert calibrate_affinity_floor(np.array([0.1, 0.2, 0.3]), 0.05) == 0.2


def test_gaussian_affinity_supports_small_maps():
    means = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    covariances = np.repeat(np.eye(3)[None] * 0.01, 2, axis=0)
    colors = np.full((2, 3), 0.5)
    gaussian_map = GaussianMap(means, covariances, colors)
    identity = Similarity(1.0, np.eye(3), np.zeros(3))
    result = verify(
        gaussian_map, gaussian_map, identity, "small-map", null_count=2,
        calibration_failures=np.zeros(19), affinity_floor=0.0,
    )
    assert result["gaussian_affinity"] is not None
