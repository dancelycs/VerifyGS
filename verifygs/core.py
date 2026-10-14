"""Core VERIFYGS scoring functions.

The implementation follows the source-to-target row-vector convention

    x_target = scale * (x_source @ rotation.T) + translation.

Ground-truth transformations are never consumed by this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GaussianMap:
    means: np.ndarray
    covariances: np.ndarray | None = None
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=np.float64)
        if means.ndim != 2 or means.shape[1] != 3 or len(means) == 0:
            raise ValueError("means must have shape [N, 3] with N > 0")
        object.__setattr__(self, "means", means)
        if self.covariances is not None:
            covariance = np.asarray(self.covariances, dtype=np.float64)
            if covariance.shape != (len(means), 3, 3):
                raise ValueError("covariances must have shape [N, 3, 3]")
            object.__setattr__(self, "covariances", covariance)
        if self.colors is not None:
            colors = np.asarray(self.colors, dtype=np.float64)
            if colors.shape != (len(means), 3):
                raise ValueError("colors must have shape [N, 3]")
            object.__setattr__(self, "colors", colors)


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation and translation must have shapes [3,3] and [3]")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def from_dict(cls, record: dict) -> "Similarity":
        return cls(
            float(record["estimated_scale"]),
            np.asarray(record["estimated_rotation"], dtype=np.float64),
            np.asarray(record["estimated_translation"], dtype=np.float64),
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points) @ self.rotation.T) + self.translation

    def transport_covariances(self, covariances: np.ndarray) -> np.ndarray:
        return self.scale**2 * np.einsum(
            "ij,njk,lk->nil", self.rotation, covariances, self.rotation
        )


def robust_frame(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the coordinate-wise median and robust 5--95% diagonal extent."""
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    low, high = np.quantile(points, (0.05, 0.95), axis=0)
    return center, max(float(np.linalg.norm(high - low)), 1e-12)


def bidirectional_support(source: np.ndarray, target: np.ndarray, radius: float) -> float:
    """Fraction of centers supported in both source-to-target directions."""
    source_distance, _ = cKDTree(target).query(source, workers=-1)
    target_distance, _ = cKDTree(source).query(target, workers=-1)
    return float(
        0.5
        * (np.mean(source_distance < radius) + np.mean(target_distance < radius))
    )


def sample_null_similarities(
    source: np.ndarray, target: np.ndarray, pair_id: str, count: int = 64
) -> list[Similarity]:
    """Deterministically sample extent-compatible random similarities."""
    if count < 1:
        raise ValueError("count must be positive")
    source_center, source_extent = robust_frame(source)
    target_center, target_extent = robust_frame(target)
    seed = int.from_bytes(hashlib.sha256(pair_id.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    scales = target_extent / source_extent * np.exp(rng.normal(0.0, 0.15, count))
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    offsets = directions * rng.uniform(0.0, 0.20, (count, 1)) * target_extent
    output = []
    for index in range(count):
        translation = (
            target_center
            + offsets[index]
            - scales[index] * (rotations[index] @ source_center)
        )
        output.append(Similarity(scales[index], rotations[index], translation))
    return output


def null_support_distribution(
    source: np.ndarray, target: np.ndarray, pair_id: str, count: int = 64
) -> np.ndarray:
    extent = robust_frame(target)[1]
    radius = 0.04 * extent
    return np.asarray(
        [
            bidirectional_support(transform.apply(source), target, radius)
            for transform in sample_null_similarities(source, target, pair_id, count)
        ],
        dtype=np.float64,
    )


def gaussian_affinity(
    source: GaussianMap,
    target: GaussianMap,
    transform: Similarity,
    neighbours: int = 4,
    trim_fraction: float = 0.30,
    color_sigma: float = 0.25,
) -> float | None:
    """Return the separate covariance--color diagnostic A(T)."""
    if any(value is None for value in (
        source.covariances, source.colors, target.covariances, target.colors
    )):
        return None
    moved = transform.apply(source.means)
    moved_covariance = transform.transport_covariances(source.covariances)
    extent = robust_frame(target.means)[1]
    regularizer = (0.01 * extent) ** 2
    identity = np.eye(3)[None]

    def directional(query_mean, query_covariance, query_color,
                    reference_mean, reference_covariance, reference_color):
        _, neighbour = cKDTree(reference_mean).query(
            query_mean, k=min(neighbours, len(reference_mean)), workers=-1
        )
        if neighbour.ndim == 1:
            neighbour = neighbour[:, None]
        delta = query_mean[:, None] - reference_mean[neighbour]
        combined = (
            query_covariance[:, None]
            + reference_covariance[neighbour]
            + regularizer * identity[:, None]
        )
        solution = np.linalg.solve(combined, delta[..., None])[..., 0]
        mahalanobis = np.einsum("nki,nki->nk", delta, solution)
        color_distance2 = np.sum(
            (query_color[:, None] - reference_color[neighbour]) ** 2, axis=2
        )
        value = np.exp(-0.5 * mahalanobis) * np.exp(
            -color_distance2 / (2.0 * color_sigma**2)
        )
        return value.max(axis=1)

    values = np.concatenate(
        (
            directional(
                moved, moved_covariance, source.colors,
                target.means, target.covariances, target.colors,
            ),
            directional(
                target.means, target.covariances, target.colors,
                moved, moved_covariance, source.colors,
            ),
        )
    )
    keep = min(len(values), max(16, round(trim_fraction * len(values))))
    return float(np.mean(np.partition(values, len(values) - keep)[-keep:]))


def failure_conformal_pvalue(score: float, calibration_failures: np.ndarray) -> float:
    """Return the finite-sample upper-tail p-value against failed candidates."""
    failures = np.asarray(calibration_failures, dtype=np.float64).reshape(-1)
    if len(failures) == 0 or not np.all(np.isfinite(failures)):
        raise ValueError("calibration_failures must contain finite scores")
    return float((1 + np.sum(failures >= score)) / (len(failures) + 1))


def calibrate_affinity_floor(
    successful_affinities: np.ndarray, quantile: float = 0.05
) -> float:
    """Calibrate the Gaussian-consistency veto from successful candidates."""
    values = np.asarray(successful_affinities, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("successful_affinities must contain finite scores")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return float(np.quantile(values, quantile, method="higher"))


def verify(
    source: GaussianMap,
    target: GaussianMap,
    transform: Similarity,
    pair_id: str,
    null_count: int = 64,
    threshold: float | None = None,
    null_q95: float | None = None,
    calibration_failures: np.ndarray | None = None,
    alpha: float = 0.05,
    affinity_floor: float | None = None,
) -> dict[str, float | bool | None]:
    """Score one fixed candidate without using solver state or ground truth."""
    moved = transform.apply(source.means)
    extent = robust_frame(target.means)[1]
    support = bidirectional_support(moved, target.means, 0.04 * extent)
    if null_q95 is None:
        null = null_support_distribution(
            source.means, target.means, pair_id=pair_id, count=null_count
        )
        q95 = float(np.quantile(null, 0.95))
    else:
        q95 = float(null_q95)
    ratio = support / max(q95, 1e-6)
    affinity = gaussian_affinity(source, target, transform)
    conformal_p = None
    accepted = None if threshold is None else bool(ratio >= threshold)
    if calibration_failures is not None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if affinity_floor is None:
            raise ValueError(
                "affinity_floor is required for the final Gaussian-constrained gate"
            )
        conformal_p = failure_conformal_pvalue(ratio, calibration_failures)
        accepted = bool(
            conformal_p <= alpha
            and affinity is not None
            and affinity >= affinity_floor
        )
    return {
        "support_04": support,
        "null_q95": q95,
        "ratio_q95": ratio,
        "gaussian_affinity": affinity,
        "failure_conformal_pvalue": conformal_p,
        "accepted": accepted,
    }
