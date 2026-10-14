"""VERIFYGS public API."""

from .core import (
    GaussianMap, Similarity, calibrate_affinity_floor,
    failure_conformal_pvalue, verify,
)

__all__ = [
    "GaussianMap", "Similarity", "calibrate_affinity_floor",
    "failure_conformal_pvalue", "verify",
]
__version__ = "1.0.0"
