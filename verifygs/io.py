"""Portable NPZ and JSON input helpers for VERIFYGS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import GaussianMap, Similarity


def load_pair(path: str | Path) -> tuple[GaussianMap, GaussianMap]:
    with np.load(Path(path), allow_pickle=False) as archive:
        def load(prefix: str) -> GaussianMap:
            covariance_key = f"{prefix}_covariances"
            color_key = f"{prefix}_colors"
            return GaussianMap(
                archive[f"{prefix}_means"],
                archive[covariance_key] if covariance_key in archive else None,
                archive[color_key] if color_key in archive else None,
            )
        return load("source"), load("target")


def load_candidate(path: str | Path) -> Similarity:
    payload = json.loads(Path(path).read_text())
    return Similarity.from_dict(payload)
