#!/usr/bin/env python3
"""Create a deterministic pair and one correct candidate for a smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    output = Path(__file__).resolve().parent
    rng = np.random.default_rng(17)
    source = rng.normal(size=(700, 3)) * np.array([1.3, 0.7, 0.35])
    covariance = np.repeat(np.diag([0.010, 0.006, 0.003])[None], len(source), axis=0)
    colors = np.clip(0.5 + 0.2 * source, 0, 1)
    rotation = Rotation.from_euler("zyx", [28, -9, 5], degrees=True).as_matrix()
    scale = 1.17
    translation = np.array([0.45, -0.30, 0.18])
    target = scale * (source @ rotation.T) + translation
    target += rng.normal(scale=0.004, size=target.shape)
    np.savez_compressed(
        output / "synthetic_pair.npz",
        source_means=source,
        source_covariances=covariance,
        source_colors=colors,
        target_means=target,
        target_covariances=scale**2 * np.einsum(
            "ij,njk,lk->nil", rotation, covariance, rotation
        ),
        target_colors=colors,
    )
    candidate = {
        "estimated_scale": scale,
        "estimated_rotation": rotation.tolist(),
        "estimated_translation": translation.tolist(),
    }
    (output / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")


if __name__ == "__main__":
    main()
