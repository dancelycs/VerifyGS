#!/usr/bin/env python3
"""Plot controlled rotation, translation, and scale sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.data.read_text())["summary"]
    panels = [
        ("rotation_deg", "Rotation perturbation (degrees)"),
        ("translation_extent", "Translation perturbation / extent"),
        ("scale_percent", "Scale perturbation (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    for axis, (key, xlabel) in zip(axes, panels):
        rows = summary[key]
        x = [row["level"] for row in rows]
        y = [row["mean"] for row in rows]
        std = [row["std"] for row in rows]
        axis.plot(x, y, "o-", lw=2, color="#b6202e")
        axis.fill_between(x, [a-b for a, b in zip(y, std)],
                          [a+b for a, b in zip(y, std)], color="#b6202e", alpha=0.15)
        axis.axhline(1.0, color="#777777", ls="--", lw=1, label="Null q95")
        axis.set_xlabel(xlabel); axis.grid(alpha=0.2)
    axes[0].set_ylabel("VerifyGS q95 support ratio")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout(pad=0.7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
