#!/usr/bin/env python3
"""Visualize the gain and cost of the Gaussian-consistency veto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    ("scannet_heldout", "ScanNet", "#3f77b5"),
    ("eth3d_transfer", "ETH3D", "#d97925"),
    ("bevgsreg_transfer", "BEVGSReg", "#6b4c9a"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    repeated = data["repeated_scene_disjoint_splits"]["records"]
    fixed = data["fixed_split"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.7,
        "pdf.fonttype": 42, "axes.linewidth": 0.75,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72))

    # (a) Paired FAR change over the same ten scene-disjoint partitions.
    ax = axes[0]
    for index, (key, label, color) in enumerate(DATASETS):
        rows = sorted(
            (row for row in repeated if row["dataset"] == key),
            key=lambda row: row["split"],
        )
        geometry = 100 * np.asarray([
            row["geometry_only"]["false_acceptance_rate"] for row in rows
        ])
        full = 100 * np.asarray([
            row["geometry_and_gaussian"]["false_acceptance_rate"] for row in rows
        ])
        left, right = index - .14, index + .14
        for first, second in zip(geometry, full):
            ax.plot([left, right], [first, second], color="#aeb5bd",
                    lw=.72, alpha=.72, zorder=1)
        ax.scatter(np.full(len(rows), left), geometry, s=18, facecolor="white",
                   edgecolor=color, linewidth=.85, zorder=2)
        ax.scatter(np.full(len(rows), right), full, s=20, facecolor=color,
                   edgecolor="white", linewidth=.45, zorder=3)
        ax.plot([left, right], [geometry.mean(), full.mean()], color=color,
                lw=2.0, zorder=4)
        ax.scatter([left, right], [geometry.mean(), full.mean()], marker="D",
                   s=28, color=color, edgecolor="white", linewidth=.55, zorder=5)
    ax.set_xticks(range(len(DATASETS)), [item[1] for item in DATASETS])
    ax.set_ylabel("False-acceptance rate (%)")
    ax.set_title("Repeated scene-disjoint calibration", fontsize=8.7,
                 weight="bold", pad=4)
    ax.grid(axis="y", alpha=.20, linewidth=.6)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7.1)
    ax.scatter([], [], s=20, facecolor="white", edgecolor="#59636e",
               label="Geometry gate")
    ax.scatter([], [], s=20, facecolor="#59636e", edgecolor="white",
               label="Full VERIFYGS")
    ax.legend(loc="upper left", fontsize=6.6, frameon=True, framealpha=.94,
              borderpad=.28, labelspacing=.25, handletextpad=.35)
    ax.text(.985, .975, "(a)", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.2, weight="bold")

    # (b) Fixed-split movement in the selective operating plane.
    ax = axes[1]
    for key, label, color in DATASETS:
        geometry = fixed[key]["geometry_only"]
        full = fixed[key]["geometry_and_gaussian"]
        x0, x1 = 100 * geometry["coverage"], 100 * full["coverage"]
        y0, y1 = 100 * (1 - geometry["precision"]), 100 * (1 - full["precision"])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.55,
                                    mutation_scale=10, shrinkA=5, shrinkB=5))
        ax.scatter(x0, y0, s=42, facecolor="white", edgecolor=color,
                   linewidth=1.25, zorder=3)
        ax.scatter(x1, y1, s=52, marker="*", facecolor=color,
                   edgecolor="white", linewidth=.55, zorder=4)
        offset = {"ScanNet": (1.0, 1.0), "ETH3D": (.7, 1.0),
                  "BEVGSReg": (.7, -2.4)}[label]
        ax.annotate(label, (x1, y1), xytext=offset, textcoords="offset points",
                    fontsize=6.9, color=color, weight="bold")
    ax.set_xlabel("Accepted coverage (%)")
    ax.set_ylabel("Error among accepted (%)")
    ax.set_title("Effect at the fixed operating point", fontsize=8.7,
                 weight="bold", pad=4)
    ax.grid(alpha=.20, linewidth=.6)
    ax.set_xlim(5, 32); ax.set_ylim(0, 39)
    ax.tick_params(labelsize=7.1)
    ax.scatter([], [], s=36, facecolor="white", edgecolor="#59636e",
               label="Geometry gate")
    ax.scatter([], [], s=44, marker="*", facecolor="#59636e",
               edgecolor="white", label="Full VERIFYGS")
    ax.legend(loc="upper right", fontsize=6.6, frameon=True, framealpha=.94,
              borderpad=.28, labelspacing=.25, handletextpad=.35)
    ax.text(.012, .975, "(b)", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, weight="bold")

    fig.tight_layout(pad=.45, w_pad=1.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=320, bbox_inches="tight")


if __name__ == "__main__":
    main()
