#!/usr/bin/env python3
"""Plot repeated-split external false-acceptance transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    aggregate = payload["aggregate"]
    targets = np.asarray(payload["target_false_acceptance_rates"]) * 100
    methods = [
        ("native", "Native confidence", "#858c94", "o"),
        ("support", "Raw support", "#3f77b5", "s"),
        ("verifygs", "VerifyGS", "#c32738", "o"),
    ]
    datasets = [("eth3d", "ETH3D"), ("bevgsreg", "BEVGSReg")]

    plt.rcParams.update({"font.size": 8.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.35), sharex=True)
    for axis, (dataset, title) in zip(axes, datasets):
        for score, label, color, marker in methods:
            mean = np.asarray([
                aggregate[f"{target / 100:.3f}"][score][dataset][
                    "false_acceptance_rate"
                ]["mean"]
                for target in targets
            ]) * 100
            std = np.asarray([
                aggregate[f"{target / 100:.3f}"][score][dataset][
                    "false_acceptance_rate"
                ]["std"]
                for target in targets
            ]) * 100
            axis.plot(
                targets, mean, color=color, marker=marker, markersize=4.2,
                linewidth=1.6, label=label, zorder=3,
            )
            axis.fill_between(
                targets, np.maximum(mean - std, 0), mean + std,
                color=color, alpha=.13, linewidth=0,
            )
        axis.plot(targets, targets, color="#4a4a4a", linestyle="--", linewidth=.9,
                  label="Calibration target" if dataset == "eth3d" else None)
        axis.set_title(title, fontsize=9.2, weight="bold", pad=3)
        axis.set_ylabel("External FAR (%)", fontsize=8.2)
        axis.grid(alpha=.2, linewidth=.6)
        axis.tick_params(labelsize=7.4)
        axis.set_xlim(.5, 10.5)
    axes[0].set_ylim(0, 40)
    axes[1].set_ylim(0, 60)
    axes[0].legend(
        loc="upper left", fontsize=6.6, frameon=True, framealpha=.94,
        borderpad=.3, labelspacing=.22, handlelength=1.6,
    )
    axes[1].set_xlabel("Target FAR on ScanNet calibration scenes (%)", fontsize=8.2)
    axes[1].set_xticks(targets, ["1", "2.5", "5", "10"])
    fig.tight_layout(pad=.45, h_pad=.55)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=320, bbox_inches="tight")


if __name__ == "__main__":
    main()
