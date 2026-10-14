#!/usr/bin/env python3
"""Plot the background-sample ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = [8, 16, 32, 64]
    rows = []
    for count in counts:
        data = json.loads((args.root / f"n{count}" / "summary.json").read_text())
        rows.append(data["results"]["ratio_q95"]["external_zero_shot"])
    fig, left = plt.subplots(figsize=(6.8, 3.2))
    right = left.twinx()
    left.plot(counts, [row["auroc"] for row in rows], "o-", lw=2, label="AUROC", color="#4c78a8")
    left.plot(counts, [row["average_precision"] for row in rows], "s-", lw=2, label="AP", color="#f58518")
    left.plot(counts, [row["recall"] for row in rows], "^-", lw=2, label="Recall", color="#54a24b")
    right.plot(counts, [100 * row["false_acceptance_rate"] for row in rows], "D--", lw=2,
               label="False acceptance", color="#b6202e")
    left.set_xlabel("Number of pair-conditioned background similarities")
    left.set_ylabel("AUROC / AP / recall")
    right.set_ylabel("False-acceptance rate (%)", color="#b6202e")
    left.set_xticks(counts); left.set_ylim(0.5, 1.0); right.set_ylim(0, 10)
    left.grid(alpha=0.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
