#!/usr/bin/env python3
"""Plot the zero-shot ROC and precision-recall curves used in the main paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


def merge(evidence_path, null_path):
    evidence = json.loads(Path(evidence_path).read_text())["records"]
    null = json.loads(Path(null_path).read_text())["records"]
    lookup = {(row["source_result"], row["pair"]): row for row in null}
    return [{**row, "ratio_q95": lookup[(row["source_result"], row["pair"])]["ratio_q95"]}
            for row in evidence]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet-evidence", required=True)
    parser.add_argument("--scannet-null", required=True)
    parser.add_argument("--eth-evidence", required=True)
    parser.add_argument("--eth-null", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Keep the ScanNet inputs in the interface so that this script remains tied
    # to the complete evaluation bundle.  The main-paper figure focuses on the
    # predeclared zero-shot transfer test; all ScanNet values remain in Table 1.
    json.loads(Path(args.protocol).read_text())
    merge(args.scannet_evidence, args.scannet_null)
    eth = merge(args.eth_evidence, args.eth_null)
    methods = [
        ("Solver consensus", "internal_consensus", "#8c8c8c"),
        ("VerifyGS", "ratio_q95", "#b6202e"),
    ]
    plt.rcParams.update({"font.size": 8.2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.15))
    label = np.asarray([row["loose_success"] for row in eth], dtype=bool)
    for method, key, color in methods:
        score = np.asarray([row[key] for row in eth])
        fpr, tpr, _ = roc_curve(label, score)
        precision, recall, _ = precision_recall_curve(label, score)
        axes[0].plot(fpr, tpr, lw=1.65, color=color,
                     label=f"{method} ({auc(fpr, tpr):.3f})")
        axes[1].plot(recall, precision, lw=1.65, color=color,
                     label=f"{method} ({average_precision_score(label, score):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=.8)
    axes[0].set_title("ROC: ETH3D zero-shot", fontsize=9.2)
    axes[1].set_title("Precision--recall: ETH3D zero-shot", fontsize=9.2)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for axis in axes:
        axis.set_xlim(0, 1); axis.set_ylim(0, 1.02); axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=6.5, frameon=True,
                    borderpad=.25, handlelength=1.5, labelspacing=.2)
    fig.tight_layout(pad=.45, h_pad=.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
