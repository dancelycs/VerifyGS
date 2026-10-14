#!/usr/bin/env python3
"""Plot per-public-solver AUROC and AP gains from VerifyGS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METHODS = [
    "bufferx", "cofinet", "fcgf", "fpfh_ransac", "geotransformer",
    "parenet", "refreshreg", "splatreg",
]
DISPLAY = [
    "BUFFER-X", "CoFiNet", "FCGF", "FPFH--RANSAC", "GeoTransformer",
    "PARE-Net", "RefreshReg", "SplatReg",
]


def load(evidence_path: Path, null_path: Path, allowed: set[str] | None):
    evidence = json.loads(evidence_path.read_text())["records"]
    null = json.loads(null_path.read_text())["records"]
    lookup = {(row["source_result"], row["pair"]): row for row in null}
    rows = []
    for row in evidence:
        key = (row["source_result"], row["pair"])
        if key not in lookup or (allowed is not None and row["pair"] not in allowed):
            continue
        method = Path(row["source_result"]).name.replace(
            "refreshreg_reconstructed", "refreshreg"
        )
        rows.append({**row, "method": method, "ratio_q95": lookup[key]["ratio_q95"]})
    return rows


def scores(rows):
    result = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        label = np.asarray([row["loose_success"] for row in selected], dtype=bool)
        result[method] = {"count": len(selected), "positive": int(label.sum())}
        for key in ("internal_consensus", "ratio_q95"):
            value = np.asarray([row[key] for row in selected])
            if len(np.unique(label)) < 2 or (key == "internal_consensus" and np.ptp(value) == 0):
                result[method][key] = None
            else:
                result[method][key] = (
                    roc_auc_score(label, value), average_precision_score(label, value)
                )
    return result


def draw(axis, title, data):
    y = np.arange(len(METHODS))
    for index, method in enumerate(METHODS):
        native = data[method]["internal_consensus"]
        verify = data[method]["ratio_q95"]
        if native is not None and verify is not None:
            axis.plot([native[0], verify[0]], [index + .12, index + .12],
                      color="#aeb4bc", lw=1.0, zorder=1)
            axis.plot([native[1], verify[1]], [index - .12, index - .12],
                      color="#c8cdd3", lw=.9, ls="--", zorder=1)
            axis.scatter(native[0], index + .12, s=20, color="#7f8791", marker="o", zorder=2)
            axis.scatter(native[1], index - .12, s=22, color="#7f8791", marker="^", zorder=2)
        elif native is None:
            axis.text(.02, index, "native N/A", fontsize=5.8, color="#69717c", va="center")
        if verify is not None:
            axis.scatter(verify[0], index + .12, s=22, color="#b6202e", marker="o", zorder=3)
            axis.scatter(verify[1], index - .12, s=24, color="#b6202e", marker="^", zorder=3)
    labels = [f"{name} ({data[m]['positive']}+)" for name, m in zip(DISPLAY, METHODS)]
    axis.set_yticks(y, labels, fontsize=6.8)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.02)
    axis.set_xticks(np.arange(0, 1.01, .2))
    axis.set_xlabel("Discrimination score", fontsize=7.4)
    axis.set_title(title, fontsize=9, weight="bold", pad=3)
    axis.grid(axis="x", alpha=.2)
    axis.tick_params(axis="x", labelsize=6.8)
    for spine in axis.spines.values():
        spine.set_color("#b9c0c9")
        spine.set_linewidth(.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet-evidence", type=Path, required=True)
    parser.add_argument("--scannet-null", type=Path, required=True)
    parser.add_argument("--eth-evidence", type=Path, required=True)
    parser.add_argument("--eth-null", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    heldout = set(protocol["scannet_test_pairs"])
    scan = scores(load(args.scannet_evidence, args.scannet_null, heldout))
    eth = scores(load(args.eth_evidence, args.eth_null, None))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65))
    draw(axes[0], "ScanNet held-out", scan)
    draw(axes[1], "ETH3D transfer validation", eth)
    handles = [
        plt.Line2D([], [], color="#7f8791", marker="o", ls="", label="Pipeline AUROC"),
        plt.Line2D([], [], color="#b6202e", marker="o", ls="", label="VerifyGS AUROC"),
        plt.Line2D([], [], color="#7f8791", marker="^", ls="", label="Pipeline AP"),
        plt.Line2D([], [], color="#b6202e", marker="^", ls="", label="VerifyGS AP"),
    ]
    axes[0].legend(handles=handles, loc="upper left", ncol=2, frameon=True,
                   fontsize=6.0, columnspacing=.65, handletextpad=.35,
                   borderpad=.25, labelspacing=.18, framealpha=.92)
    fig.tight_layout(pad=.45, w_pad=.9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
