#!/usr/bin/env python3
"""Plot external-dataset selective risk for public confidence and VerifyGS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def merge(evidence_path, null_path):
    evidence = json.loads(Path(evidence_path).read_text())["records"]
    null = json.loads(Path(null_path).read_text())["records"]
    lookup = {(row["source_result"], row["pair"]): row for row in null}
    return [{**row, "ratio_q95": lookup[(row["source_result"], row["pair"])]["ratio_q95"]}
            for row in evidence]


def curve(records, key):
    total = len(records)
    if key == "internal_consensus":
        # A constant field is unavailable, not a chance-level confidence.
        records = [row for row in records
                   if Path(row["source_result"]).name != "splatreg"]
    labels = np.asarray([row["loose_success"] for row in records], dtype=bool)
    scores = np.asarray([row[key] for row in records])
    order = np.argsort(-scores)
    sorted_label = labels[order]
    accepted = np.arange(1, len(labels) + 1)
    coverage = accepted / total
    false_discovery = np.cumsum(~sorted_label) / accepted
    return coverage, false_discovery


def point(records, key, threshold):
    labels = np.asarray([row["loose_success"] for row in records], dtype=bool)
    scores = np.asarray([row[key] for row in records])
    keep = scores >= threshold
    return float(keep.mean()), float((keep & ~labels).sum() / max(int(keep.sum()), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-evidence", required=True)
    parser.add_argument("--eth-null", required=True)
    parser.add_argument("--bev-evidence", required=True)
    parser.add_argument("--bev-null", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--risk-summary", required=True)
    parser.add_argument("--bev-risk-summary", required=True)
    parser.add_argument("--gate-summary", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    risk_summary = json.loads(Path(args.risk_summary).read_text())
    bev_risk_summary = json.loads(Path(args.bev_risk_summary).read_text())
    gate_summary = json.loads(Path(args.gate_summary).read_text())["fixed_split"]
    common = set(risk_summary["common_public_methods"])
    eth = [row for row in merge(args.eth_evidence, args.eth_null)
           if Path(row["source_result"]).name.replace(
               "refreshreg_reconstructed", "refreshreg") in common]
    bev_common = set(bev_risk_summary["common_public_methods"])
    bev = [row for row in merge(args.bev_evidence, args.bev_null)
           if Path(row["source_result"]).name.replace("buffer_x", "bufferx") in bev_common]
    methods = [("Pipeline consensus", "internal_consensus", "#7f8791"),
               ("Raw support", "support_04", "#3f77b5"),
               ("VerifyGS geometry rank", "ratio_q95", "#b6202e")]
    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.48), sharey=True)
    eth_key = (
        "eth3d_transfer_validation"
        if "eth3d_transfer_validation" in risk_summary
        else "eth3d_zero_shot"
    )
    datasets = [
        ("ETH3D transfer validation", eth, risk_summary[eth_key], "eth3d_transfer"),
        ("BEVGSReg external transfer", bev,
         bev_risk_summary["bevgsreg39_fixed_transfer"], "bevgsreg_transfer"),
    ]
    summary_keys = {
        "internal_consensus": "native_confidence_method_specific_thresholds",
        "support_04": "raw_support_single_threshold",
        "ratio_q95": "verifygs_single_threshold",
    }
    for axis, (title, records, summary, gate_key) in zip(axes, datasets):
        for name, key, color in methods:
            coverage, risk = curve(records, key)
            axis.plot(100 * coverage, 100 * risk, lw=1.5, color=color, label=name)
            if key == "internal_consensus":
                operating = summary[summary_keys[key]]
                x = operating["coverage"]
                y = 1.0 - operating["precision"]
            elif key == "ratio_q95":
                operating = gate_summary[gate_key]["geometry_and_gaussian"]
                x = operating["coverage"]
                y = 1.0 - operating["precision"]
            else:
                threshold_value = protocol["results"][key]["threshold"]
                x, y = point(records, key, threshold_value)
            axis.scatter([100 * x], [100 * y], s=25, color=color,
                         edgecolor="white", linewidth=.5, zorder=3)
        axis.set_title(title, fontsize=8.5, weight="bold", pad=3)
        axis.set_xlabel("Accepted coverage (%)", fontsize=7)
        axis.set_ylabel("Error among accepted (%)", fontsize=7)
        axis.set_xlim(0, 100); axis.set_xticks(np.arange(0, 101, 20))
        axis.set_ylim(0, 100); axis.grid(alpha=0.2)
        axis.tick_params(labelsize=6.5)
    axes[0].legend(loc="upper right", fontsize=6.1, frameon=True,
                   borderpad=.25, labelspacing=.2, handlelength=1.5,
                   framealpha=.92)
    fig.tight_layout(pad=.35, w_pad=.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
