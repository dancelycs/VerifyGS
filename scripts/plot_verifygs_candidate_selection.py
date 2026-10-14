#!/usr/bin/env python3
"""Visualize pair-level complementarity of public registration candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ORDER = [
    "icp", "fpfh_ransac", "fcgf", "predator", "cofinet",
    "geotransformer", "parenet", "cast", "bufferx", "randpcr",
    "refreshreg", "splatreg",
]
DISPLAY = {
    "icp": "ICP", "fpfh_ransac": "FPFH–RANSAC", "fcgf": "FCGF",
    "predator": "PREDATOR", "cofinet": "CoFiNet",
    "geotransformer": "GeoTransformer", "parenet": "PARE-Net",
    "cast": "CAST", "bufferx": "BUFFER-X", "randpcr": "RandPCR",
    "refreshreg": "RefreshReg", "splatreg": "SplatReg",
}


def load_rows(path: Path, allowed: set[str] | None) -> list[dict]:
    result = []
    for row in json.loads(path.read_text())["records"]:
        if allowed is not None and row["pair"] not in allowed:
            continue
        method = Path(row["source_result"]).name
        if method == "refreshreg_reconstructed":
            method = "refreshreg"
        result.append({**row, "method": method})
    return result


def build_matrix(rows: list[dict]):
    pairs = sorted({row["pair"] for row in rows})
    present = {row["method"] for row in rows}
    methods = [method for method in ORDER if method in present]
    lookup = {(row["method"], row["pair"]): row for row in rows}
    values = np.full((len(methods) + 1, len(pairs)), np.nan)
    for i, method in enumerate(methods):
        for j, pair in enumerate(pairs):
            row = lookup.get((method, pair))
            if row is not None:
                values[i, j] = 2 if row["strict_success"] else (
                    1 if row["loose_success"] else 0
                )
    for j, pair in enumerate(pairs):
        selected = max(
            (row for row in rows if row["pair"] == pair),
            key=lambda row: row["ratio_q95"],
        )
        values[-1, j] = 2 if selected["strict_success"] else (
            1 if selected["loose_success"] else 0
        )
    return pairs, methods + ["verifygs"], values


def draw(axis, title: str, rows: list[dict], cmap, norm):
    pairs, methods, values = build_matrix(rows)
    axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    axis.axhline(len(methods) - 1.5, color="#232b36", lw=1.1)
    labels = [DISPLAY.get(method, "VerifyGS selector") for method in methods]
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=6.2)
    axis.get_yticklabels()[-1].set_weight("bold")
    ticks = np.unique(np.linspace(0, len(pairs) - 1, min(6, len(pairs)), dtype=int))
    axis.set_xticks(ticks, [str(index + 1) for index in ticks], fontsize=6)
    axis.set_xlabel("Map-pair index", fontsize=7)
    axis.set_title(title, fontsize=8.4, weight="bold", pad=3)
    axis.tick_params(length=2, pad=1.5)
    for x in np.arange(-.5, len(pairs), 5):
        axis.axvline(x, color="white", lw=.35, alpha=.75)
    for spine in axis.spines.values():
        spine.set_color("#aeb6c2")
        spine.set_linewidth(.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet-null", type=Path, required=True)
    parser.add_argument("--eth-null", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    heldout = set(protocol["scannet_test_pairs"])
    scan = load_rows(args.scannet_null, heldout)
    eth = load_rows(args.eth_null, None)
    cmap = ListedColormap(["#e8ecf2", "#e9b949", "#238b75"])
    cmap.set_bad("white")
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.52),
                             gridspec_kw={"width_ratios": [1, 1]})
    draw(axes[0], "ScanNet: 41 held-out pairs", scan, cmap, norm)
    draw(axes[1], "ETH3D: 19 transfer-validation pairs", eth, cmap, norm)
    legend = [
        Patch(facecolor="#238b75", label="Strict success"),
        Patch(facecolor="#e9b949", label="Loose only"),
        Patch(facecolor="#e8ecf2", label="Failure"),
        Patch(facecolor="white", edgecolor="#aeb6c2", label="No output"),
    ]
    axes[0].legend(handles=legend, loc="upper center", ncol=2, frameon=True,
                   fontsize=5.2, columnspacing=.55, handlelength=.9,
                   handletextpad=.3, borderpad=.2, labelspacing=.15,
                   framealpha=.92)
    fig.tight_layout(pad=.32, w_pad=.65)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
