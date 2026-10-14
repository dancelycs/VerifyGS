#!/usr/bin/env python3
"""Recompute VERIFYGS scores from prepared maps and released candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verifygs.core import Similarity, null_support_distribution, verify
from verifygs.io import load_pair


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing <pair>.npz files")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Released candidate JSON file or directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-count", type=int, default=64)
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        "--calibration", type=Path,
        default=ROOT / "data" / "final_calibration.json",
        help="Final calibration JSON (default: data/final_calibration.json)",
    )
    calibration_group.add_argument(
        "--no-calibration", action="store_true",
        help="Report observable evidence without a final acceptance decision",
    )
    args = parser.parse_args()
    calibration_failures = None
    affinity_floor = None
    alpha = 0.05
    if not args.no_calibration:
        calibration = json.loads(args.calibration.read_text())
        calibration_failures = np.asarray(
            calibration["failure_ratio_q95"], dtype=np.float64
        )
        affinity_floor = float(calibration["affinity_floor"])
        alpha = float(calibration.get("alpha", alpha))
    files = ([args.candidates] if args.candidates.is_file()
             else sorted(args.candidates.glob("*.json")))
    records = []
    pair_cache = {}
    null_cache = {}
    for candidate_file in files:
        payload = json.loads(candidate_file.read_text())
        method = payload["method"]
        for row in payload["rows"]:
            if row.get("status") != "ok":
                continue
            pair_id = row["pair"]
            if pair_id not in pair_cache:
                pair_cache[pair_id] = load_pair(args.data / f"{pair_id}.npz")
                source_map, target_map = pair_cache[pair_id]
                distribution = null_support_distribution(
                    source_map.means, target_map.means, pair_id, args.null_count
                )
                null_cache[pair_id] = float(np.quantile(distribution, 0.95))
            source, target = pair_cache[pair_id]
            score = verify(
                source, target, Similarity.from_dict(row), pair_id, args.null_count,
                null_q95=null_cache[pair_id],
                calibration_failures=calibration_failures,
                affinity_floor=affinity_floor, alpha=alpha,
            )
            records.append({
                "method": method, "pair": pair_id,
                "strict_success": row.get("strict_success"),
                "loose_success": row.get("success"), **score,
            })
            print(f"{method}: {pair_id}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
