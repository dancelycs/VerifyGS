"""Command-line interface for scoring one registration candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import verify
from .io import load_candidate, load_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pair-id", help="Defaults to the NPZ filename stem")
    parser.add_argument("--null-count", type=int, default=64)
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--threshold", type=float)
    decision.add_argument(
        "--calibration", type=Path,
        help="JSON containing failure_ratio_q95, affinity_floor, and alpha",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source, target = load_pair(args.pair)
    calibration_failures = None
    affinity_floor = None
    alpha = 0.05
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text())
        calibration_failures = np.asarray(
            calibration["failure_ratio_q95"], dtype=np.float64
        )
        affinity_floor = float(calibration["affinity_floor"])
        alpha = float(calibration.get("alpha", alpha))
    result = verify(
        source, target, load_candidate(args.candidate),
        pair_id=args.pair_id or args.pair.stem,
        null_count=args.null_count, threshold=args.threshold,
        calibration_failures=calibration_failures,
        affinity_floor=affinity_floor, alpha=alpha,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
