#!/usr/bin/env python3
"""Recreate the released quantitative tables and figures from public records."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "reproduced"
ALIASES = {"buffer_x": "bufferx", "refreshreg_reconstructed": "refreshreg"}
DISPLAY = {
    "icp": "ICP", "fpfh_ransac": "FPFH+RANSAC", "fcgf": "FCGF",
    "predator": "PREDATOR", "cofinet": "CoFiNet",
    "geotransformer": "GeoTransformer", "parenet": "PARE-Net",
    "cast": "CAST", "bufferx": "BUFFER-X", "randpcr": "RandPCR",
    "refreshreg": "RefreshReg (reproduced)", "splatreg": "SplatReg",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def method(row: dict) -> str:
    name = Path(row["source_result"]).name
    return ALIASES.get(name, name)


def merged(dataset: str) -> list[dict]:
    evidence = read(DATA / dataset / "evidence.json")["records"]
    null = read(DATA / dataset / "null.json")["records"]
    lookup = {(method(row), row["pair"]): row for row in null}
    rows = []
    for row in evidence:
        key = (method(row), row["pair"])
        if key in lookup:
            rows.append({**row, "method": key[0], "ratio_q95": lookup[key]["ratio_q95"]})
    return rows


def metrics(rows: list[dict], key: str) -> dict[str, float]:
    labels = np.asarray([row["loose_success"] for row in rows], dtype=bool)
    scores = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def percentile(reference: list[float], value: float) -> float:
    ordered = np.sort(np.asarray(reference, dtype=float))
    return float(np.searchsorted(ordered, value, side="right") / len(ordered))


def add_within_pipeline_rank(rows: list[dict], calibration: list[dict]) -> None:
    by_method = defaultdict(list)
    for row in calibration:
        by_method[row["method"]].append(row["internal_consensus"])
    for row in rows:
        row["within_pipeline_rank"] = percentile(
            by_method[row["method"]], row["internal_consensus"]
        )


def write_table(name: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    directory = OUTPUT / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in columns])
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---:" if index else "---" for index in range(len(columns))) + "|"
    lines = [header, rule]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (directory / f"{name}.md").write_text("\n".join(lines) + "\n")


def discrimination_tables(scan: list[dict], eth: list[dict], protocol: dict) -> dict:
    calibration_ids = set(protocol["calibration_pairs"])
    test_ids = set(protocol["scannet_test_pairs"])
    calibration = [row for row in scan if row["pair"] in calibration_ids]
    scan_test = [row for row in scan if row["pair"] in test_ids]
    methods = {row["method"] for row in calibration}
    native = {
        name for name in methods
        if np.ptp([row["internal_consensus"] for row in calibration if row["method"] == name]) > 0
    }
    scan_paired = [row for row in scan_test if row["method"] in native]
    eth_paired = [row for row in eth if row["method"] in native]
    add_within_pipeline_rank(scan_paired, calibration)
    add_within_pipeline_rank(eth_paired, calibration)
    score_rows = []
    for label, key in (
        ("Pipeline consensus (raw pooled)", "internal_consensus"),
        ("Pipeline consensus (within-pipeline rank)", "within_pipeline_rank"),
        ("Negative trimmed Chamfer", "negative_trimmed_distance"),
        ("Bidirectional support S_0.04", "support_04"),
        ("Scene-conditioned score z", "ratio_q95"),
    ):
        scan_metric, eth_metric = metrics(scan_paired, key), metrics(eth_paired, key)
        score_rows.append({
            "score": label, "scan_auroc": scan_metric["auroc"],
            "scan_ap": scan_metric["average_precision"],
            "eth_auroc": eth_metric["auroc"], "eth_ap": eth_metric["average_precision"],
        })
    write_table("main_discrimination", score_rows, [
        ("score", "Score"), ("scan_auroc", "ScanNet AUROC"),
        ("scan_ap", "ScanNet AP"), ("eth_auroc", "ETH3D AUROC"),
        ("eth_ap", "ETH3D AP"),
    ])
    complete_rows = []
    for label, key in (
        ("Recorded pipeline score", "internal_consensus"),
        ("Strict support S_0.02", "support_02"),
        ("Support S_0.04", "support_04"),
        ("Negative trimmed distance", "negative_trimmed_distance"),
        ("Negative color residual", "negative_color_residual"),
        ("Negative covariance-shape residual", "negative_shape_residual"),
        ("Gaussian affinity", "gaussian_affinity"),
        ("Fixed multichannel diagnostic", "gaussian_verification"),
        ("Scene-conditioned ratio z", "ratio_q95"),
    ):
        scan_metric, eth_metric = metrics(scan_test, key), metrics(eth, key)
        complete_rows.append({
            "score": label, "scan_auroc": scan_metric["auroc"],
            "scan_ap": scan_metric["average_precision"],
            "eth_auroc": eth_metric["auroc"], "eth_ap": eth_metric["average_precision"],
        })
    write_table("complete_evidence", complete_rows, [
        ("score", "Evidence"), ("scan_auroc", "ScanNet AUROC"),
        ("scan_ap", "ScanNet AP"), ("eth_auroc", "ETH3D AUROC"),
        ("eth_ap", "ETH3D AP"),
    ])
    return {
        "calibration_outputs": len(calibration), "scannet_outputs": len(scan_test),
        "eth3d_outputs": len(eth), "paired_scannet_outputs": len(scan_paired),
        "paired_eth3d_outputs": len(eth_paired), "paired_methods": sorted(native),
        "main_discrimination": score_rows, "complete_evidence": complete_rows,
    }


def selection_table(datasets: dict[str, list[dict]], protocol: dict) -> dict:
    test_ids = set(protocol["scannet_test_pairs"])
    datasets = {**datasets, "scannet": [r for r in datasets["scannet"] if r["pair"] in test_ids]}
    methods = sorted({row["method"] for rows in datasets.values() for row in rows})
    table = []
    detail = {}
    for name, rows in datasets.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["pair"]].append(row)
        per_method = {}
        for candidate_method in methods:
            candidates = [r for r in rows if r["method"] == candidate_method]
            if candidates:
                per_method[candidate_method] = {
                    "strict": sum(r["strict_success"] for r in candidates),
                    "loose": sum(r["loose_success"] for r in candidates),
                    "pairs": len(candidates),
                }
        selected = [max(items, key=lambda r: r["support_04"]) for items in grouped.values()]
        oracle = {
            "strict": sum(any(r["strict_success"] for r in items) for items in grouped.values()),
            "loose": sum(any(r["loose_success"] for r in items) for items in grouped.values()),
        }
        detail[name] = {
            "methods": per_method,
            "verifygs_selection": {
                "strict": sum(r["strict_success"] for r in selected),
                "loose": sum(r["loose_success"] for r in selected),
            },
            "oracle": oracle,
        }
    for candidate_method in methods:
        row = {"method": DISPLAY.get(candidate_method, candidate_method)}
        for dataset in datasets:
            value = detail[dataset]["methods"].get(candidate_method)
            row[dataset] = "--" if value is None else f"{value['strict']}/{value['loose']}"
        table.append(row)
    for key, label in (("verifygs_selection", "VERIFYGS selection"), ("oracle", "Oracle upper bound")):
        row = {"method": label}
        for dataset in datasets:
            value = detail[dataset][key]
            row[dataset] = f"{value['strict']}/{value['loose']}"
        table.append(row)
    write_table("candidate_selection", table, [
        ("method", "Pipeline/selector"), ("scannet", "ScanNet"),
        ("eth3d", "ETH3D"), ("bevgsreg", "BEVGSReg"),
    ])
    return detail


def additional_tables(datasets: dict[str, list[dict]], protocol: dict) -> dict:
    """Write the remaining main-paper and supplementary quantitative tables."""
    paper = read(DATA / "paper_metrics.json")
    gate = read(DATA / "gaussian_conformal_gate.json")
    gate_names = {
        "scannet": "scannet_heldout",
        "eth3d": "eth3d_transfer",
        "bevgsreg": "bevgsreg_transfer",
    }
    deployment_rows = []
    for dataset in ("scannet", "eth3d", "bevgsreg"):
        for key, label in (
            ("trimmed_distance", "Trimmed distance"),
            ("raw_support", "Raw support"),
            ("verifygs", "VERIFYGS"),
        ):
            if key == "verifygs":
                result = gate["selection_then_acceptance"][gate_names[dataset]]
                value = {
                    "pair_coverage": result["coverage"],
                    "accepted_risk": result["accepted_risk"],
                    "accepted_correct": result["accepted_correct"],
                    "pairs_with_successful_candidate": result[
                        "pairs_with_successful_candidate"
                    ],
                }
            else:
                value = paper["selection_then_abstention"][dataset][key]
            deployment_rows.append({
                "dataset": dataset, "score": label,
                "coverage": 100 * value["pair_coverage"],
                "risk": 100 * value["accepted_risk"],
                "correct_solvable": (
                    f"{value['accepted_correct']}/{value['pairs_with_successful_candidate']}"
                ),
            })
    write_table("selection_abstention", deployment_rows, [
        ("dataset", "Dataset"), ("score", "Decision score"),
        ("coverage", "Coverage (%)"), ("risk", "Accepted risk (%)"),
        ("correct_solvable", "Correct/solvable"),
    ])

    transfer_rows = []
    for dataset, path, key in (
        ("ETH3D", DATA/"eth3d/risk_transfer.json", "eth3d_transfer_validation"),
        ("BEVGSReg", DATA/"bevgsreg/risk_transfer.json", "bevgsreg39_fixed_transfer"),
    ):
        block = read(path)[key]
        for score_key, label in (
            ("native_confidence_method_specific_thresholds", "Pipeline consensus"),
            ("raw_support_single_threshold", "Raw support"),
            ("verifygs_single_threshold", "VERIFYGS"),
        ):
            if score_key == "verifygs_single_threshold":
                value = gate["fixed_split"][gate_names[dataset.lower()]][
                    "geometry_and_gaussian"
                ]
            else:
                value = block[score_key]
            transfer_rows.append({
                "dataset": dataset, "score": label,
                "positive_negative": (
                    f"{value['positives']}/{value['outputs']-value['positives']}"
                ),
                "coverage": 100*value["coverage"], "risk": 100*(1-value["precision"]),
                "far": 100*value["false_acceptance_rate"], "recall": 100*value["recall"],
            })
    write_table("transferred_operating_points", transfer_rows, [
        ("dataset", "Dataset"), ("score", "Decision score"),
        ("positive_negative", "Positive/negative"), ("coverage", "Coverage (%)"),
        ("risk", "Risk (%)"), ("far", "FAR (%)"), ("recall", "Recall (%)"),
    ])

    gate_rows = []
    for dataset, key in (
        ("ScanNet", "scannet_heldout"),
        ("ETH3D", "eth3d_transfer"),
        ("BEVGSReg", "bevgsreg_transfer"),
    ):
        for decision, label in (
            ("geometry_only", "Geometry"),
            ("geometry_and_gaussian", "Full"),
        ):
            value = gate["fixed_split"][key][decision]
            gate_rows.append({
                "dataset": dataset,
                "decision": label,
                "coverage": 100 * value["coverage"],
                "precision": 100 * value["precision"],
                "far": 100 * value["false_acceptance_rate"],
                "recall": 100 * value["recall"],
            })
    write_table("gaussian_consistency_gate", gate_rows, [
        ("dataset", "Dataset"), ("decision", "Decision"),
        ("coverage", "Coverage (%)"), ("precision", "Precision (%)"),
        ("far", "FAR (%)"), ("recall", "Recall (%)"),
    ])

    paired = read(DATA / "paired_bootstrap.json")
    paired_rows = []
    for dataset, result_key, discrimination_key in (
        ("ScanNet", "scannet_heldout", "scannet_heldout"),
        ("ETH3D", "eth3d_transfer_validation", "eth3d_common_native"),
    ):
        baseline = paper["native_paired_discrimination"][discrimination_key]["scores"]
        for label, metric_key, diff_key in (
            ("AUROC", "auroc", "auroc_difference"),
            ("AP", "average_precision", "ap_difference"),
        ):
            difference = paired[result_key][diff_key]
            low, _, high = difference["bootstrap_95"]
            paired_rows.append({
                "dataset": dataset, "metric": label,
                "consensus": baseline["internal_consensus"][metric_key],
                "verifygs": baseline["ratio_q95"][metric_key],
                "difference": difference["point"],
                "interval": f"[{low:.3f}, {high:.3f}]",
                "probability": 100*difference["probability_improvement_positive"],
            })
    write_table("paired_bootstrap", paired_rows, [
        ("dataset", "Dataset"), ("metric", "Metric"),
        ("consensus", "Consensus"), ("verifygs", "VERIFYGS"),
        ("difference", "Difference"), ("interval", "95% interval"),
        ("probability", "Positive replicates (%)"),
    ])

    intervals = read(DATA / "bootstrap_intervals.json")
    interval_rows = []
    for dataset, key in (("ScanNet", "scannet_heldout"), ("ETH3D", "external_zero_shot")):
        for label, metric_key in (("AUROC", "auroc"), ("AP", "average_precision"),
                                  ("FAR", "false_acceptance_rate")):
            value = intervals[key][metric_key]
            low, _, high = value["bootstrap_95"]
            interval_rows.append({"dataset": dataset, "metric": label,
                                  "point": value["point"],
                                  "interval": f"[{low:.3f}, {high:.3f}]"})
    write_table("bootstrap_intervals", interval_rows, [
        ("dataset", "Dataset"), ("metric", "Metric"),
        ("point", "Point"), ("interval", "95% interval"),
    ])

    coverage_rows = []
    for dataset, key in (("ETH3D", "eth3d_common"), ("BEVGSReg", "bevgsreg_common")):
        for score_key, label in (("support_04", "Support"), ("ratio_q95", "VERIFYGS")):
            values = paper["matched_coverage"][key][score_key]
            coverage_rows.append({
                "dataset": dataset, "score": label,
                "20": 100*values["20pct"]["accepted_risk"],
                "30": 100*values["30pct"]["accepted_risk"],
                "40": 100*values["40pct"]["accepted_risk"],
            })
    write_table("matched_coverage_risk", coverage_rows, [
        ("dataset", "Dataset"), ("score", "Score"),
        ("20", "Risk at 20%"), ("30", "Risk at 30%"), ("40", "Risk at 40%"),
    ])

    held_out = gate["scene_and_pipeline_disjoint"]["per_method"]
    unseen_rows = []
    for specification in held_out:
        name = specification["held_out_method"]
        tp = specification["true_accepts"]
        fp = specification["false_accepts"]
        fn = specification["positives"] - tp
        tn = specification["outputs"] - specification["positives"] - fp
        unseen_rows.append({
            "method": DISPLAY.get(name, name), "positives": tp+fn,
            "confusion": f"{tp}/{fp}/{fn}/{tn}",
            "far": 100*fp/max(fp+tn,1), "precision": 100*tp/max(tp+fp,1),
            "recall": 100*tp/max(tp+fn,1),
        })
    pooled = gate["scene_and_pipeline_disjoint"]["pooled"]
    tp = pooled["true_accepts"]
    fp = pooled["false_accepts"]
    fn = pooled["positives"] - tp
    tn = pooled["outputs"] - pooled["positives"] - fp
    unseen_rows.append({
        "method":"All held-out pipelines", "positives": pooled["positives"],
        "confusion": f"{tp}/{fp}/{fn}/{tn}",
        "far": 100 * pooled["false_acceptance_rate"],
        "precision": 100 * pooled["precision"],
        "recall": 100 * pooled["recall"],
    })
    write_table("unseen_pipeline", unseen_rows, [
        ("method", "Held-out pipeline"), ("positives", "Positives"),
        ("confusion", "TP/FP/FN/TN"), ("far", "FAR (%)"),
        ("precision", "Precision (%)"), ("recall", "Recall (%)"),
    ])

    perturbation = read(DATA / "controlled_perturbations.json")["summary"]
    indexed = {key: {row["level"]: row["mean"] for row in values}
               for key, values in perturbation.items()}
    perturb_rows = [
        {"type":"Rotation", "none":indexed["rotation_deg"][0],
         "mild":indexed["rotation_deg"][10], "moderate":indexed["rotation_deg"][30],
         "severe":indexed["rotation_deg"][60]},
        {"type":"Translation", "none":indexed["translation_extent"][0],
         "mild":indexed["translation_extent"][.04], "moderate":indexed["translation_extent"][.16],
         "severe":indexed["translation_extent"][.32]},
        {"type":"Scale", "none":indexed["scale_percent"][0],
         "mild":indexed["scale_percent"][10], "moderate":indexed["scale_percent"][40],
         "severe":indexed["scale_percent"][60]},
    ]
    write_table("controlled_perturbations", perturb_rows, [
        ("type", "Type"), ("none", "None"), ("mild", "Mild"),
        ("moderate", "Moderate"), ("severe", "Severe"),
    ])

    object_metrics = read(DATA/"objaverse/multichannel.json")["metrics"]["loose_success"]
    object_ratio = metrics(merged("objaverse"), "ratio_q95")["auroc"]
    object_rows = [
        {"evidence":"Spatial support", "auroc":object_metrics["support_04"]["auroc"]},
        {"evidence":"Scene-conditioned ratio z", "auroc":object_ratio},
        {"evidence":"Gaussian affinity", "auroc":object_metrics["gaussian_affinity"]["auroc"]},
        {"evidence":"Degree-zero color agreement", "auroc":object_metrics["negative_color_residual"]["auroc"]},
        {"evidence":"Covariance-shape agreement", "auroc":object_metrics["negative_shape_residual"]["auroc"]},
        {"evidence":"Fixed multichannel null rule", "auroc":object_metrics["null_second_best_percentile"]["auroc"]},
    ]
    write_table("objaverse_stress_test", object_rows, [("evidence","Evidence"),("auroc","AUROC")])

    runtime = read(DATA / "runtime.json")
    runtime_rows = [{
        "gaussians_per_map": runtime["gaussians_per_map"],
        "candidate_ms": 1000*runtime["candidate_seconds_median"],
        "null_ms": 1000*runtime["null_seconds_median"],
        "first_ms": 1000*runtime["amortized_one_candidate_seconds_median"],
        "memory_mb": runtime["peak_process_rss_mb"],
    }]
    write_table("runtime", runtime_rows, [
        ("gaussians_per_map", "Gaussians/map"), ("candidate_ms", "Candidate (ms)"),
        ("null_ms", "64-pose null (ms)"), ("first_ms", "First decision (ms)"),
        ("memory_mb", "Peak RSS (MB)"),
    ])
    return {"generated_table_count": 10, "unseen_pipeline_pooled": unseen_rows[-1]}


def run_plots() -> list[str]:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        ("plot_verifygs_solver_gains.py", [
            "--scannet-evidence", DATA/"scannet/evidence.json", "--scannet-null", DATA/"scannet/null.json",
            "--eth-evidence", DATA/"eth3d/evidence.json", "--eth-null", DATA/"eth3d/null.json",
            "--protocol", DATA/"protocol.json", "--out", figure_dir/"solver_gains",
        ]),
        ("plot_verifygs_candidate_selection.py", [
            "--scannet-null", DATA/"scannet/null.json", "--eth-null", DATA/"eth3d/null.json",
            "--protocol", DATA/"protocol.json", "--out", figure_dir/"candidate_selection",
        ]),
        ("plot_verifygs_risk_coverage.py", [
            "--eth-evidence", DATA/"eth3d/evidence.json", "--eth-null", DATA/"eth3d/null.json",
            "--bev-evidence", DATA/"bevgsreg/evidence.json", "--bev-null", DATA/"bevgsreg/null.json",
            "--protocol", DATA/"protocol.json", "--risk-summary", DATA/"eth3d/risk_transfer.json",
            "--bev-risk-summary", DATA/"bevgsreg/risk_transfer.json",
            "--gate-summary", DATA/"gaussian_conformal_gate.json",
            "--out", figure_dir/"risk_coverage",
        ]),
        ("plot_verifygs_gaussian_gate.py", [
            "--summary", DATA/"gaussian_conformal_gate.json",
            "--out", figure_dir/"gaussian_gate_gain",
        ]),
        ("plot_verifygs_null_ablation.py", [
            "--root", DATA/"null_count_ablation", "--out", figure_dir/"null_count",
        ]),
        ("plot_verifygs_perturbations.py", [
            "--data", DATA/"controlled_perturbations.json", "--out", figure_dir/"controlled_perturbations",
        ]),
        ("plot_verifygs_calibration_robustness.py", [
            "--summary", DATA/"calibration_robustness.json", "--out", figure_dir/"calibration_robustness",
        ]),
        ("plot_verifygs_roc_pr.py", [
            "--scannet-evidence", DATA/"scannet/evidence.json", "--scannet-null", DATA/"scannet/null.json",
            "--eth-evidence", DATA/"eth3d/evidence.json", "--eth-null", DATA/"eth3d/null.json",
            "--protocol", DATA/"protocol.json", "--out", figure_dir/"roc_pr",
        ]),
    ]
    generated = []
    for script, arguments in commands:
        command = [sys.executable, str(ROOT/"scripts"/script)] + [str(v) for v in arguments]
        subprocess.run(command, check=True, cwd=ROOT)
        generated.append(Path(script).stem.replace("plot_verifygs_", ""))
    return generated


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    protocol = read(DATA / "protocol.json")
    datasets = {name: merged(name) for name in ("scannet", "eth3d", "bevgsreg")}
    report = discrimination_tables(datasets["scannet"], datasets["eth3d"], protocol)
    report["selection"] = selection_table(datasets, protocol)
    report["additional_tables"] = additional_tables(datasets, protocol)
    report["generated_figures"] = run_plots()
    gate = read(DATA / "gaussian_conformal_gate.json")
    report["final_decision"] = gate["calibration"]
    report["checks"] = {
        "scannet_complete_pool": report["scannet_outputs"] == 410,
        "eth3d_complete_pool": report["eth3d_outputs"] == 223,
        "paired_scannet_pool": report["paired_scannet_outputs"] == 369,
        "paired_eth3d_pool": report["paired_eth3d_outputs"] == 166,
    }
    if not all(report["checks"].values()):
        raise RuntimeError(f"Released-record validation failed: {report['checks']}")
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "ok", "checks": report["checks"],
                      "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
