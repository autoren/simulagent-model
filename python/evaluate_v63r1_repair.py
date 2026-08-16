#!/usr/bin/env python3
"""Run the one-change V63r1 repeat-pooling measurement repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_v63_inference import (
    aggregate_exact,
    exact_controls,
    gate_checks,
    inference_metrics,
    normalization_ok,
    read_jsonl,
    run_parallel,
    summarize_diagnostics,
    weighted_quantile,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v63_external_inference import exact_inference, load_anchor, logsumexp, smc2_inference


def average_map(results: list[dict], key: str) -> dict[str, float]:
    combined: dict[str, float] = {}
    count = len(results)
    for result in results:
        for item, value in result[key].items():
            combined[item] = combined.get(item, 0.0) + float(value) / count
    total = sum(combined.values())
    return {item: value / total for item, value in combined.items()}


def average_sequence(results: list[dict], key: str) -> list[float]:
    values = np.mean(np.asarray([result[key] for result in results], dtype=np.float64), axis=0)
    values /= values.sum()
    return values.tolist()


def pool_smc2_repeats(results: list[dict]) -> dict[str, Any]:
    """Equal-weight posterior mixture matching the frozen V53r2 aggregation rule."""
    if not results:
        raise ValueError("V63r1 cannot pool an empty repeat set")
    count = len(results)
    theta_values, theta_weights, atoms = [], [], []
    for result in results:
        theta_values.extend(map(float, result["theta_values"]))
        theta_weights.extend(float(weight) / count for weight in result["theta_weights"])
        atoms.extend({**atom, "weight": float(atom["weight"]) / count} for atom in result["atoms"])
    theta_total = sum(theta_weights)
    atom_total = sum(float(atom["weight"]) for atom in atoms)
    theta_weights = [weight / theta_total for weight in theta_weights]
    for atom in atoms:
        atom["weight"] = float(atom["weight"]) / atom_total
    return {
        "identity": average_sequence(results, "identity"),
        "theta_values": theta_values,
        "theta_weights": theta_weights,
        "joint_bins": average_map(results, "joint_bins"),
        "current_side": average_sequence(results, "current_side"),
        "next_observation": average_sequence(results, "next_observation"),
        "log_evidence": logsumexp([float(result["log_evidence"]) for result in results])
        - math.log(count),
        "atoms": atoms,
        "repeat_results": results,
    }


def merge_repeat_diagnostics(results: list[dict]) -> dict[str, Any]:
    rows = [summarize_diagnostics(result) for result in results]
    return {
        "outer_stream_count": sum(row["outer_stream_count"] for row in rows),
        "outer_unique_stream_count": sum(row["outer_unique_stream_count"] for row in rows),
        "inner_stream_count": sum(row["inner_stream_count"] for row in rows),
        "inner_unique_stream_count": sum(row["inner_unique_stream_count"] for row in rows),
        "outer_fingerprint_count": sum(row["outer_fingerprint_count"] for row in rows),
        "outer_unique_fingerprint_count": sum(row["outer_unique_fingerprint_count"] for row in rows),
        "inner_fingerprint_count": sum(row["inner_fingerprint_count"] for row in rows),
        "inner_unique_fingerprint_count": sum(row["inner_unique_fingerprint_count"] for row in rows),
        "mean_final_outer_ess_fraction": float(np.mean([
            row["mean_final_outer_ess_fraction"] for row in rows
        ])),
        "mean_distinct_theta_ancestor_fraction": float(np.mean([
            row["mean_distinct_theta_ancestor_fraction"] for row in rows
        ])),
        "move_attempts": sum(row["move_attempts"] for row in rows),
        "move_accepts": sum(row["move_accepts"] for row in rows),
    }


def pooled_exact_worker(payload: tuple) -> dict[str, Any]:
    anchor, record, truth, config = payload
    exact = exact_inference(anchor, record, config)
    low, high = config["unknownDynamicsFamily"]["continuousParameter"]["support"]
    rows = []
    repeats = int(config["smcSquared"]["independentRepeatsOnExactBenchmark"])
    for budget in config["smcSquared"]["outerThetaParticleBudgets"]:
        independent = [
            smc2_inference(anchor, record, config, int(budget), repeat, "exact")
            for repeat in range(repeats)
        ]
        pooled = pool_smc2_repeats(independent)
        rows.append({
            "budget": int(budget),
            "metrics": inference_metrics(exact, pooled, float(high - low)),
            "normalization_ok": normalization_ok(pooled),
            "truth_identity_mass": float(pooled["identity"][int(truth["identity"])]),
            "exact_max_identity_mass": max(exact["identity"]),
            "approximate_max_identity_mass": max(pooled["identity"]),
            "exact_theta_central_80_width": (
                weighted_quantile(exact["theta_values"], exact["theta_weights"], 0.9)
                - weighted_quantile(exact["theta_values"], exact["theta_weights"], 0.1)
            ),
            "approximate_unique_theta_fraction": (
                len(set(map(float, pooled["theta_values"]))) / len(pooled["theta_values"])
            ),
            "diagnostics": merge_repeat_diagnostics(independent),
        })
    return {
        "id": record["id"],
        "truth_identity": truth["identity"],
        "truth_theta": truth["theta"],
        "exact_normalization_ok": normalization_ok(exact),
        "controls": exact_controls(exact, config),
        "rows": rows,
    }


def subsection_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock", default="configs/v63r1-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v63r1-repeat-pooling-repair/evaluation"
    )
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_repair_evaluation"]:
        raise RuntimeError("V63r1 evaluator lock does not authorize the repair run")
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    source_outcome = json.loads((PROJECT_ROOT / design["source_v63_outcome_lock"]).read_text())
    original_result_path = (PROJECT_ROOT / source_outcome["result"]).resolve()
    original = json.loads(original_result_path.read_text())
    original_evaluation_lock = json.loads(
        (PROJECT_ROOT / source_outcome["evaluation_implementation_lock"]).read_text()
    )
    population_seal = json.loads(
        (PROJECT_ROOT / original_evaluation_lock["population_seal"]).read_text()
    )
    implementation = json.loads((PROJECT_ROOT / population_seal["implementation_lock"]).read_text())
    original_design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = original_design["config_payload"]
    population_manifest = json.loads((PROJECT_ROOT / population_seal["manifest"]).read_text())
    population_root = PROJECT_ROOT / population_manifest["population_root"]
    for path, digest in population_seal["file_sha256"].items():
        if file_sha256(population_root / path) != digest:
            raise RuntimeError(f"V63r1 population hash mismatch: {path}")
    for path, digest in implementation["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"V63r1 candidate source hash mismatch: {path}")
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError("V63r1 repair result already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = output_dir / "attempt.json"
    if not attempt_path.exists():
        attempt_path.write_text(json.dumps({
            "schema_version": "63r1",
            "logical_repair_attempt": 1,
            "evaluation_lock_sha256": file_sha256(lock_path),
            "original_v63_result_sha256": file_sha256(original_result_path),
        }, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    if attempt["logical_repair_attempt"] != 1:
        raise RuntimeError("V63r1 permits exactly one logical repair attempt")
    anchor = load_anchor(PROJECT_ROOT / config["externalSource"]["sealedModel"])
    public = read_jsonl(population_root / "exact-public.jsonl")
    truth = read_jsonl(population_root / "exact-truth.jsonl")
    started = time.time()
    records = run_parallel(
        pooled_exact_worker,
        [(anchor, record, target, config) for record, target in zip(public, truth, strict=True)],
        args.workers,
        output_dir / "exact-checkpoints",
    )
    exact_summary = aggregate_exact(records, config)
    sbc_summary = original["simulation_based_calibration"]
    scale_summary = original["scale_stress"]
    runtime_summary = original["runtime_crosscheck"]
    implementation_audit = json.loads((PROJECT_ROOT / implementation["implementation_audit"]).read_text())
    checks = gate_checks(
        exact_summary, sbc_summary, scale_summary, runtime_summary, config, implementation_audit
    )
    passed = all(checks.values())
    result = {
        "schema_version": "63r1",
        "experiment": "v63r1_repeat_pooling_measurement_repair",
        "passed": passed,
        "decision": (
            "authorize_preregistration_of_separate_multi_action_external_EIG_stage"
            if passed else "V63r1_pooling_repair_failed_block_active_selection"
        ),
        "measurement_repair_not_independent_replication": True,
        "original_v63_remains_failed": True,
        "gate_checks": checks,
        "exact_benchmark": exact_summary,
        "simulation_based_calibration": sbc_summary,
        "scale_stress": scale_summary,
        "runtime_crosscheck": runtime_summary,
        "reuse_bindings": {
            "original_v63_result": str(original_result_path.relative_to(PROJECT_ROOT)),
            "original_v63_result_sha256": file_sha256(original_result_path),
            "sbc_summary_sha256": subsection_sha256(sbc_summary),
            "scale_summary_sha256": subsection_sha256(scale_summary),
            "runtime_summary_sha256": subsection_sha256(runtime_summary),
            "original_runtime_result": "outputs/v63-external-unknown-dynamics/evaluation/runtime-result.json",
            "original_runtime_result_sha256": file_sha256(
                PROJECT_ROOT / "outputs/v63-external-unknown-dynamics/evaluation/runtime-result.json"
            ),
        },
        "bindings": {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "population_seal": original_evaluation_lock["population_seal"],
            "population_seal_sha256": file_sha256(
                PROJECT_ROOT / original_evaluation_lock["population_seal"]
            ),
            "candidate_source_sha256": implementation["source_sha256"],
        },
        "access": {
            "logical_repair_attempts": 1,
            "original_v63_reruns": 0,
            "exact_repair_records": len(records),
            "SBC_reruns": 0,
            "scale_reruns": 0,
            "runtime_reruns": 0,
            "candidate_truth_fields_passed_to_inference": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
        "runtime_seconds_non_gating": time.time() - started,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
