#!/usr/bin/env python3
"""One-shot evaluator for the sealed V71 sensor-codebook development census."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v71_cassandra_pomdp import parse_cassandra_pomdp_file, source_validation
from v71_exact_planning import (
    SensorCodebookKernel,
    best_open_loop_sequence,
    evaluate_policy_exact,
    finite_horizon_return_scale,
    kernel_from_parsed,
    map_control,
    plan_exact,
    plan_myopic,
    posterior_sampling_control,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _policy_root(policy: dict[str, Any], kernel: SensorCodebookKernel) -> dict[str, Any]:
    return {
        "value": float(policy["value"]),
        "selected_action": int(policy["selected_action"]),
        "selected_action_name": kernel.action_names[int(policy["selected_action"])],
        "optimal_actions": [int(action) for action in policy["optimal_actions"]],
        "optimal_action_names": [
            kernel.action_names[int(action)] for action in policy["optimal_actions"]
        ],
        "q_values": [float(value) for value in policy["q_values"]],
    }


def evaluate_record(
    kernel: SensorCodebookKernel,
    record: dict[str, Any],
    *,
    horizon: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    belief = np.asarray(record["joint_belief_latent_by_state"], dtype=np.float64)
    stats: dict[str, int] = {}
    exact = plan_exact(
        kernel, belief, horizon, tie_tolerance=tie_tolerance, stats=stats
    )
    mapped = map_control(kernel, belief, horizon, tie_tolerance=tie_tolerance)
    sampled = posterior_sampling_control(
        kernel, belief, horizon, tie_tolerance=tie_tolerance
    )
    open_loop = best_open_loop_sequence(
        kernel, belief, horizon, tie_tolerance=tie_tolerance
    )
    myopic_policy = plan_myopic(
        kernel, belief, horizon, tie_tolerance=tie_tolerance
    )
    myopic_value = evaluate_policy_exact(kernel, belief, myopic_policy, horizon)
    exact_value = float(exact["value"])
    controls = {
        "map": float(mapped["value"]),
        "posterior_sampling": float(sampled["value"]),
        "open_loop": float(open_loop["value"]),
        "myopic_one_step": float(myopic_value),
    }
    scale = finite_horizon_return_scale(kernel, horizon)
    regrets = {
        name: float((exact_value - value) / scale)
        for name, value in controls.items()
    }
    finite = [
        exact_value,
        scale,
        float(record["public_prefix_probability"]),
        *controls.values(),
        *regrets.values(),
    ]
    map_policy = mapped["policy"]
    return {
        "record_id": record["record_id"],
        "model_file": record["model_file"],
        "depth": int(record["depth"]),
        "public_action": record["public_action"],
        "public_observation": record["public_observation"],
        "public_prefix_probability": float(record["public_prefix_probability"]),
        "joint_belief_sum_error": abs(float(belief.sum()) - 1.0),
        "latent_posterior": belief.sum(axis=1).tolist(),
        "return_scale": scale,
        "exact_bayes_adaptive": _policy_root(exact, kernel),
        "map": {
            "latent": int(mapped["latent"]),
            "latent_name": mapped["latent_name"],
            "latent_mass": float(mapped["latent_mass"]),
            "selected_action": int(map_policy["selected_action"]),
            "selected_action_name": kernel.action_names[
                int(map_policy["selected_action"])
            ],
            "value": controls["map"],
            "on_support": bool(mapped["on_support"]),
            "fallback_count": 0,
        },
        "posterior_sampling": {
            "value": controls["posterior_sampling"],
            "root_action_distribution": sampled["root_action_distribution"],
            "models": sampled["models"],
            "sampled_model_persists_for_full_policy": True,
            "on_support": bool(sampled["on_support"]),
            "fallback_count": 0,
        },
        "open_loop": {
            "value": controls["open_loop"],
            "selected_actions": [int(action) for action in open_loop["selected_actions"]],
            "selected_action_names": [
                kernel.action_names[int(action)]
                for action in open_loop["selected_actions"]
            ],
            "sequence_count": int(open_loop["sequence_count"]),
        },
        "myopic_one_step": {
            "value": controls["myopic_one_step"],
            "selected_action": int(myopic_policy["selected_action"]),
            "selected_action_name": kernel.action_names[
                int(myopic_policy["selected_action"])
            ],
        },
        "normalized_regrets": regrets,
        "exact_ba_map_root_action_disagreement": int(exact["selected_action"])
        != int(map_policy["selected_action"]),
        "point_models_on_support": bool(mapped["on_support"] and sampled["on_support"]),
        "point_model_fallback_count": 0,
        "bellman_nodes": int(stats.get("bellman_nodes", 0)),
        "all_metrics_finite": all(math.isfinite(value) for value in finite),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    source_checks: dict[str, dict[str, bool]],
    *,
    expected_records: int,
) -> dict[str, Any]:
    gates = config["prospectiveDevelopmentGates"]
    threshold = float(config["prospectiveDevelopmentDesign"]["materialNormalizedRegret"])
    models = sorted({row["model_file"] for row in rows})
    roots = {row["model_file"]: row for row in rows if row["depth"] == 0}
    if set(roots) != set(models):
        raise RuntimeError("V71 requires exactly one root record per development model")
    by_model: dict[str, Any] = {}
    disagreement_models = 0
    material_map_models = 0
    material_sampling_models = 0
    for model in models:
        subset = [row for row in rows if row["model_file"] == model]
        root_disagreement = bool(
            roots[model]["exact_ba_map_root_action_disagreement"]
        )
        map_maximum = max(row["normalized_regrets"]["map"] for row in subset)
        sampling_maximum = max(
            row["normalized_regrets"]["posterior_sampling"] for row in subset
        )
        disagreement_models += root_disagreement
        material_map_models += map_maximum >= threshold
        material_sampling_models += sampling_maximum >= threshold
        by_model[model] = {
            "records": len(subset),
            "root_exact_BA_MAP_action_disagreement": root_disagreement,
            "maximum_normalized_regret": {
                control: max(row["normalized_regrets"][control] for row in subset)
                for control in (
                    "map",
                    "posterior_sampling",
                    "open_loop",
                    "myopic_one_step",
                )
            },
            "normalized_regret": {
                control: _summary(
                    [row["normalized_regrets"][control] for row in subset]
                )
                for control in (
                    "map",
                    "posterior_sampling",
                    "open_loop",
                    "myopic_one_step",
                )
            },
        }

    completed_fraction = len(rows) / expected_records
    source_rate = sum(all(checks.values()) for checks in source_checks.values()) / len(
        source_checks
    )
    belief_rate = sum(row["joint_belief_sum_error"] <= 1e-12 for row in rows) / len(rows)
    finite_rate = sum(row["all_metrics_finite"] for row in rows) / len(rows)
    support_rate = sum(row["point_models_on_support"] for row in rows) / len(rows)
    maximum_map_regret = max(row["normalized_regrets"]["map"] for row in rows)
    metrics = {
        "development_model_count": len(models),
        "retained_record_count": len(rows),
        "completed_record_fraction": completed_fraction,
        "source_validation_rate": source_rate,
        "belief_normalization_rate": belief_rate,
        "finite_metric_rate": finite_rate,
        "point_model_on_support_rate": support_rate,
        "models_with_exact_BA_MAP_root_action_disagreement": disagreement_models,
        "models_with_material_MAP_regret": material_map_models,
        "models_with_material_posterior_sampling_regret": material_sampling_models,
        "maximum_normalized_MAP_regret": maximum_map_regret,
        "record_selection_or_rejection_count": 0,
        "protected_confirmation_policy_value_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    gate_results = {
        "minimumDevelopmentModels": len(models) >= gates["minimumDevelopmentModels"],
        "minimumCompletedRecordFraction": completed_fraction
        >= gates["minimumCompletedRecordFraction"],
        "minimumSourceValidationRate": source_rate
        >= gates["minimumSourceValidationRate"],
        "minimumBeliefNormalizationRate": belief_rate
        >= gates["minimumBeliefNormalizationRate"],
        "minimumFiniteMetricRate": finite_rate >= gates["minimumFiniteMetricRate"],
        "minimumPointModelOnSupportRate": support_rate
        >= gates["minimumPointModelOnSupportRate"],
        "minimumModelsWithExactBAMAPRootActionDisagreement": disagreement_models
        >= gates["minimumModelsWithExactBAMAPRootActionDisagreement"],
        "minimumModelsWithMaterialMAPRegret": material_map_models
        >= gates["minimumModelsWithMaterialMAPRegret"],
        "minimumModelsWithMaterialPosteriorSamplingRegret": material_sampling_models
        >= gates["minimumModelsWithMaterialPosteriorSamplingRegret"],
        "minimumMaximumNormalizedMAPRegret": maximum_map_regret
        >= gates["minimumMaximumNormalizedMAPRegret"],
        "maximumRecordSelectionOrRejectionCount": 0
        <= gates["maximumRecordSelectionOrRejectionCount"],
        "maximumProtectedConfirmationPolicyValueCount": 0
        <= gates["maximumProtectedConfirmationPolicyValueCount"],
        "maximumHumanRecordAccessCount": gates["maximumHumanRecordAccessCount"] == 0,
        "maximumModelForwardPassCount": gates["maximumModelForwardPassCount"] == 0,
        "maximumAdapterTrainingRunCount": gates["maximumAdapterTrainingRunCount"] == 0,
    }
    passed = all(gate_results.values())
    return {
        "passed": passed,
        "decision": (
            "authorize_separate_protected_confirmation_preregistration_only"
            if passed
            else "stop_v71_before_any_protected_confirmation_history_or_outcome"
        ),
        "metrics": metrics,
        "gate_results": gate_results,
        "by_model": by_model,
        "full_census_normalized_regret": {
            control: _summary([row["normalized_regrets"][control] for row in rows])
            for control in (
                "map",
                "posterior_sampling",
                "open_loop",
                "myopic_one_step",
            )
        },
    }


def load_preflight(
    evaluator_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    lock = json.loads(evaluator_lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V71 evaluator lock payload hash mismatch")
    for path_key, hash_key in (
        ("evaluator", "evaluator_sha256"),
        ("planning_core", "planning_core_sha256"),
        ("belief_core", "belief_core_sha256"),
        ("source_parser", "source_parser_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V71 locked {path_key} hash mismatch")
    seal_path = PROJECT_ROOT / lock["development_census_seal"]
    if file_sha256(seal_path) != lock["development_census_seal_sha256"]:
        raise RuntimeError("V71 census seal hash mismatch")
    seal = json.loads(seal_path.read_text())
    census_path = PROJECT_ROOT / seal["census"]
    if file_sha256(census_path) != seal["census_sha256"]:
        raise RuntimeError("V71 sealed census hash mismatch")
    source_lock_path = PROJECT_ROOT / seal["source_lock"]
    if file_sha256(source_lock_path) != seal["source_lock_sha256"]:
        raise RuntimeError("V71 source lock hash mismatch")
    config = json.loads(source_lock_path.read_text())["config_payload"]
    records = read_jsonl(census_path)
    if len(records) != seal["record_count"] == lock["expected_records"]:
        raise RuntimeError("V71 census record count mismatch")
    if not lock["authorization"]["run_development_outcomes_once"]:
        raise PermissionError("V71 evaluator lock does not authorize execution")
    if lock["authorization"]["read_protected_confirmation_histories_or_outcomes"]:
        raise PermissionError("V71 evaluator unexpectedly authorizes protected access")
    return lock, config, records, census_path


def run(evaluator_lock_path: Path) -> None:
    lock, config, records, census_path = load_preflight(evaluator_lock_path)
    output_dir = PROJECT_ROOT / "outputs/v71-sensor-codebook/development-evaluation"
    attempt_path = output_dir / "attempt.json"
    rows_path = output_dir / "record-results.jsonl"
    result_path = output_dir / "result.json"
    if output_dir.exists() or attempt_path.exists() or rows_path.exists() or result_path.exists():
        raise RuntimeError("V71 development evaluation has already been attempted")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema_version": "71-sensor-codebook-development",
        "experiment": "v71_sensor_codebook_development_attempt",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "attempt_number": 1,
        "protected_confirmation_policy_value_count": 0,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")

    source_root = (
        PROJECT_ROOT
        / config["source"]["checkout"]
        / config["source"]["modelDirectory"]
    )
    development = set(config["prospectivePartition"]["developmentFresh"])
    unexpected = sorted({row["model_file"] for row in records} - development)
    if unexpected:
        raise RuntimeError(f"sealed V71 census contains non-development models: {unexpected}")
    reliability = float(config["unknownSensorFamily"]["reliability"])
    horizon = int(config["prospectiveDevelopmentDesign"]["horizonActions"])
    tolerance = float(config["prospectiveDevelopmentDesign"]["tieTolerance"])
    kernels: dict[str, SensorCodebookKernel] = {}
    validation: dict[str, dict[str, bool]] = {}
    for filename in sorted(development):
        parsed = parse_cassandra_pomdp_file(source_root / filename)
        checks = source_validation(parsed)
        if not all(checks.values()):
            raise RuntimeError(f"V71 development source no longer validates: {filename}")
        validation[filename] = checks
        kernels[filename] = kernel_from_parsed(parsed, reliability=reliability)

    started = time.perf_counter()
    rows = [
        evaluate_record(
            kernels[record["model_file"]],
            record,
            horizon=horizon,
            tie_tolerance=tolerance,
        )
        for record in records
    ]
    elapsed = time.perf_counter() - started
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    aggregate = aggregate_rows(
        rows, config, validation, expected_records=lock["expected_records"]
    )
    result = {
        "schema_version": "71-sensor-codebook-development",
        "experiment": "v71_sensor_codebook_development_exact_screen",
        **aggregate,
        "runtime_seconds": elapsed,
        "source_validation": validation,
        "records": len(rows),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "access": {
            "development_records_evaluated": len(rows),
            "records_selected_rejected_or_replaced": 0,
            "protected_confirmation_policy_value_count": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator-lock",
        default="configs/v71-sensor-codebook-development-evaluator-lock.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    lock_path = Path(args.evaluator_lock)
    if not lock_path.is_absolute():
        lock_path = PROJECT_ROOT / lock_path
    if args.preflight:
        _, config, records, census_path = load_preflight(lock_path)
        print(
            json.dumps(
                {
                    "passed": True,
                    "records": len(records),
                    "development_models": config["prospectivePartition"][
                        "developmentFresh"
                    ],
                    "census": str(census_path.relative_to(PROJECT_ROOT)),
                    "family": "binary_sensor_codebook_with_identical_point_support",
                    "protected_confirmation_policy_value_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run(lock_path)


if __name__ == "__main__":
    main()
