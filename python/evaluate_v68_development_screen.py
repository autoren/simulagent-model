#!/usr/bin/env python3
"""One-shot evaluator for the sealed V68 development-only sensitivity census."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import POMDPModel, validate_model
from v66_bayes_adaptive_reward import (
    evaluate_policy,
    map_model_policy,
    persistent_posterior_sampling_mixture,
    plan_bayes_adaptive,
    plan_information_only_policy,
    plan_myopic_reward_policy,
    static_entropy,
)
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v68_multi_environment_exact import (
    CommandChannelFamily,
    best_open_loop_sequence,
    build_command_channel_family,
    filter_action_observation_history,
    finite_horizon_return_scale,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonable_policy_root(policy: dict[str, Any], action_names: Sequence[str]) -> dict[str, Any]:
    return {
        "selected_action": int(policy["selected_action"]),
        "selected_action_name": action_names[int(policy["selected_action"])],
        "optimal_actions": [int(action) for action in policy["optimal_actions"]],
        "optimal_action_names": [action_names[int(action)] for action in policy["optimal_actions"]],
        "q_values": [float(value) for value in policy["q_values"]],
        "value": float(policy["value"]),
    }


def evaluate_record(
    model_file: str,
    primary: CommandChannelFamily,
    convergence: CommandChannelFamily,
    record: dict[str, Any],
    *,
    horizon: int,
    tie_tolerance: float,
    posterior_sampling_points: int,
    posterior_sampling_offset: float,
) -> dict[str, Any]:
    if record["model_file"] != model_file:
        raise ValueError("V68 record/model binding mismatch")
    primary_belief, primary_log_evidence = filter_action_observation_history(
        primary, record["actions"], record["observations"]
    )
    convergence_belief, convergence_log_evidence = filter_action_observation_history(
        convergence, record["actions"], record["observations"]
    )
    if abs(primary_log_evidence - float(record["log_evidence"])) > 1e-10:
        raise RuntimeError("V68 sealed primary log evidence mismatch")
    if abs(convergence_log_evidence - primary_log_evidence) > 2e-10:
        raise RuntimeError("V68 quadrature log evidence mismatch")

    primary_stats: dict[str, int] = {}
    convergence_stats: dict[str, int] = {}
    exact = plan_bayes_adaptive(
        primary.kernel,
        primary_belief,
        horizon,
        tie_tolerance=tie_tolerance,
        stats=primary_stats,
    )
    converged = plan_bayes_adaptive(
        convergence.kernel,
        convergence_belief,
        horizon,
        tie_tolerance=tie_tolerance,
        stats=convergence_stats,
    )
    map_result = map_model_policy(
        primary.kernel, primary_belief, horizon, tie_tolerance=tie_tolerance
    )
    posterior_sampling = persistent_posterior_sampling_mixture(
        primary.kernel,
        primary_belief,
        horizon,
        points=posterior_sampling_points,
        offset=posterior_sampling_offset,
        tie_tolerance=tie_tolerance,
    )
    open_loop = best_open_loop_sequence(
        primary.kernel,
        primary_belief,
        horizon,
        tie_tolerance=tie_tolerance,
    )
    myopic_policy = plan_myopic_reward_policy(
        primary.kernel, primary_belief, horizon, tie_tolerance=tie_tolerance
    )
    information_policy = plan_information_only_policy(
        primary.kernel, primary_belief, horizon, tie_tolerance=tie_tolerance
    )
    myopic_value = evaluate_policy(
        primary.kernel, primary_belief, myopic_policy, horizon
    )
    information_value = evaluate_policy(
        primary.kernel, primary_belief, information_policy, horizon
    )
    scale = finite_horizon_return_scale(primary.model, horizon)
    exact_value = float(exact["value"])
    controls = {
        "map": float(map_result["exact_environment_value"]),
        "posterior_sampling": float(posterior_sampling["value"]),
        "open_loop": float(open_loop["value"]),
        "myopic_reward": float(myopic_value),
        "information_only": float(information_value),
    }
    normalized_regrets = {
        name: float((exact_value - value) / scale) for name, value in controls.items()
    }
    map_policy = map_result["policy"]
    finite_values = [
        exact_value,
        float(converged["value"]),
        scale,
        primary_log_evidence,
        *controls.values(),
        *normalized_regrets.values(),
    ]
    return {
        "record_id": record["record_id"],
        "model_file": model_file,
        "prefix_depth": int(record["prefix_depth"]),
        "actions": list(record["actions"]),
        "observations": list(record["observations"]),
        "history_probability": float(record["history_probability"]),
        "primary_log_evidence": primary_log_evidence,
        "primary_belief_sum_error": abs(float(primary_belief.sum()) - 1.0),
        "convergence_belief_sum_error": abs(float(convergence_belief.sum()) - 1.0),
        "static_posterior_entropy_nats": static_entropy(primary_belief),
        "maximum_static_atom_mass": float(primary_belief.sum(axis=1).max()),
        "return_scale": scale,
        "exact_bayes_adaptive": _jsonable_policy_root(exact, primary.model.actions),
        "convergence_bayes_adaptive": _jsonable_policy_root(
            converged, convergence.model.actions
        ),
        "map": {
            "static_index": int(map_result["static_index"]),
            "static_mass": float(map_result["static_mass"]),
            "selected_action": int(map_policy["selected_action"]),
            "selected_action_name": primary.model.actions[int(map_policy["selected_action"])],
            "optimal_actions": [int(action) for action in map_policy["optimal_actions"]],
            "exact_environment_value": controls["map"],
        },
        "posterior_sampling": {
            "points": posterior_sampling_points,
            "offset": posterior_sampling_offset,
            "value": controls["posterior_sampling"],
            "root_action_distribution": posterior_sampling["root_action_distribution"],
        },
        "open_loop": {
            "value": controls["open_loop"],
            "selected_actions": list(open_loop["selected_actions"]),
            "selected_action_names": list(open_loop["selected_action_names"]),
            "sequence_count": int(open_loop["sequence_count"]),
        },
        "myopic_reward": {
            "value": controls["myopic_reward"],
            "selected_action": int(myopic_policy["selected_action"]),
        },
        "information_only": {
            "value": controls["information_only"],
            "selected_action": int(information_policy["selected_action"]),
        },
        "normalized_regrets": normalized_regrets,
        "primary_vs_convergence_normalized_value_error": abs(
            exact_value - float(converged["value"])
        )
        / scale,
        "primary_action_in_convergence_optimal_set": int(exact["selected_action"])
        in set(int(action) for action in converged["optimal_actions"]),
        "exact_ba_map_root_action_disagreement": int(exact["selected_action"])
        != int(map_policy["selected_action"]),
        "primary_bellman_nodes": int(primary_stats.get("bellman_nodes", 0)),
        "convergence_bellman_nodes": int(convergence_stats.get("bellman_nodes", 0)),
        "all_metrics_finite": all(math.isfinite(value) for value in finite_values),
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
    *,
    expected_record_count: int,
    confirmatory_models_scored: int = 0,
) -> dict[str, Any]:
    gates = config["gates"]
    threshold = float(gates["materialNormalizedRegret"])
    completed_fraction = len(rows) / expected_record_count if expected_record_count else 0.0
    models = sorted({row["model_file"] for row in rows})
    finite_rate = sum(bool(row["all_metrics_finite"]) for row in rows) / len(rows)
    belief_rate = sum(
        row["primary_belief_sum_error"] <= 1e-10
        and row["convergence_belief_sum_error"] <= 1e-10
        for row in rows
    ) / len(rows)
    convergence_error = max(
        row["primary_vs_convergence_normalized_value_error"] for row in rows
    )
    convergence_action_rate = sum(
        bool(row["primary_action_in_convergence_optimal_set"]) for row in rows
    ) / len(rows)
    disagreements = sum(
        bool(row["exact_ba_map_root_action_disagreement"]) for row in rows
    )
    material_counts = {
        control: sum(row["normalized_regrets"][control] >= threshold for row in rows)
        for control in ("map", "posterior_sampling", "open_loop", "myopic_reward", "information_only")
    }
    maximum_map_regret = max(row["normalized_regrets"]["map"] for row in rows)
    source_validation_rate = 1.0
    selection_rejection_count = 0
    metrics = {
        "development_model_count": len(models),
        "retained_record_count": len(rows),
        "completed_record_fraction": completed_fraction,
        "source_validation_rate": source_validation_rate,
        "belief_normalization_rate": belief_rate,
        "finite_metric_rate": finite_rate,
        "maximum_primary_vs_convergence_normalized_value_error": convergence_error,
        "primary_action_in_convergence_optimal_set_rate": convergence_action_rate,
        "exact_BA_MAP_root_action_disagreement_records": disagreements,
        "material_regret_record_counts": material_counts,
        "maximum_normalized_MAP_regret": maximum_map_regret,
        "confirmatory_models_scored": confirmatory_models_scored,
        "record_selection_or_rejection_count": selection_rejection_count,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    gate_results = {
        "minimumDevelopmentModels": len(models) >= gates["minimumDevelopmentModels"],
        "minimumRetainedRecords": len(rows) >= gates["minimumRetainedRecords"],
        "minimumCompletedRecordFraction": completed_fraction >= gates["minimumCompletedRecordFraction"],
        "minimumSourceValidationRate": source_validation_rate >= gates["minimumSourceValidationRate"],
        "minimumBeliefNormalizationRate": belief_rate >= gates["minimumBeliefNormalizationRate"],
        "minimumFiniteMetricRate": finite_rate >= gates["minimumFiniteMetricRate"],
        "maximumPrimaryVsConvergenceNormalizedValueError": convergence_error <= gates["maximumPrimaryVsConvergenceNormalizedValueError"],
        "minimumPrimaryActionInConvergenceOptimalSetRate": convergence_action_rate >= gates["minimumPrimaryActionInConvergenceOptimalSetRate"],
        "minimumExactBAMinusMAPRootActionDisagreementRecords": disagreements >= gates["minimumExactBAMinusMAPRootActionDisagreementRecords"],
        "minimumExactBAMinusMAPMaterialRegretRecords": material_counts["map"] >= gates["minimumExactBAMinusMAPMaterialRegretRecords"],
        "minimumMaximumNormalizedMAPRegret": maximum_map_regret >= gates["minimumMaximumNormalizedMAPRegret"],
        "minimumExactBAMinusOpenLoopMaterialRegretRecords": material_counts["open_loop"] >= gates["minimumExactBAMinusOpenLoopMaterialRegretRecords"],
        "minimumExactBAMinusPosteriorSamplingMaterialRegretRecords": material_counts["posterior_sampling"] >= gates["minimumExactBAMinusPosteriorSamplingMaterialRegretRecords"],
        "maximumConfirmatoryModelsScored": confirmatory_models_scored <= gates["maximumConfirmatoryModelsScored"],
        "maximumRecordSelectionOrRejectionCount": selection_rejection_count <= gates["maximumRecordSelectionOrRejectionCount"],
        "maximumHumanRecordAccessCount": gates["maximumHumanRecordAccessCount"] == 0,
        "maximumModelForwardPassCount": gates["maximumModelForwardPassCount"] == 0,
        "maximumAdapterTrainingRunCount": gates["maximumAdapterTrainingRunCount"] == 0,
    }
    by_model: dict[str, Any] = {}
    for model in models:
        subset = [row for row in rows if row["model_file"] == model]
        by_model[model] = {
            "records": len(subset),
            "root_action_disagreements": sum(
                row["exact_ba_map_root_action_disagreement"] for row in subset
            ),
            "normalized_regret": {
                control: _summary(
                    [row["normalized_regrets"][control] for row in subset]
                )
                for control in ("map", "posterior_sampling", "open_loop", "myopic_reward", "information_only")
            },
        }
    passed = all(gate_results.values())
    return {
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
        "decision": (
            "authorize_preregistration_of_confirmatory_multi_environment_design_only"
            if passed
            else "stop_unchanged_family_before_any_confirmatory_model_is_scored"
        ),
        "by_model": by_model,
        "full_census_normalized_regret": {
            control: _summary([row["normalized_regrets"][control] for row in rows])
            for control in ("map", "posterior_sampling", "open_loop", "myopic_reward", "information_only")
        },
    }


def load_preflight(
    evaluator_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    payload = {
        key: value for key, value in evaluator_lock.items() if key != "lock_payload_sha256"
    }
    if payload_hash(payload) != evaluator_lock["lock_payload_sha256"]:
        raise RuntimeError("V68 evaluator lock payload hash mismatch")
    if file_sha256(PROJECT_ROOT / evaluator_lock["evaluator"]) != evaluator_lock["evaluator_sha256"]:
        raise RuntimeError("V68 evaluator source hash mismatch")
    seal_path = PROJECT_ROOT / evaluator_lock["development_census_seal"]
    if file_sha256(seal_path) != evaluator_lock["development_census_seal_sha256"]:
        raise RuntimeError("V68 census seal hash mismatch")
    seal = json.loads(seal_path.read_text())
    census_path = PROJECT_ROOT / seal["census"]
    if file_sha256(census_path) != seal["census_sha256"]:
        raise RuntimeError("V68 sealed census hash mismatch")
    design_path = PROJECT_ROOT / seal["development_design_lock"]
    if file_sha256(design_path) != seal["development_design_lock_sha256"]:
        raise RuntimeError("V68 development design hash mismatch")
    design = json.loads(design_path.read_text())
    records = read_jsonl(census_path)
    if len(records) != seal["record_count"]:
        raise RuntimeError("V68 census record count mismatch")
    if not evaluator_lock["authorization"]["run_development_screen_once"]:
        raise PermissionError("V68 evaluator lock does not authorize execution")
    if evaluator_lock["authorization"]["score_confirmatory_models"]:
        raise PermissionError("V68 evaluator lock unexpectedly authorizes confirmatory models")
    return evaluator_lock, design["config_payload"], records, census_path


def run(evaluator_lock_path: Path) -> None:
    evaluator_lock, config, records, census_path = load_preflight(evaluator_lock_path)
    output_dir = PROJECT_ROOT / "outputs/v68-development-screening/evaluation"
    attempt_path = output_dir / "attempt.json"
    rows_path = output_dir / "record-results.jsonl"
    result_path = output_dir / "result.json"
    if output_dir.exists() or attempt_path.exists() or rows_path.exists() or result_path.exists():
        raise RuntimeError("V68 development evaluation has already been attempted")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_screen_attempt",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "attempt_number": 1,
        "confirmatory_models_scored": 0,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")

    source_dir = (
        PROJECT_ROOT
        / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/POMDP"
    )
    primary_nodes = int(config["exactPlanning"]["primaryQuadratureNodes"])
    convergence_nodes = int(config["exactPlanning"]["convergenceQuadratureNodes"])
    horizon = int(config["exactPlanning"]["horizonActions"])
    tolerance = float(config["exactPlanning"]["tieTolerance"])
    low, high = map(float, config["unknownDynamicsFamily"]["thetaSupport"])
    model_specs = {row["file"]: row for row in config["developmentModels"]}
    unexpected = sorted(set(record["model_file"] for record in records) - set(model_specs))
    if unexpected:
        raise RuntimeError(f"sealed census contains non-development models: {unexpected}")
    families: dict[str, tuple[CommandChannelFamily, CommandChannelFamily]] = {}
    source_validation: dict[str, dict[str, bool]] = {}
    for model_file, spec in model_specs.items():
        model = parse_cassandra_pomdp_file(source_dir / model_file)
        checks = validate_model(model)
        if not all(checks.values()):
            raise RuntimeError(f"development source model no longer validates: {model_file}")
        source_validation[model_file] = checks
        families[model_file] = (
            build_command_channel_family(
                model,
                spec["canonicalActionCycle"],
                quadrature_nodes=primary_nodes,
                theta_support=(low, high),
            ),
            build_command_channel_family(
                model,
                spec["canonicalActionCycle"],
                quadrature_nodes=convergence_nodes,
                theta_support=(low, high),
            ),
        )

    started = time.perf_counter()
    rows = []
    for record in records:
        primary, convergence = families[record["model_file"]]
        rows.append(
            evaluate_record(
                record["model_file"],
                primary,
                convergence,
                record,
                horizon=horizon,
                tie_tolerance=tolerance,
                posterior_sampling_points=17,
                posterior_sampling_offset=1.0 / 34.0,
            )
        )
    elapsed = time.perf_counter() - started
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    aggregate = aggregate_rows(
        rows,
        config,
        expected_record_count=len(records),
        confirmatory_models_scored=0,
    )
    result = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_only_exact_sensitivity_screen",
        "passed": aggregate["passed"],
        "decision": aggregate["decision"],
        "metrics": aggregate["metrics"],
        "gate_results": aggregate["gate_results"],
        "by_model": aggregate["by_model"],
        "full_census_normalized_regret": aggregate["full_census_normalized_regret"],
        "runtime_seconds": elapsed,
        "source_validation": source_validation,
        "records": len(rows),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "access": {
            "development_records_evaluated": len(rows),
            "records_selected_rejected_or_replaced": 0,
            "confirmatory_models_scored": 0,
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
        default="configs/v68-development-evaluator-lock.json",
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
                    "development_models": [row["file"] for row in config["developmentModels"]],
                    "census": str(census_path.relative_to(PROJECT_ROOT)),
                    "confirmatory_models_scored": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run(lock_path)


if __name__ == "__main__":
    main()
