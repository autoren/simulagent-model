#!/usr/bin/env python3
"""One-shot V68r2 development evaluator with total point-model controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import evaluate_v68_development_screen as base

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import validate_model
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v68_multi_environment_exact import CommandChannelFamily, build_command_channel_family
from v68r2_point_model_controls import (
    totalized_map_model_policy,
    totalized_persistent_posterior_sampling_mixture,
)


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
    """Call the frozen evaluator while replacing exactly its two point controls."""
    captured_map: dict[str, Any] = {}
    captured_ps: dict[str, Any] = {}

    def repaired_map(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = totalized_map_model_policy(*args, **kwargs)
        captured_map.update(result)
        return result

    def repaired_ps(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = totalized_persistent_posterior_sampling_mixture(*args, **kwargs)
        captured_ps.update(result)
        return result

    original_map = base.map_model_policy
    original_ps = base.persistent_posterior_sampling_mixture
    base.map_model_policy = repaired_map
    base.persistent_posterior_sampling_mixture = repaired_ps
    try:
        row = base.evaluate_record(
            model_file,
            primary,
            convergence,
            record,
            horizon=horizon,
            tie_tolerance=tie_tolerance,
            posterior_sampling_points=posterior_sampling_points,
            posterior_sampling_offset=posterior_sampling_offset,
        )
    finally:
        base.map_model_policy = original_map
        base.persistent_posterior_sampling_mixture = original_ps
    if not captured_map or not captured_ps:
        raise RuntimeError("V68r2 did not call both repaired point-model controls")
    row["map"].update(
        {
            "off_support_branch_count": int(captured_map["off_support_branch_count"]),
            "expected_off_support_entry_probability": float(
                captured_map["expected_off_support_entry_probability"]
            ),
            "fallback_action": int(captured_map["fallback_action"]),
            "fallback_action_name": captured_map["fallback_action_name"],
            "point_model_persists_on_supported_histories": bool(
                captured_map["point_model_persists_on_supported_histories"]
            ),
            "off_support_fallback_is_open_loop": bool(
                captured_map["off_support_fallback_is_open_loop"]
            ),
            "off_support_model_reselection": bool(
                captured_map["off_support_model_reselection"]
            ),
            "epsilon_smoothing": bool(captured_map["epsilon_smoothing"]),
        }
    )
    row["posterior_sampling"].update(
        {
            "off_support_branch_count": int(captured_ps["off_support_branch_count"]),
            "expected_off_support_entry_probability": float(
                captured_ps["expected_off_support_entry_probability"]
            ),
            "fallback_action": int(captured_ps["fallback_action"]),
            "fallback_action_name": captured_ps["fallback_action_name"],
            "sampled_model_persists_on_supported_histories": bool(
                captured_ps["sampled_model_persists_on_supported_histories"]
            ),
            "off_support_fallback_is_open_loop": bool(
                captured_ps["off_support_fallback_is_open_loop"]
            ),
            "off_support_model_resampling": bool(
                captured_ps["off_support_model_resampling"]
            ),
            "epsilon_smoothing": bool(captured_ps["epsilon_smoothing"]),
        }
    )
    return row


def load_preflight(
    evaluator_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    lock = json.loads(evaluator_lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if base.payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V68r2 evaluator lock payload hash mismatch")
    for path_key, hash_key, label in (
        ("evaluator", "evaluator_sha256", "evaluator"),
        ("repair_implementation", "repair_implementation_sha256", "repair implementation"),
        ("unchanged_V68_evaluator", "unchanged_V68_evaluator_sha256", "unchanged V68 evaluator"),
        ("source_failed_attempt", "source_failed_attempt_sha256", "failed attempt"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V68r2 {label} hash mismatch")
    seal_path = PROJECT_ROOT / lock["development_census_seal"]
    if file_sha256(seal_path) != lock["development_census_seal_sha256"]:
        raise RuntimeError("V68r2 census seal hash mismatch")
    seal = json.loads(seal_path.read_text())
    census_path = PROJECT_ROOT / seal["census"]
    if file_sha256(census_path) != seal["census_sha256"]:
        raise RuntimeError("V68r2 sealed census hash mismatch")
    design_path = PROJECT_ROOT / seal["development_design_lock"]
    if file_sha256(design_path) != seal["development_design_lock_sha256"]:
        raise RuntimeError("V68r2 unchanged development design hash mismatch")
    config = json.loads(design_path.read_text())["config_payload"]
    records = base.read_jsonl(census_path)
    if len(records) != seal["record_count"] == lock["expected_records"]:
        raise RuntimeError("V68r2 census record count mismatch")
    if not lock["authorization"]["run_repaired_development_screen_once"]:
        raise PermissionError("V68r2 evaluator lock does not authorize execution")
    if lock["authorization"]["score_confirmatory_models"]:
        raise PermissionError("V68r2 evaluator unexpectedly authorizes holdout scoring")
    return lock, config, records, census_path


def run(evaluator_lock_path: Path) -> None:
    lock, config, records, census_path = load_preflight(evaluator_lock_path)
    output_dir = PROJECT_ROOT / "outputs/v68r2-development-screening/evaluation"
    attempt_path = output_dir / "attempt.json"
    rows_path = output_dir / "record-results.jsonl"
    result_path = output_dir / "result.json"
    if output_dir.exists() or attempt_path.exists() or rows_path.exists() or result_path.exists():
        raise RuntimeError("V68r2 development evaluation has already been attempted")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema_version": "68r2-development-screening",
        "experiment": "v68r2_development_screen_attempt",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "attempt_number": 1,
        "source_failed_attempt": lock["source_failed_attempt"],
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
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    aggregate = base.aggregate_rows(
        rows,
        config,
        expected_record_count=len(records),
        confirmatory_models_scored=0,
    )

    def diagnostics(name: str, reselection_key: str) -> dict[str, Any]:
        return {
            "records_with_off_support_branches": sum(
                row[name]["off_support_branch_count"] > 0 for row in rows
            ),
            "total_off_support_branch_count": sum(
                row[name]["off_support_branch_count"] for row in rows
            ),
            "mean_expected_entry_probability": sum(
                row[name]["expected_off_support_entry_probability"] for row in rows
            )
            / len(rows),
            "maximum_expected_entry_probability": max(
                row[name]["expected_off_support_entry_probability"] for row in rows
            ),
            "model_reselection_or_resampling_count": sum(
                row[name][reselection_key] for row in rows
            ),
            "epsilon_smoothing_count": sum(row[name]["epsilon_smoothing"] for row in rows),
        }

    result = {
        "schema_version": "68r2-development-screening",
        "experiment": "v68r2_development_only_exact_sensitivity_screen",
        "passed": aggregate["passed"],
        "decision": aggregate["decision"],
        "metrics": aggregate["metrics"],
        "gate_results": aggregate["gate_results"],
        "by_model": aggregate["by_model"],
        "full_census_normalized_regret": aggregate["full_census_normalized_regret"],
        "off_support_totalization": {
            "map": diagnostics("map", "off_support_model_reselection"),
            "posterior_sampling": diagnostics(
                "posterior_sampling", "off_support_model_resampling"
            ),
        },
        "runtime_seconds": elapsed,
        "source_validation": source_validation,
        "records": len(rows),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "source_failed_attempt": lock["source_failed_attempt"],
        "source_failed_attempt_sha256": lock["source_failed_attempt_sha256"],
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
        "--evaluator-lock", default="configs/v68r2-development-evaluator-lock.json"
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
                    "repair": "fixed_first_canonical_action_for_all_point_model_controls_after_exact_off_support_observation",
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
