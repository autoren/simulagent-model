#!/usr/bin/env python3
"""Independently audit and freeze the complete V70 confirmatory outcome."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v70_confirmatory_aggregation import aggregate_confirmatory_rows


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    evaluator_lock_path = PROJECT_ROOT / "configs/v70-confirmatory-evaluator-lock.json"
    attempt_path = PROJECT_ROOT / "outputs/v70-confirmatory/evaluation/attempt.json"
    rows_path = PROJECT_ROOT / "outputs/v70-confirmatory/evaluation/record-results.jsonl"
    result_path = PROJECT_ROOT / "outputs/v70-confirmatory/evaluation/result.json"
    audit_path = PROJECT_ROOT / "outputs/v70-confirmatory/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v70-confirmatory-outcome-lock.json"
    if lock_path.exists():
        raise RuntimeError("V70 confirmatory outcome already frozen")

    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    evaluator_payload = {
        key: value
        for key, value in evaluator_lock.items()
        if key != "lock_payload_sha256"
    }
    attempt = json.loads(attempt_path.read_text())
    rows = read_jsonl(rows_path)
    result = json.loads(result_path.read_text())
    errors: list[str] = []

    dependency_hashes_ok = payload_hash(evaluator_payload) == evaluator_lock[
        "lock_payload_sha256"
    ]
    for path_key, hash_key in (
        ("evaluator", "evaluator_sha256"),
        ("aggregation", "aggregation_sha256"),
        ("family_implementation", "family_implementation_sha256"),
        ("point_control_implementation", "point_control_implementation_sha256"),
        (
            "unchanged_exact_record_evaluator",
            "unchanged_exact_record_evaluator_sha256",
        ),
        ("reporting_lock", "reporting_lock_sha256"),
    ):
        dependency_hashes_ok = bool(
            dependency_hashes_ok
            and file_sha256(PROJECT_ROOT / evaluator_lock[path_key])
            == evaluator_lock[hash_key]
        )
    authorization_ok = bool(
        dependency_hashes_ok
        and evaluator_lock["authorization"]["run_confirmatory_outcome_once"]
        and not evaluator_lock["authorization"][
            "modify_prior_locks_code_census_reporting_or_gates"
        ]
        and not evaluator_lock["authorization"]["rescore_development_models"]
        and not evaluator_lock["authorization"]["drop_or_replace_models"]
    )
    if not authorization_ok:
        errors.append("V70 evaluator lock, dependency hashes, or authorization failed")

    reporting = json.loads(
        (PROJECT_ROOT / evaluator_lock["reporting_lock"]).read_text()
    )
    census_seal_path = PROJECT_ROOT / reporting["census_seal"]
    census_seal = json.loads(census_seal_path.read_text())
    census_path = PROJECT_ROOT / census_seal["census"]
    census = read_jsonl(census_path)
    design_path = PROJECT_ROOT / census_seal["confirmatory_design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    chain_ok = bool(
        file_sha256(census_seal_path) == reporting["census_seal_sha256"]
        and file_sha256(census_path) == census_seal["census_sha256"]
        and file_sha256(design_path)
        == census_seal["confirmatory_design_lock_sha256"]
        and len(census) == census_seal["record_count"] == 244
    )
    if not chain_ok:
        errors.append("V70 reporting, census, or design hash chain failed")

    files_ok = bool(
        attempt["attempt_number"]
        == evaluator_lock["expected_attempt_number"]
        == 1
        and attempt["evaluator_lock_sha256"] == file_sha256(evaluator_lock_path)
        and attempt["census_sha256"] == file_sha256(census_path)
        and attempt["expected_records"]
        == evaluator_lock["expected_records"]
        == len(rows)
        == result["records"]
        == 244
        and attempt["expected_confirmatory_models"]
        == evaluator_lock["expected_confirmatory_models"]
        == result["access"]["confirmatory_models_scored"]
        == 9
        and result["attempt"] == str(attempt_path.relative_to(PROJECT_ROOT))
        and result["attempt_sha256"] == file_sha256(attempt_path)
        and result["record_results"] == str(rows_path.relative_to(PROJECT_ROOT))
        and result["record_results_sha256"] == file_sha256(rows_path)
    )
    if not files_ok:
        errors.append("V70 attempt, rows, or result hash binding failed")

    census_by_id = {row["record_id"]: row for row in census}
    row_by_id = {row["record_id"]: row for row in rows}
    complete_census_ok = bool(
        len(census_by_id) == len(census) == 244
        and len(row_by_id) == len(rows) == 244
        and set(row_by_id) == set(census_by_id)
        and all(
            row_by_id[record_id]["model_file"] == census_row["model_file"]
            and row_by_id[record_id]["stratum"] == census_row["stratum"]
            and row_by_id[record_id]["prefix_depth"] == census_row["prefix_depth"]
            for record_id, census_row in census_by_id.items()
        )
    )
    if not complete_census_ok:
        errors.append("V70 outcome rows do not exactly match the sealed census")

    source_validation = result["source_validation"]
    recomputed = aggregate_confirmatory_rows(
        rows,
        config,
        expected_record_count=244,
        source_validation=source_validation,
        record_selection_or_rejection_count=0,
        development_models_rescored=0,
    )
    aggregate_keys = (
        "passed",
        "decision",
        "metrics",
        "gate_results",
        "qualified_models",
        "posterior_sampling_material_models",
        "by_model",
        "by_stratum",
        "full_census_normalized_regret",
        "Tier_B_cheese_pair",
    )
    aggregate_ok = all(recomputed[key] == result[key] for key in aggregate_keys)
    if not aggregate_ok:
        errors.append("V70 independent complete-census aggregation differs")

    expected_decision = (
        "confirm_multi_environment_replication_for_project_authored_V69_family"
        if result["passed"]
        else "report_complete_negative_or_mixed_confirmatory_result_without_tuning"
    )
    decision_ok = bool(
        result["passed"] == all(result["gate_results"].values())
        and result["decision"] == expected_decision
        and len(result["gate_results"])
        == len(config["confirmatoryGates"])
    )
    if not decision_ok:
        errors.append("V70 frozen gate truth and decision are inconsistent")

    neutral_rows = copy.deepcopy(rows)
    for row in neutral_rows:
        for control in ("map", "posterior_sampling"):
            row[control]["off_support_branch_count"] = 0
            row[control]["expected_off_support_entry_probability"] = 0.0
    neutral = aggregate_confirmatory_rows(
        neutral_rows,
        config,
        expected_record_count=244,
        source_validation=source_validation,
        record_selection_or_rejection_count=0,
        development_models_rescored=0,
    )
    fallback_non_gating_ok = bool(
        neutral["passed"] == recomputed["passed"]
        and neutral["decision"] == recomputed["decision"]
        and neutral["gate_results"] == recomputed["gate_results"]
        and neutral["metrics"] == recomputed["metrics"]
        and neutral["qualified_models"] == recomputed["qualified_models"]
        and neutral["posterior_sampling_material_models"]
        == recomputed["posterior_sampling_material_models"]
    )
    if not fallback_non_gating_ok:
        errors.append("V70 fallback diagnostics changed a primary decision quantity")

    expected_models = {spec["file"] for spec in config["confirmatoryModels"]}
    access = result["access"]
    firewall_ok = bool(
        set(source_validation) == expected_models
        and all(checks and all(checks.values()) for checks in source_validation.values())
        and access["confirmatory_records_evaluated"] == 244
        and access["records_selected_rejected_or_replaced"] == 0
        and access["development_models_rescored"] == 0
        and access["SMC2_runs"] == 0
        and access["human_records"] == 0
        and access["model_forward_passes"] == 0
        and access["adapter_training_runs"] == 0
        and attempt["development_models_rescored"] == 0
        and census_seal["selection_rejection_or_replacement_count"] == 0
        and census_seal["development_models_rescored"] == 0
    )
    if not firewall_ok:
        errors.append("V70 source validation, selection, or access firewall failed")

    checks = {
        "locked_evaluator_dependencies_and_one_shot_authorization": authorization_ok,
        "reporting_census_and_design_hash_chain": chain_ok,
        "attempt_rows_and_result_hash_binding": files_ok,
        "all_244_rows_exactly_match_the_sealed_census": complete_census_ok,
        "independent_complete_census_aggregate_recomputation": aggregate_ok,
        "frozen_gate_truth_and_decision_consistency": decision_ok,
        "fallback_diagnostics_are_non_decisional": fallback_non_gating_ok,
        "source_selection_and_access_firewall": firewall_ok,
    }
    audit = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_complete_v70_confirmatory_outcome"
            if not errors
            else "reject_v70_confirmatory_outcome_integrity"
        ),
        "scientific_outcome_passed": bool(result["passed"]),
        "scientific_decision": result["decision"],
        "errors": errors,
        "checks": checks,
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_outcome_lock",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed": bool(result["passed"]),
            "decision": result["decision"],
            "records": result["records"],
            "confirmatory_models": result["metrics"]["confirmatory_model_count"],
            "paired_MAP_qualifying_models": result["metrics"][
                "paired_MAP_qualifying_model_count"
            ],
            "paired_MAP_qualifying_structurally_related_models": result["metrics"][
                "paired_MAP_qualifying_structurally_related_model_count"
            ],
            "paired_MAP_qualifying_novel_models": result["metrics"][
                "paired_MAP_qualifying_novel_model_count"
            ],
            "material_posterior_sampling_models": result["metrics"][
                "material_posterior_sampling_model_count"
            ],
            "maximum_normalized_MAP_regret": result["metrics"][
                "maximum_normalized_MAP_regret"
            ],
            "qualified_models": result["qualified_models"],
            "posterior_sampling_material_models": result[
                "posterior_sampling_material_models"
            ],
        },
        "authorization": {
            "modify_or_rerun_V70": False,
            "drop_replace_or_rescore_confirmatory_models": False,
            "revise_frozen_gates_after_outcome": False,
            "report_and_synthesize_complete_result": True,
            "develop_new_family_only_under_new_preregistration_and_fresh_data": True,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
