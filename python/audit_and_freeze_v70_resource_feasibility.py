#!/usr/bin/env python3
"""Audit and freeze metadata-only V70 resource feasibility."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    budget_path = PROJECT_ROOT / "configs/v70-confirmatory-resource-budget.json"
    plan_path = PROJECT_ROOT / "docs/v70-confirmatory-resource-feasibility-plan.md"
    design_path = PROJECT_ROOT / "configs/v70-confirmatory-design-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v70-confirmatory/resource-audit.json"
    lock_path = PROJECT_ROOT / "configs/v70-confirmatory-resource-lock.json"
    if lock_path.exists():
        raise RuntimeError("V70 resource feasibility already frozen")
    budget = json.loads(budget_path.read_text())
    design = json.loads(design_path.read_text())
    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    design_ok = bool(
        payload_hash(design_payload) == design["lock_payload_sha256"]
        and design["authorization"]["audit_and_freeze_resource_feasibility"]
        and not design["authorization"]["construct_confirmatory_census"]
        and not design["authorization"]["run_confirmatory_outcomes"]
        and not design["authorization"]["drop_or_replace_models"]
    )
    if not design_ok:
        errors.append("V70 design lock or resource-only authorization failed")

    source_lock = json.loads(
        (PROJECT_ROOT / design["source_feasibility_lock"]).read_text()
    )
    inventory_path = PROJECT_ROOT / source_lock["source_inventory"]
    inventory = json.loads(inventory_path.read_text())
    inventory_ok = bool(
        file_sha256(inventory_path) == source_lock["source_inventory_sha256"]
        and len(design["config_payload"]["confirmatoryModels"]) == 9
    )
    if not inventory_ok:
        errors.append("locked source inventory or nine-model design binding failed")

    dimensions = {row["file"]: row for row in inventory["models"]}
    convergence_nodes = int(
        design["config_payload"]["exactPlanning"]["convergenceQuadratureNodes"]
    )
    rows: list[dict[str, Any]] = []
    for spec in design["config_payload"]["confirmatoryModels"]:
        meta = dimensions[spec["file"]]
        actions = int(meta["actions"])
        states = int(meta["states"])
        observations = int(meta["observations"])
        action_observations = actions * observations
        rows.append(
            {
                "file": spec["file"],
                "stratum": spec["stratum"],
                "states": states,
                "actions": actions,
                "observations": observations,
                "census_record_upper_bound": 1 + action_observations,
                "per_record_bellman_branch_node_upper_bound": (
                    1 + action_observations + action_observations**2
                ),
                "convergence_transition_tensor_bytes": (
                    2 * convergence_nodes * actions * states * states * 8
                ),
                "convergence_joint_belief_bytes": (
                    2 * convergence_nodes * states * 8
                ),
            }
        )

    totals = {
        "total_census_record_upper_bound": sum(
            row["census_record_upper_bound"] for row in rows
        ),
        "maximum_per_model_census_record_upper_bound": max(
            row["census_record_upper_bound"] for row in rows
        ),
        "maximum_per_record_bellman_branch_node_upper_bound": max(
            row["per_record_bellman_branch_node_upper_bound"] for row in rows
        ),
        "dual_resolution_total_branch_node_upper_bound": sum(
            2
            * row["census_record_upper_bound"]
            * row["per_record_bellman_branch_node_upper_bound"]
            for row in rows
        ),
        "maximum_per_model_convergence_transition_tensor_bytes": max(
            row["convergence_transition_tensor_bytes"] for row in rows
        ),
        "total_convergence_transition_tensor_bytes": sum(
            row["convergence_transition_tensor_bytes"] for row in rows
        ),
        "maximum_per_model_convergence_joint_belief_bytes": max(
            row["convergence_joint_belief_bytes"] for row in rows
        ),
    }
    thresholds = budget["nonOutcomeThresholds"]
    bound_checks = {
        "total_census_record_upper_bound": (
            totals["total_census_record_upper_bound"]
            <= thresholds["maximumTotalCensusRecordUpperBound"]
        ),
        "per_model_census_record_upper_bound": (
            totals["maximum_per_model_census_record_upper_bound"]
            <= thresholds["maximumPerModelCensusRecordUpperBound"]
        ),
        "per_record_bellman_branch_node_upper_bound": (
            totals["maximum_per_record_bellman_branch_node_upper_bound"]
            <= thresholds["maximumPerRecordBellmanBranchNodeUpperBound"]
        ),
        "dual_resolution_total_branch_node_upper_bound": (
            totals["dual_resolution_total_branch_node_upper_bound"]
            <= thresholds["maximumDualResolutionTotalBranchNodeUpperBound"]
        ),
        "per_model_convergence_transition_tensor_bytes": (
            totals["maximum_per_model_convergence_transition_tensor_bytes"]
            <= thresholds["maximumPerModelConvergenceTransitionTensorBytes"]
        ),
        "total_convergence_transition_tensor_bytes": (
            totals["total_convergence_transition_tensor_bytes"]
            <= thresholds["maximumTotalConvergenceTransitionTensorBytes"]
        ),
        "per_model_convergence_joint_belief_bytes": (
            totals["maximum_per_model_convergence_joint_belief_bytes"]
            <= thresholds["maximumPerModelConvergenceJointBeliefBytes"]
        ),
    }
    bounds_ok = all(bound_checks.values())
    if not bounds_ok:
        errors.append("one or more frozen resource-only bounds failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v70-confirmatory-census-seal.json",
            "outputs/v70-confirmatory/census.jsonl",
            "python/evaluate_v70_confirmatory.py",
            "configs/v70-confirmatory-evaluator-lock.json",
            "outputs/v70-confirmatory/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V70 census or evaluation exists before resource lock")

    checks = {
        "design_binding_and_resource_only_authorization": design_ok,
        "locked_inventory_and_nine_model_binding": inventory_ok,
        "all_frozen_resource_bounds": bounds_ok,
        "census_and_evaluation_absent": downstream_absent,
        "no_policy_value_EIG_regret_or_history_construction": True,
    }
    audit = {
        "schema_version": "70-confirmatory-resource-feasibility",
        "experiment": "v70_confirmatory_resource_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "authorize_complete_nine_model_census_construction"
            if not errors
            else "defer_entire_V70_confirmation_without_model_dropping"
        ),
        "errors": errors,
        "checks": checks,
        "bound_checks": bound_checks,
        "model_bounds": rows,
        "totals": totals,
        "access": {
            "confirmatory_source_metadata_records_read": 9,
            "confirmatory_histories_constructed": 0,
            "confirmatory_policy_values_computed": 0,
            "confirmatory_EIG_values_computed": 0,
            "development_models_rescored": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "70-confirmatory-resource-feasibility",
        "experiment": "v70_confirmatory_resource_lock",
        "confirmatory_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "confirmatory_design_lock_sha256": file_sha256(design_path),
        "resource_budget": str(budget_path.relative_to(PROJECT_ROOT)),
        "resource_budget_sha256": file_sha256(budget_path),
        "resource_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "resource_plan_sha256": file_sha256(plan_path),
        "resource_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "resource_audit_sha256": file_sha256(audit_path),
        "model_bounds": rows,
        "totals": totals,
        "authorization": {
            "modify_design_resource_budget_or_model_set": False,
            "construct_and_seal_complete_confirmatory_census": True,
            "write_reporting_or_evaluator": False,
            "run_confirmatory_outcomes": False,
            "drop_or_replace_models": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
