#!/usr/bin/env python3
"""Audit and freeze metadata-only V71 development resource feasibility."""
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
    budget_path = (
        PROJECT_ROOT
        / "configs/v71-sensor-codebook-development-resource-budget.json"
    )
    plan_path = (
        PROJECT_ROOT
        / "docs/v71-sensor-codebook-development-resource-feasibility-plan.md"
    )
    source_lock_path = PROJECT_ROOT / "configs/v71-sensor-codebook-source-lock.json"
    auditor_path = (
        PROJECT_ROOT
        / "python/audit_and_freeze_v71_development_resource_feasibility.py"
    )
    audit_path = (
        PROJECT_ROOT / "outputs/v71-sensor-codebook/development-resource-audit.json"
    )
    lock_path = (
        PROJECT_ROOT
        / "configs/v71-sensor-codebook-development-resource-lock.json"
    )
    if lock_path.exists():
        raise RuntimeError("V71 development resource feasibility is already frozen")

    budget = json.loads(budget_path.read_text())
    source_lock = json.loads(source_lock_path.read_text())
    source_payload = {
        key: value for key, value in source_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    source_ok = bool(
        payload_hash(source_payload) == source_lock["lock_payload_sha256"]
        and source_lock["authorization"][
            "audit_and_freeze_development_resource_feasibility"
        ]
        and not source_lock["authorization"]["construct_development_census"]
        and not source_lock["authorization"]["run_development_outcomes"]
        and not source_lock["authorization"]["read_protected_confirmation_outcomes"]
    )
    if not source_ok:
        errors.append("V71 source lock or resource-only authorization failed")

    inventory_path = PROJECT_ROOT / source_lock["source_inventory"]
    inventory = json.loads(inventory_path.read_text())
    development = [
        row
        for row in inventory["selected_models"]
        if row["role"] == "developmentFresh"
    ]
    inventory_ok = bool(
        file_sha256(inventory_path) == source_lock["source_inventory_sha256"]
        and len(development) == 3
        and all(not row["failed_validation_checks"] for row in development)
    )
    if not inventory_ok:
        errors.append("locked inventory or three-model development binding failed")

    latent_count = int(budget["constants"]["latentCount"])
    control_count = int(budget["constants"]["controlCountIncludingExact"])
    rows: list[dict[str, Any]] = []
    for source in sorted(development, key=lambda row: row["file"]):
        states = int(source["states"])
        actions = int(source["actions"])
        observations = int(source["observations"])
        action_observations = actions * observations
        census = 1 + action_observations
        branch_nodes = 1 + action_observations + action_observations**2
        rows.append(
            {
                "file": source["file"],
                "states": states,
                "actions": actions,
                "observations": observations,
                "census_record_upper_bound": census,
                "per_record_bellman_branch_node_upper_bound": branch_nodes,
                "all_planner_branch_node_upper_bound": (
                    control_count * census * branch_nodes
                ),
                "joint_transition_bytes": (
                    latent_count * actions * states * states * 8
                ),
                "joint_observation_bytes": (
                    latent_count * actions * states * observations * 8
                ),
                "joint_belief_bytes": latent_count * states * 8,
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
        "maximum_per_model_all_planner_branch_node_upper_bound": max(
            row["all_planner_branch_node_upper_bound"] for row in rows
        ),
        "total_all_planner_branch_node_upper_bound": sum(
            row["all_planner_branch_node_upper_bound"] for row in rows
        ),
        "maximum_per_model_joint_transition_bytes": max(
            row["joint_transition_bytes"] for row in rows
        ),
        "maximum_per_model_joint_observation_bytes": max(
            row["joint_observation_bytes"] for row in rows
        ),
        "maximum_per_model_joint_belief_bytes": max(
            row["joint_belief_bytes"] for row in rows
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
        "per_model_all_planner_branch_node_upper_bound": (
            totals["maximum_per_model_all_planner_branch_node_upper_bound"]
            <= thresholds["maximumPerModelAllPlannerBranchNodeUpperBound"]
        ),
        "total_all_planner_branch_node_upper_bound": (
            totals["total_all_planner_branch_node_upper_bound"]
            <= thresholds["maximumTotalAllPlannerBranchNodeUpperBound"]
        ),
        "per_model_joint_transition_bytes": (
            totals["maximum_per_model_joint_transition_bytes"]
            <= thresholds["maximumPerModelJointTransitionBytes"]
        ),
        "per_model_joint_observation_bytes": (
            totals["maximum_per_model_joint_observation_bytes"]
            <= thresholds["maximumPerModelJointObservationBytes"]
        ),
        "per_model_joint_belief_bytes": (
            totals["maximum_per_model_joint_belief_bytes"]
            <= thresholds["maximumPerModelJointBeliefBytes"]
        ),
    }
    bounds_ok = all(bound_checks.values())
    if not bounds_ok:
        errors.append("one or more frozen dimensions-only resource bounds failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v71-sensor-codebook-development-census-seal.json",
            "outputs/v71-sensor-codebook/development-census.jsonl",
            "python/evaluate_v71_sensor_codebook_development.py",
            "configs/v71-sensor-codebook-development-evaluator-lock.json",
            "outputs/v71-sensor-codebook/development-evaluation",
            "outputs/v71-sensor-codebook/protected-confirmation-evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V71 census, evaluator, or outcomes exist before resource lock")

    checks = {
        "source_lock_and_resource_only_authorization": source_ok,
        "locked_valid_three_model_development_inventory": inventory_ok,
        "all_frozen_dimensions_only_resource_bounds": bounds_ok,
        "census_evaluator_and_outcomes_absent": downstream_absent,
        "zero_history_policy_value_action_regret_EIG_or_protected_access": True,
    }
    access = {
        "development_source_metadata_records_read": 3,
        "development_histories_constructed": 0,
        "policy_values_computed": 0,
        "optimal_actions_computed": 0,
        "regrets_computed": 0,
        "EIG_values_computed": 0,
        "protected_confirmation_records_read": 0,
    }
    audit = {
        "schema_version": "71-sensor-codebook-development-resource-feasibility",
        "experiment": "v71_development_resource_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "authorize_complete_three_model_development_census_construction"
            if not errors
            else "defer_entire_v71_development_without_model_dropping"
        ),
        "errors": errors,
        "checks": checks,
        "bound_checks": bound_checks,
        "model_bounds": rows,
        "totals": totals,
        "access": access,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "71-sensor-codebook-development-resource-feasibility",
        "experiment": "v71_sensor_codebook_development_resource_lock",
        "source_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_lock_sha256": file_sha256(source_lock_path),
        "resource_budget": str(budget_path.relative_to(PROJECT_ROOT)),
        "resource_budget_sha256": file_sha256(budget_path),
        "resource_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "resource_plan_sha256": file_sha256(plan_path),
        "resource_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "resource_auditor_sha256": file_sha256(auditor_path),
        "resource_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "resource_audit_sha256": file_sha256(audit_path),
        "model_bounds": rows,
        "totals": totals,
        "authorization": {
            "modify_source_family_partition_resource_budget_or_gates": False,
            "construct_and_seal_complete_development_census": True,
            "write_development_evaluator": False,
            "run_development_outcomes": False,
            "read_protected_confirmation_histories_or_outcomes": False,
            "drop_replace_repair_renormalize_or_simplify_models": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
