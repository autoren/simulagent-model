#!/usr/bin/env python3
"""Audit and freeze the metadata-only V73 IMPRL source selection."""
from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v73-active-sensing-source-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v73-active-sensing-source-feasibility-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v73_source_feasibility.py"
    prior_lock_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-outcome-lock.json"
    )
    checkout = PROJECT_ROOT / "data/v73-active-sensing/source-checkouts/imprl"
    source_path = checkout / "imprl/envs/structural_envs/k_out_of_n_infinite.py"
    finite_path = checkout / "imprl/envs/structural_envs/k_out_of_n_finite.py"
    config_source_path = (
        checkout
        / "imprl/envs/structural_envs/env_configs/hard-4-of-4_infinite.yaml"
    )
    license_path = checkout / "LICENSE"
    inventory_path = PROJECT_ROOT / "outputs/v73-active-sensing/source-inventory.json"
    audit_path = PROJECT_ROOT / "outputs/v73-active-sensing/source-audit.json"
    lock_path = PROJECT_ROOT / "configs/v73-active-sensing-source-lock.json"
    if lock_path.exists():
        raise RuntimeError("V73 source feasibility is already frozen")

    config = json.loads(config_path.read_text())
    prior_lock = json.loads(prior_lock_path.read_text())
    prior_payload = {
        key: value for key, value in prior_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(prior_payload) == prior_lock["lock_payload_sha256"]
        and not prior_lock["outcome"]["passed_development_gates"]
        and prior_lock["authorization"][
            "begin_successor_only_after_fresh_preregistration_and_structural_dominance_audit"
        ]
        and not prior_lock["authorization"][
            "inspect_or_score_V72_protected_confirmation_sources"
        ]
    )
    if not authorization_ok:
        errors.append("V72 outcome lock does not authorize a materially new successor")

    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual_commit = completed.stdout.strip()
    selected = config["selectedDevelopmentSource"]
    commit_ok = actual_commit == selected["commit"]
    if not commit_ok:
        errors.append("V73 IMPRL checkout commit drifted")

    source = source_path.read_text()
    source_config = config_source_path.read_text()
    license_text = license_path.read_text()
    source_structure_ok = bool(
        'self.action_map = {0: "do_nothing", 1: "replace", 2: "inspect"}' in source
        and "self.transition_model[c, 0, :, :] = self.deterioration_table" in source
        and "self.replacement_table[c] @ self.deterioration_table" in source
        and "self.observation_model[c, 2, :, :] = inspection_model[c]" in source
        and "failure_cost = self.system_replacement_reward * self.FAILURE_PENALTY_FACTOR" in source
        and "Apache License" in license_text
    )
    if not source_structure_ok:
        errors.append("V73 IMPRL structural or license declarations drifted")

    required_config_fragments = (
        "replacement_rewards: [-50, -30, -80, -90]",
        "inspection_rewards: [-5, -3, -8, -4]",
        "mobilisation_reward: -4",
        "failure_penalty_factor: 3",
        "initial_belief: [0.6, 0.4, 0.0]",
        "obs_accuracies: [0.8, 0.85, 0.9, 0.9]",
        "discount_factor: 0.8",
        "replacement_accuracies: [0.95, 0.9, 0.98, 1]",
        "failure_obs_accuracies: [0.95, 0.9, 0.98, 1]",
        "[[0.72, 0.28, 0.0]",
        "[0.0, 0.78, 0.22]",
    )
    source_parameters_ok = all(
        fragment in source_config for fragment in required_config_fragments
    )
    blueprint = config["fixedAdapterBlueprint"]
    frozen = blueprint["sourceGrounding"]
    blueprint_ok = bool(
        frozen["sourceComponentTransitionRows"]
        == [[0.72, 0.28, 0.0], [0.0, 0.78, 0.22], [0.0, 0.0, 1.0]]
        and frozen["replacementReward"] == -90.0
        and frozen["inspectionReward"] == -4.0
        and frozen["mobilisationReward"] == -4.0
        and frozen["projectedSingleComponentFailureReward"] == -270.0
        and blueprint["projectAuthoredLayer"]["beaconHarvestable"] is False
        and blueprint["finiteModel"]["planningHorizonActions"] == 5
    )
    if not source_parameters_ok or not blueprint_ok:
        errors.append("V73 source parameters or fixed adapter blueprint drifted")

    access = {
        "source_auditor_invocations": 2,
        "failed_source_auditor_invocations": 1,
        "failed_source_auditor_causes": [
            "the initial executable audit expected generic outcome-lock field names instead of the frozen V72 names"
        ],
        "repository_landing_pages_previewed": 3,
        "pinned_repositories_cloned": 1,
        "source_files_inspected": 4,
        "candidate_simulator_runs": 0,
        "candidate_policy_values_computed": 0,
        "candidate_optimal_actions_computed": 0,
        "candidate_regrets_computed": 0,
        "candidate_EIG_values_computed": 0,
        "protected_source_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    inventory = {
        "schema_version": "73-active-sensing-source-feasibility",
        "experiment": "v73_imprl_source_inventory",
        "selection_basis": "source structure and declared parameters only",
        "repository": selected["repository"],
        "commit": actual_commit,
        "license": selected["license"],
        "selected_environment": selected["environment"],
        "selected_configuration": selected["configuration"],
        "selected_component_index_zero_based": selected["componentIndexZeroBased"],
        "source_files": {
            str(path.relative_to(checkout)): file_sha256(path)
            for path in (source_path, finite_path, config_source_path, license_path)
        },
        "fixed_adapter_blueprint": blueprint,
        "access": access,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    checks = {
        "v72_authorizes_materially_new_successor": authorization_ok,
        "pinned_commit_matches": commit_ok,
        "source_structure_and_license_match": source_structure_ok,
        "selected_source_parameters_match": source_parameters_ok,
        "fixed_adapter_blueprint_matches": blueprint_ok,
        "zero_candidate_outcome_access": all(
            access[key] == 0
            for key in (
                "candidate_simulator_runs",
                "candidate_policy_values_computed",
                "candidate_optimal_actions_computed",
                "candidate_regrets_computed",
                "candidate_EIG_values_computed",
                "protected_source_access_count",
                "human_record_access_count",
                "model_forward_pass_count",
                "adapter_training_run_count",
            )
        ),
    }
    audit = {
        "schema_version": "73-active-sensing-source-feasibility",
        "experiment": "v73_imprl_source_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_source_and_authorize_adapter_structural_audit"
            if not errors
            else "stop_v73_before_adapter_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "73-active-sensing-source-feasibility",
        "experiment": "v73_imprl_source_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "prior_lock": str(prior_lock_path.relative_to(PROJECT_ROOT)),
        "prior_lock_sha256": file_sha256(prior_lock_path),
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "repository_commit": actual_commit,
        "authorization": {
            "modify_source_selection_or_blueprint": False,
            "implement_adapter_and_structural_tests": True,
            "run_preregistered_structural_dominance_audit": True,
            "compute_exact_BA_MAP_PS_or_myopic_outcomes": False,
            "select_or_inspect_confirmation_sources": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
