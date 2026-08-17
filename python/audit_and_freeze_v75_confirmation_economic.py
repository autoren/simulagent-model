#!/usr/bin/env python3
"""Audit and freeze V75's source-level economic screen before adapter code."""
from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v68_cassandra_pomdp import parse_cassandra_pomdp_text


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-economic.json"
    plan_path = PROJECT_ROOT / "docs/v75-active-sensing-confirmation-economic-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v75_confirmation_economic.py"
    prior_lock_path = PROJECT_ROOT / "configs/v74-active-sensing-development-outcome-lock.json"
    checkout = PROJECT_ROOT / "data/v75-active-sensing-confirmation/source-checkouts/nova"
    source_path = checkout / "tests/benchmarks/algorithms/domains/paint_95.pomdp"
    license_path = checkout / "LICENSE"
    inventory_path = PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/source-economic-inventory.json"
    audit_path = PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/source-economic-audit.json"
    lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-economic-lock.json"
    if lock_path.exists():
        raise RuntimeError("V75 confirmation economic feasibility is already frozen")

    forbidden_future_paths = (
        PROJECT_ROOT / "python/v75_nova_paint_source.py",
        PROJECT_ROOT / "python/test_v75_nova_paint_source.py",
        PROJECT_ROOT / "python/evaluate_v75_nova_paint_confirmation.py",
    )
    config = json.loads(config_path.read_text())
    prior_lock = json.loads(prior_lock_path.read_text())
    prior_payload = {
        key: value for key, value in prior_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    prior_authorization_ok = bool(
        payload_hash(prior_payload) == prior_lock["lock_payload_sha256"]
        and prior_lock["outcome"]["passed_all_development_gates"]
        and prior_lock["authorization"][
            "design_confirmation_successor_after_fresh_preregistration"
        ]
        and not prior_lock["authorization"]["compute_confirmation_policy_values"]
        and not prior_lock["authorization"]["modify_or_rerun_V74"]
    )
    if not prior_authorization_ok:
        errors.append("V74 outcome lock does not authorize a freshly preregistered successor")

    preimplementation_ok = not any(path.exists() for path in forbidden_future_paths)
    if not preimplementation_ok:
        errors.append("V75 economic screen did not precede adapter implementation")

    selected = config["selectedConfirmationSource"]
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    commit_ok = actual_commit == selected["commit"]
    source_hash_ok = file_sha256(source_path) == selected["sourceFileSha256"]
    if not commit_ok or not source_hash_ok:
        errors.append("V75 NOVA checkout or paint source hash drifted")

    source_text = source_path.read_text()
    license_text = license_path.read_text()
    source_declarations_ok = all(
        fragment in source_text
        for fragment in (
            "discount: 0.95",
            "states: NFL-NBL-NPA NFL-NBL-PA FL-NBL-PA FL-BL-NPA",
            "actions: paint inspect ship reject",
            "observations: NBL BL",
            "0.5 0.0 0.0 0.5",
            "O: inspect : NFL-NBL-NPA : NBL 0.75",
            "O: inspect : FL-BL-NPA : BL 0.75",
            "R: ship : NFL-NBL-PA : * : * 1.0",
            "R: reject : FL-BL-NPA : * : * 1.0",
        )
    ) and "MIT License" in license_text
    if not source_declarations_ok:
        errors.append("V75 source structure or MIT license declaration drifted")

    model = parse_cassandra_pomdp_text(source_text, name="nova_paint_95")
    normalized = bool(
        np.isclose(model.initial.sum(), 1.0, atol=1e-12, rtol=0.0)
        and np.allclose(model.transition.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)
        and np.allclose(model.observation.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)
        and np.isfinite(model.reward).all()
    )

    source = config["sourceGrounding"]
    layer = config["prospectiveProjectLayer"]
    gates = config["economicGates"]
    p = float(source["inspectionAccuracy"])
    paint_success = float(source["paintSuccessProbability"])
    discount = float(source["discount"])
    paired_correct = p * p + (1.0 - p) * (1.0 - p)
    reject_contribution = (
        discount**2 * 0.5 * (2.0 * paired_correct - 1.0)
    )
    conditional_good_painted_ship = 2.0 * paint_success - 1.0
    ship_contribution = discount**3 * 0.5 * (
        paired_correct * conditional_good_painted_ship
        + (1.0 - paired_correct) * -1.0
    )
    fixed_value = reject_contribution + ship_contribution
    best_open_loop_value = 0.0
    raw_advantage = fixed_value - best_open_loop_value
    return_scale = 2.0 * sum(discount**step for step in range(4))
    normalized_advantage = raw_advantage / return_scale
    normalized_margin = normalized_advantage - float(
        gates["minimumNormalizedFixedPolicyOverBestOpenLoopAdvantage"]
    )

    checks = {
        "v74_authorizes_fresh_preregistered_successor": prior_authorization_ok,
        "economic_screen_precedes_adapter": preimplementation_ok,
        "pinned_commit_and_source_hash_match": commit_ok and source_hash_ok,
        "source_structure_and_MIT_license_match": source_declarations_ok,
        "source_model_is_normalized": normalized is gates["requiredSourceNormalization"],
        "source_is_outcome_untouched_but_not_discovery_clean": bool(
            selected["freshRelativeToV74Environment"]
            and not selected["sourceDiscoveryClean"]
        ),
        "source_inspection_accuracy_passes": p >= float(
            gates["minimumSourceInspectionAccuracy"]
        ),
        "point_model_marginal_observation_support": bool(
            p > 0.0 and p < 1.0 and gates["requiredPointModelMarginalObservationSupport"]
        ),
        "fixed_policy_value_strictly_positive": fixed_value > 0.0,
        "raw_economic_advantage_passes": raw_advantage >= float(
            gates["minimumRawFixedPolicyOverBestOpenLoopAdvantage"]
        ),
        "normalized_economic_advantage_passes": normalized_advantage >= float(
            gates["minimumNormalizedFixedPolicyOverBestOpenLoopAdvantage"]
        ),
        "normalized_margin_passes": normalized_margin >= float(
            gates["minimumNormalizedMarginAboveThreshold"]
        ),
        "beacon_is_nonharvestable": layer["beaconHarvestable"]
        is gates["requiredCalibrationBeaconHarvestable"],
        "best_open_loop_is_zero": best_open_loop_value
        == float(gates["requiredBestOpenLoopValue"]),
        "policy_optimizer_calls_are_zero": all(
            gates[name] == 0
            for name in (
                "maximumAdapterImplementationCount",
                "maximumOptimalPlannerCalls",
                "maximumMAPCalls",
                "maximumPosteriorSamplingCalls",
                "maximumPriorPolicyOutcomeAccessCount",
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    access = {
        "candidate_repository_metadata_reads": 4,
        "pinned_repositories_cloned": 2,
        "candidate_source_files_inspected": 4,
        "rejected_candidate_policy_value_count": 0,
        "closed_form_economic_screen_count": 1,
        "adapter_implementation_count": 0,
        "optimal_planner_calls": 0,
        "MAP_calls": 0,
        "posterior_sampling_calls": 0,
        "candidate_regrets_computed": 0,
        "prior_policy_outcome_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    inventory = {
        "schema_version": "75-active-sensing-confirmation-economic",
        "experiment": "v75_nova_paint_source_inventory",
        "repository": selected["repository"],
        "commit": actual_commit,
        "license": selected["license"],
        "source_files": {
            str(source_path.relative_to(checkout)): file_sha256(source_path),
            str(license_path.relative_to(checkout)): file_sha256(license_path),
        },
        "source_shape": {
            "states": len(model.states),
            "actions": len(model.actions),
            "observations": len(model.observations),
        },
        "frozen_source_grounding": source,
        "frozen_project_layer": layer,
        "prior_exposure": selected["priorExposure"],
        "rejected_candidate": {
            "repository": "https://github.com/abaisero/gym-pomdps",
            "environment": "shopping_2",
            "reason": "deterministic movement observations create zero-likelihood histories under the wrong shared codebook",
            "policy_values_computed": False,
        },
        "access": access,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    audit = {
        "schema_version": "75-active-sensing-confirmation-economic",
        "experiment": "v75_nova_paint_prospective_economic_screen",
        "passed": not errors and not failed,
        "decision": (
            "freeze_economic_pass_and_authorize_exact_replication_preregistration"
            if not errors and not failed
            else "freeze_negative_feasibility_and_stop_before_adapter"
        ),
        "errors": errors,
        "failed_gates": failed,
        "checks": checks,
        "metrics": {
            "source_inspection_accuracy": p,
            "paired_correct_probability": paired_correct,
            "conditional_good_painted_ship_reward": conditional_good_painted_ship,
            "discounted_reject_contribution": reject_contribution,
            "discounted_paint_ship_contribution": ship_contribution,
            "fixed_economic_policy_value": fixed_value,
            "best_open_loop_value": best_open_loop_value,
            "raw_fixed_policy_over_open_loop_advantage": raw_advantage,
            "return_scale": return_scale,
            "normalized_fixed_policy_over_open_loop_advantage": normalized_advantage,
            "normalized_margin_above_threshold": normalized_margin,
        },
        "claim_boundary": config["claimBoundary"],
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lock = {
        "schema_version": "75-active-sensing-confirmation-economic",
        "experiment": "v75_nova_paint_confirmation_economic_lock",
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
        "outcome": {
            "passed_all_economic_gates": audit["passed"],
            "failed_gate_count": len(failed) + len(errors),
            "fixed_policy_value": fixed_value,
            "best_open_loop_value": best_open_loop_value,
            "normalized_advantage": normalized_advantage,
            "normalized_margin": normalized_margin,
            "source_discovery_clean": False,
            "source_policy_outcome_untouched": True,
        },
        "authorization": {
            "modify_source_economic_blueprint_or_gates": False,
            "preregister_exact_replication_design": audit["passed"],
            "implement_adapter_and_structural_tests": False,
            "run_policy_outcomes": False,
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
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
