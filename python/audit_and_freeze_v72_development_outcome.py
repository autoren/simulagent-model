#!/usr/bin/env python3
"""Audit, report, and freeze the negative V72 RockSample development result."""
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


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    evaluator_lock_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-evaluator-lock.json"
    )
    attempt_path = (
        PROJECT_ROOT / "outputs/v72-active-sensing/development-evaluation/attempt.json"
    )
    result_path = (
        PROJECT_ROOT / "outputs/v72-active-sensing/development-evaluation/result.json"
    )
    report_path = PROJECT_ROOT / "docs/v72-active-sensing-development-results.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v72_development_outcome.py"
    audit_path = (
        PROJECT_ROOT / "outputs/v72-active-sensing/development-outcome-audit.json"
    )
    lock_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-outcome-lock.json"
    )
    if lock_path.exists():
        raise RuntimeError("V72 development outcome is already frozen")

    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    lock_payload = {
        key: value
        for key, value in evaluator_lock.items()
        if key != "lock_payload_sha256"
    }
    attempt = json.loads(attempt_path.read_text())
    result = json.loads(result_path.read_text())
    row = result["row"]
    errors: list[str] = []

    lock_ok = bool(
        payload_hash(lock_payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_development_outcomes_once"]
        and not evaluator_lock["authorization"][
            "inspect_or_select_protected_confirmation_sources"
        ]
        and not evaluator_lock["authorization"][
            "compute_protected_confirmation_policy_values"
        ]
        and file_sha256(PROJECT_ROOT / evaluator_lock["evaluator"])
        == evaluator_lock["evaluator_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator_lock["exporter"])
        == evaluator_lock["exporter_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator_lock["planning_core"])
        == evaluator_lock["planning_core_sha256"]
    )
    if not lock_ok:
        errors.append("V72 evaluator lock or dependency drifted")

    access_ok = bool(
        attempt["attempt_number"] == evaluator_lock["expected_attempt_number"] == 1
        and attempt["model_count"] == evaluator_lock["expected_model_count"] == 1
        and attempt["protected_confirmation_policy_value_count"]
        == evaluator_lock["expected_protected_confirmation_policy_value_count"]
        == 0
        and attempt["candidate_EIG_value_count"] == 0
        and attempt["V71_protected_access_count"] == 0
        and attempt["human_record_access_count"] == 0
        and attempt["model_forward_pass_count"] == 0
        and attempt["adapter_training_run_count"] == 0
        and result["access"] == attempt
    )
    if not access_ok:
        errors.append("V72 one-shot or protected-access firewall failed")

    gate_ok = bool(
        not result["passed"]
        and result["decision"]
        == "freeze_negative_result_and_stop_V72_before_protected_discovery"
        and len(result["gates"]) == 9
        and sum(bool(value) for value in result["gates"].values()) == 2
        and result["gates"]["common_support_zero_fallback"]
        and result["gates"]["exact_root_margin"]
        and not result["gates"]["exact_root_action"]
        and not result["gates"]["MAP_root_action"]
        and not result["gates"]["MAP_material_regret"]
        and not result["gates"]["posterior_sampling_material_regret"]
        and not result["gates"]["exact_over_open_loop_advantage"]
    )
    if not gate_ok:
        errors.append("V72 negative gate pattern did not reproduce")

    # Independent scalar reexecution of the selected open-loop path:
    # west (0), sample known-good reference (10), east (0), east/exit (5).
    independent_dominant_path_value = 0.95 * 10.0 + 0.95**3 * 5.0
    outcome_ok = bool(
        row["exact"]["root_action"] == "west"
        and row["map"]["root_action"] == "west"
        and row["posterior_sampling"]["root_action_distribution"]["west"] == 1.0
        and row["open_loop"]["selected_actions"]
        == ["west", "sample", "east", "east"]
        and close(row["exact"]["value"], independent_dominant_path_value)
        and close(row["map"]["exact_environment_value"], independent_dominant_path_value)
        and close(
            row["posterior_sampling"]["exact_environment_value"],
            independent_dominant_path_value,
        )
        and close(row["open_loop"]["value"], independent_dominant_path_value)
        and close(row["map"]["normalized_regret"], 0.0)
        and close(row["posterior_sampling"]["normalized_regret"], 0.0)
        and close(row["open_loop"]["normalized_exact_advantage"], 0.0)
        and row["support"]["point_model_supports_identical"]
        and row["support"]["point_model_on_support_rate"] == 1.0
        and row["support"]["fallback_count"] == 0
        and row["bellman_nodes"] == 772
    )
    if not outcome_ok:
        errors.append("V72 dominant known-reward path or control agreement failed")

    protected_absent = not (
        PROJECT_ROOT / "outputs/v72-active-sensing/protected-confirmation-evaluation"
    ).exists()
    if not protected_absent:
        errors.append("V72 protected confirmation outcome exists after failed development")

    checks = {
        "frozen_evaluator_and_dependency_integrity": lock_ok,
        "one_attempt_zero_protected_EIG_human_or_model_access": access_ok,
        "seven_of_nine_scientific_gates_failed": gate_ok,
        "independent_dominant_known_reward_path": outcome_ok,
        "protected_confirmation_absent": protected_absent,
    }
    audit = {
        "schema_version": "72-active-sensing-development-evaluation",
        "experiment": "v72_development_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_negative_development_result_and_close_V72"
            if not errors
            else "reject_v72_development_outcome"
        ),
        "errors": errors,
        "checks": checks,
        "independent_dominant_path_value": independent_dominant_path_value,
        "access": attempt,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    report = f"""# V72 external RockSample development results

## Bottom line

V72 is a valid and informative negative development result. The engineered oracle proved that the planner can express the intended shared-support active-sensing mechanism, but the prospectively selected external RockSample blueprint did not instantiate it. Seven of nine noncompensatory development gates failed, so V72 stops before any protected-source discovery or confirmation outcome.

## What happened

Exact Bayes-adaptive, MAP certainty equivalence, persistent posterior sampling, and best open loop all selected the same root action: `west`. The best open-loop sequence was `west → sample → east → east`, with exact value `{row['exact']['value']}`. The independent audit reproduced this directly as `0.95 × 10 + 0.95³ × 5 = {independent_dominant_path_value}`.

The sequence moves to the known-good reference rock, samples its guaranteed `+10` reward, and then exits for `+5`. Because the calibration reference was itself a harvestable source of reward, uncertainty about sensor labels was irrelevant to the best plan.

- MAP normalized regret: `{row['map']['normalized_regret']}`.
- Posterior-sampling normalized regret: `{row['posterior_sampling']['normalized_regret']}`.
- Exact-over-open-loop normalized advantage: `{row['open_loop']['normalized_exact_advantage']}`.
- Exact root margin: `{row['exact']['root_action_margin']}`.
- Calibration-channel mutual information was nonzero (`{row['calibration_mutual_information_nats']}` nats), but it had no decision value.

Both point models retained identical observation support, their on-support rate was `1.0`, and fallback count was zero. Thus this is another control-relevance boundary, not a fallback artifact or implementation failure.

## Gate result

Only common-support/zero-fallback and the root-margin gate passed. The required exact and MAP root actions, reference-then-target branch structure, distinct final controls, MAP regret, posterior-sampling regret, and adaptive-over-open-loop advantage all failed.

## Correct successor constraint

A successor must prevent the calibration reference from being a rewarding control target. Suitable designs include a known-bad reference, a non-harvestable calibration beacon, or an observation-only reference state. Before any policy outcome, a structural dominance audit must enumerate immediately harvestable known rewards and reject a design if an open-loop route can bypass the sensing decision. Sensor discriminability and the final good/bad control threshold must also be certified from source parameters before evaluator locking.

No V72 parameter, reward, horizon, model, control, or gate may be changed retrospectively. No protected confirmation source was selected or scored; no EIG, SMC2, human, model, or adapter work occurred.
"""
    report_path.write_text(report)

    lock = {
        "schema_version": "72-active-sensing-development-evaluation",
        "experiment": "v72_rocksample_development_outcome_lock",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "outcome_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "outcome_auditor_sha256": file_sha256(auditor_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "report_sha256": file_sha256(report_path),
        "outcome": {
            "passed_development_gates": False,
            "passed_gate_count": 2,
            "failed_gate_count": 7,
            "development_model_count": 1,
            "maximum_normalized_MAP_regret": 0.0,
            "maximum_normalized_posterior_sampling_regret": 0.0,
            "protected_confirmation_policy_value_count": 0,
            "V71_protected_access_count": 0,
        },
        "authorization": {
            "modify_or_rerun_V71_or_V72": False,
            "inspect_or_score_V72_protected_confirmation_sources": False,
            "tune_V72_sensor_reward_horizon_model_controls_or_gates": False,
            "begin_successor_only_after_fresh_preregistration_and_structural_dominance_audit": True,
            "reuse_V72_development_model_for_successor_outcomes": False,
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
