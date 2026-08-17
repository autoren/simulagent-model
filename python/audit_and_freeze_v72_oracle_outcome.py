#!/usr/bin/env python3
"""Independently audit and freeze the V72 engineered-oracle outcome."""
from __future__ import annotations

import hashlib
import json
import math
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
        PROJECT_ROOT / "configs/v72-active-sensing-oracle-evaluator-lock.json"
    )
    attempt_path = PROJECT_ROOT / "outputs/v72-active-sensing/oracle-evaluation/attempt.json"
    result_path = PROJECT_ROOT / "outputs/v72-active-sensing/oracle-evaluation/result.json"
    report_path = PROJECT_ROOT / "docs/v72-active-sensing-oracle-results.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v72_oracle_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v72-active-sensing/oracle-outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v72-active-sensing-oracle-outcome-lock.json"
    if lock_path.exists():
        raise RuntimeError("V72 oracle outcome is already frozen")

    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    lock_payload = {
        key: value
        for key, value in evaluator_lock.items()
        if key != "lock_payload_sha256"
    }
    attempt = json.loads(attempt_path.read_text())
    result = json.loads(result_path.read_text())
    errors: list[str] = []

    lock_ok = bool(
        payload_hash(lock_payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_engineered_oracle_outcomes_once"]
        and not evaluator_lock["authorization"]["inspect_external_candidate_metadata"]
        and not evaluator_lock["authorization"][
            "compute_external_candidate_policy_values_actions_regrets_or_EIG"
        ]
        and file_sha256(PROJECT_ROOT / evaluator_lock["evaluator"])
        == evaluator_lock["evaluator_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator_lock["fixture_core"])
        == evaluator_lock["fixture_core_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator_lock["planning_core"])
        == evaluator_lock["planning_core_sha256"]
    )
    if not lock_ok:
        errors.append("V72 evaluator lock or frozen dependency drifted")

    access_ok = bool(
        attempt["attempt_number"] == evaluator_lock["expected_attempt_number"] == 1
        and attempt["fixture_count"] == evaluator_lock["expected_fixture_count"] == 2
        and attempt["external_candidate_metadata_records_read"] == 0
        and attempt["external_candidate_policy_values_computed"] == 0
        and attempt["V71_protected_access_count"] == 0
        and attempt["human_record_access_count"] == 0
        and attempt["model_forward_pass_count"] == 0
        and attempt["adapter_training_run_count"] == 0
        and result["access"] == attempt
    )
    if not access_ok:
        errors.append("V72 attempt count or access firewall failed")

    positive = result["fixtures"]["positive"]
    negative = result["fixtures"]["negative_control"]
    expected_mi = math.log(2.0) + 0.9 * math.log(0.9) + 0.1 * math.log(0.1)
    structural_ok = bool(
        all(result["gates"].values())
        and result["passed"]
        and result["claim_boundary"]
        == "engineered mechanism oracle only; not scientific evidence"
        and close(
            positive["structural"]["calibration_mutual_information_nats"],
            expected_mi,
        )
        and close(
            positive["structural"][
                "inspection_state_mutual_information_given_codebook_nats"
            ],
            expected_mi,
        )
        and positive["structural"]["point_model_supports_identical"]
        and positive["structural"]["point_model_on_support_rate"] == 1.0
        and positive["structural"]["fallback_count"] == 0
        and negative["structural"]["point_model_supports_identical"]
        and negative["structural"]["point_model_on_support_rate"] == 1.0
        and negative["structural"]["fallback_count"] == 0
    )
    if not structural_ok:
        errors.append("V72 shared-support information or claim boundary failed")

    positive_ok = bool(
        positive["exact"]["root_action"] == "calibrate"
        and close(positive["exact"]["value"], 2.6)
        and close(positive["exact"]["root_action_margin"], 5.6)
        and positive["map"]["root_action"] == "inspect"
        and close(positive["map"]["exact_environment_value"], -6.0)
        and close(positive["posterior_sampling"]["exact_environment_value"], -6.0)
        and close(positive["open_loop"]["value"], -3.0)
        and close(positive["myopic"]["exact_environment_value"], -3.0)
        and close(positive["map"]["normalized_regret"], 8.6 / 90.0)
        and close(positive["posterior_sampling"]["normalized_regret"], 8.6 / 90.0)
        and positive["exact"]["distinct_terminal_repair_actions"]
        == ["repair_A", "repair_B"]
        and len(positive["exact"]["calibration_branches"]) == 2
        and all(
            branch["second_action"] == "inspect"
            for branch in positive["exact"]["calibration_branches"]
        )
    )
    if not positive_ok:
        errors.append("V72 positive mechanism oracle did not reproduce fixed expectations")

    negative_ok = bool(
        negative["exact"]["root_action"] == "repair_A"
        and negative["map"]["root_action"] == "repair_A"
        and close(negative["exact"]["value"], 5.0)
        and close(negative["map"]["exact_environment_value"], 5.0)
        and close(negative["posterior_sampling"]["exact_environment_value"], 5.0)
        and close(negative["map"]["normalized_regret"], 0.0)
        and close(negative["posterior_sampling"]["normalized_regret"], 0.0)
    )
    if not negative_ok:
        errors.append("V72 dominant-action negative control failed")

    external_absent = not (
        PROJECT_ROOT / "outputs/v72-active-sensing/external-source-inventory.json"
    ).exists()
    if not external_absent:
        errors.append("External source inventory predates the frozen oracle outcome")

    checks = {
        "frozen_evaluator_and_dependency_integrity": lock_ok,
        "one_attempt_zero_external_protected_human_or_model_access": access_ok,
        "shared_support_information_and_non_evidence_boundary": structural_ok,
        "positive_active_sensing_mechanism": positive_ok,
        "dominant_action_negative_control": negative_ok,
        "external_inventory_absent_until_oracle_freeze": external_absent,
    }
    audit = {
        "schema_version": "72-active-sensing-oracle",
        "experiment": "v72_oracle_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_engineered_mechanism_and_authorize_metadata_only_external_discovery"
            if not errors
            else "reject_oracle_and_forbid_external_discovery"
        ),
        "errors": errors,
        "checks": checks,
        "independent_expectations": {
            "binary_symmetric_channel_information_nats": expected_mi,
            "positive_exact_value": 2.6,
            "positive_MAP_and_posterior_sampling_value": -6.0,
            "positive_normalized_regret": 8.6 / 90.0,
            "negative_exact_MAP_and_posterior_sampling_value": 5.0,
        },
        "access": attempt,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    report = f"""# V72 shared-support active-sensing oracle results

## Bottom line

The preregistered engineered mechanism check passed all {len(result['gates'])} oracle gates in its single authorized run. This verifies that the locked exact-planning implementation can express a fallback-free sensor-codebook problem where preserving uncertainty changes the best action. It is an implementation oracle, not external, development, or confirmation evidence.

## Positive mechanism fixture

- Exact Bayes-adaptive root action: `calibrate`.
- Exact value: `{positive['exact']['value']}`; root action margin: `{positive['exact']['root_action_margin']}`.
- After either calibration observation, the exact policy chose `inspect`; across the next observations it used both `repair_A` and `repair_B`.
- MAP and persistent posterior sampling both chose `inspect` at the root and had exact-environment value `-6.0`.
- Exact minus MAP and posterior-sampling regret was `8.6`, or `{positive['map']['normalized_regret']}` of the locked finite-horizon return scale.
- Best open-loop and myopic exact-environment values were both `-3.0`.

The two codebooks had identical full observation support. Calibration and inspection each carried `{positive['structural']['calibration_mutual_information_nats']}` nats in the relevant binary channel, all point-model branches stayed on-support, and fallback count was zero.

The seven-state Markov representation makes a repeated `calibrate` action in the already-calibrated phase repeat the reference reading. This behavior was declared in the implementation diagnostics. At the locked three-action horizon it was legal but noncompetitive: every exact branch inspected second.

## Dominant-action negative control

The control chose `repair_A` under exact Bayes-adaptive, MAP, posterior sampling, and myopic control. Exact, MAP, and posterior-sampling values were all `5.0`; both normalized regrets were exactly `0.0`. Thus shared-support uncertainty alone did not manufacture an adaptive advantage when one control action dominated.

## Claim boundary and next authorization

This result establishes mechanism availability only. The oracle used two engineered fixtures, no human records, no model or adapter calls, no SMC2, no V71 protected access, and no external candidate metadata or outcomes.

The next authorized step is metadata-only discovery of fresh external active-sensing sources. Candidate policy values, optimal actions, regrets, and expected information gain remain forbidden until an external source inventory, structural feasibility gate, partition, and evaluator are separately frozen.
"""
    report_path.write_text(report)

    lock = {
        "schema_version": "72-active-sensing-oracle",
        "experiment": "v72_active_sensing_oracle_outcome_lock",
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
            "passed_all_oracle_gates": True,
            "oracle_gate_count": len(result["gates"]),
            "scientific_evidence": False,
            "external_candidate_metadata_record_count": 0,
            "external_candidate_policy_value_count": 0,
            "V71_protected_access_count": 0,
        },
        "authorization": {
            "modify_or_rerun_V71": False,
            "modify_or_rerun_V72_oracle": False,
            "describe_oracle_as_scientific_external_development_or_confirmation_evidence": False,
            "inspect_fresh_external_candidate_metadata": True,
            "freeze_external_source_inventory_and_structural_partition": True,
            "compute_external_candidate_policy_values_actions_regrets_or_EIG": False,
            "read_V71_protected_models_histories_or_outcomes": False,
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
