#!/usr/bin/env python3
"""Audit and freeze the outcome-free V72 active-sensing oracle design."""
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
    config_path = PROJECT_ROOT / "configs/v72-active-sensing-oracle-design.json"
    plan_path = PROJECT_ROOT / "docs/v72-active-sensing-oracle-plan.md"
    v71_path = PROJECT_ROOT / "configs/v71-sensor-codebook-development-outcome-lock.json"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v72_oracle_design.py"
    audit_path = PROJECT_ROOT / "outputs/v72-active-sensing/oracle-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v72-active-sensing-oracle-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V72 oracle design is already frozen")

    config = json.loads(config_path.read_text())
    v71 = json.loads(v71_path.read_text())
    v71_payload = {key: value for key, value in v71.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    v71_ok = bool(
        payload_hash(v71_payload) == v71["lock_payload_sha256"]
        and not v71["outcome"]["passed_development_gates"]
        and v71["outcome"]["protected_confirmation_policy_value_count"] == 0
        and v71["authorization"][
            "begin_new_family_only_after_fresh_source_and_preregistration"
        ]
        and not v71["authorization"]["modify_or_rerun_V71"]
        and not v71["authorization"][
            "read_V71_protected_confirmation_histories_or_outcomes"
        ]
    )
    if not v71_ok:
        errors.append("V71 closure does not authorize a separate preregistered family")

    shared = config["sharedParameters"]
    fixture_ok = bool(
        shared["latentCodebooks"] == ["canonical", "reversed"]
        and shared["latentPrior"] == [0.5, 0.5]
        and shared["hiddenConditions"] == ["A", "B"]
        and shared["conditionPrior"] == [0.5, 0.5]
        and shared["sensorReliability"] == 0.9
        and shared["horizonActions"] == 3
        and shared["actionsInTieBreakOrder"]
        == ["calibrate", "inspect", "repair_A", "repair_B"]
        and len(shared["statesInSourceOrder"]) == 7
        and config["positiveFixture"]["rewardRules"]["correctRepair"] == 10.0
        and config["positiveFixture"]["rewardRules"]["wrongRepair"] == -20.0
        and config["negativeControlFixture"]["rewardRules"][
            "repair_AEveryCondition"
        ]
        > config["negativeControlFixture"]["rewardRules"][
            "repair_BEveryCondition"
        ]
    )
    if not fixture_ok:
        errors.append("V72 positive or negative fixture parameters drifted")

    gates = config["oracleGates"]
    gates_ok = bool(
        gates["requiredPositiveExactRootAction"] == "calibrate"
        and gates["requiredPositiveMAPRootAction"] == "inspect"
        and gates["minimumPositiveExactRootActionMargin"] == 1.0
        and gates["requiredPositiveSecondActionAfterEveryCalibrationObservation"]
        == "inspect"
        and gates["minimumDistinctPositiveTerminalRepairActionsAcrossReachableHistories"]
        == 2
        and gates["minimumPositiveNormalizedMAPRegret"] == 0.05
        and gates["minimumPositiveNormalizedPosteriorSamplingRegret"] == 0.05
        and gates["maximumNegativeControlNormalizedMAPRegret"] == 1e-12
        and gates["maximumNegativeControlNormalizedPosteriorSamplingRegret"] == 1e-12
        and gates["minimumPointModelOnSupportRate"] == 1.0
        and gates["maximumFallbackCount"] == 0
        and gates["maximumExternalCandidatePolicyValueCount"] == 0
        and gates["maximumV71ProtectedAccessCount"] == 0
    )
    if not gates_ok:
        errors.append("V72 discriminative positive/negative oracle gates are incomplete")

    boundary = config["claimBoundary"]
    authorization = config["stageAuthorization"]
    boundary_ok = bool(
        config["decisionRule"]["oraclePassIsScientificEvidence"] is False
        and config["decisionRule"]["oraclePassAuthorizesExternalOutcomeEvaluation"]
        is False
        and boundary["engineeredMechanismOracle"]
        and not boundary["externalBenchmarkEvidence"]
        and not boundary["developmentEvidence"]
        and not boundary["confirmationEvidence"]
        and not boundary["SMC2"]
        and not boundary["humanData"]
        and not boundary["modelAccess"]
        and not boundary["adapterTraining"]
        and authorization["auditAndFreezeOracleDesign"]
        and not authorization["implementAndAuditOracleEvaluator"]
        and not authorization["runOracleOutcomes"]
        and not authorization["inspectExternalCandidateMetadata"]
        and not authorization["computeExternalCandidateOutcomes"]
    )
    if not boundary_ok:
        errors.append("V72 oracle claim boundary or design-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/v72_active_sensing_oracles.py",
            "python/evaluate_v72_active_sensing_oracle.py",
            "configs/v72-active-sensing-oracle-evaluator-lock.json",
            "outputs/v72-active-sensing/oracle-evaluation",
            "configs/v72-active-sensing-oracle-outcome-lock.json",
            "outputs/v72-active-sensing/external-source-inventory.json",
        )
    )
    if not downstream_absent:
        errors.append("V72 implementation, outcome, or external inventory predates design lock")

    checks = {
        "frozen_negative_V71_and_new_family_authorization": v71_ok,
        "fixed_positive_and_negative_oracle_parameters": fixture_ok,
        "discriminative_noncompensatory_oracle_gates": gates_ok,
        "oracle_only_claim_and_design_stage_firewall": boundary_ok,
        "implementation_outcome_and_external_inventory_absent": downstream_absent,
        "zero_policy_value_action_regret_EIG_or_external_candidate_access": True,
    }
    audit = {
        "schema_version": "72-active-sensing-oracle-design",
        "experiment": "v72_oracle_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_oracle_design_and_authorize_implementation_only"
            if not errors
            else "reject_v72_oracle_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "oracle_policy_values_computed": 0,
            "oracle_optimal_actions_computed": 0,
            "external_candidate_metadata_records_read": 0,
            "external_candidate_policy_values_computed": 0,
            "V71_protected_access_count": 0
        }
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "72-active-sensing-oracle-design",
        "experiment": "v72_active_sensing_oracle_design_lock",
        "V71_outcome_lock": str(v71_path.relative_to(PROJECT_ROOT)),
        "V71_outcome_lock_sha256": file_sha256(v71_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_or_rerun_V71": False,
            "modify_V72_oracle_design_or_gates": False,
            "implement_and_audit_oracle_evaluator": True,
            "run_oracle_outcomes": False,
            "inspect_external_candidate_metadata": False,
            "compute_external_candidate_policy_values_actions_regrets_or_EIG": False,
            "read_V71_protected_models_histories_or_outcomes": False
        }
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
