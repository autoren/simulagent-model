#!/usr/bin/env python3
"""Audit and freeze V70 primary reporting and fallback diagnostics."""
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
    config_path = PROJECT_ROOT / "configs/v70-confirmatory-reporting.json"
    plan_path = PROJECT_ROOT / "docs/v70-confirmatory-reporting-plan.md"
    census_path = PROJECT_ROOT / "configs/v70-confirmatory-census-seal.json"
    audit_path = PROJECT_ROOT / "outputs/v70-confirmatory/reporting-audit.json"
    lock_path = PROJECT_ROOT / "configs/v70-confirmatory-reporting-lock.json"
    if lock_path.exists():
        raise RuntimeError("V70 reporting already frozen")
    config = json.loads(config_path.read_text())
    census = json.loads(census_path.read_text())
    census_payload = {
        key: value for key, value in census.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    census_ok = bool(
        payload_hash(census_payload) == census["lock_payload_sha256"]
        and census["authorization"]["write_and_audit_confirmatory_reporting_specification"]
        and not census["authorization"]["write_confirmatory_evaluator_before_reporting_lock"]
        and not census["authorization"]["run_confirmatory_outcomes"]
        and not census["authorization"]["drop_or_replace_models"]
        and census["record_count"] == 244
        and len(census["model_counts"]) == 9
    )
    if not census_ok:
        errors.append("V70 census seal or reporting-only authorization failed")

    design = json.loads(
        (PROJECT_ROOT / census["confirmatory_design_lock"]).read_text()
    )["config_payload"]
    threshold = float(design["normalization"]["materialNormalizedRegret"])
    primary = config["primaryDecision"]
    primary_ok = bool(
        threshold == 0.005
        and "same_record" in primary["qualifyingMAPRecord"]
        and primary["qualifyingMAPModel"] == "at_least_one_qualifying_MAP_record"
        and "same_conservatively_qualified_model_set"
        in primary["pairedGateInterpretation"]
        and primary["includeEverySealedRecord"]
        and primary["fallbackAffectedRecordsRemainInPrimary"]
        and primary["recordCountsCannotCompensateForModelLevelGateFailure"]
    )
    if not primary_ok:
        errors.append("V70 primary paired model-level decision semantics are incomplete")

    diagnostics = config["fallbackDiagnostics"]
    diagnostics_ok = bool(
        diagnostics["primaryStatus"] == "secondary_non_decisional_diagnostics_only"
        and "off_support_branch_count_greater_than_zero"
        in diagnostics["affectedRecordRule"]
        and "expected_off_support_entry_probability_greater_than_zero"
        in diagnostics["affectedRecordRule"]
        and all(
            item in diagnostics["forbiddenUses"]
            for item in (
                "change_any_primary_gate",
                "remove_a_record_from_the_primary_result",
                "reverse_a_primary_decision",
                "select_or_drop_a_model",
            )
        )
    )
    if not diagnostics_ok:
        errors.append("V70 fallback diagnostics could affect the primary result")

    tier_ok = bool(
        config["tierReporting"]["TierA"].startswith("primary_nine_model")
        and config["tierReporting"]["TierB"].startswith("side_by_side_cheese")
        and not config["tierReporting"]["externalUncertaintyFamilyClaim"]
        and config["firewall"]["mergeTierAAndTierBClaims"] == "forbidden"
    )
    if not tier_ok:
        errors.append("V70 Tier A and Tier B reporting boundaries are not separate")

    persistence_ok = bool(
        all(config["outcomePersistence"].values())
        and config["stageAuthorization"]["auditAndFreezeReporting"]
        and not config["stageAuthorization"]["writeAndAuditEvaluator"]
        and not config["stageAuthorization"]["runConfirmatoryOutcome"]
    )
    if not persistence_ok:
        errors.append("V70 durable persistence or reporting-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/evaluate_v70_confirmatory.py",
            "configs/v70-confirmatory-evaluator-lock.json",
            "outputs/v70-confirmatory/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V70 evaluator or outcome exists before reporting lock")

    checks = {
        "census_binding_and_reporting_only_authorization": census_ok,
        "paired_model_level_primary_decision": primary_ok,
        "secondary_non_decisional_fallback_diagnostics": diagnostics_ok,
        "separate_Tier_A_and_Tier_B_reporting": tier_ok,
        "durable_outcome_persistence_and_stage_firewall": persistence_ok,
        "evaluator_and_outcome_absent": downstream_absent,
    }
    audit = {
        "schema_version": "70-confirmatory-reporting",
        "experiment": "v70_confirmatory_reporting_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_reporting_and_authorize_evaluator_implementation_only"
            if not errors
            else "reject_v70_confirmatory_reporting"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "confirmatory_rows_evaluated": 0,
            "confirmatory_policy_values_computed": 0,
            "development_models_rescored": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "70-confirmatory-reporting",
        "experiment": "v70_confirmatory_reporting_lock",
        "census_seal": str(census_path.relative_to(PROJECT_ROOT)),
        "census_seal_sha256": file_sha256(census_path),
        "reporting_config": str(config_path.relative_to(PROJECT_ROOT)),
        "reporting_config_sha256": file_sha256(config_path),
        "reporting_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "reporting_plan_sha256": file_sha256(plan_path),
        "reporting_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "reporting_audit_sha256": file_sha256(audit_path),
        "material_normalized_regret": threshold,
        "authorization": {
            "modify_design_resource_census_reporting_or_gates": False,
            "write_and_audit_confirmatory_evaluator": True,
            "run_confirmatory_outcome": False,
            "rescore_development_models": False,
            "drop_or_replace_models": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
