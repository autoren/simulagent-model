#!/usr/bin/env python3
"""Audit and freeze the V70 confirmatory multi-environment design."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v70-confirmatory-multi-environment.json"
    plan_path = PROJECT_ROOT / "docs/v70-confirmatory-multi-environment-plan.md"
    outcome_path = PROJECT_ROOT / "configs/v69-development-outcome-lock.json"
    source_path = PROJECT_ROOT / "configs/v68-source-feasibility-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v70-confirmatory/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v70-confirmatory-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V70 confirmatory design already frozen")
    config = json.loads(config_path.read_text())
    outcome = json.loads(outcome_path.read_text())
    outcome_payload = {
        key: value for key, value in outcome.items() if key != "lock_payload_sha256"
    }
    source = json.loads(source_path.read_text())
    inventory_path = PROJECT_ROOT / source["source_inventory"]
    inventory = json.loads(inventory_path.read_text())
    errors: list[str] = []

    outcome_ok = bool(
        payload_hash(outcome_payload) == outcome["lock_payload_sha256"]
        and outcome["outcome"]["passed"]
        and outcome["outcome"]["confirmatory_models_scored"] == 0
        and outcome["authorization"]["preregister_confirmatory_multi_environment_design"]
        and not outcome["authorization"]["construct_confirmatory_census_before_design_lock"]
        and not outcome["authorization"]["score_confirmatory_models_before_all_new_locks"]
    )
    if not outcome_ok:
        errors.append("V69 positive outcome does not authorize V70 design only")

    inventory_ok = bool(
        file_sha256(inventory_path) == source["source_inventory_sha256"]
        and source["prospective_role_census"][
            "confirmatoryStructurallyRelatedButOutcomeUntouched"
        ]
        == 3
        and source["prospective_role_census"]["confirmatoryNovelAndOutcomeUntouched"]
        == 6
    )
    if not inventory_ok:
        errors.append("pinned source inventory or prospective role census drifted")

    expected_roles = {
        row["file"]: (
            "structurally_related"
            if row["role"] == "confirmatoryStructurallyRelatedButOutcomeUntouched"
            else "novel"
        )
        for row in inventory["models"]
        if row["role"].startswith("confirmatory") and row["source_valid"]
    }
    inventory_actions = {
        row["file"]: set(row["action_labels"])
        for row in inventory["models"]
        if row["file"] in expected_roles
    }
    assigned = {row["file"]: row for row in config["confirmatoryModels"]}
    roles = Counter(row["stratum"] for row in config["confirmatoryModels"])
    models_ok = bool(
        len(assigned) == len(config["confirmatoryModels"]) == 9
        and set(assigned) == set(expected_roles)
        and roles == {"structurally_related": 3, "novel": 6}
        and all(assigned[name]["stratum"] == expected_roles[name] for name in assigned)
        and all(
            len(row["canonicalActionCycle"])
            == len(set(row["canonicalActionCycle"]))
            and set(row["canonicalActionCycle"]) == inventory_actions[row["file"]]
            for row in config["confirmatoryModels"]
        )
    )
    if not models_ok:
        errors.append("V70 does not retain all nine role-assigned models and action alphabets")

    v69_config = json.loads(
        (PROJECT_ROOT / "configs/v69-development-design-lock.json").read_text()
    )["config_payload"]
    family_ok = bool(
        config["unknownDynamicsFamily"]["identityNames"]
        == v69_config["unknownDynamicsFamily"]["identityNames"]
        and config["unknownDynamicsFamily"]["identityPrior"]
        == v69_config["unknownDynamicsFamily"]["identityPrior"]
        and config["unknownDynamicsFamily"]["thetaSupport"]
        == v69_config["unknownDynamicsFamily"]["thetaSupport"]
        and config["unknownDynamicsFamily"]["transitionDefinition"]
        == v69_config["unknownDynamicsFamily"]["transitionDefinition"]
        and config["exactPlanning"]["horizonActions"] == 3
        and config["exactPlanning"]["primaryQuadratureNodes"] == 65
        and config["exactPlanning"]["convergenceQuadratureNodes"] == 129
        and config["normalization"]["materialNormalizedRegret"]
        == v69_config["gates"]["materialNormalizedRegret"]
    )
    if not family_ok:
        errors.append("V70 family, exact planning, or materiality differs from V69")

    gates = config["confirmatoryGates"]
    gates_ok = bool(
        gates["minimumConfirmatoryModels"] == 9
        and gates["minimumStructurallyRelatedModels"] == 3
        and gates["minimumNovelModels"] == 6
        and gates["minimumModelsWithExactBAMAPRootActionDisagreement"] == 3
        and gates[
            "minimumStructurallyRelatedModelsWithExactBAMAPRootActionDisagreement"
        ]
        == 1
        and gates["minimumNovelModelsWithExactBAMAPRootActionDisagreement"] == 2
        and gates["minimumModelsWithMaterialMAPRegret"] == 3
        and gates["minimumStructurallyRelatedModelsWithMaterialMAPRegret"] == 1
        and gates["minimumNovelModelsWithMaterialMAPRegret"] == 2
        and gates["minimumMaximumNormalizedMAPRegret"] == 0.01
        and gates["maximumRecordSelectionOrRejectionCount"] == 0
    )
    if not gates_ok:
        errors.append("V70 model-level or noncompensatory gates are incomplete")

    boundary_ok = bool(
        config["claimBoundary"]["TierAProjectAuthoredFamily"]
        and config["claimBoundary"]["TierBExternalSourcePairReportedSeparately"]
        and not config["claimBoundary"]["externalUncertaintyFamilyClaim"]
        and config["firewall"]["mergeTierAAndTierBClaims"] == "forbidden"
        and config["stageAuthorization"]["auditAndFreezeConfirmatoryDesign"]
        and not config["stageAuthorization"]["auditAndFreezeResourceFeasibility"]
        and not config["stageAuthorization"]["runConfirmatoryOutcomes"]
    )
    if not boundary_ok:
        errors.append("V70 claim boundary, firewall, or design-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v70-confirmatory-resource-lock.json",
            "configs/v70-confirmatory-census-seal.json",
            "python/evaluate_v70_confirmatory.py",
            "configs/v70-confirmatory-evaluator-lock.json",
            "outputs/v70-confirmatory/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V70 resource, census, or outcome artifacts exist before design lock")

    checks = {
        "positive_development_outcome_and_design_only_authorization": outcome_ok,
        "pinned_source_inventory_and_role_census": inventory_ok,
        "all_nine_untouched_models_and_action_alphabets_frozen": models_ok,
        "unchanged_V69_family_planner_and_materiality": family_ok,
        "model_level_stratified_noncompensatory_gates": gates_ok,
        "separate_Tier_boundaries_and_design_only_firewall": boundary_ok,
        "downstream_artifacts_absent": downstream_absent,
    }
    audit = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_confirmatory_design_and_authorize_resource_feasibility_only"
            if not errors
            else "reject_v70_confirmatory_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "confirmatory_source_metadata_records_read": 9,
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
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_design_lock",
        "source_positive_development_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_positive_development_outcome_lock_sha256": file_sha256(outcome_path),
        "source_feasibility_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_feasibility_lock_sha256": file_sha256(source_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_confirmatory_design_or_prior_artifacts": False,
            "audit_and_freeze_resource_feasibility": True,
            "construct_confirmatory_census": False,
            "write_and_audit_confirmatory_evaluator": False,
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
