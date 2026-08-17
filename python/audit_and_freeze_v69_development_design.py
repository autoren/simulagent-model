#!/usr/bin/env python3
"""Audit and freeze the V69 dominant-remapping development design."""
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
    config_path = PROJECT_ROOT / "configs/v69-development-screening.json"
    plan_path = PROJECT_ROOT / "docs/v69-development-screening-plan.md"
    outcome_path = PROJECT_ROOT / "configs/v68r2-development-outcome-lock.json"
    source_path = PROJECT_ROOT / "configs/v68-source-feasibility-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v69-development-screening/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v69-development-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V69 development design already frozen")
    config = json.loads(config_path.read_text())
    outcome = json.loads(outcome_path.read_text())
    outcome_payload = {
        key: value for key, value in outcome.items() if key != "lock_payload_sha256"
    }
    source = json.loads(source_path.read_text())
    errors: list[str] = []

    outcome_ok = bool(
        payload_hash(outcome_payload) == outcome["lock_payload_sha256"]
        and not outcome["outcome"]["passed"]
        and outcome["outcome"]["confirmatory_models_scored"] == 0
        and outcome["authorization"]["preregister_materially_revised_development_family"]
        and not outcome["authorization"]["run_revised_family_before_new_locks"]
        and not outcome["authorization"]["score_confirmatory_models_under_unchanged_family"]
    )
    if not outcome_ok:
        errors.append("V68r2 outcome does not authorize a revised design only")

    source_ok = bool(
        source["prospective_role_census"]["confirmatoryNovelAndOutcomeUntouched"] == 6
        and source["prospective_role_census"][
            "confirmatoryStructurallyRelatedButOutcomeUntouched"
        ]
        == 3
        and not source["authorization"][
            "run_confirmatory_policy_EIG_regret_or_SMC2_outcomes"
        ]
        and file_sha256(PROJECT_ROOT / source["parser"]) == source["parser_sha256"]
    )
    if not source_ok:
        errors.append("source feasibility lock or holdout firewall drifted")

    old_config = json.loads(
        (PROJECT_ROOT / "configs/v68-development-design-lock.json").read_text()
    )["config_payload"]
    family = config["unknownDynamicsFamily"]
    revision_ok = bool(
        family["transitionDefinition"]
        == "T_identity_theta(command)=theta*T_source(cycle_remapped_command_for_identity)+(1-theta)*T_source(command)"
        and family["thetaSupport"] == [0.6, 0.95]
        and family["identityPrior"] == [0.5, 0.5]
        and len(family["identityNames"]) == 2
        and config["developmentModels"]
        == [
            {"file": row["file"], "canonicalActionCycle": row["canonicalActionCycle"]}
            for row in old_config["developmentModels"]
        ]
        and config["gates"] == old_config["gates"]
        and config["normalization"]["returnScale"]
        == old_config["normalization"]["returnScale"]
        and config["normalization"]["normalizedRegret"]
        == old_config["normalization"]["normalizedRegret"]
    )
    if not revision_ok:
        errors.append("V69 is not the single frozen dominant-remapping revision")

    protocol_ok = bool(
        config["publicPrefixCensus"]["depths"] == [0, 1]
        and config["publicPrefixCensus"]["retainEveryReachableActionObservationHistory"]
        and not config["publicPrefixCensus"]["selectionOrRejection"]
        and config["exactPlanning"]["horizonActions"] == 3
        and config["exactPlanning"]["primaryQuadratureNodes"] == 65
        and config["exactPlanning"]["convergenceQuadratureNodes"] == 129
        and config["firewall"]["scoreConfirmatoryModels"] == "forbidden"
        and config["stageAuthorization"]["auditAndFreezeDevelopmentDesign"]
        and not config["stageAuthorization"]["writeAndAuditExactInfrastructure"]
        and not config["stageAuthorization"]["runDevelopmentScreen"]
    )
    if not protocol_ok:
        errors.append("V69 census, exact-planning, firewall, or design-only rule failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/v69_dominant_remapping.py",
            "configs/v69-development-implementation-lock.json",
            "configs/v69-development-census-seal.json",
            "python/evaluate_v69_development_screen.py",
            "configs/v69-development-evaluator-lock.json",
            "outputs/v69-development-screening/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V69 implementation, census, or evaluation exists before design lock")

    checks = {
        "negative_outcome_and_revision_only_authorization": outcome_ok,
        "source_feasibility_and_holdout_firewall": source_ok,
        "single_material_dominant_remapping_revision_with_unchanged_gates": revision_ok,
        "complete_census_exact_planning_and_design_only_protocol": protocol_ok,
        "downstream_artifacts_absent": downstream_absent,
    }
    audit = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_dominant_remapping_design_and_authorize_implementation_only"
            if not errors
            else "reject_v69_development_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "development_records_evaluated": 0,
            "confirmatory_models_scored": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_design_lock",
        "source_negative_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_negative_outcome_lock_sha256": file_sha256(outcome_path),
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
            "modify_design_or_prior_artifacts": False,
            "write_and_audit_exact_infrastructure": True,
            "construct_development_census": False,
            "write_and_audit_development_evaluator": False,
            "run_development_screen": False,
            "score_confirmatory_models": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
