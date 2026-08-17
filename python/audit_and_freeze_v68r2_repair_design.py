#!/usr/bin/env python3
"""Audit and freeze the V68r2 all-point-model-control repair design."""
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
    config_path = PROJECT_ROOT / "configs/v68r2-development-point-control-repair.json"
    plan_path = PROJECT_ROOT / "docs/v68r2-development-point-control-repair-plan.md"
    failed_path = PROJECT_ROOT / "configs/v68r1-development-failed-attempt-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v68r2-development-screening/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68r2-development-repair-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68r2 repair design already frozen")
    config = json.loads(config_path.read_text())
    failed = json.loads(failed_path.read_text())
    failed_payload = {
        key: value for key, value in failed.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    failed_ok = bool(
        payload_hash(failed_payload) == failed["lock_payload_sha256"]
        and failed["authorization"]["preregister_all_point_model_control_totalization"]
        and not failed["authorization"]["run_V68r2_before_new_locks"]
        and not failed["authorization"]["score_confirmatory_models"]
        and failed["record_results_persisted"] == 0
        and not failed["aggregate_result_persisted"]
        and failed["confirmatory_models_scored"] == 0
    )
    if not failed_ok:
        errors.append("V68r1 failed-attempt lock does not authorize design-only V68r2")

    unchanged_paths = {
        "design": PROJECT_ROOT / config["unchangedDevelopmentDesignLock"],
        "implementation": PROJECT_ROOT / config["unchangedImplementationLock"],
        "census": PROJECT_ROOT / config["unchangedCensusSeal"],
        "evaluator": PROJECT_ROOT / config["unchangedV68EvaluatorLock"],
    }
    unchanged = {name: json.loads(path.read_text()) for name, path in unchanged_paths.items()}
    unchanged_ok = bool(
        unchanged["census"]["record_count"] == 59
        and unchanged["census"]["selection_rejection_or_replacement_count"] == 0
        and unchanged["census"]["confirmatory_models_scored"] == 0
        and len(unchanged["design"]["config_payload"]["gates"]) == 19
        and unchanged["implementation"]["implementation_sha256"]
        == file_sha256(PROJECT_ROOT / unchanged["implementation"]["implementation"])
        and unchanged["evaluator"]["evaluator_sha256"]
        == file_sha256(PROJECT_ROOT / unchanged["evaluator"]["evaluator"])
    )
    if not unchanged_ok:
        errors.append("unchanged V68 design, implementation, census, evaluator, or gates drifted")

    repair = config["repair"]
    repair_ok = bool(
        repair["scope"]
        == ["MAP_point_model_control", "persistent_posterior_sampling_point_model_control"]
        and "exact_zero" in repair["offSupportTrigger"]
        and "first_action" in repair["offSupportRule"]
        and repair["MAPSelection"] == "unchanged_first_argmax_static_atom"
        and "17_point" in repair["posteriorSamplingSelection"]
        and not repair["fallbackUsesObservations"]
        and not repair["fallbackResamplesOrReselectsModel"]
        and not repair["fallbackUsesExactPosteriorForActionSelection"]
        and not repair["epsilonSmoothing"]
        and not repair["sourceArrayMutation"]
    )
    if not repair_ok:
        errors.append("V68r2 totalization rule is incomplete or expands beyond point controls")

    boundary_ok = bool(
        config["claimBoundary"]["developmentOnly"]
        and not config["claimBoundary"]["exactBayesAdaptivePlannerChanged"]
        and config["firewall"]["scoreConfirmatoryModels"] == "forbidden"
        and config["stageAuthorization"]["auditAndFreezeRepairDesign"]
        and not config["stageAuthorization"]["writeAndAuditRepairImplementation"]
        and not config["stageAuthorization"]["runRepairedDevelopmentScreen"]
    )
    if not boundary_ok:
        errors.append("V68r2 boundary or design-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/v68r2_point_model_controls.py",
            "python/evaluate_v68r2_development_screen.py",
            "configs/v68r2-development-implementation-lock.json",
            "configs/v68r2-development-evaluator-lock.json",
            "outputs/v68r2-development-screening/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V68r2 downstream artifacts exist before design lock")

    checks = {
        "failed_attempt_binding_and_design_only_authorization": failed_ok,
        "unchanged_design_implementation_census_evaluator_and_gates": unchanged_ok,
        "complete_all_point_model_control_totalization_rule": repair_ok,
        "repair_boundary_and_firewall": boundary_ok,
        "downstream_artifacts_absent": downstream_absent,
    }
    audit = {
        "schema_version": "68r2-development-screening",
        "experiment": "v68r2_repair_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_all_point_model_control_repair_and_authorize_implementation_only"
            if not errors
            else "reject_v68r2_repair_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "additional_development_records_evaluated": 0,
            "confirmatory_models_scored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "68r2-development-screening",
        "experiment": "v68r2_development_repair_design_lock",
        "failed_attempt_lock": str(failed_path.relative_to(PROJECT_ROOT)),
        "failed_attempt_lock_sha256": file_sha256(failed_path),
        "unchanged_artifacts": {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
            for name, path in unchanged_paths.items()
        },
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_prior_failures_or_unchanged_artifacts": False,
            "write_and_audit_all_point_control_repair": True,
            "write_and_audit_repaired_evaluator": False,
            "run_repaired_development_screen": False,
            "score_confirmatory_models": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
