#!/usr/bin/env python3
"""Audit and freeze the V68r1 off-support totalization repair design."""
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
    config_path = PROJECT_ROOT / "configs/v68r1-development-off-support-repair.json"
    plan_path = PROJECT_ROOT / "docs/v68r1-development-off-support-repair-plan.md"
    audit_path = PROJECT_ROOT / "outputs/v68r1-development-screening/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68r1-development-repair-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68r1 repair design already frozen")
    config = json.loads(config_path.read_text())
    failed_path = PROJECT_ROOT / config["sourceFailedAttemptLock"]
    failed = json.loads(failed_path.read_text())
    payload = {key: value for key, value in failed.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    failed_ok = bool(
        payload_hash(payload) == failed["lock_payload_sha256"]
        and failed["authorization"]["preregister_off_support_totalization_repair"]
        and not failed["authorization"]["run_repaired_development_screen_before_new_locks"]
        and not failed["authorization"]["score_confirmatory_models"]
        and failed["record_results_persisted"] == 0
        and not failed["aggregate_result_persisted"]
        and failed["confirmatory_models_scored"] == 0
    )
    if not failed_ok:
        errors.append("V68 failed-attempt lock or repair-only authorization failed")

    unchanged_paths = {
        "design": PROJECT_ROOT / config["unchangedDevelopmentDesignLock"],
        "implementation": PROJECT_ROOT / config["unchangedImplementationLock"],
        "census": PROJECT_ROOT / config["unchangedCensusSeal"],
    }
    unchanged = {name: json.loads(path.read_text()) for name, path in unchanged_paths.items()}
    unchanged_ok = bool(
        unchanged["census"]["record_count"] == 59
        and unchanged["census"]["selection_rejection_or_replacement_count"] == 0
        and unchanged["census"]["confirmatory_models_scored"] == 0
        and unchanged["implementation"]["implementation_sha256"]
        == file_sha256(PROJECT_ROOT / unchanged["implementation"]["implementation"])
        and len(unchanged["design"]["config_payload"]["gates"]) == 19
    )
    if not unchanged_ok:
        errors.append("unchanged V68 design, implementation, census, or gates do not bind")

    repair = config["repair"]
    repair_ok = bool(
        repair["scope"] == "persistent_posterior_sampling_control_only"
        and "17_point" in repair["sampleSelection"]
        and "exact_zero" in repair["offSupportTrigger"]
        and "first_action" in repair["offSupportRule"]
        and not repair["fallbackUsesObservations"]
        and not repair["fallbackResamplesModel"]
        and not repair["fallbackUsesExactPosteriorForActionSelection"]
        and not repair["epsilonSmoothing"]
        and not repair["sourceArrayMutation"]
        and len(repair["diagnostics"]) == 3
    )
    if not repair_ok:
        errors.append("V68r1 totalization rule is incomplete or expands beyond the failed control")

    boundary = config["claimBoundary"]
    stage = config["stageAuthorization"]
    boundary_ok = bool(
        boundary["developmentOnly"]
        and boundary["posteriorSamplingControlTotalization"]
        and not boundary["exactBayesAdaptivePlannerChanged"]
        and not boundary["MAPControlChanged"]
        and not boundary["sealedCensusChanged"]
        and not any(
            boundary[key]
            for key in ("confirmatoryReplication", "SMC2", "humanData", "modelAccess", "adapterTraining")
        )
        and set(config["firewall"].values()) == {"forbidden"}
        and stage["auditAndFreezeRepairDesign"]
        and not any(value for key, value in stage.items() if key != "auditAndFreezeRepairDesign")
    )
    if not boundary_ok:
        errors.append("V68r1 repair boundary, firewall, or design-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/v68r1_posterior_sampling.py",
            "python/evaluate_v68r1_development_screen.py",
            "configs/v68r1-development-implementation-lock.json",
            "configs/v68r1-development-evaluator-lock.json",
            "outputs/v68r1-development-screening/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V68r1 implementation or evaluation exists before repair design lock")

    checks = {
        "failed_attempt_binding_and_repair_only_authorization": failed_ok,
        "unchanged_design_implementation_census_and_gates": unchanged_ok,
        "complete_posterior_sampling_only_totalization_rule": repair_ok,
        "repair_boundary_firewall_and_design_only_authorization": boundary_ok,
        "repair_implementation_and_evaluation_absent": downstream_absent,
    }
    audit = {
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_repair_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_off_support_repair_and_authorize_implementation_only"
            if not errors
            else "reject_v68r1_repair_design"
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
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_development_repair_design_lock",
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
            "modify_failed_V68_or_unchanged_artifacts": False,
            "write_and_audit_off_support_repair": True,
            "write_and_audit_repaired_evaluator": False,
            "run_repaired_development_screen": False,
            "score_confirmatory_models": False,
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
