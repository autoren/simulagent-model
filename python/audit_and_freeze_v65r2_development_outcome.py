#!/usr/bin/env python3
"""Freeze V65r2 as rejected after a pre-evaluation development access violation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v65r2-design-lock.json"
    violation_path = (
        PROJECT_ROOT
        / "outputs/v65r2-extinct-identity-repair/development-access-violation.json"
    )
    audit_path = PROJECT_ROOT / "outputs/v65r2-extinct-identity-repair/development-outcome-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r2-development-outcome-lock.json"
    if output_path.exists():
        raise RuntimeError("V65r2 development outcome already frozen")
    design = json.loads(design_path.read_text())
    violation = json.loads(violation_path.read_text())
    errors: list[str] = []

    incident_sources_bound = all(
        file_sha256(PROJECT_ROOT / relative) == digest
        for relative, digest in violation["source_sha256_at_incident"].items()
    )
    violation_integrity = bool(
        violation["design_lock_sha256"] == file_sha256(design_path)
        and violation["stage"] == "implementation_development_before_implementation_lock"
        and violation["access"]["sealed_public_records_with_candidate_EIG_scored"] == 1
        and violation["access"]["candidate_EIG_actions_scored_on_sealed_subset"] == 4
        and not violation["access"]["candidate_EIG_values_printed_or_persisted"]
        and violation["access"]["V65r2_evaluation_attempts"] == 0
        and not violation["V65r2_evaluation_authorized"]
        and not violation["passed"]
        and violation["decision"]
        == "reject_v65r2_before_implementation_lock_or_evaluation"
        and incident_sources_bound
    )
    if not violation_integrity:
        errors.append("V65r2 development access violation is incomplete or unbound")

    firewall_value = design["repair_payload"]["firewall"][
        "scoreCandidateEIGDuringDesignOrImplementationAudit"
    ]
    firewall_violation_confirmed = bool(
        firewall_value == "forbidden_on_the_sealed_subset"
        and violation["incident"]["record_id"]
        == design["repair_payload"]["mandatoryImplementationFixtures"][
            "sealedOneIdentityZeroSupport"
        ]
        and violation["incident"]["action"].endswith("called_score_all_actions")
    )
    if not firewall_violation_confirmed:
        errors.append("frozen V65r2 firewall does not match the recorded incident")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v65r2-implementation-lock.json",
            "configs/v65r2-evaluation-implementation-lock.json",
            "configs/v65r2-outcome-lock.json",
            "outputs/v65r2-extinct-identity-repair/evaluation",
        )
    )
    no_external_access = bool(
        violation["access"]["truth_fields_accessed"] == 0
        and violation["access"]["human_records"] == 0
        and violation["access"]["model_forward_passes"] == 0
        and violation["access"]["adapter_training_runs"] == 0
        and violation["access"]["V65r1_evaluation_reruns"] == 0
    )
    if not downstream_absent or not no_external_access:
        errors.append("V65r2 progressed downstream or crossed an additional access boundary")

    checks = {
        "incident_sources_and_access_bound": violation_integrity,
        "frozen_firewall_violation_confirmed": firewall_violation_confirmed,
        "implementation_lock_evaluator_and_evaluation_absent": downstream_absent,
        "no_truth_human_model_training_or_V65r1_rerun_access": no_external_access,
    }
    audit = {
        "schema_version": "65r2",
        "experiment": "v65r2_development_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_rejected_v65r2_and_authorize_v65r3_preregistration_only"
            if not errors and all(checks.values())
            else "reject_v65r2_development_outcome_record"
        ),
        "errors": errors,
        "checks": checks,
        "incident": violation["incident"],
        "access": violation["access"],
        "boundary": {
            "algorithmic_repair_accuracy_evaluated": False,
            "implementation_lock_written": False,
            "V65r2_evaluation_attempted": False,
            "Bayes_adaptive_reward_planning_authorized": False,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "65r2",
        "experiment": "v65r2_rejected_development_outcome_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "development_access_violation": str(violation_path.relative_to(PROJECT_ROOT)),
        "development_access_violation_sha256": file_sha256(violation_path),
        "development_outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "development_outcome_audit_sha256": file_sha256(audit_path),
        "outcome_auditor": "python/audit_and_freeze_v65r2_development_outcome.py",
        "outcome_auditor_sha256": file_sha256(Path(__file__).resolve()),
        "decision": "reject_v65r2_before_implementation_lock_or_evaluation",
        "authorization": {
            "modify_or_continue_v65r2": False,
            "preregister_v65r3_same_algorithmic_repair_with_synthetic_only_EIG_implementation_tests": True,
            "run_v65r3_before_preregistration_and_locks": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "decision": lock["decision"],
                "V65r2_evaluation_attempts": violation["access"][
                    "V65r2_evaluation_attempts"
                ],
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
