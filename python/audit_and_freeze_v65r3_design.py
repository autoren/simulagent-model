#!/usr/bin/env python3
"""Audit and freeze the V65r3 synthetic-only implementation access repair."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    repair_path = PROJECT_ROOT / "configs/v65r3-synthetic-only-implementation-repair.json"
    plan_path = PROJECT_ROOT / "docs/v65r3-synthetic-only-implementation-plan.md"
    audit_path = PROJECT_ROOT / "outputs/v65r3-synthetic-only-implementation/design-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r3-design-lock.json"
    if output_path.exists():
        raise RuntimeError("V65r3 design already frozen")
    repair = json.loads(repair_path.read_text())
    outcome_path = PROJECT_ROOT / repair["sourceDevelopmentOutcomeLock"]
    outcome = json.loads(outcome_path.read_text())
    v65r2_design_path = PROJECT_ROOT / outcome["design_lock"]
    v65r2_design = json.loads(v65r2_design_path.read_text())
    base = v65r2_design["config_payload"]
    errors: list[str] = []

    outcome_payload = {
        key: value for key, value in outcome.items() if key != "lock_payload_sha256"
    }
    outcome_ok = bool(
        payload_hash(outcome_payload) == outcome["lock_payload_sha256"]
        and outcome["authorization"][
            "preregister_v65r3_same_algorithmic_repair_with_synthetic_only_EIG_implementation_tests"
        ]
        and not outcome["authorization"]["modify_or_continue_v65r2"]
        and not outcome["authorization"]["reward_planning"]
        and file_sha256(PROJECT_ROOT / outcome["development_access_violation"])
        == outcome["development_access_violation_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["development_outcome_audit"])
        == outcome["development_outcome_audit_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["outcome_auditor"])
        == outcome["outcome_auditor_sha256"]
        and file_sha256(v65r2_design_path) == outcome["design_lock_sha256"]
    )
    if not outcome_ok:
        errors.append("V65r2 rejected outcome or V65r3 preregistration authorization is invalid")

    v65r2_repair = v65r2_design["repair_payload"]
    algorithm_unchanged = bool(
        repair["algorithmicRepair"].startswith("identical_to_the_frozen_v65r2")
        and base["extinctIdentityRepair"] == v65r2_repair["repair"]
        and base["durableAttemptProtocol"] == v65r2_repair["durableAttemptProtocol"]
        and repair["durableAttemptProtocol"] == "unchanged_from_v65r2_preregistration"
    )
    if not algorithm_unchanged:
        errors.append("V65r3 changes the V65r2 algorithmic or durable-attempt repair")

    access = repair["accessRepair"]
    access_ok = bool(
        set(access["sealedFatalRecordAllowedOperationsBeforeEvaluatorLock"])
        == {
            "boolean_identity_support",
            "SMC2_identity_evidence_and_posterior_inference",
            "posterior_normalization",
            "identity_mass_and_atom_exclusion_checks",
            "work_and_random_stream_diagnostics",
        }
        and set(access["sealedFatalRecordForbiddenOperationsBeforeEvaluatorLock"])
        == {
            "Rao_Blackwellized_candidate_predictive_scoring",
            "candidate_EIG_scoring",
            "candidate_action_selection",
            "exact_candidate_EIG_reference",
            "selection_regret_or_gate_computation",
        }
        and access["implementationEIGFixtures"].startswith("synthetic_public_histories_only")
        and access["sealedRecordFirewall"].startswith("assert_record_id_and_full_public_history")
        and repair["mandatoryImplementationFixtures"]["sealedSupportOnly"]
        == v65r2_repair["mandatoryImplementationFixtures"]["sealedOneIdentityZeroSupport"]
    )
    synthetic = repair["mandatoryImplementationFixtures"]["syntheticEIG"]
    synthetic_ok = bool(
        synthetic
        == {
            "record_id": "v65r3-synthetic-eig-fixture",
            "prefix_length": 2,
            "initial_observation": "left",
            "actions": ["n", "e"],
            "observations": ["left", "neither"],
        }
    )
    if not access_ok or not synthetic_ok:
        errors.append("V65r3 synthetic-only candidate-scoring firewall is incomplete")

    invariant_keys = (
        "subset",
        "smcSquared",
        "gates",
        "seeds",
        "controls",
        "exactReference",
        "comparisonTargets",
    )
    invariant_hashes = {key: payload_hash(base[key]) for key in invariant_keys}
    invariants_ok = bool(
        invariant_hashes == v65r2_design["unchanged_payload_sha256"]
        and base["subset"]["records"] == 48
        and base["smcSquared"]["outerThetaParticleBudgets"] == [31, 127, 509]
        and base["smcSquared"]["innerStateParticleBudget"] == 127
        and base["smcSquared"]["independentRepeatsPerBudget"] == 3
    )
    if not invariants_ok:
        errors.append("V65r3 does not preserve all original evaluation invariants")

    boundary = repair["claimBoundary"]
    stage = repair["stageAuthorization"]
    boundary_ok = bool(
        boundary["targetedRepairOfFailedV65r1AndRejectedV65r2"]
        and not any(
            boundary[key]
            for key in (
                "independentExactBenchmarkReplication",
                "sequentialApproximateAdaptiveRollout",
                "rewardPlanning",
                "formalVerification",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
        and set(repair["firewall"].values()) == {"forbidden"}
        and stage["writeAndAuditRepairImplementation"]
        and not any(
            value
            for key, value in stage.items()
            if key != "writeAndAuditRepairImplementation"
        )
    )
    if not boundary_ok:
        errors.append("V65r3 boundary, firewall, or implementation-only authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v65r3-implementation-lock.json",
            "configs/v65r3-evaluation-implementation-lock.json",
            "configs/v65r3-outcome-lock.json",
            "python/v65r3_smc2_eig.py",
            "python/evaluate_v65r3_eig.py",
            "outputs/v65r3-synthetic-only-implementation/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V65r3 implementation or evaluation exists before design lock")

    checks = {
        "V65r2_rejected_outcome_and_V65r3_authorization": outcome_ok,
        "algorithmic_and_durable_attempt_repair_unchanged": algorithm_unchanged,
        "synthetic_only_implementation_EIG_firewall": access_ok and synthetic_ok,
        "all_original_evaluation_invariants_bound": invariants_ok,
        "claim_boundary_firewall_and_implementation_only_authorization": boundary_ok,
        "V65r3_downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": "65r3",
        "experiment": "v65r3_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v65r3_design_and_authorize_repair_implementation_only"
            if not errors and all(checks.values())
            else "reject_v65r3_design"
        ),
        "errors": errors,
        "checks": checks,
        "unchanged_payload_sha256": invariant_hashes,
        "access": {
            "sealed_public_records_loaded": 0,
            "sealed_candidate_EIG_scores": 0,
            "truth_fields_accessed": 0,
            "V65r1_evaluation_reruns": 0,
            "V65r2_evaluation_attempts": 0,
            "V65r3_evaluation_attempts": 0,
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

    effective = copy.deepcopy(base)
    effective["schemaVersion"] = "65r3"
    effective["implementationAccessRepair"] = copy.deepcopy(access)
    effective["claimBoundary"].update(boundary)
    lock = {
        "schema_version": "65r3",
        "experiment": "v65r3_design_lock",
        "source_v65r2_development_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_v65r2_development_outcome_lock_sha256": file_sha256(outcome_path),
        "source_v65r2_design_lock": str(v65r2_design_path.relative_to(PROJECT_ROOT)),
        "source_v65r2_design_lock_sha256": file_sha256(v65r2_design_path),
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "repair_payload": repair,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "config_payload": effective,
        "unchanged_payload_sha256": invariant_hashes,
        "authorization": {
            "modify_or_rerun_v65r1": False,
            "modify_or_continue_v65r2": False,
            "modify_v65r3_design": False,
            "write_and_audit_repair_implementation": True,
            "write_and_audit_durable_evaluator": False,
            "run_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "checks": checks,
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
