#!/usr/bin/env python3
"""Audit and freeze the V67 independent bounded-verification design."""
from __future__ import annotations

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
    config_path = PROJECT_ROOT / "configs/v67-independent-bounded-policy-verification.json"
    plan_path = PROJECT_ROOT / "docs/v67-independent-bounded-policy-verification-plan.md"
    audit_path = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v67-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V67 design already frozen")
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceV66OutcomeLock"]
    source = json.loads(source_path.read_text())
    errors: list[str] = []

    payload = {key: value for key, value in source.items() if key != "lock_payload_sha256"}
    source_ok = bool(
        payload_hash(payload) == source["lock_payload_sha256"]
        and source["decision"] == "authorize_independent_bounded_policy_verification_only"
        and source["authorization"]["preregister_independent_bounded_policy_verification"]
        and not source["authorization"]["modify_or_rerun_v66"]
        and file_sha256(PROJECT_ROOT / source["record_cells"])
        == source["record_cells_sha256"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
    )
    if not source_ok:
        errors.append("V66 outcome binding or independent-verification authorization failed")

    policies = config["sourcePolicies"]
    census_ok = bool(
        policies["records"] == 48
        and policies["policyKinds"] == ["exact_policy", "pooled_SMC2_policy"]
        and policies["policiesPerKind"] == 48
        and policies["totalPolicies"] == 96
        and policies["selection"].startswith("all_frozen_V66")
        and not policies["sourceMutation"]
        and not policies["sourceEvaluationRerun"]
        and "one_record_was_inspected" in policies["sourceInspectionBeforeDesignLock"]
    )
    if not census_ok:
        errors.append("V67 must exhaustively bind and disclose all 96 source policies")

    independent = config["independentExecutor"]
    semantics = config["verifiedSemantics"]
    independence_ok = bool(
        semantics["horizonActions"] == 3
        and semantics["discount"] == 0.95
        and "257_node" in semantics["environment"]
        and "fixed" in semantics["staticModelPersistence"]
        and "does_not_call_v62" in independent["sourceParser"]
        and "does_not_import_or_call_v64" in independent["policyExecution"]
        and independent["storedApproximateBranchProbabilities"].startswith("ignored")
        and len(config["reachableChecks"]) == 10
    )
    if not independence_ok:
        errors.append("V67 independent execution semantics are incomplete")

    storm = config["probabilisticVerification"]
    storm_ok = bool(
        storm["version"] == "1.13.0"
        and storm["externalProcessRequired"]
        and storm["properties"] == {
            "termination": 'P=? [F "done"]',
            "return": 'R=? [F "done"]',
        }
        and "conditioned" in storm["transitionReward"]
    )
    if not storm_ok:
        errors.append("V67 external Storm semantics are incomplete")

    implementation = config["implementationAudit"]
    bundle = config["verificationBundle"]
    audit_ok = bool(
        not implementation["sealedPoliciesExecutedBeforeImplementationAndEvaluatorLocks"]
        and len(implementation["analyticFixtures"]) == 6
        and len(implementation["mutants"]) == 14
        and implementation["minimumAnalyticFixturePassRate"] == 1.0
        and implementation["minimumMutantKillRate"] == 1.0
        and bundle["expectedModels"] == 96
        and len(bundle["requiredFilesPerPolicy"]) == 5
        and bundle["constructionBeforeImplementationLock"] == "forbidden"
        and bundle["postSealMutation"] == "forbidden"
    )
    if not audit_ok:
        errors.append("V67 implementation audit or bundle protocol is incomplete")

    gates = config["gates"]
    gates_ok = bool(
        len(gates) == 25
        and gates["minimumCompletedPolicyFraction"] == 1.0
        and gates["minimumPolicyCount"] == 96
        and gates["minimumPolicyCountPerKind"] == 48
        and gates["minimumSourcePolicyHashMatchRate"] == 1.0
        and gates["minimumPositiveObservationBranchTotalityRate"] == 1.0
        and gates["maximumNonterminalDeadlockCount"] == 0
        and gates["maximumIndependentExecutorErrorAgainstFrozenV66Value"] <= 1e-10
        and gates["maximumStormReturnErrorAgainstIndependentExecutor"] <= 1e-9
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and gates["minimumAnalyticFixturePassRate"] == 1.0
        and all(gates[key] == 0 for key in (
            "maximumVerificationBundleHashMismatchCount",
            "maximumSourceResultMutationCount",
            "maximumToolVersionMismatchCount",
            "maximumUnexpectedVerificationAttemptCount",
            "maximumTruthFieldAccessCount",
            "maximumHumanRecordAccessCount",
            "maximumModelForwardPassCount",
            "maximumAdapterTrainingRunCount",
        ))
    )
    if not gates_ok:
        errors.append("V67 noncompensatory gates are incomplete")

    boundary = config["claimBoundary"]
    stage = config["stageAuthorization"]
    boundary_ok = bool(
        boundary["allFrozenV66ExactAndPooledPolicies"]
        and boundary["boundedHorizon"] == 3
        and boundary["exactPosteriorExecution"]
        and boundary["policyExecutionNotPlannerAlgorithmVerification"]
        and not any(boundary[key] for key in (
            "plannerOptimality", "infiniteHorizon", "formalSafetyProperty",
            "parameterUniformGuarantee", "independentBenchmarkReplication",
            "humanData", "modelAccess", "adapterTraining",
        ))
        and set(config["firewall"].values()) == {"forbidden"}
        and stage["auditAndFreezeDesign"]
        and not any(value for key, value in stage.items() if key != "auditAndFreezeDesign")
    )
    if not boundary_ok:
        errors.append("V67 claim boundary, firewall, or design-only authorization is invalid")

    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in (
        "configs/v67-implementation-lock.json",
        "configs/v67-verification-bundle-seal.json",
        "configs/v67-evaluation-implementation-lock.json",
        "configs/v67-outcome-lock.json",
        "python/v67_verification.py",
        "python/evaluate_v67_verification.py",
        "outputs/v67-independent-bounded-policy-verification/bundle",
        "outputs/v67-independent-bounded-policy-verification/verification",
    ))
    if not downstream_absent:
        errors.append("V67 downstream artifacts exist before design lock")

    checks = {
        "V66_success_and_verification_authorization": source_ok,
        "exhaustive_disclosed_96_policy_census": census_ok,
        "independent_exact_execution_semantics": independence_ok,
        "external_Storm_semantics": storm_ok,
        "implementation_audit_and_bundle_protocol": audit_ok,
        "twenty_five_noncompensatory_gates": gates_ok,
        "claim_boundary_firewall_and_design_only_authorization": boundary_ok,
        "V67_downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": "67",
        "experiment": "v67_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": "freeze_v67_design_and_authorize_independent_implementation_only" if not errors else "reject_v67_design",
        "errors": errors,
        "checks": checks,
        "access": {
            "source_policy_rows_inspected_for_interface": 1,
            "source_policies_selected_rejected_changed_or_executed": 0,
            "truth_fields_accessed": 0,
            "V66_evaluation_reruns": 0,
            "V67_verification_attempts": 0,
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
        "schema_version": "67",
        "experiment": "v67_design_lock",
        "source_v66_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v66_outcome_lock_sha256": file_sha256(source_path),
        "source_v66_record_cells": source["record_cells"],
        "source_v66_record_cells_sha256": source["record_cells_sha256"],
        "source_v66_result": source["result"],
        "source_v66_result_sha256": source["result_sha256"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_or_rerun_v66": False,
            "modify_v67_design": False,
            "write_and_audit_independent_implementation": True,
            "load_and_execute_all_source_policies": False,
            "build_verification_bundle": False,
            "write_and_audit_durable_evaluator": False,
            "run_verification": False,
            "truth_field_access": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"design_lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
