#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox.json"
    parent_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-outcome-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v166.md"
    plan_path = PROJECT_ROOT / "docs/v168-fixed-ontology-reversible-sandbox-plan.md"
    protocol_path = PROJECT_ROOT / "python/v168_fixed_ontology_reversible_sandbox.py"
    tests_path = PROJECT_ROOT / "python/test_v168_fixed_ontology_reversible_sandbox.py"
    runner_path = PROJECT_ROOT / "python/run_v168_fixed_ontology_reversible_sandbox.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v168_fixed_ontology_reversible_sandbox_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v168_fixed_ontology_reversible_sandbox.py"
    audit_path = PROJECT_ROOT / "outputs/v168-fixed-ontology-reversible-sandbox/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-lock.json"
    output_root = PROJECT_ROOT / "outputs/v168-fixed-ontology-reversible-sandbox/census"
    outcome_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V168 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    ontology = config["fixedOntology"]
    protocol = config["transactionProtocol"]
    population = config["population"]
    gates = config["sandboxGates"]
    authority = config["authorityBoundary"]
    exposure = config["preLockExposure"]
    checks = {
        "corrected_V167_is_exact_and_authorizes_only_separate_sandbox_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["V167_scientific_planner_gates_passed"]
            and parent["authorization"]["preregister_fixed_ontology_reversible_sandbox"]
            and not parent["authorization"]["run_sandbox_without_separate_lock"]
            and not parent["authorization"]["run_local_or_API_model"]
            and not parent["authorization"]["register_provisional_primitive"]
        ),
        "fixed_trusted_ontology_is_small_typed_and_excludes_provisional_concepts": bool(
            ontology["entityType"] == "Device"
            and ontology["entityIds"] == ["D1", "D2", "D3"]
            and set(ontology["mutableFields"]) == {"mode", "quota", "owner_team"}
            and ontology["systemManagedFields"] == ["revision"]
            and not ontology["provisionalConceptsAllowed"]
        ),
        "reversible_transaction_protocol_is_complete": bool(
            protocol["proposalContainsExpectedRevisionForEveryAndOnlyTouchedEntity"]
            and protocol["duplicateEntityFieldOperationRejected"]
            and not protocol["previewMutatesState"]
            and protocol["previewBindsBaseStatePatchAndExpectedPostState"]
            and protocol["commitRequiresExactPreviewTokenAndUnchangedBaseState"]
            and protocol["commitIsAtomic"]
            and protocol["independentVerificationAfterCommit"]
            and protocol["verificationFailureTriggersAutomaticRollback"]
            and protocol["explicitRollbackSupported"]
            and protocol["provenanceLogHashChainedAndAppendOnly"]
        ),
        "balanced_development_scenarios_and_faults_are_frozen": bool(
            population["split"] == "development_only"
            and len(population["scenarios"]) == 11
            and population["recordsPerScenario"] == 12
            and population["requiredRecordCount"] == 132
            and population["projectAuthoredSynthetic"]
            and not population["evaluationPopulationCreated"]
            and config["faultModel"]["faultInjectionIsNotARealServiceOrExternalSideEffect"]
        ),
        "exact_safety_recovery_and_integrity_gates_are_noncompensatory": bool(
            gates["requiredExpectedDispositionAccuracy"] == 1.0
            and gates["requiredExactFinalTargetState"] == 1.0
            and gates["requiredRejectedStateImmutability"] == 1.0
            and gates["requiredPreviewNonmutation"] == 1.0
            and gates["requiredPreviewCommitParity"] == 1.0
            and gates["requiredAtomicMultiEntityCommit"] == 1.0
            and gates["requiredExplicitRollbackRecovery"] == 1.0
            and gates["requiredVerificationFailureRollbackRecovery"] == 1.0
            and gates["requiredInvariantPreservation"] == 1.0
            and gates["requiredZeroUnauthorizedCommitMutation"] == 1.0
            and gates["requiredFaultDetection"] == 1.0
            and gates["requiredProvenanceChainValidity"] == 1.0
        ),
        "authority_real_effect_and_model_boundaries_are_closed": bool(
            authority["stateStoreIsLocalInMemorySimulation"]
            and authority["onlyFixedTrustedOntologyMayBeUsed"]
            and authority["provisionalConceptsMayNotEnterSandbox"]
            and authority["learnedConfidenceCannotAuthorizeCommit"]
            and authority["independentVerifierCannotBeBypassed"]
            and not authority["realServiceOrToolTargetExists"]
            and authority["realExecutionCount"] == 0
            and exposure["implementationTestCensusRunCount"] == 1
            and exposure["formalCensusRunCount"] == 0
            and all(value == 0 for key, value in exposure.items() if key not in {"implementationTestCensusRunCount", "formalCensusRunCount"})
            and not config["decisionRule"]["passAuthorizesRealServiceOrExecution"]
            and not config["decisionRule"]["passAuthorizesProvisionalOntologyIntegration"]
            and not config["decisionRule"]["passAuthorizesModelOrEvaluationPopulation"]
        ),
        "required_locked_files_exist": all(path.is_file() for path in (
            config_path, parent_path, roadmap_path, plan_path, protocol_path, tests_path,
            runner_path, verifier_path, auditor_path,
        )),
        "formal_census_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "168-fixed-ontology-reversible-sandbox-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_simulated_sandbox_census" if passed else "reject_V168_design",
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V167r1_outcome": parent_path,
        "roadmap": roadmap_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "168-fixed-ontology-reversible-sandbox-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_ontology_invariants_population_faults_protocol_metrics_gates_or_decision": False,
            "run_formal_simulated_census_once": True,
            "create_or_open_evaluation_population": False,
            "load_or_run_local_or_API_model": False,
            "use_or_register_provisional_concept": False,
            "call_real_service_or_tool": False,
            "perform_external_side_effect_or_real_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
