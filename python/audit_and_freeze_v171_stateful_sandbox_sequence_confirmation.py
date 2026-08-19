#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v171_stateful_sandbox_sequence_confirmation import build_sequences, compose_config


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation.json"
    parent_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-outcome-lock.json"
    source_outcome_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-outcome-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v168.md"
    plan_path = PROJECT_ROOT / "docs/v171-stateful-sandbox-sequence-confirmation-plan.md"
    protocol_path = PROJECT_ROOT / "python/v171_stateful_sandbox_sequence_confirmation.py"
    tests_path = PROJECT_ROOT / "python/test_v171_stateful_sandbox_sequence_confirmation.py"
    runner_path = PROJECT_ROOT / "python/run_v171_stateful_sandbox_sequence_confirmation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v171_stateful_sandbox_sequence_confirmation_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v171_stateful_sandbox_sequence_confirmation.py"
    audit_path = PROJECT_ROOT / "outputs/v171-stateful-sandbox-sequence-confirmation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v171-stateful-sandbox-sequence-confirmation/census"
    outcome_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V171 is already preregistered, run, or frozen")

    design = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    source_outcome = json.loads(source_outcome_path.read_text())
    source_lock_path = PROJECT_ROOT / source_outcome["sandbox_lock"]
    source_lock = json.loads(source_lock_path.read_text())
    source_protocol_path = PROJECT_ROOT / source_lock["protocol"]
    config = compose_config(design, source_lock["config_payload"])
    fixtures = build_sequences(config)
    population = design["population"]
    recovery = design["recoveryPolicy"]
    gates = design["confirmationGates"]
    exposure = design["preLockExposure"]
    authority = design["authorityBoundary"]
    fixed = design["fixedSourceContract"]
    scenarios = population["scenarioFamilies"]
    variants = population["variantIndices"]

    checks = {
        "V170_is_strong_frozen_and_authorizes_stateful_confirmation": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_integrity_passed"]
            and parent["outcome"]["strong_confirmation"]
            and parent["authorization"]["advance_to_stateful_sandbox_confirmation"]
            and not parent["authorization"]["design_cross_track_integration_now"]
        ),
        "V168_positive_source_contract_and_protocol_are_exact": bool(
            valid_lock(source_outcome)
            and source_outcome["outcome"]["passed"]
            and source_outcome["outcome"]["scientific_sandbox_gates_passed"]
            and valid_lock(source_lock)
            and file_sha256(source_lock_path) == source_outcome["sandbox_lock_sha256"]
            and file_sha256(source_protocol_path) == source_lock["protocol_sha256"]
        ),
        "V168_ontology_and_transaction_semantics_are_reused_without_source_edit": bool(
            fixed["reuseV168FixedOntologyExactly"]
            and fixed["reuseV168InvariantsExactly"]
            and fixed["reuseV168ProposalValidationExactly"]
            and fixed["reuseV168AtomicPatchAndRevisionSemanticsExactly"]
            and fixed["reuseV168PreviewBindingAndVerificationSemanticsExactly"]
            and not fixed["modifyV168SourceOrLockedArtifacts"]
            and set(fixed["addedHarnessScopeOnly"]) == {
                "durable transaction lifecycle status",
                "write-ahead recovery journal",
                "simulated restart and crash recovery",
                "idempotent replay and repeated rollback guards",
                "provenance verification before recovery or continuation",
            }
        ),
        "population_is_complete_bounded_and_outcome_independent": bool(
            population["split"] == "fresh_procedural_confirmation"
            and len(scenarios) == 11
            and len(set(scenarios)) == 11
            and variants == list(range(200, 212))
            and len(fixtures) == 132
            and len({row["sequence_id"] for row in fixtures}) == 132
            and {row["scenario"] for row in fixtures} == set(scenarios)
            and not population["selectionOrExclusionAfterOutcomeInspectionAllowed"]
            and population["projectAuthoredSynthetic"]
            and not population["humanAuthored"]
            and not population["modelGenerated"]
        ),
        "recovery_policy_is_fail_closed_and_requires_continuation": bool(
            recovery["unverifiedPreparedOrAppliedTransaction"].startswith("restore complete before-state")
            and recovery["verifiedTransactionWhoseStateMatchesExpectedPostState"].startswith("finalize retained")
            and recovery["provenanceFailure"].startswith("fail closed")
            and recovery["terminalTransactionReplay"].startswith("reject idempotently")
            and recovery["repeatedRollback"].startswith("return already_rolled_back")
            and recovery["postRecoveryContinuationRequired"]
        ),
        "all_safety_recovery_and_integrity_gates_are_noncompensatory": bool(
            all(
                value == 1.0
                for key, value in gates.items()
                if key.startswith("required") and key not in {
                    "requiredSequenceCount", "requiredScenarioCount", "requiredSequencesPerScenario"
                }
            )
            and gates["requiredSequenceCount"] == 132
            and gates["requiredScenarioCount"] == 11
            and gates["requiredSequencesPerScenario"] == 12
        ),
        "prelock_exposure_and_authority_boundaries_hold": bool(
            exposure["implementationUnitSequenceCount"] == 3
            and exposure["formalSequenceRunCount"] == 0
            and exposure["aggregateFormalMetricInspectionCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key not in {
                    "implementationUnitSequenceCount",
                    "formalSequenceRunCount",
                    "aggregateFormalMetricInspectionCount",
                }
            )
            and authority["stateAndDurabilityAreLocalInMemorySimulation"]
            and authority["onlyFixedTrustedV168OntologyMayBeUsed"]
            and authority["provisionalConceptsMayNotEnterSandbox"]
            and authority["learnedConfidenceCannotAuthorizeCommit"]
            and authority["provenanceVerifierCannotBeBypassed"]
            and not authority["realServiceOrToolTargetExists"]
            and authority["realExecutionCount"] == 0
            and not design["decisionRule"]["passAuthorizesImmediateIntegrationRun"]
            and not design["decisionRule"]["passAuthorizesRealServiceOrExecution"]
            and not design["decisionRule"]["passAuthorizesProvisionalOntologyIntegration"]
            and not design["decisionRule"]["passAuthorizesModelOrLanguagePopulation"]
        ),
        "required_locked_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                parent_path,
                source_outcome_path,
                source_lock_path,
                source_protocol_path,
                roadmap_path,
                plan_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
        "formal_census_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "171-stateful-sandbox-sequence-confirmation-design-audit",
        "experiment": design["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_stateful_sequence_census" if passed else "reject_V171_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "population_identity_sha256": payload_hash(fixtures),
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V170_outcome": parent_path,
        "source_V168_outcome": source_outcome_path,
        "source_V168_lock": source_lock_path,
        "source_V168_protocol": source_protocol_path,
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
        "schema_version": "171-stateful-sandbox-sequence-confirmation-lock",
        "experiment": design["experiment"],
        "config_payload": design,
        "V168_config_payload": source_lock["config_payload"],
        "composed_config_payload": config,
        "population_identity_sha256": payload_hash(fixtures),
        "authorization": {
            "modify_V168_source_contract_population_recovery_metrics_gates_or_decision": False,
            "run_formal_stateful_sequence_census_once": True,
            "select_exclude_or_tune_after_outcome_inspection": False,
            "design_or_run_cross_track_integration_now": False,
            "run_model_use_provisional_concept_call_real_service_or_execute": False,
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
