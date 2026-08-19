#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v208-external-behavioral-abstention-source-census.json"
    parent_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v208-external-behavioral-abstention-source-census-plan.md"
    protocol_path = PROJECT_ROOT / "python/v208_external_behavioral_abstention_source_census.py"
    tests_path = PROJECT_ROOT / "python/test_v208_external_behavioral_abstention_source_census.py"
    runner_path = PROJECT_ROOT / "python/run_v208_external_behavioral_abstention_source_census.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v208_external_behavioral_abstention_source_census_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v208_external_behavioral_abstention_source_census.py"
    audit_path = PROJECT_ROOT / "outputs/v208-external-behavioral-abstention-source-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v208-external-behavioral-abstention-source-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v208-external-behavioral-abstention-source-census/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v208-external-behavioral-abstention-source-census-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V208 already started")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    candidates = config["candidates"]
    repositories = [candidate["codeRepository"] for candidate in candidates]
    allowed = config["allowedEvaluationReads"]
    prelock = config["preLockExposure"]
    gates = config["qualificationGates"]
    landing_keys = {
        "pairedOrMatchedControlsClaimed",
        "explicitMachineReadablePairIdentityFieldClaimed",
        "explicitPreExecutionPhaseIdentityClaimed",
        "deterministicActAbstainGoldFieldClaimed",
        "promptLabelSeparationClaimed",
        "balancedActAbstainControlsClaimed",
        "textOnlyShadowClassificationSupported",
        "goldIndependentOfLLMJudgeClaimed",
        "runtimeInteractionRequired",
    }
    checks = {
        "V207r2_is_valid_and_preserves_AgentAbstain_negative": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["V207_scientific_outcome_available"]
            and not parent["outcome"]["V207r1_scientific_feasibility_passed"]
            and parent["outcome"]["V207r1_transport_integrity_passed"]
        ),
        "fresh_six_source_roster_is_fixed_and_repository_distinct": bool(
            len(candidates) == gates["minimumCandidateCount"] == 6
            and len(repositories) == len(set(repositories))
            and all("agentabstain" not in repository.casefold() for repository in repositories)
        ),
        "landing_fact_schema_is_complete_before_metadata_census": all(
            set(candidate["landingFacts"]) == landing_keys
            and candidate["declaredPublicUnitCount"] >= 0
            and candidate["declaredPairCount"] >= 0
            and candidate["declaredScenarioCount"] >= 0
            for candidate in candidates
        ),
        "qualification_is_strict_and_noncompensatory": bool(
            gates["minimumDeclaredPublicUnitCount"] == 40
            and gates["minimumDeclaredPairCount"] == 20
            and gates["minimumDeclaredScenarioCount"] == 2
            and gates["requiredPairedOrMatchedControls"]
            and gates["requiredExplicitMachineReadablePairIdentityField"]
            and gates["requiredExplicitPreExecutionPhaseIdentity"]
            and gates["requiredDeterministicActAbstainGoldField"]
            and gates["requiredPromptLabelSeparation"]
            and gates["requiredBalancedActAbstainControls"]
            and gates["requiredTextOnlyShadowClassification"]
            and gates["requiredGoldIndependentOfLLMJudge"]
            and not gates["maximumRuntimeInteractionRequired"]
        ),
        "evaluation_reads_are_metadata_only_with_zero_blob_and_task_payload": bool(
            allowed["githubRepositoryObjectMetadata"]
            and allowed["githubRecursiveTreePathsOnly"]
            and allowed["huggingFaceDatasetHeadOnly"]
            and allowed["huggingFaceCardDataObjectMetadataOnly"]
            and allowed["githubREADMEOrOtherBlobBodyReadCount"] == 0
            and allowed["githubLicenseBlobBodyReadCount"] == 0
            and allowed["huggingFaceTreeOrPayloadReadCount"] == 0
            and allowed["taskInstructionExampleDialogueRationaleResponseReadCount"] == 0
            and allowed["modelOrPolicyEvaluationCount"] == 0
        ),
        "prelock_exposure_is_explicit_and_no_payload_or_model_was_opened": bool(
            prelock["incidentalIllustrativeTaskExamplePreviewCount"] == 1
            and prelock["candidateRecursiveRepositoryTreeReadCount"] == 0
            and prelock["candidateDatasetHeadReadCount"] == 0
            and prelock["candidateDatasetCardMetadataReadCount"] == 0
            and prelock["candidateTaskPayloadFileReadCount"] == 0
            and prelock["manualFullTaskInstructionExampleDialogueRationaleResponseReadCount"] == 0
            and prelock["modelOrPolicyEvaluationCount"] == 0
        ),
        "decision_does_not_authorize_language_or_model": bool(
            not config["decisionRule"]["passAuthorizesImmediateTaskTextAccessOrModelRun"]
            and not config["decisionRule"]["passAuthorizesAPITrainingToolServiceRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_absent": bool(
            all(path.is_file() for path in (config_path, parent_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "208-external-behavioral-abstention-source-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V208_metadata_census" if passed else "reject_V208_design",
        "checks": checks,
        "prelock_exposure": prelock,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V207r2_outcome": parent_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "208-external-behavioral-abstention-source-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_roster_landing_facts_gates_or_decision": False,
            "run_one_metadata_only_source_census": True,
            "read_blob_task_language_example_dialogue_rationale_response": False,
            "model_API_training_tool_service_registration_authority_action_or_execution": False,
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
