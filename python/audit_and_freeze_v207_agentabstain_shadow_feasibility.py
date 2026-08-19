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
    config_path = PROJECT_ROOT / "configs/v207-agentabstain-shadow-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v207-agentabstain-shadow-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v207_agentabstain_shadow_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v207_agentabstain_shadow_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v207_agentabstain_shadow_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v207_agentabstain_shadow_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v207_agentabstain_shadow_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v207-agentabstain-shadow-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v207-agentabstain-shadow-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v207-agentabstain-shadow-feasibility/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v207-agentabstain-shadow-feasibility-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V207 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV206OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    source = config["source"]
    allowed = config["allowedMetadataReads"]
    qualifications = config["qualificationGates"]
    access = config["accessGates"]
    prelock = config["preLockExposure"]
    parent_agent = next(
        record for record in parent["outcome"]["summary"]["records"] if record["candidate_id"] == "AGENT_ABSTAIN"
    )
    checks = {
        "V206_is_valid_negative_and_AgentAbstain_identity_is_exact": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_feasibility_passed"]
            and source["codeCommitFromV206"] == parent_agent["commit"]
            and source["codeREADMESha256FromV206"] == parent_agent["readme"]["sha256"]
            and source["codeLicenseSha256FromV206"] == parent_agent["license_file"]["sha256"]
        ),
        "behavioral_track_is_explicitly_separate_from_V205": bool(
            "not V205 likelihood validation" in config["claimBoundary"]
            and not config["decisionRule"]["passAuthorizesImmediateTaskTextAccessOrModelRun"]
        ),
        "allowed_reads_are_schema_only_and_bounded": bool(
            allowed["githubTreePathsOnly"]
            and allowed["datasetTreePathsAndObjectMetadataOnly"]
            and allowed["datasetCardYAMLFrontMatterOnly"]
            and not allowed["persistFetchedCodeOrCardText"]
            and allowed["persistOnlyHashesByteCountsPathsAndExtractedSchemaIdentifiers"]
            and allowed["maximumAllowedCodeSchemaFiles"] == access["maximumCodeSchemaFileReadCount"]
        ),
        "pair_preexecution_schema_and_shadow_gates_are_noncompensatory": bool(
            qualifications["minimumTreeIdentifiedCompletePairCount"] == 40
            and qualifications["minimumTreeIdentifiedPreExecutionPairCount"] == 20
            and qualifications["minimumTreeIdentifiedPreExecutionScenarioCount"] == 2
            and qualifications["requiredBothPairSidesIdentifiableWithoutTaskPayload"]
            and qualifications["requiredGoldDecisionIdentifiableWithoutLLMJudge"]
            and qualifications["requiredIdentityGoldScenarioAndPromptSchemaFields"]
            and qualifications["requiredRationaleSeparableFromPrompt"]
            and qualifications["requiredPreExecutionSubsetSelectableBeforeTaskText"]
            and qualifications["requiredShadowNoToolNoExecutionEvaluationPath"]
        ),
        "contamination_uncertainty_is_not_hidden": bool(
            prelock["observedPublicReleaseMonth"] == "2026-07"
            and "no contamination-free claim" in qualifications["requiredContaminationTreatment"]
            and "no training or finetuning" in qualifications["requiredContaminationTreatment"]
        ),
        "prelock_dataset_schema_task_language_and_outcome_access_is_zero": bool(
            prelock["exactDatasetHeadReadCount"] == 0
            and prelock["datasetTreeOrCardReadCount"] == 0
            and prelock["codeTreeOrSchemaFileReadCount"] == 0
            and prelock["taskInstructionExampleDialogueRationaleReadCount"] == 0
            and prelock["modelOrPolicyEvaluationCount"] == 0
        ),
        "future_selection_is_text_blind_and_requires_a_separate_extraction_lock": bool(
            not config["futureSelectionIfQualified"]["selectionMayReadTaskText"]
            and config["futureSelectionIfQualified"]["futureTextExtractionRequiresSeparateLock"]
            and config["futureSelectionIfQualified"]["minimumPairs"] >= qualifications["minimumTreeIdentifiedPreExecutionPairCount"]
        ),
        "forbidden_task_language_model_tool_and_execution_access_is_zero": all(
            value == 0
            for key, value in access.items()
            if key.startswith("maximum") and key != "maximumCodeSchemaFileReadCount"
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    parent_path,
                    PROJECT_ROOT / config["roadmap"],
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "207-agentabstain-shadow-metadata-schema-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V207_metadata_schema_audit" if passed else "reject_V207_design",
        "checks": checks,
        "prelock_exposure": prelock,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V206_outcome": parent_path,
        "roadmap": PROJECT_ROOT / config["roadmap"],
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "207-agentabstain-shadow-metadata-schema-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_source_paths_patterns_thresholds_selection_or_decision": False,
            "run_one_metadata_schema_feasibility_audit": True,
            "open_task_payload_instruction_example_dialogue_or_rationale": False,
            "model_tool_execution_API_training_registration_authority_or_side_effect": False,
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
