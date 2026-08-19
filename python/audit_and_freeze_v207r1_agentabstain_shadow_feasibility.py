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
    config_path = PROJECT_ROOT / "configs/v207r1-agentabstain-shadow-feasibility.json"
    scientific_path = PROJECT_ROOT / "configs/v207-agentabstain-shadow-feasibility.json"
    parent_lock_path = PROJECT_ROOT / "configs/v207-agentabstain-shadow-feasibility-lock.json"
    failure_path = PROJECT_ROOT / "outputs/v207-agentabstain-shadow-feasibility/transport-failure.json"
    plan_path = PROJECT_ROOT / "docs/v207r1-agentabstain-shadow-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v207r1_agentabstain_shadow_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v207r1_agentabstain_shadow_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v207r1_agentabstain_shadow_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v207r1_agentabstain_shadow_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v207r1_agentabstain_shadow_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v207r1-agentabstain-shadow-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v207r1-agentabstain-shadow-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v207r1-agentabstain-shadow-feasibility/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v207r1-agentabstain-shadow-feasibility-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V207r1 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    scientific = json.loads(scientific_path.read_text())
    parent = json.loads(parent_lock_path.read_text())
    failure = json.loads(failure_path.read_text())
    repair = config["repairPolicy"]
    prelock = config["preLockExposure"]
    prior = failure["cumulative_pre_repair_access"]
    checks = {
        "parent_V207_lock_is_valid_exact_and_unmodified": bool(
            valid_lock(parent)
            and file_sha256(parent_lock_path) == config["parentV207DesignLockSha256"]
            and file_sha256(scientific_path) == config["unchangedScientificConfigSha256"]
            and parent["config_payload"] == scientific
            and parent["config_sha256"] == file_sha256(scientific_path)
        ),
        "V207_is_a_technical_nonresult": bool(
            not failure["scientific_outcome_available"]
            and failure["failed_stage"] == "dataset_tree_metadata_transport"
            and failure["failure"]["http_status"] == 400
            and failure["failure"]["requested_limit"] == 10000
            and not failure["failure"]["evaluation_outputs_persisted"]
        ),
        "prior_metadata_exposure_is_exactly_carried_forward": bool(
            prelock["priorCodeTreeMetadataReadCount"] == prior["code_tree_metadata_read_count"]
            and prelock["priorCodeSchemaFileReadCount"] == prior["code_schema_file_read_count"]
            and prelock["priorDatasetHeadReadCount"] == prior["dataset_head_read_count"]
            and prelock["priorDatasetTreeMetadataRequestCount"] == prior["dataset_tree_metadata_request_count"]
            and prelock["priorDatasetTreeMetadataSuccessCount"] == prior["dataset_tree_metadata_success_count"]
            and prelock["priorDatasetTreeMetadataHTTP400Count"] == prior["dataset_tree_metadata_http_400_count"]
            and prelock["priorDatasetCardHeaderReadCount"] == prior["dataset_card_header_read_count"]
            and all(
                prior[key] == 0
                for key in (
                    "dataset_task_payload_file_read_count",
                    "task_instruction_example_dialogue_rationale_read_count",
                    "protected_access_count",
                    "model_load_count",
                    "model_generation_count",
                    "model_API_call_count",
                    "training_run_count",
                    "tool_call_count",
                    "service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
        ),
        "repair_is_bounded_cursor_transport_only": bool(
            repair["huggingFaceTreePageSize"] == 1000
            and repair["minimumExpectedPageCount"] == 2
            and repair["maximumAllowedPageCount"] == 20
            and repair["followOnlyExplicitRelNextLinks"]
            and repair["requireSameHTTPSHostAndPinnedRevisionPath"]
            and repair["rejectRepeatedPageURLOrCursorCycle"]
            and repair["requireTerminalPageWithoutNextLink"]
            and repair["combineOnlyTreeObjectMetadata"]
            and repair["persistNoResponseBodyOrCursorURL"]
            and repair["countEveryPhysicalPageRead"]
            and repair["preserveOneLogicalDatasetTreeCensus"]
        ),
        "scientific_and_firewall_rules_are_explicitly_unchanged": bool(
            repair["preserveEveryScientificQualificationGate"]
            and repair["preserveTaskLanguageModelToolAndExecutionFirewall"]
            and "no change to source" in config["claimBoundary"]
            and not config["decisionRule"]["passAuthorizesImmediateTaskTextAccessOrModelRun"]
            and not config["decisionRule"]["passAuthorizesToolsExecutionAPITrainingRegistrationAuthorityOrSideEffects"]
        ),
        "V207r1_prelock_new_access_is_zero": bool(
            prelock["V207r1DatasetTreePageReadCount"] == 0
            and prelock["V207r1DatasetCardHeaderReadCount"] == 0
            and prelock["V207r1TaskInstructionExampleDialogueRationaleReadCount"] == 0
            and prelock["V207r1ModelOrPolicyEvaluationCount"] == 0
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    scientific_path,
                    parent_lock_path,
                    failure_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    PROJECT_ROOT / config["roadmap"],
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "207r1-agentabstain-shadow-metadata-schema-transport-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V207r1_paginated_metadata_schema_audit" if passed else "reject_V207r1_repair_design",
        "checks": checks,
        "prelock_exposure": prelock,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "unchanged_scientific_config": scientific_path,
        "parent_V207_design_lock": parent_lock_path,
        "V207_transport_failure": failure_path,
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
        "schema_version": "207r1-agentabstain-shadow-metadata-schema-transport-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "scientific_config_payload": scientific,
        "authorization": {
            "modify_scientific_source_patterns_thresholds_selection_or_decision": False,
            "run_one_paginated_metadata_schema_feasibility_audit": True,
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
