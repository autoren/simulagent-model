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
    config_path = PROJECT_ROOT / "configs/v206-fresh-external-analogue-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v206-fresh-external-analogue-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v206_fresh_external_analogue_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v206_fresh_external_analogue_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v206_fresh_external_analogue_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v206_fresh_external_analogue_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v206_fresh_external_analogue_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v206-fresh-external-analogue-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v206-fresh-external-analogue-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v206-fresh-external-analogue-feasibility/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v206-fresh-external-analogue-feasibility-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V206 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV205OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    candidates = config["candidates"]
    gates = config["metadataQualificationGates"]
    access = config["accessGates"]
    discovery = config["leadDiscovery"]
    candidate_repositories = [candidate["repository"] for candidate in candidates]
    checks = {
        "V205_is_valid_positive_and_authorizes_only_separate_source_feasibility": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_oracle_passed"]
            and parent["authorization"]["preregister_separate_fresh_source_feasibility_design_only"]
            and not parent["authorization"]["external_candidate_language_or_model_run"]
        ),
        "candidate_set_is_fixed_unique_and_fresh_relative_to_registry": bool(
            len(candidates) == gates["requiredCandidateCount"] == 6
            and len(set(candidate_repositories)) == len(candidates)
            and len({candidate["candidateId"] for candidate in candidates}) == len(candidates)
            and not set(candidate_repositories).intersection(config["priorExposureRepositoryUrls"])
        ),
        "metadata_rules_require_the_complete_same_source_conjunction": bool(
            all(
                gates[key]
                for key in (
                    "requiredOfficialLicense",
                    "requiredFreshRepositoryAndDomainFamily",
                    "requiredExplicitOpenWorldOutsideInvalidOrAbstentionRegime",
                    "requiredActionDependentInformationGathering",
                    "requiredInEpisodeReferenceCalibrationOrCrossSensorPathway",
                    "requiredSafeDeferAbstainOrHoldAction",
                    "requiredDelayedStateDependentOrIrreversibleConsequence",
                    "requiredExactSimulatorOrValidatedGenerativeLikelihoodPath",
                    "requiredAllCriticalElementsSourceNative",
                )
            )
            and gates["minimumQualifiedRepositoryDistinctFamilies"] == 1
        ),
        "prelock_exact_source_and_outcome_access_is_zero": bool(
            discovery["exactRepositoryHeadReadCountBeforeLock"] == 0
            and discovery["exactOfficialREADMEFetchCountBeforeLock"] == 0
            and discovery["exactLicenseFetchCountBeforeLock"] == 0
            and discovery["candidateImplementationOrTaskRecordReadCountBeforeLock"] == 0
            and discovery["policyOrSimulatorEvaluationCountBeforeLock"] == 0
        ),
        "access_contract_is_metadata_only": bool(
            access["requiredRepositoryHeadReadCount"] == len(candidates)
            and access["requiredOfficialREADMEFetchAttemptCount"] == len(candidates)
            and access["requiredLicenseFetchAttemptCount"] == len(candidates)
            and all(value == 0 for key, value in access.items() if not key.startswith("required"))
        ),
        "positive_result_still_requires_a_separate_structural_lock": bool(
            not config["decisionRule"]["passAuthorizesImmediateRepositoryCloneArchiveOrImplementationInspection"]
            and not config["decisionRule"]["passAuthorizesLanguageTaskRecordOrModelEvaluation"]
            and not config["decisionRule"]["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
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
        "schema_version": "206-fresh-external-analogue-source-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V206_official_metadata_census" if passed else "reject_V206_design",
        "checks": checks,
        "prelock_exposure": discovery,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V205_outcome": parent_path,
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
        "schema_version": "206-fresh-external-analogue-source-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_candidates_paths_evidence_rules_gates_or_decision": False,
            "run_one_official_metadata_feasibility_census": True,
            "clone_download_archive_open_implementation_or_read_task_language": False,
            "model_API_training_registration_authority_action_or_execution": False,
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
