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
    config_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair.json"
    plan_path = PROJECT_ROOT / "docs/v207r2-agentabstain-outcome-verification-repair-plan.md"
    protocol_path = PROJECT_ROOT / "python/v207r2_agentabstain_outcome_verification_repair.py"
    tests_path = PROJECT_ROOT / "python/test_v207r2_agentabstain_outcome_verification_repair.py"
    runner_path = PROJECT_ROOT / "python/run_v207r2_agentabstain_outcome_verification_repair.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v207r2_agentabstain_outcome_verification_repair_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v207r2_agentabstain_outcome_verification_repair.py"
    audit_path = PROJECT_ROOT / "outputs/v207r2-agentabstain-outcome-verification-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair-lock.json"
    output_root = PROJECT_ROOT / "outputs/v207r2-agentabstain-outcome-verification-repair/repair"
    outcome_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V207r2 already started")

    config = json.loads(config_path.read_text())
    inputs = {
        "source_V207r1_lock": PROJECT_ROOT / config["sourceV207r1Lock"],
        "source_failed_outcome_audit": PROJECT_ROOT / config["sourceFailedOutcomeAudit"],
        "source_summary": PROJECT_ROOT / config["sourceSummary"],
        "source_result": PROJECT_ROOT / config["sourceResult"],
        "source_results_document": PROJECT_ROOT / config["sourceResultsDocument"],
    }
    source_lock = json.loads(inputs["source_V207r1_lock"].read_text())
    failed = json.loads(inputs["source_failed_outcome_audit"].read_text())
    dependency_keys = [key for key in source_lock if not key.endswith("_sha256") and f"{key}_sha256" in source_lock]
    checks = {
        "V207r1_lock_and_dependencies_are_restored_exactly": bool(
            valid_lock(source_lock)
            and all(file_sha256(PROJECT_ROOT / source_lock[key]) == source_lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "V207r1_failed_only_expected_dependency_check": bool(
            not failed["passed"]
            and sorted(key for key, value in failed["checks"].items() if not value)
            == sorted(config["repairContract"]["requiredFalseChecks"])
        ),
        "scope_is_verification_only": bool(
            not config["repairContract"]["sourceArtifactsMayBeModified"]
            and not config["repairContract"]["networkMetadataMayBeReadAgain"]
            and not config["repairContract"]["scientificEvaluationOrModelMayBeRerun"]
        ),
        "prelock_new_access_is_zero": all(value == 0 for value in config["preLockExposure"].values()),
        "authority_remains_closed": not config["decisionRule"]["passAuthorizesTaskLanguageModelAPITrainingToolAuthorityActionOrExecution"],
        "required_files_exist_and_outputs_absent": bool(
            all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path, *inputs.values()))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "207r2-agentabstain-outcome-verification-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_V207r2_verification_repair_design" if passed else "reject_V207r2_verification_repair_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        **inputs,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "207r2-agentabstain-outcome-verification-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_sources_or_rerun": False,
            "run_one_local_stored_artifact_verification": True,
            "task_language_model_API_training_tool_authority_action_or_execution": False,
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
