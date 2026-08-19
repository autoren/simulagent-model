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
    config_path = PROJECT_ROOT / "configs/v209r1-controlled-language-observation-pomdp-shape-repair.json"
    plan_path = PROJECT_ROOT / "docs/v209r1-controlled-language-observation-pomdp-shape-repair-plan.md"
    protocol_path = PROJECT_ROOT / "python/v209r1_dynamic_regime_shape_repair.py"
    tests_path = PROJECT_ROOT / "python/test_v209r1_dynamic_regime_shape_repair.py"
    runner_path = PROJECT_ROOT / "python/run_v209r1_dynamic_regime_shape_repair.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v209r1_dynamic_regime_shape_repair_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v209r1_dynamic_regime_shape_repair.py"
    audit_path = PROJECT_ROOT / "outputs/v209r1-controlled-language-observation-pomdp-shape-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v209r1-controlled-language-observation-pomdp-shape-repair-lock.json"
    output_root = PROJECT_ROOT / "outputs/v209r1-controlled-language-observation-pomdp-shape-repair/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v209r1-controlled-language-observation-pomdp-shape-repair-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V209r1 is already preregistered, run, or frozen")

    repair_config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / repair_config["parentV209DesignLock"]
    failure_path = PROJECT_ROOT / repair_config["parentV209TechnicalFailure"]
    parent_lock = json.loads(parent_path.read_text())
    failure = json.loads(failure_path.read_text())
    repair = repair_config["repair"]
    exposure = repair_config["preLockExposure"]
    checks = {
        "parent_V209_design_lock_is_valid_and_authorized_one_run": bool(
            valid_lock(parent_lock)
            and parent_lock["authorization"]["run_exact_single_model_free_oracle_evaluation"]
            and not parent_lock["authorization"]["read_external_language_load_or_run_model_or_access_protected_data"]
        ),
        "technical_failure_is_exactly_pre_result_comparator_shape_validation": bool(
            failure["design_lock_sha256"] == file_sha256(parent_path)
            and failure["oracle_attempt_count"] == 1
            and failure["scientific_result_count"] == 0
            and not failure["summary_written"]
            and not failure["result_written"]
            and failure["failure_phase"] == "closed_world_comparator_kernel_construction"
            and not failure["scientific_elements_opened_before_failure"]["comparator_policy_or_regret_was_computed"]
            and not failure["scientific_elements_opened_before_failure"]["scientific_gate_was_evaluated"]
        ),
        "repair_scope_is_only_dynamic_regime_dimension_validation": bool(
            repair_config["schemaVersion"] == "209r1-controlled-language-observation-POMDP-shape-repair-design"
            and repair["permittedChangedFunction"] == "LanguageKernel.__post_init__"
            and repair["permittedChangedBehavior"] == "one-regime and two-regime comparator kernels pass shape validation"
            and repair["maximumChangedScientificParameterCount"] == 0
            and repair["maximumChangedGateCount"] == 0
            and repair["maximumChangedComparatorCount"] == 0
            and repair["maximumChangedDecisionRuleCount"] == 0
        ),
        "exposure_and_access_are_honest_and_bounded": bool(
            exposure["failedOracleAttemptCount"] == 1
            and exposure["scientificResultCount"] == 0
            and exposure["fullExactPolicyComputedInMemoryCount"] == 1
            and exposure["fullExactPolicyReturnedOrWrittenCount"] == 0
            and exposure["comparatorPolicyOrRegretComputedCount"] == 0
            and exposure["scientificGateEvaluationCount"] == 0
            and all(
                exposure[key] == 0
                for key in (
                    "externalLanguageRecordReadCount",
                    "modelLoadOrGenerationCount",
                    "APICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
        ),
        "repair_authorizes_only_tests_and_one_repaired_oracle": bool(
            repair_config["authorization"]["runUnitTestsAfterRepairLock"]
            and repair_config["authorization"]["runOneRepairedExactOracleEvaluation"]
            and not repair_config["authorization"]["modifyParentScientificDesign"]
            and not repair_config["authorization"]["openExternalLanguageOrRunModel"]
            and not repair_config["authorization"]["APITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_repaired_outputs_are_absent": bool(
            all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path, parent_path, failure_path))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "209r1-controlled-language-observation-POMDP-shape-repair-design-audit",
        "experiment": repair_config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V209r1_repaired_oracle" if passed else "reject_V209r1_repair",
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "repair_config": config_path,
        "parent_V209_design_lock": parent_path,
        "parent_V209_technical_failure": failure_path,
        "plan": plan_path,
        "repair_protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "209r1-controlled-language-observation-POMDP-shape-repair-lock",
        "experiment": repair_config["experiment"],
        "repair_config_payload": repair_config,
        "parent_config_payload_sha256": parent_lock["config_sha256"],
        "authorization": {
            "modify_parent_scientific_design": False,
            "run_unit_tests_and_one_repaired_oracle": True,
            "open_external_language_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
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
