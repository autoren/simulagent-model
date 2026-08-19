#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v110_open_world_deterministic_validation import split_secondary_development


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v110-open-world-deterministic-validation.json"
    parent_path = PROJECT_ROOT / "configs/v109-open-world-typed-choice-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v110-open-world-deterministic-validation-plan.md"
    protocol_path = PROJECT_ROOT / "python/v110_open_world_deterministic_validation.py"
    tests_path = PROJECT_ROOT / "python/test_v110_open_world_deterministic_validation.py"
    runner_path = PROJECT_ROOT / "python/run_v110_open_world_deterministic_validation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v110_deterministic_validation_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v110_deterministic_validation.py"
    audit_path = PROJECT_ROOT / "outputs/v110-open-world-deterministic-validation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v110-open-world-deterministic-validation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v110-open-world-deterministic-validation/development-evaluation"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V110 design is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    v109_lock = json.loads((PROJECT_ROOT / parent["implementation_lock"]).read_text())
    v109_result_path = PROJECT_ROOT / parent["result"]
    v109_result = json.loads(v109_result_path.read_text())
    baseline_outcome_path = PROJECT_ROOT / v109_lock["baseline_outcome"]
    baseline_outcome = json.loads(baseline_outcome_path.read_text())
    baseline_lock_path = PROJECT_ROOT / baseline_outcome["benchmark_lock"]
    baseline_lock = json.loads(baseline_lock_path.read_text())
    selected_population_path = PROJECT_ROOT / v109_lock["interface_config_payload"]["selectedPopulation"]
    selected_population = json.loads(selected_population_path.read_text())
    structural = {row["population_id"]: row for row in selected_population["selected_population"]}
    observed_ids = {
        identifier for identifier, row in v109_result["fixtures"].items()
        if row["kind"] == "observed_model_blind_holdback"
    }
    controls = [
        row for row in v109_result["fixtures"].values()
        if row["kind"] == "controlled_missing_observation"
    ]
    structural_records = [{
        "record_id": identifier, "class_label": structural[identifier]["class_label"],
    } for identifier in sorted(observed_ids)]
    split = split_secondary_development(structural_records, config)
    policies = set(config["policies"])
    required = set(config["developmentOutcomeGates"]["requiredPolicyNames"])
    checks = {
        "V109_is_exact_complete_mechanically_valid_and_semantically_nonqualifying": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["condition_completed"]
            and parent["outcome"]["interface_pass"]
            and not parent["outcome"]["semantic_pass"]
            and file_sha256(v109_result_path) == parent["result_sha256"]
            and not parent["authorization"]["read_protected_test_or_run_model_before_separate_lock"]
        ),
        "new_secondary_split_is_balanced_disjoint_and_identity_only": bool(
            len(observed_ids) == 128 and split["counts"] == {"calibration": 64, "evaluation": 64}
            and all(count == 16 for values in split["class_counts"].values() for count in values.values())
            and not ({row["record_id"] for row in split["calibration"]}
                     & {row["record_id"] for row in split["evaluation"]})
            and config["secondaryDevelopmentSplit"]["splitUsesOnlyFrozenRecordIdentifierAndClass"]
        ),
        "exact_required_policy_comparison_and_primary_policy_are_frozen": bool(
            policies == required and len(policies) == 9
            and config["primaryPolicy"] == "llm_plus_validation"
            and "direct_llm" in policies and "calibrated_abstention" in policies
            and "deterministic_novelty_gate" in policies and "oracle" in policies
        ),
        "threshold_grids_selection_rules_and_noncompensatory_gates_are_frozen": bool(
            len(config["calibratedAbstention"]["thresholdGrid"]) == 13
            and config["retrievalCalibration"]["reuseExactV106VectorizerAndThresholdGrids"]
            and config["developmentOutcomeGates"]["minimumPrimaryNovelStatusRecall"] == 0.50
            and config["developmentOutcomeGates"]["maximumPrimaryFalseKnownAcceptanceRate"] == 0.10
            and config["developmentOutcomeGates"]["maximumActualExecutionCount"] == 0
        ),
        "V109_controls_and_complete_safe_hypothesis_boundary_are_exact": bool(
            len(controls) == 64 and all(
                row["response_valid"] and row["parsed_response"]["status"] == "ABSTAIN"
                for row in controls
            )
            and config["authorityBoundary"]["allPoliciesAreCounterfactualShadowPolicies"]
            and config["authorityBoundary"]["LLMRemainsEvidenceOnly"]
            and config["authorityBoundary"]["completeSafeHypothesisUniverseRetained"]
            and not config["authorityBoundary"]["modelOrGateDefinesCapability"]
            and not config["authorityBoundary"]["modelOrGateExecutesAction"]
        ),
        "inputs_are_exact_and_protected_test_remains_identity_only": bool(
            file_sha256(PROJECT_ROOT / baseline_lock["source_archive"]) == baseline_lock["source_archive_sha256"]
            and file_sha256(PROJECT_ROOT / baseline_lock["development_language"]) == baseline_lock["development_language_sha256"]
            and file_sha256(PROJECT_ROOT / baseline_lock["visible_catalog"]) == baseline_lock["visible_catalog_sha256"]
            and file_sha256(selected_population_path) == v109_lock["interface_config_payload"]["selectedPopulationSha256"]
            and baseline_lock["protected_test_identity_only"]["file_open_count"] == 0
            and not config["inputs"]["readProtectedTestLanguage"]
            and not config["inputs"]["manualLanguageOrRawResponseInspection"]
        ),
        "pass_does_not_directly_authorize_test_induction_model_or_execution": bool(
            not config["decisionRule"]["passAuthorizesProtectedTestImmediately"]
            and not config["decisionRule"]["passAuthorizesSchemaInductionImmediately"]
            and not config["decisionRule"]["passAuthorizesAPITrainingPlanningOrExecution"]
        ),
        "locked_code_runtime_and_output_absence_hold": bool(
            all(path.is_file() for path in (
                plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
            ))
            and metadata.version("scikit-learn") == "1.9.0"
            and metadata.version("numpy") == "2.5.1" and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "110-open-world-deterministic-validation-design-audit",
        "experiment": "v110_open_world_deterministic_validation_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_deterministic_secondary_development_analysis" if passed else "reject_V110_analysis",
        "checks": checks,
        "prospective_split_summary": {
            "counts": split["counts"], "class_counts": split["class_counts"],
            "membership_sha256": split["membership_sha256"], "contains_language": False,
        },
        "access": {
            "source_archive_read_count": 0, "development_language_read_count": 0,
            "V109_result_automatic_read_count": 1, "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_typed_choice_outcome": parent_path,
        "V109_implementation_lock": PROJECT_ROOT / parent["implementation_lock"],
        "V109_result": v109_result_path, "baseline_outcome": baseline_outcome_path,
        "baseline_lock": baseline_lock_path,
        "source_archive": PROJECT_ROOT / baseline_lock["source_archive"],
        "development_language": PROJECT_ROOT / baseline_lock["development_language"],
        "visible_catalog": PROJECT_ROOT / baseline_lock["visible_catalog"],
        "controlled_identifiers": PROJECT_ROOT / baseline_lock["controlled_identifiers"],
        "selected_population": selected_population_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "110-open-world-deterministic-validation-lock",
        "experiment": "v110_open_world_deterministic_validation_lock",
        "config_payload": config,
        "baseline_config_payload": baseline_lock["config_payload"],
        "authorization": {
            "modify_split_policies_thresholds_selection_metrics_gates_or_decision": False,
            "run_one_deterministic_secondary_development_analysis_once": True,
            "emit_source_language_or_raw_model_response": False,
            "read_protected_test_language": False,
            "load_or_generate_with_model": False,
            "run_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
