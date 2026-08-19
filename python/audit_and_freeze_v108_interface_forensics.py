#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v108-open-world-interface-forensics.json"
    parent_path = PROJECT_ROOT / "configs/v107-open-world-local-model-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v108-open-world-interface-forensics-plan.md"
    protocol_path = PROJECT_ROOT / "python/v108_open_world_interface_forensics.py"
    tests_path = PROJECT_ROOT / "python/test_v108_open_world_interface_forensics.py"
    runner_path = PROJECT_ROOT / "python/run_v108_open_world_interface_forensics.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v108_interface_forensics_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v108_interface_forensics.py"
    audit_path = PROJECT_ROOT / "outputs/v108-open-world-interface-forensics/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v108-open-world-interface-forensics-lock.json"
    output_root = PROJECT_ROOT / "outputs/v108-open-world-interface-forensics/forensics"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V108 diagnostic is already frozen or materialized")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    implementation_lock = json.loads((PROJECT_ROOT / parent["implementation_lock"]).read_text())
    baseline_outcome = json.loads((PROJECT_ROOT / implementation_lock["parent_baseline_outcome"]).read_text())
    interface_config = implementation_lock["interface_config_payload"]
    population_path = PROJECT_ROOT / interface_config["selectedPopulation"]
    membership_path = PROJECT_ROOT / baseline_outcome["development_split_membership"]
    result_path = PROJECT_ROOT / parent["result"]
    gates = config["formatDominanceGates"]
    canonicalization = config["canonicalization"]
    checks = {
        "V107_outcome_is_exact_nonqualifying_and_model_branch_closed": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["condition_completed"]
            and not parent["outcome"]["quality_gate_pass"]
            and parent["authorization"]["close_model_branch_without_protected_test"]
            and not parent["authorization"]["preregister_one_identical_protected_test_run"]
        ),
        "exact_existing_outputs_text_free_structure_and_membership_are_frozen": bool(
            file_sha256(result_path) == parent["result_sha256"]
            and file_sha256(population_path) == interface_config["selectedPopulationSha256"]
            and file_sha256(membership_path) == baseline_outcome["development_split_membership_sha256"]
            and config["inputs"]["useExactFrozenV107RawResponses"]
            and config["inputs"]["useV101TextFreeStructuralGroundTruth"]
            and config["inputs"]["useV106EvaluationMembership"]
            and not config["inputs"]["readDevelopmentOrProtectedLanguage"]
            and not config["inputs"]["manualRawResponseInspection"]
        ),
        "diagnostic_canonicalization_is_narrow_nonsemantic_and_nonretrospective": bool(
            canonicalization["diagnosticOnly"]
            and canonicalization["uniqueLocalIntentNameToQualifiedIntentId"]
            and canonicalization["removeNovelScenarioOnlyWhenStatusKnownAndScenarioExactlyMatchesResolvedIntentScenario"]
            and not canonicalization["changeStatus"]
            and not canonicalization["inferIntentFromUtteranceOrGold"]
            and not canonicalization["changeConfidence"]
            and not canonicalization["retryOrRegenerate"]
            and not canonicalization["replaceFrozenV107Score"]
        ),
        "format_dominance_thresholds_and_zero_access_limits_are_frozen": bool(
            gates["minimumInvalidObservedCanonicalizableFraction"] == 0.75
            and gates["minimumCounterfactualKnownExactIntentAccuracy"] == 0.70
            and gates["minimumCounterfactualObservedExactDecisionAccuracy"] == 0.65
            and gates["maximumDevelopmentLanguageReadCount"] == 0
            and gates["maximumProtectedTestLanguageReadCount"] == 0
            and gates["maximumModelLoadCount"] == 0
            and gates["maximumModelGenerationCount"] == 0
        ),
        "diagnostic_cannot_rescore_retry_test_or_expand_model_authority": bool(
            not config["decisionRule"]["passAuthorizesV107RescoreOrRetry"]
            and not config["decisionRule"]["passAuthorizesProtectedTestOrModelAccess"]
            and not config["decisionRule"]["passAuthorizesAPITrainingPlanningOrExecution"]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "108-open-world-interface-forensics-design-audit",
        "experiment": "v108_open_world_interface_forensics_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_aggregate_existing_output_diagnostic" if passed else "reject_V108_diagnostic",
        "checks": checks,
        "access": {
            "existing_raw_response_semantic_inspection_count": 0,
            "development_language_read_count": 0, "protected_test_language_read_count": 0,
            "manual_raw_response_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
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
        "config": config_path, "parent_model_outcome": parent_path,
        "implementation_lock": PROJECT_ROOT / parent["implementation_lock"],
        "baseline_outcome": PROJECT_ROOT / implementation_lock["parent_baseline_outcome"],
        "model_result": result_path, "selected_population": population_path,
        "development_membership": membership_path,
        "visible_catalog": PROJECT_ROOT / implementation_lock["visible_catalog"],
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "108-open-world-interface-forensics-lock",
        "experiment": "v108_open_world_interface_forensics_lock",
        "config_payload": config,
        "baseline_config_payload": implementation_lock["baseline_config_payload"],
        "interface_config_payload": interface_config,
        "authorization": {
            "modify_categories_transformations_thresholds_or_decision": False,
            "automatically_parse_existing_raw_responses_once": True,
            "emit_raw_response_text_or_individual_identifiers": False,
            "read_development_or_protected_test_language": False,
            "manually_inspect_raw_response_text": False,
            "load_or_run_any_model": False,
            "replace_rescore_retry_or_rerun_V107": False,
            "run_API_model_or_train_adapter": False,
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
