#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from collections import Counter
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v111_open_world_separability_audit import enumerate_rules


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v111-open-world-separability-audit.json"
    parent_path = PROJECT_ROOT / "configs/v110-open-world-deterministic-validation-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v111-open-world-separability-audit-plan.md"
    protocol_path = PROJECT_ROOT / "python/v111_open_world_separability_audit.py"
    tests_path = PROJECT_ROOT / "python/test_v111_open_world_separability_audit.py"
    runner_path = PROJECT_ROOT / "python/run_v111_open_world_separability_audit.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v111_separability_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v111_separability_audit.py"
    audit_path = PROJECT_ROOT / "outputs/v111-open-world-separability-audit/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v111-open-world-separability-audit-lock.json"
    output_root = PROJECT_ROOT / "outputs/v111-open-world-separability-audit/development-census"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V111 design is already frozen or evaluated")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_analysis_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_analysis_lock = json.loads(parent_analysis_lock_path.read_text())
    parent_result_path = PROJECT_ROOT / parent["result"]
    parent_result = json.loads(parent_result_path.read_text())
    membership_path = PROJECT_ROOT / parent_result["output_integrity"]["secondary_membership"]["path"]
    membership = json.loads(membership_path.read_text())
    rows = membership["membership"]
    subset_counts = Counter(row["subset"] for row in rows)
    class_subset_counts = Counter((row["subset"], row["class_label"]) for row in rows)
    rules = enumerate_rules(config)

    checks = {
        "V110_is_exact_complete_and_scientifically_nonqualifying": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and not parent["outcome"]["quality_gate_pass"]
            and file_sha256(parent_analysis_lock_path) == parent["analysis_lock_sha256"]
            and file_sha256(parent_result_path) == parent["result_sha256"]
            and not parent["authorization"]["read_protected_test_before_separate_lock"]
            and not parent["authorization"]["proceed_to_schema_or_mechanic_induction"]
            and not parent["authorization"]["proceed_to_richer_sequential_decision_problem"]
        ),
        "V110_membership_is_exact_balanced_identity_only_and_reused": bool(
            file_sha256(membership_path) == parent_result["output_integrity"]["secondary_membership"]["sha256"]
            and membership["contains_language"] is False
            and subset_counts == {"calibration": 64, "evaluation": 64}
            and all(count == 16 for count in class_subset_counts.values())
            and len(class_subset_counts) == 8
            and config["inputs"]["reuseExactV110CalibrationEvaluationMembership"]
        ),
        "feature_rule_grids_and_calibration_only_selection_are_frozen": bool(
            len(config["registeredFeatures"]) == 12
            and len(config["registeredRuleFamilies"]) == 10
            and len(rules) == 1343
            and len(config["thresholdGrids"]["score"]) == 11
            and len(config["thresholdGrids"]["margin"]) == 10
            and len(config["thresholdGrids"]["proposal"]) == 11
            and config["calibrationSelection"]["minimumNovelPrecisionForFeasibleCandidate"] == 0.70
            and config["calibrationSelection"]["minimumNovelRecallForFeasibleCandidate"] == 0.50
            and config["calibrationSelection"]["maximumNonNovelFalsePositiveRateForFeasibleCandidate"] == 0.10
        ),
        "evaluation_gates_are_noncompensatory_and_oracle_is_diagnostic_only": bool(
            config["evaluationGates"]["minimumSelectedRuleNovelPrecision"] == 0.70
            and config["evaluationGates"]["minimumSelectedRuleNovelRecall"] == 0.50
            and config["evaluationGates"]["maximumSelectedRuleNonNovelFalsePositiveRate"] == 0.10
            and not config["decisionRule"]["evaluationOracleMayAuthorizePolicy"]
        ),
        "aggregate_only_zero_new_model_and_protected_boundary_is_frozen": bool(
            config["inputs"]["useExactFrozenV109ParsedPredictions"]
            and config["inputs"]["useExactV106CharacterTfidfRepresentation"]
            and not config["inputs"]["readProtectedTestLanguage"]
            and not config["inputs"]["manualLanguageOrRawResponseInspection"]
            and not config["inputs"]["persistIndividualFeaturesPredictionsOrIdentifiers"]
            and not config["decisionRule"]["passAuthorizesProtectedTestInductionPlanningAPITrainingOrExecution"]
        ),
        "locked_code_runtime_and_output_absence_hold": bool(
            all(path.is_file() for path in (
                plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
            ))
            and metadata.version("scikit-learn") == "1.9.0"
            and metadata.version("numpy") == "2.5.1"
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "111-open-world-separability-audit-design-audit",
        "experiment": "v111_existing_evidence_deterministic_novelty_separability_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_aggregate_only_development_census" if passed else "reject_V111_census",
        "checks": checks,
        "prospective_summary": {
            "candidate_rule_count": len(rules),
            "subset_counts": dict(sorted(subset_counts.items())),
            "class_subset_counts": {
                f"{subset}:{label}": count
                for (subset, label), count in sorted(class_subset_counts.items())
            },
            "contains_language_or_individual_features": False,
        },
        "access": {
            "source_archive_read_count": 0, "development_language_read_count": 0,
            "V109_result_automatic_read_count": 0, "V110_result_automatic_read_count": 1,
            "protected_test_language_read_count": 0,
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

    dependency_paths = {
        "config": config_path,
        "parent_outcome": parent_path,
        "parent_analysis_lock": parent_analysis_lock_path,
        "parent_result": parent_result_path,
        "V109_result": PROJECT_ROOT / parent_analysis_lock["V109_result"],
        "baseline_outcome": PROJECT_ROOT / parent_analysis_lock["baseline_outcome"],
        "baseline_lock": PROJECT_ROOT / parent_analysis_lock["baseline_lock"],
        "source_archive": PROJECT_ROOT / parent_analysis_lock["source_archive"],
        "development_language": PROJECT_ROOT / parent_analysis_lock["development_language"],
        "visible_catalog": PROJECT_ROOT / parent_analysis_lock["visible_catalog"],
        "secondary_membership": membership_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "111-open-world-separability-audit-lock",
        "experiment": "v111_existing_evidence_deterministic_novelty_separability_lock",
        "config_payload": config,
        "baseline_config_payload": parent_analysis_lock["baseline_config_payload"],
        "authorization": {
            "modify_membership_features_rules_thresholds_selection_gates_or_decision": False,
            "run_one_aggregate_only_development_census_once": True,
            "persist_individual_features_predictions_identifiers_language_or_raw_response": False,
            "read_protected_test_language": False,
            "load_or_generate_with_model": False,
            "run_API_model_or_train_adapter": False,
            "proceed_to_schema_induction_or_sequential_planning": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependency_paths.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
