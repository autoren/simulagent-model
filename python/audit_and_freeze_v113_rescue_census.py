#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v113_known_disagreement_rescue import enumerate_rules


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v113-known-disagreement-rescue-census.json"
    parent_path = PROJECT_ROOT / "configs/v112r1-full-policy-aggregation-recovery-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v113-known-disagreement-rescue-census-plan.md"
    protocol_path = PROJECT_ROOT / "python/v113_known_disagreement_rescue.py"
    tests_path = PROJECT_ROOT / "python/test_v113_known_disagreement_rescue.py"
    runner_path = PROJECT_ROOT / "python/run_v113_known_disagreement_rescue_census.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v113_rescue_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v113_rescue_census.py"
    audit_path = PROJECT_ROOT / "outputs/v113-known-disagreement-rescue-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v113-known-disagreement-rescue-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v113-known-disagreement-rescue-census/historical-census"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V113 census is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    r1_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    r1_lock = json.loads(r1_lock_path.read_text())
    r1_result_path = PROJECT_ROOT / parent["result"]
    rules = enumerate_rules(config)
    checks = {
        "V112r1_is_exact_novelty_positive_and_full_policy_negative": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["novel_evidence_pass"]
            and not parent["outcome"]["quality_gate_pass"]
            and parent["authorization"]["redesign_policy_on_new_population"]
            and not parent["authorization"]["read_protected_test_before_separate_lock"]
            and file_sha256(r1_lock_path) == parent["analysis_lock_sha256"]
            and file_sha256(r1_result_path) == parent["result_sha256"]
        ),
        "registered_feature_rule_and_threshold_space_is_exact": bool(
            len(config["registeredFeatures"]) == 5
            and len(config["registeredRuleFamilies"]) == 9
            and len(rules) == 239
            and len({str(sorted(rule.items())) for rule in rules}) == 239
            and config["rescueActionConfidence"] == 0.75
        ),
        "selection_uses_all_seventeen_inherited_gates_and_is_historical_only": bool(
            len(parent["outcome"]["summary"]["quality_gates"]) == 17
            and config["selection"]["evaluationRole"].startswith("historical policy design only")
        ),
        "novelty_protected_model_and_authority_boundaries_are_frozen": bool(
            config["inputs"]["useExactPreservedV112Fixtures"]
            and config["inputs"]["useExactV112FreshLanguageAsHistoricalDevelopmentOnly"]
            and not config["inputs"]["readProtectedTestLanguage"]
            and not config["inputs"]["manualLanguageOrRawResponseInspection"]
            and not config["inputs"]["persistIndividualFeaturesPredictionsIdentifiersLanguageOrRawResponse"]
            and not config["inputs"]["loadOrGenerateModel"]
            and not config["decisionRule"]["passAuthorizesProtectedTestImmediately"]
            and not config["decisionRule"]["passAuthorizesSchemaInductionSequentialPlanningNewModelAPITrainingActionOrExecution"]
        ),
        "locked_code_and_output_absence_hold": bool(
            all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "113-known-disagreement-rescue-census-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_and_authorize_one_historical_rescue_census" if passed else "reject_V113_census",
        "checks": checks, "prospective_candidate_count": len(rules),
        "access": {
            "preserved_fixture_read_count": 0, "historical_language_read_count": 0,
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
    dependencies = {
        "config": config_path, "parent_outcome": parent_path,
        "V112r1_lock": r1_lock_path, "V112r1_result": r1_result_path,
        "fresh_language": PROJECT_ROOT / r1_lock["fresh_language"],
        "fixture_manifest": PROJECT_ROOT / r1_lock["fixture_manifest"],
        "source_archive": PROJECT_ROOT / r1_lock["source_archive"],
        "visible_catalog": PROJECT_ROOT / r1_lock["visible_catalog"],
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "113-known-disagreement-rescue-census-lock",
        "experiment": config["experiment"], "config_payload": config,
        "V112_config_payload": r1_lock["V112_config_payload"],
        "baseline_config_payload": r1_lock["baseline_config_payload"],
        "preserved_access": r1_lock["preserved_access"],
        "authorization": {
            "modify_features_rules_thresholds_confidence_metrics_gates_selection_or_decision": False,
            "run_one_aggregate_only_historical_census": True,
            "persist_individual_features_predictions_identifiers_language_or_raw_response": False,
            "read_protected_test_load_or_generate_model_run_API_train_or_execute": False,
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
