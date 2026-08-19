#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v112r1-full-policy-aggregation-recovery.json"
    plan_path = PROJECT_ROOT / "docs/v112r1-full-policy-aggregation-recovery-plan.md"
    protocol_path = PROJECT_ROOT / "python/v112r1_full_policy_aggregation.py"
    tests_path = PROJECT_ROOT / "python/test_v112r1_full_policy_aggregation.py"
    runner_path = PROJECT_ROOT / "python/run_v112r1_full_policy_aggregation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v112r1_aggregation_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v112r1_aggregation_recovery.py"
    manifest_path = PROJECT_ROOT / "outputs/v112r1-full-policy-aggregation-recovery/design/preserved-fixture-manifest.json"
    audit_path = PROJECT_ROOT / "outputs/v112r1-full-policy-aggregation-recovery/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v112r1-full-policy-aggregation-recovery-lock.json"
    result_root = PROJECT_ROOT / "outputs/v112r1-full-policy-aggregation-recovery/recovered-evaluation"
    if manifest_path.exists() or audit_path.exists() or lock_path.exists() or result_root.exists():
        raise RuntimeError("V112r1 recovery is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent_lock_path = PROJECT_ROOT / config["parentV112ImplementationLock"]
    failure_path = PROJECT_ROOT / config["parentFailure"]
    language_path = PROJECT_ROOT / config["freshLanguage"]
    raw_dir = PROJECT_ROOT / config["rawFixtureDirectory"]
    parent_lock = json.loads(parent_lock_path.read_text())
    failure = json.loads(failure_path.read_text())
    raw_paths = sorted(raw_dir.glob("*.json"))
    manifest_rows = [
        {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for path in raw_paths
    ]
    manifest = {"fixture_count": len(manifest_rows), "fixtures": manifest_rows}
    write_json(manifest_path, manifest)
    registered = config["registeredFailure"]
    checks = {
        "parent_lock_is_exact": payload_hash({key: value for key, value in parent_lock.items() if key != "lock_payload_sha256"}) == parent_lock["lock_payload_sha256"],
        "failure_is_exactly_the_registered_post_generation_aggregation_error": bool(
            failure["stage"] == registered["stage"]
            and failure["exception_type"] == registered["exceptionType"]
            and failure["exception_message"] == registered["exceptionMessage"]
            and failure["completed_fixture_count"] == registered["requiredCompletedFixtureCount"]
            and failure["attempt"]["model_load_count"] == registered["requiredModelLoadCount"]
            and failure["attempt"]["model_generation_count"] == registered["requiredModelGenerationCount"]
            and failure["active_fixture"] is None and not failure["result_artifact_written"]
        ),
        "all_preserved_fixtures_are_manifested_once": bool(
            len(raw_paths) == len({path.name for path in raw_paths}) == 240
            and manifest["fixture_count"] == 240
        ),
        "fresh_language_and_no_parent_result_are_exact": bool(
            language_path.is_file()
            and len(language_path.read_text().splitlines()) == 192
            and not (raw_dir.parent / "result.json").exists()
        ),
        "repair_is_mechanical_only_and_forbids_new_model_or_inspection": bool(
            not config["authorizedRepair"]["changesPopulationLanguagePromptModelResponsePolicyThresholdConfidenceMetricGateOrDecision"]
            and not config["authorizedRepair"]["newModelLoadOrGeneration"]
            and not config["authorizedRepair"]["manualLanguageOrRawResponseInspection"]
        ),
        "all_recovery_code_exists": all(path.is_file() for path in (
            plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
        )),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "112r1-full-policy-aggregation-recovery-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_and_authorize_one_aggregation_only_recovery" if passed else "reject_recovery",
        "checks": checks,
        "access": {
            "preserved_fixture_automatic_hash_count": 240,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_lock": parent_lock_path, "parent_failure": failure_path,
        "fresh_language": language_path, "fixture_manifest": manifest_path,
        "source_archive": PROJECT_ROOT / parent_lock["source_archive"],
        "visible_catalog": PROJECT_ROOT / parent_lock["visible_catalog"],
        "fresh_population": PROJECT_ROOT / parent_lock["fresh_population"],
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "112r1-full-policy-aggregation-recovery-lock",
        "experiment": config["experiment"], "config_payload": config,
        "V112_config_payload": parent_lock["config_payload"],
        "baseline_config_payload": parent_lock["baseline_config_payload"],
        "preserved_access": failure["attempt"],
        "authorization": {
            "modify_V112_population_language_prompt_model_response_policy_metric_gate_or_decision": False,
            "run_one_aggregation_only_recovery": True,
            "load_or_generate_with_model": False,
            "manually_inspect_language_or_raw_response": False,
            "read_protected_test_run_API_train_or_execute": False,
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
