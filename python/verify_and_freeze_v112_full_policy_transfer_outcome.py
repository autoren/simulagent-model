#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v106_open_world_benchmark import (
    build_declared_training_records, character_retrieval_observations, fit_character_retrieval,
)
from run_v112_open_world_full_policy_transfer import (
    decision_for, evaluate_policy_outputs, payload_hash, read_jsonl,
)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v112-open-world-full-policy-transfer-lock.json"
    result_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/model-policy-transfer/result.json"
    access_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/model-policy-transfer/access.json"
    doc_path = PROJECT_ROOT / "docs/v112-open-world-full-policy-transfer-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v112_full_policy_transfer_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v112-open-world-full-policy-transfer-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V112 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V112 result document before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    language_path = PROJECT_ROOT / result["output_integrity"]["fresh_language"]["path"]
    records = read_jsonl(language_path)
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    source_records, _ = parse_massive_archive(
        archive_bytes, lock["config_payload"]["extraction"]["expectedLocaleMemberSuffix"],
    )
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training = build_declared_training_records(source_records, catalog)
    retrieval_spec = lock["baseline_config_payload"]["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training, retrieval_spec["vectorizer"])
    retrieval = character_retrieval_observations(fitted, records)
    summary = evaluate_policy_outputs(
        records, result["fixtures"], retrieval, access,
        lock["config_payload"], lock["baseline_config_payload"],
    )
    quality_pass = all(summary["quality_gates"].values())
    novel_names = (
        "novel_evidence_precision", "novel_evidence_recall",
        "novel_evidence_non_novel_false_positive_rate", "novel_evidence_ECE",
    )
    novel_pass = all(summary["quality_gates"][key] for key in novel_names)
    access_pass = all(summary["access_gates"].values())
    expected_decision = decision_for(quality_pass, novel_pass, access_pass)
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "source_inventory", "source_archive",
        "excluded_population", "visible_catalog", "choice_catalog", "model_manifest",
        "baseline_lock", "V109_result", "fresh_population", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    fixture_checks = all(
        row["raw_response_sha256"] == hashlib.sha256(row["raw_response"].encode()).hexdigest()
        and row["permanently_non_authoritative"] and not row["safe_hypothesis_universe_pruned"]
        and not row["capability_defined"] and not row["executable"]
        for row in result["fixtures"].values()
    )
    checks = {
        "lock_and_all_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "fresh_language_identity_is_exact": file_sha256(language_path) == result["output_integrity"]["fresh_language"]["sha256"] == result["fresh_language_sha256"],
        "all_fixture_responses_and_authority_boundaries_are_exact": fixture_checks,
        "summary_quality_access_and_decision_reconstruct_exactly": bool(
            summary == result["summary"] and quality_pass == result["quality_pass"]
            and novel_pass == result["novel_evidence_pass"] and access_pass == result["access_pass"]
            and expected_decision == result["decision"]
        ),
        "condition_completed_exactly_once": bool(
            result["completed_condition"] and len(result["fixtures"]) == 240
            and access["model_load_count"] == 1 and access["model_generation_count"] == 240
        ),
        "zero_protected_manual_API_training_service_effect_and_execution": bool(
            all(access[key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            ))
            and summary["actual_execution_count"] == 0 and summary["true_hypothesis_retention"] == 1.0
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "112-open-world-full-policy-transfer-outcome-audit",
        "experiment": "v112_fresh_massive_non_authoritative_novelty_evidence_policy_outcome_audit",
        "passed": passed, "quality_gate_pass": quality_pass, "novel_evidence_pass": novel_pass,
        "decision": expected_decision, "checks": checks, "independent_summary": summary,
        "additional_access": {
            "fresh_development_language_read_count": 1, "source_archive_read_count": 1,
            "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "analysis_lock": lock_path, "result": result_path, "access": access_path,
        "fresh_language": language_path, "verifier": verifier_path,
        "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "112-open-world-full-policy-transfer-outcome-lock",
        "experiment": "v112_fresh_massive_non_authoritative_novelty_evidence_policy_outcome_lock",
        "outcome": {
            "passed": True, "quality_gate_pass": quality_pass,
            "novel_evidence_pass": novel_pass, "decision": expected_decision,
            "summary": summary,
        },
        "authorization": {
            "modify_rerun_retry_or_retune_V112": False,
            "preregister_protected_test_confirmation": bool(quality_pass),
            "seek_new_contrastive_or_multiturn_evidence": not novel_pass,
            "redesign_policy_on_new_population": bool(novel_pass and not quality_pass),
            "read_protected_test_before_separate_lock": False,
            "proceed_to_schema_or_mechanic_induction": False,
            "proceed_to_richer_sequential_decision_problem": False,
            "run_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
