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
from v114_rescued_policy_transfer import evaluate_transfer
from run_v114_rescued_policy_transfer import payload_hash, read_jsonl, transfer_flags


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v114-rescued-policy-transfer-lock.json"
    result_path = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/model-policy-transfer/result.json"
    access_path = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/model-policy-transfer/access.json"
    doc_path = PROJECT_ROOT / "docs/v114-rescued-policy-transfer-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v114_rescued_policy_transfer_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v114-rescued-policy-transfer-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V114 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V114 result document before freezing")

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
    summary = evaluate_transfer(
        records, result["fixtures"], fitted, retrieval, access,
        lock["V112_config_payload"], lock["config_payload"], lock["baseline_config_payload"],
    )
    flags = transfer_flags(summary)
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "V112r1_lock", "V112_lock",
        "source_inventory", "source_archive", "V101_population", "V112_population",
        "visible_catalog", "choice_catalog", "model_manifest", "baseline_lock", "V109_result",
        "fresh_population", "plan", "protocol", "tests", "runner", "verifier", "auditor",
        "design_audit",
    )
    fixture_checks = all(
        row["raw_response_sha256"] == hashlib.sha256(row["raw_response"].encode()).hexdigest()
        and row["permanently_non_authoritative"] and not row["safe_hypothesis_universe_pruned"]
        and not row["capability_defined"] and not row["executable"]
        for row in result["fixtures"].values()
    )
    population = json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())
    v101 = json.loads((PROJECT_ROOT / lock["V101_population"]).read_text())
    v112 = json.loads((PROJECT_ROOT / lock["V112_population"]).read_text())
    selected_ids = {row["candidate_id"] for row in population["selected_population"]}
    excluded_ids = {
        row["candidate_id"]
        for source in (v101, v112) for row in source["selected_population"]
    }
    checks = {
        "lock_and_all_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "fresh_population_and_language_identity_are_exact_and_disjoint": bool(
            not (selected_ids & excluded_ids)
            and all(row["source_partition"] == "test" for row in population["selected_population"])
            and file_sha256(language_path) == result["output_integrity"]["fresh_language"]["sha256"] == result["fresh_language_sha256"]
        ),
        "all_fixture_responses_and_authority_boundaries_are_exact": fixture_checks,
        "paired_summary_flags_and_decision_reconstruct_exactly": bool(
            summary == result["summary"]
            and all(result[key] == value for key, value in flags.items())
            and summary["paired_rescue_diagnostics"]["individual_record_emission_count"] == 0
            and summary["paired_rescue_diagnostics"]["novel_evidence_exactly_unchanged"]
        ),
        "condition_completed_exactly_once_and_shared_by_both_policies": bool(
            result["completed_condition"] and len(result["fixtures"]) == 240
            and access["model_load_count"] == 1 and access["model_generation_count"] == 240
            and lock["authorization"]["feed_each_model_response_to_both_paired_policies"]
        ),
        "zero_protected_manual_API_training_service_effect_and_execution": bool(
            all(access[key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            ))
            and summary["actual_execution_count"] == 0
            and summary["true_hypothesis_retention"] == 1.0
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "114-rescued-policy-transfer-outcome-audit",
        "experiment": lock["config_payload"]["experiment"], "passed": passed,
        **flags, "checks": checks, "independent_summary": summary,
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
        "schema_version": "114-rescued-policy-transfer-outcome-lock",
        "experiment": "v114_record_disjoint_massive_rescued_policy_transfer_outcome_lock",
        "outcome": {
            "passed": True, **flags,
            "full_policy_qualification": {
                "V112_baseline_pass": flags["base_policy_pass"],
                "V114_rescued_pass": flags["rescued_policy_pass"],
            },
            "rescue_mechanism_qualification": {
                "status": flags["mechanism_status"],
                "opportunity_sufficient": flags["opportunity_sufficient"],
                "diagnostics": summary["paired_rescue_diagnostics"],
            },
            "summary": summary,
        },
        "authorization": {
            "modify_rerun_retry_or_retune_V114": False,
            "preregister_sandboxed_typed_induction_feasibility": flags["preregister_sandboxed_typed_induction_feasibility"],
            "proceed_immediately_to_schema_or_mechanic_induction": False,
            "open_original_protected_test": False,
            "proceed_to_richer_sequential_decision_problem": False,
            "run_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
