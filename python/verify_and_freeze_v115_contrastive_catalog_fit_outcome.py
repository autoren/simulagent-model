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
from v109_open_world_typed_choice import validate_and_expand_choice
from v115_contrastive_catalog_fit import (
    classify_v115, evaluate_v115, reviewed_choice, validate_and_expand_contrastive,
)
from run_v115_contrastive_catalog_fit import payload_hash, read_jsonl


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v115-contrastive-catalog-fit-lock.json"
    result_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/model-contrastive/result.json"
    access_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/model-contrastive/access.json"
    doc_path = PROJECT_ROOT / "docs/v115-contrastive-catalog-fit-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v115_contrastive_catalog_fit_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v115-contrastive-catalog-fit-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V115 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V115 result document before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    config = lock["config_payload"]
    v112_config = lock["V112_config_payload"]
    language_path = PROJECT_ROOT / result["output_integrity"]["fresh_language"]["path"]
    records = read_jsonl(language_path)
    record_by_id = {row["record_id"]: row for row in records}
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    source_records, _ = parse_massive_archive(
        archive_bytes, config["extraction"]["expectedLocaleMemberSuffix"],
    )
    visible_catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training = build_declared_training_records(source_records, visible_catalog)
    retrieval_spec = lock["baseline_config_payload"]["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training, retrieval_spec["vectorizer"])
    retrieval = character_retrieval_observations(fitted, records)
    choice_catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())

    fixture_integrity = True
    for name, fixture in result["fixtures"].items():
        observed = name in record_by_id
        first = fixture["pass_one"]
        second = fixture["pass_two"]
        parsed_one, valid_one, reason_one = validate_and_expand_choice(
            first["raw_response"], choice_catalog, v112_config,
        )
        candidate = reviewed_choice(
            parsed_one, retrieval[name]["nearest_intent"] if observed else None,
            choice_catalog, observed,
        )
        parsed_two, evidence_two, valid_two, reason_two = validate_and_expand_contrastive(
            second["raw_response"], candidate, choice_catalog, config,
        )
        fixture_integrity = fixture_integrity and bool(
            fixture["name"] == name
            and fixture["kind"] == ("observed_fresh_contrastive" if observed else "controlled_missing_observation")
            and fixture["candidate_choice_id"] == candidate["choice_id"]
            and first["raw_response_sha256"] == hashlib.sha256(first["raw_response"].encode()).hexdigest()
            and second["raw_response_sha256"] == hashlib.sha256(second["raw_response"].encode()).hexdigest()
            and first["parsed_response"] == parsed_one and first["response_valid"] == valid_one
            and first["validation_reason"] == reason_one
            and second["parsed_response"] == parsed_two and second["evidence"] == evidence_two
            and second["response_valid"] == valid_two and second["validation_reason"] == reason_two
            and fixture["permanently_non_authoritative"]
            and not fixture["safe_hypothesis_universe_pruned"]
            and not fixture["capability_defined"] and not fixture["executable"]
        )

    summary = evaluate_v115(
        records, result["fixtures"], retrieval, access, v112_config, config,
        lock["baseline_config_payload"],
    )
    flags = classify_v115(summary)
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "V112_lock", "source_inventory",
        "source_archive", "V101_population", "V112_population", "V114_population",
        "visible_catalog", "choice_catalog", "model_manifest", "baseline_lock", "V109_result",
        "fresh_population", "plan", "protocol", "tests", "runner", "verifier", "auditor",
        "design_audit",
    )
    population = json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())
    selected_ids = {row["candidate_id"] for row in population["selected_population"]}
    excluded_ids: set[str] = set()
    for key in ("V101_population", "V112_population", "V114_population"):
        excluded = json.loads((PROJECT_ROOT / lock[key]).read_text())
        excluded_ids.update(row["candidate_id"] for row in excluded["selected_population"])
    expected_fixture_count = config["condition"]["observedFixtureCount"] + config["condition"]["controlledMissingObservationCount"]
    checks = {
        "lock_and_all_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "fresh_population_and_language_are_exact_test_only_and_disjoint": bool(
            not (selected_ids & excluded_ids)
            and all(row["source_partition"] == "test" for row in population["selected_population"])
            and file_sha256(language_path) == result["output_integrity"]["fresh_language"]["sha256"] == result["fresh_language_sha256"]
        ),
        "both_passes_revalidate_and_all_authority_boundaries_hold": fixture_integrity,
        "independent_summary_flags_and_decision_reconstruct_exactly": bool(
            summary == result["summary"]
            and all(result[key] == value for key, value in flags.items())
            and result["decision"] == flags["decision"]
            and not flags["schema_induction_authorized"]
        ),
        "one_model_load_and_exactly_480_no_retry_generations_completed": bool(
            result["completed_condition"] and len(result["fixtures"]) == expected_fixture_count
            and access["model_load_count"] == 1
            and access["model_generation_count"] == config["condition"]["totalGenerationCount"] == 480
            and config["condition"]["retryCount"] == 0
        ),
        "zero_protected_manual_API_training_induction_service_effect_and_execution": bool(
            all(access[key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            ))
            and summary["actual_execution_count"] == 0
            and summary["true_hypothesis_retention"] == 1.0
            and summary["individual_evidence_emission_count"] == 0
            and not lock["authorization"]["begin_schema_induction_or_richer_sequential_planning"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "115-contrastive-catalog-fit-outcome-audit",
        "experiment": config["experiment"], "passed": passed,
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
        "schema_version": "115-contrastive-catalog-fit-outcome-lock",
        "experiment": "v115_two_pass_contrastive_catalog_fit_outcome_lock",
        "outcome": {"passed": True, **flags, "summary": summary},
        "authorization": {
            "modify_rerun_retry_or_retune_V115": False,
            "seek_genuinely_independent_confirmation_source": flags["seek_independent_source_transfer"],
            "open_original_protected_test": False,
            "begin_schema_or_capability_induction": False,
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
