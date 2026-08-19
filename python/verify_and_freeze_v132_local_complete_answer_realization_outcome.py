#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v132_local_complete_answer_realization import evaluate_realization


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v132-local-complete-answer-realization-lock.json"
    result_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/model-realization/result.json"
    access_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/model-realization/access.json"
    language_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/selected-language/records.jsonl"
    prompt_catalog_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/selected-language/prompt-choice-catalog.json"
    doc_path = PROJECT_ROOT / "docs/v132-local-complete-answer-realization-results.md"
    audit_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v132-local-complete-answer-realization-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v132_local_complete_answer_realization_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V132 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V132 results document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text()); access = json.loads(access_path.read_text())
    config = lock["config_payload"]
    population = json.loads((PROJECT_ROOT / lock["fixture_population"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    v130 = json.loads((PROJECT_ROOT / lock["V130_config"]).read_text())
    expected = evaluate_realization(population, result["fixtures"], catalog, baseline, v130, access, config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)),
        "condition_completed_exactly_once": bool(result["completed_condition"] and len(result["fixtures"]) == 264 and access["model_load_count"] == 1 and access["model_generation_count"] == 264),
        "independent_aggregate_recomputation_exact": result["summary"] == expected,
        "decision_exact": result["decision"] == expected["decision"],
        "selected_artifact_integrity": bool(result["output_integrity"]["selected_language"]["sha256"] == file_sha256(language_path) and result["output_integrity"]["prompt_choice_catalog"]["sha256"] == file_sha256(prompt_catalog_path)),
        "access_and_authority_hold": bool(expected["access_pass"] and expected["true_hypothesis_retention"] == 1.0 and expected["actual_execution_count"] == 0),
    }
    passed = all(checks.values())
    audit = {"schema_version": "132-local-complete-answer-realization-outcome-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "recomputed_summary": expected}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "result": result_path, "access": access_path, "selected_language": language_path, "prompt_choice_catalog": prompt_catalog_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    positive = expected["outcome_pass"]
    outcome: dict[str, Any] = {
        "schema_version": "132-local-complete-answer-realization-outcome-lock",
        "experiment": "v132_local_complete_answer_realization_outcome_lock",
        "outcome": {"passed": True, "audit_pass": True, "evidence_pass": expected["evidence_pass"], "downstream_pass": expected["downstream_pass"], "access_pass": expected["access_pass"], "realization_pass": positive, "decision": expected["decision"], "summary": expected},
        "authorization": {
            "modify_rerun_retry_retune_or_mine_V132": False,
            "preregister_independent_external_confirmation": positive,
            "claim_human_equivalence_or_repeated_sample_independence": False,
            "open_protected_or_begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
