#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v112_open_world_full_policy_transfer import payload_hash
from run_v113_known_disagreement_rescue_census import reconstruct


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v113-known-disagreement-rescue-census-lock.json"
    result_path = PROJECT_ROOT / "outputs/v113-known-disagreement-rescue-census/historical-census/result.json"
    doc_path = PROJECT_ROOT / "docs/v113-known-disagreement-rescue-census-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v113_rescue_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v113-known-disagreement-rescue-census/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v113-known-disagreement-rescue-census-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V113 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V113 result document before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    census, metadata = reconstruct(lock)
    feasible = census["feasible_candidate_count"] > 0
    expected_decision = (
        "freeze_selected_rescue_for_separately_locked_new_population_transfer"
        if feasible else "close_simple_disagreement_rescue_family_require_new_evidence_or_policy_structure"
    )
    dependency_keys = (
        "config", "parent_outcome", "V112r1_lock", "V112r1_result", "fresh_language",
        "fixture_manifest", "source_archive", "visible_catalog", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "census_metadata_feasibility_and_decision_reconstruct_exactly": bool(
            census == result["analysis"] and metadata == result["metadata"]
            and feasible == result["feasible_rescue_exists"] and expected_decision == result["decision"]
        ),
        "aggregate_only_no_individual_emission": census["individual_feature_prediction_identifier_language_or_response_emission_count"] == 0,
        "zero_protected_manual_model_API_training_service_effect_and_execution": all(
            result["access"][key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "model_load_count", "model_generation_count", "LLM_API_call_count",
                "adapter_training_run_count", "real_service_call_count", "external_side_effect_count",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "113-known-disagreement-rescue-census-outcome-audit",
        "experiment": lock["config_payload"]["experiment"], "passed": passed,
        "feasible_rescue_exists": feasible, "decision": expected_decision,
        "checks": checks, "independent_analysis": census, "independent_metadata": metadata,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "analysis_lock": lock_path, "result": result_path, "verifier": verifier_path,
        "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "113-known-disagreement-rescue-census-outcome-lock",
        "experiment": "v113_known_disagreement_rescue_census_outcome_lock",
        "outcome": {
            "passed": True, "feasible_rescue_exists": feasible,
            "decision": expected_decision, "selected": census["selected"],
            "feasible_candidate_count": census["feasible_candidate_count"],
        },
        "authorization": {
            "modify_rerun_or_retune_V113": False,
            "preregister_selected_rescue_on_new_disjoint_population": bool(feasible),
            "seek_new_evidence_or_policy_structure": not feasible,
            "read_protected_test_before_separate_lock": False,
            "proceed_to_schema_or_mechanic_induction": False,
            "proceed_to_richer_sequential_decision_problem": False,
            "run_additional_local_or_API_model_or_train_adapter": False,
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
