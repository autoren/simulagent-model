#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v151_local_proposal_query_ranking import evaluate, parse_proposal


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v151-local-proposal-query-ranking-lock.json"
    result_path = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/model-realization/result.json"
    access_path = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/model-realization/access.json"
    doc_path = PROJECT_ROOT / "docs/v151-local-proposal-query-ranking-results.md"
    audit_path = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v151-local-proposal-query-ranking-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v151_local_proposal_query_ranking_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V151 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V151 results document before freezing the outcome")

    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    hidden = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    answers = json.loads((PROJECT_ROOT / lock["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["interaction_catalog"]).read_text())
    witness_config = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    oracle_config = json.loads((PROJECT_ROOT / lock["oracle_config"]).read_text())
    expected = evaluate(result["fixtures"], hidden, answers, catalog, witness_config, oracle_config, access, config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    raw_fields = {"raw_response", "prompt", "payload", "conversation", "thinking_trace", "final_text"}
    reconstructed = []
    for row in result["fixtures"].values():
        if row["proposal_valid"]:
            reparsed = parse_proposal(json.dumps(row["normalized_proposal"]), catalog, config)
            reconstructed.append(
                reparsed["proposal_valid"]
                and all(
                    reparsed[key] == row[key]
                    for key in (
                        "validation_reason",
                        "normalized_proposal",
                        "evidence_status",
                        "candidate_state_ids",
                        "query_ranking",
                        "confidence",
                    )
                )
            )
        else:
            reconstructed.append(
                row["normalized_proposal"] is None
                and row["evidence_status"] == "NEEDS_CLARIFICATION"
                and row["candidate_state_ids"] == []
                and row["query_ranking"] == config["fallbackQueryRanking"]
                and row["confidence"] == 0.0
            )

    checks = {
        "preregistration_lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "completed_exact_development_request_realization": bool(
            result["completed_condition"]
            and len(result["fixtures"]) == config["population"]["requestFixtureCount"]
            and access["model_generation_count"] == config["population"]["requestFixtureCount"]
            and access["model_load_count"] == 1
            and access["tokenizer_load_count"] == 1
        ),
        "summary_and_decision_exact": result["summary"] == expected and result["decision"] == expected["decision"],
        "all_normalized_proposals_reconstruct_or_invalid_fail_closed": bool(reconstructed and all(reconstructed)),
        "no_raw_prompt_payload_conversation_trace_or_final_text_persisted": all(
            not (raw_fields & set(row)) and not row["raw_response_persisted"]
            for row in result["fixtures"].values()
        ),
        "all_resource_diagnostics_are_finite_and_bounded": all(
            isinstance(row["prompt_token_count"], int)
            and 0 < row["prompt_token_count"] <= config["prompt"]["maximumPromptTokens"]
            and isinstance(row["generated_token_count"], int)
            and 0 <= row["generated_token_count"] <= config["model"]["maximumNewTokens"]
            and isinstance(row["generation_seconds"], (int, float))
            and not isinstance(row["generation_seconds"], bool)
            and math.isfinite(row["generation_seconds"])
            and row["generation_seconds"] >= 0.0
            for row in result["fixtures"].values()
        ),
        "all_outputs_permanently_non_authoritative_nonexecuting": all(
            row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and not row["capability_defined_or_registered"]
            and not row["executable"]
            and row["actual_execution_count"] == 0
            for row in result["fixtures"].values()
        ),
        "zero_closed_answer_evaluation_retry_raw_API_training_services_side_effects_execution": bool(
            access["closed_answer_model_generation_count"] == 0
            and access["evaluation_fixture_model_generation_count"] == 0
            and access["retry_count"] == 0
            and access["manual_raw_response_inspection_count"] == 0
            and access["persisted_raw_response_count"] == 0
            and access["API_call_count"] == 0
            and access["training_run_count"] == 0
            and access["real_service_call_count"] == 0
            and access["external_side_effect_count"] == 0
            and access["actual_execution_count"] == 0
        ),
        "all_access_gates_pass": all(expected["access_gates"].values()),
        "confidence_remains_diagnostic_only": expected["calibration_diagnostics"]["confidence_is_diagnostic_not_fitted_or_authoritative"],
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "151-local-proposal-query-ranking-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "qualified": expected["qualified"],
        "decision": expected["decision"],
        "metrics": expected["metrics"],
        "calibration_diagnostics": expected["calibration_diagnostics"],
        "qualification_gates": expected["qualification_gates"],
        "access_gates": expected["access_gates"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "access": access_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "151-local-proposal-query-ranking-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "realization_completed": True,
            "qualified": expected["qualified"],
            "decision": expected["decision"],
            "metrics": expected["metrics"],
            "calibration_diagnostics": expected["calibration_diagnostics"],
            "qualification_gates": expected["qualification_gates"],
        },
        "authorization": {
            "retain_as_project_authored_synthetic_development_evidence_only": True,
            "preregister_separate_V149_evaluation_realization": expected["qualified"],
            "run_or_open_V149_evaluation_before_separate_preregistration": False,
            "modify_retry_rerun_reprompt_tune_threshold_fit_or_mine_V151": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
