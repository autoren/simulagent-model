#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v151_local_proposal_query_ranking import evaluate, parse_proposal
from v151r1_local_proposal_query_ranking_recovery import interrupted_fail_closed, recovery_evaluation_config


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v151r1-local-proposal-query-ranking-recovery-lock.json"
    result_path = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/model-recovery/result.json"
    access_path = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/model-recovery/access.json"
    doc_path = PROJECT_ROOT / "docs/v151r1-local-proposal-query-ranking-recovery-results.md"
    audit_path = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v151r1-local-proposal-query-ranking-recovery-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v151r1_local_proposal_query_ranking_recovery_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V151r1 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V151r1 results document before freezing the outcome")

    lock = json.loads(lock_path.read_text())
    recovery = lock["recovery_config_payload"]
    base = lock["base_V151_config_payload"]
    evaluation_config = recovery_evaluation_config(base, recovery)
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    hidden = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    answers = json.loads((PROJECT_ROOT / lock["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["interaction_catalog"]).read_text())
    witness = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    oracle = json.loads((PROJECT_ROOT / lock["oracle_config"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["retained_partial_manifest"]).read_text())
    expected = evaluate(result["fixtures"], hidden, answers, catalog, witness, oracle, access, evaluation_config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]

    retained_exact = []
    for record in manifest["persisted_records"]:
        source_path = PROJECT_ROOT / record["path"]
        retained_exact.append(
            file_sha256(source_path) == record["sha256"]
            and result["fixtures"].get(record["fixture_id"]) == json.loads(source_path.read_text())
        )
    interrupted_expected = interrupted_fail_closed(manifest["interrupted_fixture_id"], catalog, base)
    recovery_artifact_checks = []
    recovery_dir = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/model-recovery/raw-fixtures"
    for fixture_id in manifest["never_started_fixture_ids"]:
        ordinal = manifest["all_fixture_ids"].index(fixture_id)
        path = recovery_dir / f"{ordinal:03d}-{fixture_id}.json"
        recovery_artifact_checks.append(path.is_file() and result["fixtures"].get(fixture_id) == json.loads(path.read_text()))

    reconstructed = []
    for fixture_id, row in result["fixtures"].items():
        if fixture_id == manifest["interrupted_fixture_id"]:
            reconstructed.append(row == interrupted_expected)
        elif row["proposal_valid"]:
            reparsed = parse_proposal(json.dumps(row["normalized_proposal"]), catalog, base)
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
                and row["query_ranking"] == base["fallbackQueryRanking"]
                and row["confidence"] == 0.0
            )
    raw_fields = {"raw_response", "prompt", "payload", "conversation", "thinking_trace", "final_text"}
    regular_rows = [row for key, row in result["fixtures"].items() if key != manifest["interrupted_fixture_id"]]
    checks = {
        "recovery_lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "completed_exact_58_plus_1_plus_37_census": bool(
            result["completed_condition"]
            and len(result["fixtures"]) == 96
            and result["retained_fixture_count"] == 58
            and result["technical_fail_closed_fixture_count"] == 1
            and result["recovery_generated_fixture_count"] == 37
            and set(result["fixtures"]) == set(manifest["all_fixture_ids"])
        ),
        "all_58_retained_outputs_byte_source_exact": bool(retained_exact and all(retained_exact)),
        "interrupted_fixture_exact_registered_fail_closed_output": result["fixtures"].get(manifest["interrupted_fixture_id"]) == interrupted_expected,
        "all_37_recovery_outputs_have_exact_durable_artifacts": bool(recovery_artifact_checks and all(recovery_artifact_checks)),
        "summary_and_decision_exact": result["summary"] == expected and result["decision"] == expected["decision"],
        "all_normalized_proposals_reconstruct_or_invalid_fail_closed": bool(reconstructed and all(reconstructed)),
        "exact_total_and_recovery_access_counts": bool(
            access["model_generation_count"] == 96
            and access["recovery_model_generation_count"] == 37
            and access["prior_interrupted_attempt_count"] == 1
            and access["model_load_count"] == 2
            and access["tokenizer_load_count"] == 2
            and access["recovery_model_load_count"] == 1
            and access["technical_fail_closed_fixture_count"] == 1
        ),
        "no_raw_prompt_payload_conversation_trace_or_final_text_persisted": all(
            not (raw_fields & set(row)) and not row["raw_response_persisted"]
            for row in result["fixtures"].values()
        ),
        "regular_resource_diagnostics_finite_and_bounded": all(
            isinstance(row["prompt_token_count"], int)
            and 0 < row["prompt_token_count"] <= base["prompt"]["maximumPromptTokens"]
            and isinstance(row["generated_token_count"], int)
            and 0 <= row["generated_token_count"] <= base["model"]["maximumNewTokens"]
            and isinstance(row["generation_seconds"], (int, float))
            and not isinstance(row["generation_seconds"], bool)
            and math.isfinite(row["generation_seconds"])
            and row["generation_seconds"] >= 0.0
            for row in regular_rows
        ),
        "all_outputs_permanently_non_authoritative_nonexecuting": all(
            row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and not row["capability_defined_or_registered"]
            and not row["executable"]
            and row["actual_execution_count"] == 0
            for row in result["fixtures"].values()
        ),
        "zero_closed_answer_evaluation_retry_raw_API_training_services_side_effects_execution": all(
            access[key] == 0
            for key in (
                "closed_answer_model_generation_count",
                "evaluation_fixture_model_generation_count",
                "retry_count",
                "manual_raw_response_inspection_count",
                "persisted_raw_response_count",
                "API_call_count",
                "training_run_count",
                "real_service_call_count",
                "external_side_effect_count",
                "actual_execution_count",
            )
        ),
        "all_recovery_access_gates_pass": all(expected["access_gates"].values()),
        "confidence_remains_diagnostic_only": expected["calibration_diagnostics"]["confidence_is_diagnostic_not_fitted_or_authoritative"],
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "151r1-local-proposal-query-ranking-recovery-outcome-audit",
        "experiment": recovery["experiment"],
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
        "schema_version": "151r1-local-proposal-query-ranking-recovery-outcome-lock",
        "experiment": recovery["experiment"],
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "recovery_completed": True,
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
            "modify_retry_rerun_reprompt_tune_threshold_fit_or_mine_V151r1": False,
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
