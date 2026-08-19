#!/usr/bin/env python3
"""Independently verify and freeze the V91 rank-only outcome."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v91_rank_only_protocol import (
    aggregate_model_rows,
    evaluate_gates,
    score_response,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v91-rank-only-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v91-rank-only/evaluation/result.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v91_outcome.py"
    document_path = PROJECT_ROOT / "docs/v91-rank-only-results.md"
    research_path = PROJECT_ROOT / "docs/research-direction.md"
    audit_path = PROJECT_ROOT / "outputs/v91-rank-only/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v91-rank-only-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V91 outcome is already frozen")

    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    config = implementation["config_payload"]
    result = json.loads(result_path.read_text())
    planner = json.loads(
        (PROJECT_ROOT / implementation["planner_invariance_result"]).read_text()
    )
    corpus_path = PROJECT_ROOT / implementation["corpus"]
    records = [
        json.loads(line) for line in corpus_path.read_text().splitlines() if line
    ]
    records_by_id = {record["id"]: record for record in records}

    raw_dir = result_path.parent / "raw-fixtures"
    raw_paths = sorted(raw_dir.glob("*.json"))
    raw_rows: dict[str, dict[str, Any]] = {}
    raw_artifacts: list[dict[str, str]] = []
    for raw_path in raw_paths:
        row = json.loads(raw_path.read_text())
        raw_rows[row["id"]] = row
        raw_artifacts.append(
            {
                "path": str(raw_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(raw_path),
            }
        )

    fixture_rows = list(result["fixtures"].values())
    reconstructed_metrics = aggregate_model_rows(fixture_rows, records)
    reconstructed_gates = evaluate_gates(
        reconstructed_metrics, planner, config, result["final_access"]
    )
    rescored_exact = True
    stable_score_keys = (
        "id",
        "source_record_id",
        "service",
        "label_kind",
        "response",
        "parsed",
        "exact_json",
        "exact_keys",
        "list_well_formed",
        "raw_allowed_only",
        "raw_unique",
        "raw_full_permutation",
        "raw_priority",
        "completed_priority",
        "canonical_complete_set",
        "canonical_NONE_retained",
        "gold_intent",
        "gold_rank",
        "top1",
        "top2",
        "reciprocal_rank",
        "candidate_count",
        "authoritative_state_fingerprint_before",
        "authoritative_state_fingerprint_after",
        "authoritative_state_preserved",
        "permanently_non_deployable",
        "executable",
        "belief_authority",
        "action_authority",
        "pruning_authority",
    )
    for row in fixture_rows:
        rescored = score_response(records_by_id[row["id"]], row["response"])
        if any(rescored[key] != row[key] for key in stable_score_keys):
            rescored_exact = False
            break

    active = [row for row in fixture_rows if row["label_kind"] == "active"]
    none = [row for row in fixture_rows if row["label_kind"] == "none"]
    service_role_counts = Counter(
        f"{row['service']}::{row['label_kind']}::top1_{str(row['top1']).lower()}"
        for row in fixture_rows
    )
    access = result["final_access"]
    dependency_keys = (
        "design_lock",
        "corpus_seal",
        "corpus",
        "protocol",
        "tests",
        "runner",
        "census_harness",
        "model_manifest",
        "planner_outcome_lock",
        "planner_implementation_lock",
        "planner_invariance_result",
        "implementation_auditor",
        "implementation_audit",
    )
    grammar = reconstructed_metrics["controls"]["identifier_exact_match_grammar"]
    checks = {
        "implementation_lock_and_every_frozen_dependency_are_exact": bool(
            payload_hash(implementation_payload)
            == implementation["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / implementation[key])
                == implementation[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "result_raw_fixtures_and_embedded_fixtures_are_exact": bool(
            len(raw_paths) == 64
            and len(result["fixtures"]) == 64
            and raw_rows == result["fixtures"]
        ),
        "every_response_rescores_and_all_metrics_and_gates_reconstruct": bool(
            rescored_exact
            and reconstructed_metrics == result["metrics"]
            and reconstructed_gates == result["gates"]
        ),
        "condition_completed_once_but_failed_registered_utility_gates": bool(
            result["completed_condition"]
            and result["quality_gate_pass"] is False
            and result["passed"] is False
            and result["decision"]
            == "freeze_nonqualifying_ranker_and_retain_deterministic_exhaustive_order"
        ),
        "canonical_completion_state_and_authority_safety_are_perfect": bool(
            sum(row["canonical_complete_set"] for row in fixture_rows) == 64
            and sum(row["canonical_NONE_retained"] for row in fixture_rows) == 64
            and sum(row["authoritative_state_preserved"] for row in fixture_rows)
            == 64
            and all(not row["executable"] for row in fixture_rows)
            and all(not row["belief_authority"] for row in fixture_rows)
            and all(not row["action_authority"] for row in fixture_rows)
            and all(not row["pruning_authority"] for row in fixture_rows)
        ),
        "exact_planner_permutation_invariance_and_certification_reconstruct": bool(
            planner == result["planner_invariance"]
            and planner["fixture_count"] == 4
            and planner["permutation_count"] == 480
            and planner["invariant_permutation_count"] == 480
            and planner["invariance_rate"] == 1.0
            and planner["action_mismatch_count"] == 0
            and planner["maximum_absolute_value_error"] <= 1e-12
            and planner["execution_certificate_violation_count"] == 0
            and planner["model_output_access_count"] == 0
        ),
        "registered_model_ranking_counts_reconstruct": bool(
            len(active) == 32
            and len(none) == 32
            and sum(row["top1"] for row in active) == 25
            and sum(row["top1"] for row in none) == 5
            and sum(row["top1"] for row in fixture_rows) == 30
            and sum(row["top2"] for row in fixture_rows) == 38
            and sum(row["raw_full_permutation"] for row in fixture_rows) == 10
            and close(reconstructed_metrics["mean_reciprocal_rank"], 2 / 3)
            and close(reconstructed_metrics["mean_gold_rank"], 1.9375)
        ),
        "best_nonoracle_control_beats_model_on_both_registered_comparisons": bool(
            reconstructed_metrics["best_nonoracle_MRR_control"]
            == "identifier_exact_match_grammar"
            and reconstructed_metrics["best_nonoracle_mean_rank_control"]
            == "identifier_exact_match_grammar"
            and close(grammar["mean_reciprocal_rank"], 0.7526041666666666)
            and close(grammar["mean_gold_rank"], 1.609375)
            and close(
                reconstructed_metrics[
                    "MRR_improvement_over_best_nonoracle_control"
                ],
                -0.0859375,
            )
            and close(
                reconstructed_metrics[
                    "mean_rank_reduction_versus_best_nonoracle_control"
                ],
                -0.328125,
            )
        ),
        "access_is_one_local_load_64_generations_and_zero_expansive_use": bool(
            access["model_load_count"] == 1
            and access["model_generation_count"] == 64
            and access["new_model_weight_download_count"] == 0
            and access["LLM_API_call_count"] == 0
            and access["adapter_training_run_count"] == 0
            and access["manual_utterance_inspection_count"] == 0
            and access["pruned_hypothesis_count"] == 0
            and access["early_stopping_count"] == 0
            and access["belief_update_from_model_count"] == 0
            and access["action_selection_from_model_count"] == 0
            and access["real_service_call_count"] == 0
            and access["external_side_effect_count"] == 0
        ),
        "results_document_and_research_direction_exist": bool(
            document_path.is_file() and research_path.is_file()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "91-rank-only-outcome-audit",
        "experiment": "v91_rank_only_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_verified_V91_nonqualifying_ranker"
            if passed
            else "reject_V91_outcome"
        ),
        "checks": checks,
        "verified_metrics": reconstructed_metrics,
        "service_role_top1_counts": dict(sorted(service_role_counts.items())),
        "additional_access": {
            "source_language_record_access_count": 64,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": result["claim_boundary"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "91-rank-only-outcome-lock",
        "experiment": "v91_rank_only_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "raw_fixture_artifacts": raw_artifacts,
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "results_document": str(document_path.relative_to(PROJECT_ROOT)),
        "results_document_sha256": file_sha256(document_path),
        "research_direction": str(research_path.relative_to(PROJECT_ROOT)),
        "research_direction_sha256": file_sha256(research_path),
        "outcome": {
            "passed": True,
            "condition_qualified": False,
            "decision": result["decision"],
            "metrics": reconstructed_metrics,
            "planner_invariance": planner,
            "access": access,
        },
        "authorization": {
            "modify_or_rerun_V91": False,
            "retain_complete_deterministic_schema_enumeration": True,
            "retain_operational_NONE_and_immutable_authoritative_state": True,
            "use_local_model_as_candidate_generator_or_search_scheduler": False,
            "prune_or_early_stop_search": False,
            "add_larger_local_or_API_model_for_this_branch": False,
            "train_adapter_or_learn_likelihood_for_this_branch": False,
            "grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
            "report_and_synthesize_frozen_results": True
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
