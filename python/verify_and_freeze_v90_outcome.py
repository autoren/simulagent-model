#!/usr/bin/env python3
"""Independently verify and freeze the V90 comparison outcome."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v90_capacity_generation_protocol import (
    aggregate,
    evaluate_condition_gates,
    paired_correctness_transitions,
    quality_gate_pass,
)


CONDITION_IDS = (
    "qwen35_4b_4bit",
    "qwen35_27b_4bit",
    "qwen38_27b_4bit",
    "qwen38_27b_8bit",
)

CONTRASTS = {
    "capacity_qwen35_4b_to_27b": ("qwen35_4b_4bit", "qwen35_27b_4bit"),
    "generation_qwen35_to_qwen38_27b_4bit": (
        "qwen35_27b_4bit",
        "qwen38_27b_4bit",
    ),
    "quantization_qwen38_4bit_to_8bit": (
        "qwen38_27b_4bit",
        "qwen38_27b_8bit",
    ),
}

SCALAR_METRICS = (
    "exact_JSON_parse_rate",
    "ontology_conformance_rate",
    "mandatory_NONE_inclusion_rate",
    "gold_active_intent_coverage_rate",
    "intent_candidate_set_exact_rate",
    "none_only_intent_exact_rate",
    "state_slot_key_recall",
    "state_slot_key_exact_rate",
    "mean_intent_candidate_count",
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def active_complementarity(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, int]:
    counts = {
        "both_correct": 0,
        "left_only_correct": 0,
        "right_only_correct": 0,
        "neither_correct": 0,
    }
    for record_id in sorted(left):
        if left[record_id]["active_intent"] == "NONE":
            continue
        pair = (
            bool(left[record_id]["gold_active_intent_covered"]),
            bool(right[record_id]["gold_active_intent_covered"]),
        )
        key = {
            (True, True): "both_correct",
            (True, False): "left_only_correct",
            (False, True): "right_only_correct",
            (False, False): "neither_correct",
        }[pair]
        counts[key] += 1
    return counts


def main() -> None:
    implementation_path = (
        PROJECT_ROOT / "configs/v90-capacity-generation-implementation-lock.json"
    )
    comparison_path = (
        PROJECT_ROOT / "outputs/v90-capacity-generation/comparison/result.json"
    )
    analyzer_path = PROJECT_ROOT / "python/analyze_v90_capacity_generation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v90_outcome.py"
    document_path = PROJECT_ROOT / "docs/v90-capacity-generation-results.md"
    research_path = PROJECT_ROOT / "docs/research-direction.md"
    audit_path = PROJECT_ROOT / "outputs/v90-capacity-generation/outcome-audit.json"
    failed_attempt_path = (
        PROJECT_ROOT
        / "outputs/v90-capacity-generation/outcome-audit-attempt-0.json"
    )
    lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V90 outcome is already frozen")

    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    comparison = json.loads(comparison_path.read_text())
    config = implementation["config_payload"]
    failed_attempt = json.loads(failed_attempt_path.read_text())

    results: dict[str, dict[str, Any]] = {}
    result_paths: dict[str, Path] = {}
    raw_fixture_artifacts: list[dict[str, str]] = []
    raw_exact = True
    for condition_id in CONDITION_IDS:
        path = (
            PROJECT_ROOT
            / "outputs/v90-capacity-generation/evaluation"
            / condition_id
            / "result.json"
        )
        result = json.loads(path.read_text())
        results[condition_id] = result
        result_paths[condition_id] = path
        raw_dir = path.parent / "raw-fixtures"
        raw_paths = sorted(raw_dir.glob("*.json"))
        if len(raw_paths) != 48:
            raw_exact = False
        raw_rows: dict[str, dict[str, Any]] = {}
        for raw_path in raw_paths:
            raw_row = json.loads(raw_path.read_text())
            raw_rows[raw_row["id"]] = raw_row
            raw_fixture_artifacts.append(
                {
                    "path": str(raw_path.relative_to(PROJECT_ROOT)),
                    "sha256": file_sha256(raw_path),
                }
            )
        if raw_rows != result["fixtures"]:
            raw_exact = False

    reconstructed_metrics: dict[str, dict[str, Any]] = {}
    reconstructed_gates: dict[str, dict[str, bool]] = {}
    reconstruction_exact = True
    for condition_id, result in results.items():
        rows = list(result["fixtures"].values())
        records = {
            row["id"]: {
                "id": row["id"],
                "service": row["service"],
                "gold": row["gold"],
                "allowed_intent_ids": row["allowed_intent_ids"],
                "allowed_slot_ids": row["allowed_slot_ids"],
            }
            for row in rows
        }
        metrics = aggregate(rows, records)
        gates = evaluate_condition_gates(metrics, config, result["final_access"])
        reconstructed_metrics[condition_id] = metrics
        reconstructed_gates[condition_id] = gates
        reconstruction_exact = bool(
            reconstruction_exact
            and metrics == result["metrics"]
            and gates == result["gates"]
            and quality_gate_pass(gates) == result["quality_gate_pass"]
        )

    contrast_exact = True
    for contrast_name, (left_id, right_id) in CONTRASTS.items():
        observed = comparison["contrasts"][contrast_name]
        left = results[left_id]
        right = results[right_id]
        deltas = {
            metric: float(right["metrics"][metric])
            - float(left["metrics"][metric])
            for metric in SCALAR_METRICS
        }
        contrast_exact = bool(
            contrast_exact
            and observed["left_condition"] == left_id
            and observed["right_condition"] == right_id
            and observed["right_minus_left_metric_deltas"] == deltas
            and observed["paired_correctness_transitions"]
            == paired_correctness_transitions(left["fixtures"], right["fixtures"])
            and observed["active_intent_coverage_complementarity"]
            == active_complementarity(left["fixtures"], right["fixtures"])
            and close(
                observed["right_minus_left_elapsed_seconds"],
                right["final_access"]["elapsed_seconds"]
                - left["final_access"]["elapsed_seconds"],
            )
            and observed["right_minus_left_peak_active_memory_bytes"]
            == right["final_access"]["peak_active_memory_bytes"]
            - left["final_access"]["peak_active_memory_bytes"]
        )

    expected_result_hashes = comparison["condition_results"]
    manifest_exact = all(
        results[condition_id]["model_manifest_sha256"]
        == implementation["model_manifests"][condition_id]["manifest_sha256"]
        for condition_id in CONDITION_IDS
    )
    results_exact = all(
        expected_result_hashes[condition_id]["path"]
        == str(result_paths[condition_id].relative_to(PROJECT_ROOT))
        and expected_result_hashes[condition_id]["sha256"]
        == file_sha256(result_paths[condition_id])
        for condition_id in CONDITION_IDS
    )

    total_access = comparison["access"]
    checks = {
        "implementation_lock_and_frozen_dependencies_are_exact": bool(
            payload_hash(implementation_payload)
            == implementation["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / implementation[key])
                == implementation[f"{key}_sha256"]
                for key in (
                    "design_lock",
                    "corpus_seal",
                    "corpus",
                    "protocol",
                    "tests",
                    "runner",
                    "implementation_auditor",
                    "implementation_audit",
                )
            )
        ),
        "disclosed_verifier_attempt_only_failed_on_manifest_hash_field_selection": bool(
            failed_attempt["passed"] is False
            and failed_attempt["decision"] == "reject_V90_outcome"
            and [
                key
                for key, value in failed_attempt["checks"].items()
                if not value
            ]
            == ["analyzer_and_four_result_hashes_are_exact"]
        ),
        "analyzer_and_four_result_hashes_are_exact": bool(
            comparison["analyzer"] == str(analyzer_path.relative_to(PROJECT_ROOT))
            and comparison["analyzer_sha256"] == file_sha256(analyzer_path)
            and results_exact
            and manifest_exact
        ),
        "all_raw_fixture_artifacts_equal_embedded_results": bool(
            raw_exact and len(raw_fixture_artifacts) == 192
        ),
        "all_metrics_gates_and_quality_decisions_reconstruct": reconstruction_exact,
        "all_registered_paired_contrasts_reconstruct": contrast_exact,
        "all_four_conditions_completed_but_none_qualified": bool(
            all(result["completed_condition"] for result in results.values())
            and all(len(result["fixtures"]) == 48 for result in results.values())
            and not any(result["quality_gate_pass"] for result in results.values())
        ),
        "observed_capacity_generation_and_quantization_boundary_reconstructs": bool(
            close(results["qwen35_4b_4bit"]["metrics"]["gold_active_intent_coverage_rate"], 21 / 24)
            and close(results["qwen35_27b_4bit"]["metrics"]["gold_active_intent_coverage_rate"], 19 / 24)
            and close(results["qwen38_27b_4bit"]["metrics"]["gold_active_intent_coverage_rate"], 21 / 24)
            and close(results["qwen38_27b_8bit"]["metrics"]["gold_active_intent_coverage_rate"], 21 / 24)
            and close(results["qwen35_4b_4bit"]["metrics"]["state_slot_key_exact_rate"], 11 / 48)
            and close(results["qwen35_27b_4bit"]["metrics"]["state_slot_key_exact_rate"], 12 / 48)
            and close(results["qwen38_27b_4bit"]["metrics"]["state_slot_key_exact_rate"], 12 / 48)
            and close(results["qwen38_27b_8bit"]["metrics"]["state_slot_key_exact_rate"], 14 / 48)
            and close(results["qwen38_27b_4bit"]["metrics"]["mandatory_NONE_inclusion_rate"], 45 / 48)
            and close(results["qwen38_27b_8bit"]["metrics"]["mandatory_NONE_inclusion_rate"], 1.0)
        ),
        "no_qualifying_27b_makes_combination_ineligible_and_unexecuted": bool(
            comparison["qualifying_27b_conditions"] == []
            and comparison["selected_27b_condition"] is None
            and comparison["combination_diagnostic"]["eligible"] is False
            and comparison["combination_diagnostic"]["executed"] is False
            and comparison["combination_diagnostic"]["authorizes_live_cascade"] is False
            and comparison["decision"] == "retain_model_free_authoritative_boundary"
        ),
        "model_usage_policy_retains_only_non_authoritative_shadow_models": bool(
            comparison["model_usage_policy"]["authoritative_semantic_boundary"]
            == "model_free_schema_renderer_and_deterministic_validator"
            and comparison["model_usage_policy"]["qwen35_4b_4bit"]
            == "retain_only_as_low_cost_frozen_shadow_baseline"
            and comparison["model_usage_policy"]["all_27b_conditions"]
            == "freeze_as_nonqualifying_shadow_evidence"
            and comparison["model_usage_policy"]["belief_or_action_authority"] is False
            and comparison["model_usage_policy"]["deployment_or_execution"] is False
        ),
        "access_counts_match_four_independent_local_runs_and_zero_expansive_access": bool(
            total_access["model_load_count"] == 4
            and total_access["model_generation_count"] == 192
            and total_access["LLM_API_call_count"] == 0
            and total_access["adapter_training_run_count"] == 0
            and total_access["manual_utterance_inspection_count"] == 0
            and total_access["real_service_call_count"] == 0
            and total_access["external_side_effect_count"] == 0
            and all(
                result["final_access"]["model_load_count"] == 1
                and result["final_access"]["model_generation_count"] == 48
                for result in results.values()
            )
        ),
        "results_document_and_research_direction_exist": bool(
            document_path.is_file() and research_path.is_file()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "90-capacity-generation-outcome-audit",
        "experiment": "v90_capacity_generation_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_verified_V90_model_free_boundary"
            if passed
            else "reject_V90_outcome"
        ),
        "checks": checks,
        "verified_condition_metrics": {
            condition_id: {
                metric: reconstructed_metrics[condition_id][metric]
                for metric in SCALAR_METRICS
            }
            for condition_id in CONDITION_IDS
        },
        "verified_decision": comparison["decision"],
        "claim_boundary": comparison["claim_boundary"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "90-capacity-generation-outcome-lock",
        "experiment": "v90_capacity_generation_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "condition_results": {
            condition_id: {
                "path": str(result_paths[condition_id].relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(result_paths[condition_id]),
            }
            for condition_id in CONDITION_IDS
        },
        "failed_outcome_audit_attempt": str(
            failed_attempt_path.relative_to(PROJECT_ROOT)
        ),
        "failed_outcome_audit_attempt_sha256": file_sha256(failed_attempt_path),
        "raw_fixture_artifacts": raw_fixture_artifacts,
        "analyzer": str(analyzer_path.relative_to(PROJECT_ROOT)),
        "analyzer_sha256": file_sha256(analyzer_path),
        "comparison_result": str(comparison_path.relative_to(PROJECT_ROOT)),
        "comparison_result_sha256": file_sha256(comparison_path),
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
            "decision": comparison["decision"],
            "qualifying_27b_conditions": [],
            "combination_diagnostic_eligible": False,
            "condition_summaries": comparison["condition_summaries"],
            "contrasts": comparison["contrasts"],
            "model_usage_policy": comparison["model_usage_policy"],
            "access": comparison["access"],
        },
        "authorization": {
            "modify_or_rerun_V90": False,
            "retain_model_free_authoritative_boundary": True,
            "retain_qwen35_4b_only_as_frozen_shadow_baseline": True,
            "adopt_any_27b_or_8bit_condition": False,
            "construct_small_large_union_or_cascade": False,
            "run_API_model_or_train_adapter_for_this_branch": False,
            "grant_any_model_belief_or_action_authority": False,
            "deploy_or_execute_any_model_output": False,
            "perform_real_service_call_or_external_side_effect": False,
            "report_and_synthesize_frozen_results": True,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
