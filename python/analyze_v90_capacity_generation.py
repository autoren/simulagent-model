#!/usr/bin/env python3
"""Compute the preregistered V90 paired model comparison exactly once."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v90_capacity_generation_protocol import paired_correctness_transitions


CONDITION_IDS = (
    "qwen35_4b_4bit",
    "qwen35_27b_4bit",
    "qwen38_27b_4bit",
    "qwen38_27b_8bit",
)

SUMMARY_METRICS = (
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


def metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Return right-minus-left deltas for all registered scalar quality metrics."""
    return {
        metric: float(right[metric]) - float(left[metric])
        for metric in SUMMARY_METRICS
    }


def active_complementarity(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Count paired active-intent coverage outcomes without constructing a union."""
    counts = {
        "both_correct": 0,
        "left_only_correct": 0,
        "right_only_correct": 0,
        "neither_correct": 0,
    }
    if set(left) != set(right):
        raise ValueError("paired condition fixture identities differ")
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


def load_condition(condition_id: str) -> tuple[Path, dict[str, Any]]:
    path = (
        PROJECT_ROOT
        / "outputs/v90-capacity-generation/evaluation"
        / condition_id
        / "result.json"
    )
    return path, json.loads(path.read_text())


def main() -> None:
    implementation_lock_path = (
        PROJECT_ROOT / "configs/v90-capacity-generation-implementation-lock.json"
    )
    analyzer_path = PROJECT_ROOT / "python/analyze_v90_capacity_generation.py"
    output_dir = PROJECT_ROOT / "outputs/v90-capacity-generation/comparison"
    result_path = output_dir / "result.json"
    if output_dir.exists():
        raise RuntimeError("V90 comparison may run only once")

    implementation = json.loads(implementation_lock_path.read_text())
    implementation_payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    if payload_hash(implementation_payload) != implementation["lock_payload_sha256"]:
        raise RuntimeError("V90 implementation lock drifted")
    for key in (
        "design_lock",
        "corpus_seal",
        "corpus",
        "protocol",
        "tests",
        "runner",
        "implementation_auditor",
        "implementation_audit",
    ):
        if file_sha256(PROJECT_ROOT / implementation[key]) != implementation[f"{key}_sha256"]:
            raise RuntimeError(f"V90 locked dependency drifted: {key}")

    configured_ids = tuple(
        condition["id"] for condition in implementation["config_payload"]["modelConditions"]
    )
    if configured_ids != CONDITION_IDS:
        raise RuntimeError("V90 condition order or identity drifted")

    paths: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    fixtures: dict[str, dict[str, dict[str, Any]]] = {}
    for condition_id in CONDITION_IDS:
        path, result = load_condition(condition_id)
        if result["condition"]["id"] != condition_id:
            raise RuntimeError(f"V90 condition identity mismatch: {condition_id}")
        if not result["completed_condition"]:
            raise RuntimeError(f"V90 condition is incomplete: {condition_id}")
        if len(result["fixtures"]) != 48:
            raise RuntimeError(f"V90 fixture count mismatch: {condition_id}")
        paths[condition_id] = path
        results[condition_id] = result
        fixtures[condition_id] = result["fixtures"]

    fixture_ids = [set(fixtures[condition_id]) for condition_id in CONDITION_IDS]
    if any(ids != fixture_ids[0] for ids in fixture_ids[1:]):
        raise RuntimeError("V90 paired fixture identities differ")

    comparison_specs = {
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
    contrasts: dict[str, Any] = {}
    for name, (left_id, right_id) in comparison_specs.items():
        contrasts[name] = {
            "left_condition": left_id,
            "right_condition": right_id,
            "right_minus_left_metric_deltas": metric_deltas(
                results[left_id]["metrics"], results[right_id]["metrics"]
            ),
            "paired_correctness_transitions": paired_correctness_transitions(
                fixtures[left_id], fixtures[right_id]
            ),
            "active_intent_coverage_complementarity": active_complementarity(
                fixtures[left_id], fixtures[right_id]
            ),
            "right_minus_left_elapsed_seconds": (
                results[right_id]["final_access"]["elapsed_seconds"]
                - results[left_id]["final_access"]["elapsed_seconds"]
            ),
            "right_minus_left_peak_active_memory_bytes": (
                results[right_id]["final_access"]["peak_active_memory_bytes"]
                - results[left_id]["final_access"]["peak_active_memory_bytes"]
            ),
        }

    condition_summaries: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        result = results[condition_id]
        condition_summaries[condition_id] = {
            "role": result["condition"]["role"],
            "quantization_bits": result["condition"]["quantizationBits"],
            "weight_bytes": result["condition"]["weightBytes"],
            "completed_condition": result["completed_condition"],
            "quality_gate_pass": result["quality_gate_pass"],
            "failed_quality_gates": sorted(
                key
                for key in (
                    "exact_JSON_parse",
                    "ontology_conformance",
                    "mandatory_NONE_inclusion",
                    "permanent_non_deployable",
                    "gold_active_intent_coverage",
                    "intent_candidate_set_exact",
                    "NONE_only_intent_exact",
                    "per_service_intent_candidate_recall",
                    "state_slot_key_recall",
                    "state_slot_key_exact",
                    "intent_exact_improvement_over_exhaustive",
                    "mean_intent_candidate_count",
                )
                if not result["gates"][key]
            ),
            "metrics": {
                metric: result["metrics"][metric] for metric in SUMMARY_METRICS
            },
            "per_service_intent_candidate_recall": result["metrics"][
                "per_service_intent_candidate_recall"
            ],
            "runtime": {
                "elapsed_seconds": result["final_access"]["elapsed_seconds"],
                "model_load_seconds": result["final_access"]["model_load_seconds"],
                "peak_active_memory_bytes": result["final_access"][
                    "peak_active_memory_bytes"
                ],
            },
            "access": result["final_access"],
        }

    qualifying_27b = [
        condition_id
        for condition_id in CONDITION_IDS
        if "27b" in condition_id and results[condition_id]["quality_gate_pass"]
    ]
    if qualifying_27b:
        selected_27b = min(
            qualifying_27b,
            key=lambda condition_id: results[condition_id]["condition"]["weightBytes"],
        )
        combination = {
            "eligible": True,
            "selected_large_condition": selected_27b,
            "executed": False,
            "reason": "eligible_pair_requires_a_separately_frozen_union_evaluation_before_execution",
            "authorizes_live_cascade": False,
        }
        decision = "preregister_separate_posterior_integration_shadow_study"
    else:
        selected_27b = None
        combination = {
            "eligible": False,
            "selected_large_condition": None,
            "executed": False,
            "reason": "no_27B_condition_passed_every_independent_noncompensatory_gate",
            "authorizes_live_cascade": False,
        }
        decision = "retain_model_free_authoritative_boundary"

    total_access = {
        "model_load_count": sum(
            result["final_access"]["model_load_count"] for result in results.values()
        ),
        "model_generation_count": sum(
            result["final_access"]["model_generation_count"] for result in results.values()
        ),
        "LLM_API_call_count": sum(
            result["final_access"]["LLM_API_call_count"] for result in results.values()
        ),
        "adapter_training_run_count": sum(
            result["final_access"]["adapter_training_run_count"]
            for result in results.values()
        ),
        "manual_utterance_inspection_count": sum(
            result["final_access"]["manual_utterance_inspection_count"]
            for result in results.values()
        ),
        "real_service_call_count": sum(
            result["final_access"]["real_service_call_count"]
            for result in results.values()
        ),
        "external_side_effect_count": sum(
            result["final_access"]["external_side_effect_count"]
            for result in results.values()
        ),
    }
    result = {
        "schema_version": "90-capacity-generation-comparison-result",
        "experiment": "v90_fresh_external_local_capacity_generation_shadow",
        "passed": True,
        "decision": decision,
        "condition_results": {
            condition_id: {
                "path": str(paths[condition_id].relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(paths[condition_id]),
            }
            for condition_id in CONDITION_IDS
        },
        "condition_summaries": condition_summaries,
        "contrasts": contrasts,
        "qualifying_27b_conditions": qualifying_27b,
        "selected_27b_condition": selected_27b,
        "combination_diagnostic": combination,
        "model_usage_policy": {
            "authoritative_semantic_boundary": "model_free_schema_renderer_and_deterministic_validator",
            "qwen35_4b_4bit": "retain_only_as_low_cost_frozen_shadow_baseline",
            "all_27b_conditions": "freeze_as_nonqualifying_shadow_evidence",
            "small_large_combination": "not_eligible_and_not_executed",
            "LLM_API": "not_authorized_or_needed_for_this_result",
            "adapter_training": "not_authorized_or_needed_for_this_result",
            "belief_or_action_authority": False,
            "deployment_or_execution": False,
        },
        "access": total_access,
        "claim_boundary": (
            "paired offline local-model shadow comparison on one frozen 48-record fresh "
            "population; no API, training, manual source-language inspection, retries, "
            "belief authority, deployment, service call, or external side effect"
        ),
        "analyzer": str(analyzer_path.relative_to(PROJECT_ROOT)),
        "analyzer_sha256": file_sha256(analyzer_path),
    }
    output_dir.mkdir(parents=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "condition_summaries": condition_summaries,
                "contrasts": contrasts,
                "combination_diagnostic": combination,
                "access": total_access,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
