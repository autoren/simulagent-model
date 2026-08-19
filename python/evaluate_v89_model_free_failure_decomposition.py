#!/usr/bin/env python3
"""Compute the one frozen, identifier-only V89 failure decomposition."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v88_external_candidate_protocol import aggregate, evaluate_gates, set_precision, set_recall


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def counterfactual_row(base: dict[str, Any], intents: set[str], slots: set[str], *, structured: bool) -> dict[str, Any]:
    gold_i = set(base["gold"]["intent_candidates"])
    gold_s = set(base["gold"]["state_slot_key_candidates"])
    active = base["active_intent"]
    return {
        **base,
        "exact_json": structured,
        "ontology_conformant": structured,
        "mandatory_NONE_included": "NONE" in intents,
        "gold_active_intent_covered": active == "NONE" or active in intents,
        "intent_candidate_precision": set_precision(intents, gold_i),
        "intent_candidate_recall": set_recall(intents, gold_i),
        "intent_candidate_exact": intents == gold_i,
        "none_only_intent_exact": active != "NONE" or intents == {"NONE"},
        "state_slot_key_precision": set_precision(slots, gold_s),
        "state_slot_key_recall": set_recall(slots, gold_s),
        "state_slot_key_exact": slots == gold_s,
        "intent_candidate_count": len(intents),
        "permanently_non_deployable": True,
        "intent_candidates": sorted(intents),
        "state_slot_key_candidates": sorted(slots),
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    lock_payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(lock_payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V89 implementation lock drifted")
    if not lock["authorization"]["evaluate_identifier_only_decomposition_once"]:
        raise RuntimeError("V89 evaluation is not authorized")
    for key in ("design_lock", "parent_outcome_lock", "parent_result", "protocol", "evaluator", "implementation_auditor"):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V89 locked dependency drifted: {key}")
    output_dir = PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/evaluation"
    if output_dir.exists():
        raise RuntimeError("V89 decomposition may run only once")

    parent = json.loads((PROJECT_ROOT / lock["parent_outcome_lock"]).read_text())
    parent_result = json.loads((PROJECT_ROOT / lock["parent_result"]).read_text())
    rows = [json.loads((PROJECT_ROOT / item["path"]).read_text()) for item in parent["raw_fixture_artifacts"]]
    records = {row["id"]: {
        "id": row["id"], "service": row["service"], "gold": row["gold"],
        "allowed_intent_ids": row["allowed_intent_ids"], "allowed_slot_ids": row["allowed_slot_ids"],
    } for row in rows}

    views: dict[str, list[dict[str, Any]]] = {"strict_frozen_outcome": rows}
    serialization, state_strict, intent_strict, serialization_state, serialization_intent, full = [], [], [], [], [], []
    for row in rows:
        actual_i = set(row["intent_candidates"]) if row["ontology_conformant"] else set()
        actual_s = set(row["state_slot_key_candidates"]) if row["ontology_conformant"] else set()
        gold_i = set(row["gold"]["intent_candidates"])
        gold_s = set(row["gold"]["state_slot_key_candidates"])
        serialized_i = actual_i if row["ontology_conformant"] else gold_i
        serialized_s = actual_s if row["ontology_conformant"] else gold_s
        serialization.append(counterfactual_row(row, serialized_i, serialized_s, structured=True))
        state_strict.append(counterfactual_row(row, actual_i, gold_s, structured=row["ontology_conformant"]))
        intent_strict.append(counterfactual_row(row, gold_i, actual_s, structured=row["ontology_conformant"]))
        serialization_state.append(counterfactual_row(row, serialized_i, gold_s, structured=True))
        serialization_intent.append(counterfactual_row(row, gold_i, serialized_s, structured=True))
        full.append(counterfactual_row(row, gold_i, gold_s, structured=True))
    views.update({
        "perfect_serialization_upper_bound": serialization,
        "perfect_state_oracle_with_strict_intents": state_strict,
        "perfect_intent_oracle_with_strict_state": intent_strict,
        "perfect_serialization_plus_state_oracle": serialization_state,
        "perfect_serialization_plus_intent_oracle": serialization_intent,
        "full_oracle": full,
    })
    zero_access = {"model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "manual_utterance_inspection_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}
    metrics = {name: aggregate(view, records) for name, view in views.items()}
    gate_views = {name: evaluate_gates(metric, lock["original_config"], zero_access) for name, metric in metrics.items()}
    conforming = [row for row in rows if row["ontology_conformant"]]
    joint = defaultdict(int)
    for row in conforming:
        joint[f"intent_{'exact' if row['intent_candidate_exact'] else 'wrong'}__state_{'exact' if row['state_slot_key_exact'] else 'wrong'}"] += 1
    malformed = defaultdict(int)
    for row in rows:
        if not row["ontology_conformant"]:
            role = "NONE" if row["active_intent"] == "NONE" else "active"
            malformed[f"{row['service']}::{role}"] += 1
    cardinality = defaultdict(list)
    for row in rows:
        cardinality[len(row["gold"]["state_slot_key_candidates"])].append(row)
    cardinality_diagnostic = {
        str(size): {
            "record_count": len(bucket),
            "strict_state_exact_rate": sum(item["state_slot_key_exact"] for item in bucket) / len(bucket),
            "strict_state_recall": sum(item["state_slot_key_recall"] for item in bucket) / len(bucket),
        }
        for size, bucket in sorted(cardinality.items())
    }
    intent_gate_names = (
        "exact_JSON_parse", "ontology_conformance", "mandatory_NONE_inclusion",
        "gold_active_intent_coverage", "intent_candidate_set_exact", "NONE_only_intent_exact",
        "per_service_intent_candidate_recall", "intent_exact_improvement_over_exhaustive",
        "mean_intent_candidate_count",
    )
    serialization_semantic_gate_names = (
        "gold_active_intent_coverage", "intent_candidate_set_exact", "NONE_only_intent_exact",
        "per_service_intent_candidate_recall", "state_slot_key_recall", "state_slot_key_exact",
        "intent_exact_improvement_over_exhaustive", "mean_intent_candidate_count",
    )
    serialization_sufficient = all(gate_views["perfect_serialization_upper_bound"][key] for key in serialization_semantic_gate_names)
    serialization_plus_state_intent_sufficient = all(gate_views["perfect_serialization_plus_state_oracle"][key] for key in intent_gate_names)
    result = {
        "schema_version": "89-model-free-failure-decomposition-result",
        "experiment": "v89_v88r1_failure_decomposition_without_language_or_model_access",
        "passed": True,
        "decision": (
            "rule_out_serialization_only_and_preregister_model_free_state_accumulation_feasibility"
            if not serialization_sufficient and serialization_plus_state_intent_sufficient
            else "pause_external_local_model_integration"
        ),
        "strict_parent_metrics_reconstructed": metrics["strict_frozen_outcome"] == parent_result["metrics"],
        "view_metrics": metrics,
        "view_gates": gate_views,
        "diagnostics": {
            "joint_exactness_among_conforming": dict(sorted(joint.items())),
            "malformed_by_service_and_label_role": dict(sorted(malformed.items())),
            "state_target_cardinality": cardinality_diagnostic,
            "serialization_only_upper_bound_satisfies_all_semantic_gates": serialization_sufficient,
            "serialization_plus_state_oracle_satisfies_all_intent_gates": serialization_plus_state_intent_sufficient,
            "serialization_only_failed_semantic_gates": [key for key in serialization_semantic_gate_names if not gate_views["perfect_serialization_upper_bound"][key]],
        },
        "access": {"source_language_access_count": 0, **zero_access},
        "claim_boundary": "identifier-only optimistic failure decomposition; no source language, prompt, model, API, training, deployment, service call, or execution access",
    }
    output_dir.mkdir(parents=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "decision": result["decision"], "diagnostics": result["diagnostics"], "view_summary": {name: {"intent_exact": value["intent_candidate_set_exact_rate"], "state_exact": value["state_slot_key_exact_rate"], "state_recall": value["state_slot_key_recall"]} for name, value in metrics.items()}, "access": result["access"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
