#!/usr/bin/env python3
"""Independently verify and freeze the single V81 factorized-model outcome."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


LABEL_KEYS = [
    "schedule_review",
    "send_summary",
    "alex_chen",
    "alex_kim",
    "out_of_ontology",
]
CONFIDENCE_KEYS = {
    "confidence", "confidences", "probability", "probabilities", "score", "scores"
}
ACTION_KEYS = {"action", "actions", "tool", "tools", "tool_call", "tool_calls"}
CANDIDATE_KEYS = {"candidate", "candidates", "candidate_id", "candidate_ids"}


def payload_hash(value: dict[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def nested_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(nested_keys(child))
    return keys


def compose(labels: dict[str, bool]) -> list[str]:
    if labels.get("out_of_ontology", False):
        return ["none_of_the_above"]
    candidates: list[str] = []
    for recipient in ("alex_chen", "alex_kim"):
        for operation in ("schedule_review", "send_summary"):
            if labels.get(operation, False) and labels.get(recipient, False):
                candidates.append(f"{operation}__{recipient}")
    candidates.append("none_of_the_above")
    return candidates


def score(record: dict[str, Any], raw: str, config: dict[str, Any]) -> dict[str, Any]:
    value: Any = None
    parse_ok = True
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parse_ok = False
    keys = nested_keys(value) if parse_ok else []
    confidence_count = sum(key.lower() in CONFIDENCE_KEYS for key in keys)
    action_count = sum(key.lower() in ACTION_KEYS for key in keys)
    candidate_field_count = sum(key.lower() in CANDIDATE_KEYS for key in keys)
    labels = {
        key: value[key]
        for key in LABEL_KEYS
        if isinstance(value, dict) and key in value and type(value[key]) is bool
    }
    complete = len(labels) == len(LABEL_KEYS)
    consistent = bool(
        complete
        and labels["out_of_ontology"]
        == (not labels["schedule_review"] and not labels["send_summary"])
    )
    valid = bool(
        parse_ok
        and isinstance(value, dict)
        and list(value) == LABEL_KEYS
        and complete
        and consistent
        and confidence_count == action_count == candidate_field_count == 0
    )
    candidates = compose(labels) if valid else []
    gold_labels = record["goldLabels"]
    gold_candidates = record["goldCandidateIds"]
    return {
        "id": record["id"],
        "stratum": record["stratum"],
        "exact_json_parse": parse_ok,
        "schema_valid": valid,
        "labels": labels,
        "ontology_consistent": consistent,
        "candidate_ids": candidates,
        "candidate_count": len(candidates),
        "none_of_the_above_included": "none_of_the_above" in candidates,
        "canonical_order": bool(candidates)
        and candidates
        == [
            candidate
            for candidate in config["candidateIdsInRequiredOrder"]
            if candidate in candidates
        ],
        "confidence_or_probability_field_count": confidence_count,
        "action_or_tool_field_count": action_count,
        "candidate_id_field_count": candidate_field_count,
        "label_accuracy": (
            sum(labels.get(key) == value for key, value in gold_labels.items())
            / len(gold_labels)
            if valid
            else 0.0
        ),
        "exact_label_vector": labels == gold_labels if valid else False,
        "out_of_ontology_label_correct": bool(
            labels.get("out_of_ontology") == gold_labels["out_of_ontology"]
        )
        if valid
        else False,
        "gold_candidate_recall": len(set(candidates) & set(gold_candidates))
        / len(gold_candidates),
        "exact_candidate_set": candidates == gold_candidates,
    }


def mean(values: list[float | bool]) -> float:
    return float(sum(values) / len(values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    return {
        "record_count": len(rows),
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "exact_json_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "schema_validity_rate": mean([row["schema_valid"] for row in rows]),
        "exact_label_vector_accuracy": mean([row["exact_label_vector"] for row in rows]),
        "mean_label_accuracy": mean([row["label_accuracy"] for row in rows]),
        "out_of_ontology_label_accuracy": mean(
            [row["out_of_ontology_label_correct"] for row in strata["out_of_ontology"]]
        ),
        "none_of_the_above_inclusion_rate": mean(
            [row["none_of_the_above_included"] for row in rows]
        ),
        "mean_gold_candidate_recall": mean(
            [row["gold_candidate_recall"] for row in rows]
        ),
        "per_stratum_mean_gold_candidate_recall": {
            name: mean([row["gold_candidate_recall"] for row in members])
            for name, members in sorted(strata.items())
        },
        "exact_candidate_set_accuracy": mean([row["exact_candidate_set"] for row in rows]),
        "clear_exact_candidate_set_accuracy": mean(
            [row["exact_candidate_set"] for row in strata["clear"]]
        ),
        "out_of_ontology_exact_candidate_set_accuracy": mean(
            [row["exact_candidate_set"] for row in strata["out_of_ontology"]]
        ),
        "canonical_order_rate": mean([row["canonical_order"] for row in rows]),
        "mean_candidate_count": mean([row["candidate_count"] for row in rows]),
        "confidence_or_probability_field_count": sum(
            row["confidence_or_probability_field_count"] for row in rows
        ),
        "action_or_tool_field_count": sum(row["action_or_tool_field_count"] for row in rows),
        "candidate_id_field_count": sum(row["candidate_id_field_count"] for row in rows),
    }


def gate(metrics: dict[str, Any], config: dict[str, Any], access: dict[str, int]) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "complete_record_and_stratum_census": bool(
            metrics["record_count"] == gates["requiredRecordCount"]
            and metrics["stratum_counts"] == gates["requiredStratumCounts"]
        ),
        "exact_JSON_parse_rate": metrics["exact_json_parse_rate"] >= gates["minimumExactJSONParseRate"],
        "schema_validity_rate": metrics["schema_validity_rate"] >= gates["minimumSchemaValidityRate"],
        "exact_label_vector_accuracy": metrics["exact_label_vector_accuracy"] >= gates["minimumExactLabelVectorAccuracy"],
        "mean_label_accuracy": metrics["mean_label_accuracy"] >= gates["minimumMeanLabelAccuracy"],
        "out_of_ontology_label_accuracy": metrics["out_of_ontology_label_accuracy"] >= gates["minimumOutOfOntologyLabelAccuracy"],
        "none_of_the_above_inclusion_rate": metrics["none_of_the_above_inclusion_rate"] >= 1.0,
        "mean_gold_candidate_recall": metrics["mean_gold_candidate_recall"] >= gates["minimumMeanGoldCandidateRecall"],
        "per_stratum_gold_candidate_recall": all(
            value >= gates["minimumPerStratumMeanGoldCandidateRecall"]
            for value in metrics["per_stratum_mean_gold_candidate_recall"].values()
        ),
        "exact_candidate_set_accuracy": metrics["exact_candidate_set_accuracy"] >= gates["minimumExactCandidateSetAccuracy"],
        "clear_exact_candidate_set_accuracy": metrics["clear_exact_candidate_set_accuracy"] >= gates["minimumClearExactCandidateSetAccuracy"],
        "out_of_ontology_exact_candidate_set_accuracy": metrics["out_of_ontology_exact_candidate_set_accuracy"] >= gates["minimumOutOfOntologyExactCandidateSetAccuracy"],
        "canonical_order_rate": metrics["canonical_order_rate"] >= gates["minimumCanonicalOrderRate"],
        "bounded_mean_candidate_count": metrics["mean_candidate_count"] <= gates["maximumMeanCandidateCount"],
        "zero_confidence_or_probability_fields": metrics["confidence_or_probability_field_count"] <= gates["maximumConfidenceOrProbabilityFieldCount"],
        "zero_action_or_tool_fields": metrics["action_or_tool_field_count"] <= gates["maximumActionOrToolFieldCount"],
        "zero_candidate_ID_fields_from_model": metrics["candidate_id_field_count"] <= gates["maximumCandidateIdFieldCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_generation_count"] <= gates["maximumModelGenerationCount"]
            and access["API_call_count"] <= gates["maximumAPICallCount"]
            and access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"]
            and access["human_record_access_count"] <= gates["maximumHumanRecordAccessCount"]
            and access["real_tool_call_count"] <= gates["maximumRealToolCallCount"]
            and access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"]
        ),
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v81-factorized-local-candidate/evaluation"
    result_path = evaluation_dir / "result.json"
    access_path = evaluation_dir / "access.json"
    audit_path = PROJECT_ROOT / "outputs/v81-factorized-local-candidate/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v81_factorized_outcome.py"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V81 outcome is already frozen")
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation.items() if key != "lock_payload_sha256"
    }
    config = implementation["config_payload"]
    records = [
        json.loads(line)
        for line in (PROJECT_ROOT / implementation["corpus"]).read_text().splitlines()
        if line
    ]
    fixture_paths = sorted((evaluation_dir / "raw-fixtures").glob("*.json"))
    fixtures = [json.loads(path.read_text()) for path in fixture_paths]
    by_id = {fixture["id"]: fixture for fixture in fixtures}
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    rows = [score(record, by_id[record["id"]]["raw_response"], config) for record in records]
    metrics = aggregate(rows)
    gates = gate(metrics, config, access)
    fields = (
        "id", "stratum", "exact_json_parse", "schema_valid", "labels",
        "ontology_consistent", "candidate_ids", "candidate_count",
        "none_of_the_above_included", "canonical_order",
        "confidence_or_probability_field_count", "action_or_tool_field_count",
        "candidate_id_field_count", "label_accuracy", "exact_label_vector",
        "out_of_ontology_label_correct", "gold_candidate_recall", "exact_candidate_set",
    )
    checks = {
        "implementation_lock_payload_valid": payload_hash(implementation_payload)
        == implementation["lock_payload_sha256"],
        "exactly_one_fixture_per_locked_record_in_order": bool(
            len(fixtures) == len(records) == 24
            and [fixture["id"] for fixture in fixtures] == [record["id"] for record in records]
            and len(by_id) == 24
        ),
        "independent_per_record_parse_compose_and_score_reproduced": all(
            all(close(row[field], by_id[row["id"]][field]) for field in fields)
            for row in rows
        ),
        "independent_metrics_reproduced": close(metrics, result["metrics"]),
        "independent_gates_reproduced": gates == result["gates"],
        "failure_decision_matches_noncompensatory_rule": bool(
            not all(gates.values())
            and not result["passed"]
            and result["decision"]
            == "freeze_V81_failure_without_edits_or_rerun_and_stop_local_candidate_integration"
        ),
        "one_load_twenty_four_generations_zero_external_access": bool(
            access["attempt_number"] == 1
            and access["model_load_count"] == 1
            and access["model_generation_count"] == 24
            and all(
                access[key] == 0
                for key in (
                    "API_call_count", "adapter_training_run_count",
                    "human_record_access_count", "real_tool_call_count",
                    "external_side_effect_count",
                )
            )
        ),
        "result_attempt_matches_final_access": result["attempt"] == access,
        "zero_model_or_external_access_during_verification": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "81-factorized-local-candidate-outcome-audit",
        "experiment": "v81_factorized_local_candidate_independent_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_verified_negative_and_stop_local_candidate_integration"
            if passed else "reject_V81_outcome_closure"
        ),
        "checks": checks,
        "independent_metrics": metrics,
        "independent_gates": gates,
        "access": {
            "model_load_count": 0, "model_generation_count": 0,
            "API_call_count": 0, "adapter_training_run_count": 0,
            "human_record_access_count": 0, "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    outcome_lock = {
        "schema_version": "81-factorized-local-candidate-outcome-lock",
        "experiment": "v81_factorized_local_candidate_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "access": str(access_path.relative_to(PROJECT_ROOT)),
        "access_sha256": file_sha256(access_path),
        "raw_fixture_manifest": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in fixture_paths
        },
        "outcome_verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "outcome_verifier_sha256": file_sha256(verifier_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed": False,
            "exact_JSON_parse_rate": metrics["exact_json_parse_rate"],
            "schema_validity_rate": metrics["schema_validity_rate"],
            "exact_label_vector_accuracy": metrics["exact_label_vector_accuracy"],
            "mean_label_accuracy": metrics["mean_label_accuracy"],
            "exact_candidate_set_accuracy": metrics["exact_candidate_set_accuracy"],
            "out_of_ontology_label_accuracy": metrics["out_of_ontology_label_accuracy"],
        },
        "authorization": {
            "modify_or_rerun_V81": False,
            "run_V81_local_model_again": False,
            "continue_local_model_candidate_integration": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "design_materially_different_non_authoritative_LLM_role": True,
        },
    }
    outcome_lock["lock_payload_sha256"] = payload_hash(outcome_lock)
    lock_path.write_text(json.dumps(outcome_lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
