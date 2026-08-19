#!/usr/bin/env python3
"""Independently reparse, rescore, and freeze the one-shot V80 outcome."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


CONFIDENCE_KEYS = {
    "confidence",
    "confidences",
    "probability",
    "probabilities",
    "score",
    "scores",
}
ACTION_KEYS = {
    "action",
    "actions",
    "tool",
    "tools",
    "tool_call",
    "tool_calls",
}


def payload_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def nested_keys(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.append(str(key))
            result.extend(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(nested_keys(child))
    return result


def independently_score(
    record: dict[str, Any], raw: str, config: dict[str, Any]
) -> dict[str, Any]:
    value: Any = None
    parse_ok = True
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parse_ok = False
    keys = nested_keys(value) if parse_ok else []
    confidence_count = sum(key.lower() in CONFIDENCE_KEYS for key in keys)
    action_count = sum(key.lower() in ACTION_KEYS for key in keys)
    candidate_value = value.get("candidate_ids") if isinstance(value, dict) else None
    candidates = candidate_value if isinstance(candidate_value, list) else []
    strings = [candidate for candidate in candidates if isinstance(candidate, str)]
    ordered = config["candidateIdsInRequiredOrder"]
    allowed = set(ordered)
    unknown_count = sum(candidate not in allowed for candidate in strings)
    duplicate_count = len(strings) - len(set(strings))
    canonical = bool(strings) and strings == [
        candidate for candidate in ordered if candidate in strings
    ]
    contract = config["outputContract"]
    schema_ok = bool(
        parse_ok
        and isinstance(value, dict)
        and list(value) == contract["exactTopLevelKeys"]
        and isinstance(candidate_value, list)
        and len(candidates) == len(strings)
        and contract["minimumCandidates"]
        <= len(strings)
        <= contract["maximumCandidates"]
        and unknown_count == 0
        and duplicate_count == 0
        and canonical
        and "none_of_the_above" in strings
        and confidence_count == 0
        and action_count == 0
    )
    predicted = strings if schema_ok else []
    gold = record["goldCandidateIds"]
    return {
        "id": record["id"],
        "stratum": record["stratum"],
        "exact_json_parse": parse_ok,
        "schema_valid": schema_ok,
        "candidate_ids": strings,
        "candidate_count": len(strings),
        "none_of_the_above_included": "none_of_the_above" in strings,
        "canonical_order": canonical,
        "unknown_candidate_id_count": unknown_count,
        "duplicate_candidate_id_count": duplicate_count,
        "confidence_or_probability_field_count": confidence_count,
        "action_or_tool_field_count": action_count,
        "gold_candidate_recall": len(set(predicted) & set(gold)) / len(gold),
        "exact_candidate_set": predicted == gold,
    }


def mean(values: list[float | bool]) -> float:
    return float(sum(values) / len(values))


def independently_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    return {
        "record_count": len(rows),
        "stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in rows).items())
        ),
        "exact_json_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "schema_validity_rate": mean([row["schema_valid"] for row in rows]),
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
        "exact_candidate_set_accuracy": mean(
            [row["exact_candidate_set"] for row in rows]
        ),
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
        "action_or_tool_field_count": sum(
            row["action_or_tool_field_count"] for row in rows
        ),
        "unknown_candidate_id_count": sum(
            row["unknown_candidate_id_count"] for row in rows
        ),
        "duplicate_candidate_id_count": sum(
            row["duplicate_candidate_id_count"] for row in rows
        ),
    }


def independently_gate(
    metrics: dict[str, Any], gates: dict[str, Any], access: dict[str, int]
) -> dict[str, bool]:
    return {
        "complete_record_and_stratum_census": bool(
            metrics["record_count"] == gates["requiredRecordCount"]
            and metrics["stratum_counts"] == gates["requiredStratumCounts"]
        ),
        "exact_JSON_parse_rate": metrics["exact_json_parse_rate"]
        >= gates["minimumExactJSONParseRate"],
        "schema_validity_rate": metrics["schema_validity_rate"]
        >= gates["minimumSchemaValidityRate"],
        "none_of_the_above_inclusion_rate": metrics[
            "none_of_the_above_inclusion_rate"
        ]
        >= gates["minimumNoneOfTheAboveInclusionRate"],
        "mean_gold_candidate_recall": metrics["mean_gold_candidate_recall"]
        >= gates["minimumMeanGoldCandidateRecall"],
        "per_stratum_gold_candidate_recall": all(
            value >= gates["minimumPerStratumMeanGoldCandidateRecall"]
            for value in metrics["per_stratum_mean_gold_candidate_recall"].values()
        ),
        "exact_candidate_set_accuracy": metrics["exact_candidate_set_accuracy"]
        >= gates["minimumExactCandidateSetAccuracy"],
        "clear_exact_candidate_set_accuracy": metrics[
            "clear_exact_candidate_set_accuracy"
        ]
        >= gates["minimumClearExactCandidateSetAccuracy"],
        "out_of_ontology_exact_candidate_set_accuracy": metrics[
            "out_of_ontology_exact_candidate_set_accuracy"
        ]
        >= gates["minimumOutOfOntologyExactCandidateSetAccuracy"],
        "canonical_order_rate": metrics["canonical_order_rate"]
        >= gates["minimumCanonicalOrderRate"],
        "bounded_mean_candidate_count": metrics["mean_candidate_count"]
        <= gates["maximumMeanCandidateCount"],
        "zero_confidence_or_probability_fields": metrics[
            "confidence_or_probability_field_count"
        ]
        <= gates["maximumConfidenceOrProbabilityFieldCount"],
        "zero_action_or_tool_fields": metrics["action_or_tool_field_count"]
        <= gates["maximumActionOrToolFieldCount"],
        "zero_unknown_candidate_ids": metrics["unknown_candidate_id_count"]
        <= gates["maximumUnknownCandidateIdCount"],
        "zero_duplicate_candidate_ids": metrics["duplicate_candidate_id_count"]
        <= gates["maximumDuplicateCandidateIdCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_forward_pass_count"]
            <= gates["maximumModelForwardPassCount"]
            and access["API_call_count"] <= gates["maximumAPICallCount"]
            and access["adapter_training_run_count"]
            <= gates["maximumAdapterTrainingRunCount"]
            and access["human_record_access_count"]
            <= gates["maximumHumanRecordAccessCount"]
            and access["real_tool_call_count"] <= gates["maximumRealToolCallCount"]
            and access["external_side_effect_count"]
            <= gates["maximumExternalSideEffectCount"]
        ),
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def main() -> None:
    implementation_lock_path = (
        PROJECT_ROOT / "configs/v80-local-candidate-generation-implementation-lock.json"
    )
    evaluation_dir = PROJECT_ROOT / "outputs/v80-local-candidate-generation/evaluation"
    result_path = evaluation_dir / "result.json"
    access_path = evaluation_dir / "access.json"
    audit_path = PROJECT_ROOT / "outputs/v80-local-candidate-generation/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v80-local-candidate-generation-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v80_local_candidate_outcome.py"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V80 local-model outcome is already frozen")

    lock = json.loads(implementation_lock_path.read_text())
    lock_payload = {
        key: value for key, value in lock.items() if key != "lock_payload_sha256"
    }
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    records = [
        json.loads(line)
        for line in (PROJECT_ROOT / lock["corpus"]).read_text().splitlines()
        if line
    ]
    fixture_paths = sorted((evaluation_dir / "raw-fixtures").glob("*.json"))
    fixture_values = [json.loads(path.read_text()) for path in fixture_paths]
    fixtures_by_id = {fixture["id"]: fixture for fixture in fixture_values}
    independently_scored = [
        independently_score(record, fixtures_by_id[record["id"]]["raw_response"], config)
        for record in records
    ]
    metrics = independently_aggregate(independently_scored)
    gates = independently_gate(metrics, config["gates"], access)
    per_record_fields = (
        "id",
        "stratum",
        "exact_json_parse",
        "schema_valid",
        "candidate_ids",
        "candidate_count",
        "none_of_the_above_included",
        "canonical_order",
        "unknown_candidate_id_count",
        "duplicate_candidate_id_count",
        "confidence_or_probability_field_count",
        "action_or_tool_field_count",
        "gold_candidate_recall",
        "exact_candidate_set",
    )
    record_reproduction = all(
        all(close(row[field], fixtures_by_id[row["id"]][field]) for field in per_record_fields)
        for row in independently_scored
    )
    access_exact = bool(
        access["attempt_number"] == 1
        and access["model_load_count"] == 1
        and access["model_forward_pass_count"] == 24
        and all(
            access[key] == 0
            for key in (
                "API_call_count",
                "adapter_training_run_count",
                "human_record_access_count",
                "real_tool_call_count",
                "external_side_effect_count",
            )
        )
    )
    checks = {
        "implementation_lock_payload_valid": payload_hash(lock_payload)
        == lock["lock_payload_sha256"],
        "result_and_access_artifacts_exist": result_path.is_file()
        and access_path.is_file(),
        "exactly_one_fixture_per_locked_record_in_locked_order": bool(
            len(fixture_paths) == len(records) == 24
            and [fixture["id"] for fixture in fixture_values]
            == [record["id"] for record in records]
            and len(fixtures_by_id) == 24
        ),
        "independent_per_record_parse_and_score_reproduced": record_reproduction,
        "independent_metrics_reproduced": close(metrics, result["metrics"]),
        "independent_gates_reproduced": gates == result["gates"],
        "failure_decision_matches_noncompensatory_rule": bool(
            not all(gates.values())
            and not result["passed"]
            and result["decision"]
            == "freeze_local_candidate_generation_failure_without_prompt_edits_or_rerun"
        ),
        "one_load_twenty_four_generations_and_zero_external_access": access_exact,
        "result_attempt_matches_final_access": result["attempt"] == access,
        "no_model_API_training_human_or_tool_access_during_verification": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "80-local-candidate-generation-outcome-audit",
        "experiment": "v80_local_candidate_generation_independent_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_verified_negative_and_allow_only_fresh_successor_preregistration"
            if passed
            else "reject_V80_outcome_closure"
        ),
        "checks": checks,
        "independent_metrics": metrics,
        "independent_gates": gates,
        "access": {
            "model_load_count": 0,
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    fixture_manifest = {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in fixture_paths
    }
    outcome_lock = {
        "schema_version": "80-local-candidate-generation-outcome-lock",
        "experiment": "v80_local_candidate_generation_outcome_lock",
        "implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "access": str(access_path.relative_to(PROJECT_ROOT)),
        "access_sha256": file_sha256(access_path),
        "raw_fixture_manifest": fixture_manifest,
        "outcome_verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "outcome_verifier_sha256": file_sha256(verifier_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed": False,
            "exact_JSON_parse_rate": metrics["exact_json_parse_rate"],
            "schema_validity_rate": metrics["schema_validity_rate"],
            "mean_gold_candidate_recall": metrics["mean_gold_candidate_recall"],
            "exact_candidate_set_accuracy": metrics[
                "exact_candidate_set_accuracy"
            ],
            "out_of_ontology_exact_candidate_set_accuracy": metrics[
                "out_of_ontology_exact_candidate_set_accuracy"
            ],
            "mean_candidate_count": metrics["mean_candidate_count"],
        },
        "authorization": {
            "modify_or_rerun_V80_prompt_corpus_model_parser_decoding_or_gates": False,
            "run_V80_local_model_again": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "design_and_preregister_fresh_successor": True,
            "run_fresh_successor_model_before_its_implementation_lock": False,
        },
    }
    outcome_lock["lock_payload_sha256"] = payload_hash(outcome_lock)
    lock_path.write_text(json.dumps(outcome_lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
