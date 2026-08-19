#!/usr/bin/env python3
"""Independently reconstruct and freeze the negative V88r1 outcome."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[key], b[key], tolerance) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tolerance
    return a == b


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def precision(predicted: set[str], gold: set[str]) -> float:
    return len(predicted & gold) / len(predicted) if predicted else (1.0 if not gold else 0.0)


def recall(predicted: set[str], gold: set[str]) -> float:
    return len(predicted & gold) / len(gold) if gold else (1.0 if not predicted else 0.0)


def rescore(record: dict[str, Any], response: str) -> dict[str, Any]:
    parsed = None
    exact_json = False
    exact_keys = False
    well_formed = False
    unique = False
    conformant = False
    intents: list[str] = []
    slots: list[str] = []
    try:
        parsed = json.loads(response.strip())
        exact_json = isinstance(parsed, dict)
    except (json.JSONDecodeError, TypeError):
        pass
    if exact_json:
        exact_keys = set(parsed) == {"intent_candidates", "state_slot_key_candidates"}
        raw_intents = parsed.get("intent_candidates")
        raw_slots = parsed.get("state_slot_key_candidates")
        well_formed = bool(
            isinstance(raw_intents, list) and raw_intents and isinstance(raw_slots, list)
            and all(isinstance(item, str) for item in raw_intents + raw_slots)
        )
        if well_formed:
            intents, slots = raw_intents, raw_slots
            unique = len(intents) == len(set(intents)) and len(slots) == len(set(slots))
            conformant = bool(
                exact_keys and unique and set(intents) <= set(record["allowed_intent_ids"])
                and set(slots) <= set(record["allowed_slot_ids"])
            )
    predicted_intents = set(intents) if conformant else set()
    predicted_slots = set(slots) if conformant else set()
    gold_intents = set(record["gold"]["intent_candidates"])
    gold_slots = set(record["gold"]["state_slot_key_candidates"])
    active = record["gold"]["active_intent"]
    return {
        "exact_json": exact_json, "exact_keys": exact_keys, "lists_well_formed": well_formed,
        "unique_lists": unique, "ontology_conformant": conformant,
        "intent_candidates": intents, "state_slot_key_candidates": slots,
        "mandatory_NONE_included": "NONE" in predicted_intents,
        "gold_active_intent_covered": active == "NONE" or active in predicted_intents,
        "intent_candidate_precision": precision(predicted_intents, gold_intents),
        "intent_candidate_recall": recall(predicted_intents, gold_intents),
        "intent_candidate_exact": predicted_intents == gold_intents,
        "none_only_intent_exact": active != "NONE" or predicted_intents == {"NONE"},
        "state_slot_key_precision": precision(predicted_slots, gold_slots),
        "state_slot_key_recall": recall(predicted_slots, gold_slots),
        "state_slot_key_exact": predicted_slots == gold_slots,
        "intent_candidate_count": len(predicted_intents),
    }


def control(record: dict[str, Any], kind: str) -> dict[str, float | bool]:
    gold_i = set(record["gold"]["intent_candidates"])
    gold_s = set(record["gold"]["state_slot_key_candidates"])
    if kind == "exhaustive":
        pred_i, pred_s = set(record["allowed_intent_ids"]), set(record["allowed_slot_ids"])
    elif kind == "none_only":
        pred_i, pred_s = {"NONE"}, set()
    elif kind == "empty_state_gold_intent":
        pred_i, pred_s = gold_i, set()
    else:
        pred_i, pred_s = gold_i, gold_s
    return {
        "intent_exact": pred_i == gold_i, "intent_precision": precision(pred_i, gold_i),
        "intent_recall": recall(pred_i, gold_i), "state_exact": pred_s == gold_s,
        "state_precision": precision(pred_s, gold_s), "state_recall": recall(pred_s, gold_s),
    }


def main() -> None:
    impl_path = PROJECT_ROOT / "configs/v88r1-name-preservation-repair-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/evaluation/result.json"
    access_path = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/evaluation/access.json"
    raw_root = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/evaluation/raw-fixtures"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v88r1_external_candidate_outcome.py"
    doc_path = PROJECT_ROOT / "docs/v88r1-external-intent-candidate-results.md"
    audit_path = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v88r1-external-intent-candidate-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V88r1 outcome is already frozen")

    impl = json.loads(impl_path.read_text())
    impl_payload = {key: value for key, value in impl.items() if key != "lock_payload_sha256"}
    config = impl["config_payload"]
    result = json.loads(result_path.read_text())
    retry_access = json.loads(access_path.read_text())
    records = [json.loads(line) for line in (PROJECT_ROOT / impl["corpus"]).read_text().splitlines() if line]
    records_by_name = {record["id"]: record for record in records}
    raw_rows = [json.loads(path.read_text()) for path in sorted(raw_root.glob("*.json"))]
    raw_by_name = {row["name"]: row for row in raw_rows}

    rescored = []
    row_checks = []
    per_service = defaultdict(list)
    for name, record in records_by_name.items():
        raw = raw_by_name[name]
        score = rescore(record, raw["response"])
        row_checks.append(all(close(raw[key], value) for key, value in score.items()))
        merged = {**score, "service": record["service"], "active_intent": record["gold"]["active_intent"]}
        rescored.append(merged)
        per_service[record["service"]].append(merged["intent_candidate_recall"])

    control_kinds = ("exhaustive", "none_only", "empty_state_gold_intent", "oracle")
    control_metrics = {}
    for kind in control_kinds:
        rows = [control(record, kind) for record in records]
        control_metrics[kind] = {
            "intent_exact_rate": mean([float(row["intent_exact"]) for row in rows]),
            "intent_precision": mean([float(row["intent_precision"]) for row in rows]),
            "intent_recall": mean([float(row["intent_recall"]) for row in rows]),
            "state_exact_rate": mean([float(row["state_exact"]) for row in rows]),
            "state_precision": mean([float(row["state_precision"]) for row in rows]),
            "state_recall": mean([float(row["state_recall"]) for row in rows]),
        }
    active_rows = [row for row in rescored if row["active_intent"] != "NONE"]
    none_rows = [row for row in rescored if row["active_intent"] == "NONE"]
    intent_exact_rate = mean([float(row["intent_candidate_exact"]) for row in rescored])
    metrics = {
        "record_count": len(rescored),
        "service_count": len(per_service),
        "active_intent_label_count": len({row["active_intent"] for row in active_rows}),
        "none_record_count": len(none_rows),
        "exact_JSON_parse_rate": mean([float(row["exact_json"]) for row in rescored]),
        "ontology_conformance_rate": mean([float(row["ontology_conformant"]) for row in rescored]),
        "mandatory_NONE_inclusion_rate": mean([float(row["mandatory_NONE_included"]) for row in rescored]),
        "permanent_non_deployable_rate": 1.0,
        "gold_active_intent_coverage_rate": mean([float(row["gold_active_intent_covered"]) for row in active_rows]),
        "intent_candidate_set_exact_rate": intent_exact_rate,
        "intent_candidate_precision": mean([row["intent_candidate_precision"] for row in rescored]),
        "intent_candidate_recall": mean([row["intent_candidate_recall"] for row in rescored]),
        "none_only_intent_exact_rate": mean([float(row["intent_candidate_exact"]) for row in none_rows]),
        "per_service_intent_candidate_recall": {key: mean(value) for key, value in sorted(per_service.items())},
        "state_slot_key_precision": mean([row["state_slot_key_precision"] for row in rescored]),
        "state_slot_key_recall": mean([row["state_slot_key_recall"] for row in rescored]),
        "state_slot_key_exact_rate": mean([float(row["state_slot_key_exact"]) for row in rescored]),
        "intent_exact_improvement_over_exhaustive": intent_exact_rate - control_metrics["exhaustive"]["intent_exact_rate"],
        "mean_intent_candidate_count": mean([float(row["intent_candidate_count"]) for row in rescored]),
        "controls": control_metrics,
    }
    gates = config["gates"]
    reconstructed_gates = {
        "required_record_count": metrics["record_count"] == gates["requiredRecordCount"],
        "required_service_count": metrics["service_count"] == gates["requiredServiceCount"],
        "required_active_intent_label_count": metrics["active_intent_label_count"] == gates["requiredActiveIntentLabelCount"],
        "required_NONE_record_count": metrics["none_record_count"] == gates["requiredNoneRecordCount"],
        "exact_JSON_parse": metrics["exact_JSON_parse_rate"] >= gates["minimumExactJSONParseRate"],
        "ontology_conformance": metrics["ontology_conformance_rate"] >= gates["minimumOntologyConformanceRate"],
        "mandatory_NONE_inclusion": metrics["mandatory_NONE_inclusion_rate"] >= gates["minimumMandatoryNoneInclusionRate"],
        "permanent_non_deployable": metrics["permanent_non_deployable_rate"] >= gates["minimumPermanentNonDeployableRate"],
        "gold_active_intent_coverage": metrics["gold_active_intent_coverage_rate"] >= gates["minimumGoldActiveIntentCoverageRate"],
        "intent_candidate_set_exact": metrics["intent_candidate_set_exact_rate"] >= gates["minimumIntentCandidateSetExactRate"],
        "NONE_only_intent_exact": metrics["none_only_intent_exact_rate"] >= gates["minimumNoneOnlyIntentExactRate"],
        "per_service_intent_candidate_recall": all(value >= gates["minimumPerServiceIntentCandidateRecallRate"] for value in metrics["per_service_intent_candidate_recall"].values()),
        "state_slot_key_recall": metrics["state_slot_key_recall"] >= gates["minimumStateSlotKeyRecallRate"],
        "state_slot_key_exact": metrics["state_slot_key_exact_rate"] >= gates["minimumStateSlotKeyExactRate"],
        "intent_exact_improvement_over_exhaustive": metrics["intent_exact_improvement_over_exhaustive"] >= gates["minimumIntentExactImprovementOverExhaustive"],
        "mean_intent_candidate_count": metrics["mean_intent_candidate_count"] <= gates["maximumMeanIntentCandidateCount"],
        "model_load_budget": retry_access["model_load_count"] <= gates["maximumModelLoadCount"],
        "model_generation_budget": retry_access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": retry_access["LLM_API_call_count"] <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": retry_access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "zero_manual_utterance_inspection": retry_access["manual_utterance_inspection_count"] <= gates["maximumManualUtteranceInspectionCount"],
        "zero_real_service_calls": retry_access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": retry_access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
        "cumulative_model_load_budget": result["cumulative_access"]["model_load_count"] <= impl["cumulative_resource_budget"]["maximum_model_load_count"],
        "cumulative_model_generation_budget": result["cumulative_access"]["model_generation_count"] <= impl["cumulative_resource_budget"]["maximum_model_generation_count"],
    }
    valid_rows = [row for row in rescored if row["ontology_conformant"]]
    valid_active = [row for row in valid_rows if row["active_intent"] != "NONE"]
    diagnostic = {
        "ontology_conforming_record_count": len(valid_rows),
        "malformed_by_service": {
            service: sum(not row["exact_json"] for row in service_rows)
            for service, service_rows in sorted((key, [row for row in rescored if row["service"] == key]) for key in per_service)
        },
        "conditional_intent_exact_rate": mean([float(row["intent_candidate_exact"]) for row in valid_rows]),
        "conditional_active_intent_coverage_rate": mean([float(row["gold_active_intent_covered"]) for row in valid_active]),
        "conditional_state_slot_key_recall": mean([row["state_slot_key_recall"] for row in valid_rows]),
        "conditional_state_slot_key_exact_rate": mean([float(row["state_slot_key_exact"]) for row in valid_rows]),
    }
    checks = {
        "implementation_lock_and_all_frozen_dependencies_exact": bool(
            payload_hash(impl_payload) == impl["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / impl[key]) == impl[f"{key}_sha256"] for key in (
                "repair_design_lock", "original_implementation_lock", "corpus_seal", "corpus",
                "protocol", "runner", "census_harness", "tests", "implementation_auditor"
            ))
        ),
        "all_48_raw_fixtures_present_unique_and_match_result": bool(
            len(records) == len(raw_rows) == len(raw_by_name) == 48
            and set(raw_by_name) == set(records_by_name) == set(result["fixtures"])
            and all(close(raw_by_name[name], result["fixtures"][name]) for name in raw_by_name)
        ),
        "every_raw_fixture_independently_rescores": all(row_checks),
        "metrics_and_controls_independently_reconstruct": close(metrics, result["metrics"]),
        "all_registered_gates_independently_reconstruct": reconstructed_gates == result["gates"],
        "negative_decision_is_exact_and_terminal": bool(
            not result["passed"]
            and result["decision"] == "freeze_negative_V88r1_without_any_further_retry_or_change"
            and not impl["authorization"]["rerun_after_any_outcome_or_execution_failure"]
        ),
        "retry_and_cumulative_access_exact_with_zero_API_training_inspection_execution_or_side_effect": bool(
            retry_access["model_load_count"] == 1 and retry_access["model_generation_count"] == 48
            and result["cumulative_access"]["model_load_count"] == 2
            and result["cumulative_access"]["model_generation_count"] == 49
            and all(result["cumulative_access"][key] == 0 for key in (
                "LLM_API_call_count", "adapter_training_run_count", "manual_utterance_inspection_count",
                "real_service_call_count", "external_side_effect_count"
            ))
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "88r1-external-intent-candidate-outcome-audit",
        "experiment": "v88r1_external_intent_candidate_outcome_audit",
        "passed": passed,
        "decision": "freeze_verified_negative_V88r1_outcome" if passed else "reject_V88r1_outcome_artifact",
        "checks": checks,
        "independent_metrics": metrics,
        "post_outcome_model_free_diagnostic": diagnostic,
        "claim_boundary": result["claim_boundary"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "88r1-external-intent-candidate-outcome-lock",
        "experiment": "v88r1_external_intent_candidate_outcome_lock",
        "implementation_lock": str(impl_path.relative_to(PROJECT_ROOT)), "implementation_lock_sha256": file_sha256(impl_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "access": str(access_path.relative_to(PROJECT_ROOT)), "access_sha256": file_sha256(access_path),
        "raw_fixture_artifacts": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
            for path in sorted(raw_root.glob("*.json"))
        ],
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)), "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)), "audit_sha256": file_sha256(audit_path),
        "results_document": str(doc_path.relative_to(PROJECT_ROOT)), "results_document_sha256": file_sha256(doc_path),
        "outcome": {
            "passed": False,
            "decision": result["decision"],
            "metrics": result["metrics"],
            "post_outcome_model_free_diagnostic": diagnostic,
            "cumulative_access": result["cumulative_access"],
        },
        "authorization": {
            "modify_rerun_repair_or_resume_V88_or_V88r1": False,
            "retain_V83_V86_model_free_interface_and_frozen_Bayesian_core": True,
            "deploy_or_execute_any_model_output": False,
            "use_local_model_to_define_live_hypothesis_set_or_state": False,
            "run_API_fallback_or_capacity_comparator": False,
            "train_adapter_or_learned_likelihood": False,
            "preregister_model_free_failure_decomposition_only": True,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
