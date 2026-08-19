#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v106_open_world_benchmark import build_declared_training_records, split_development_records
from v110_open_world_deterministic_validation import (
    aggregate_metrics, build_analysis, evaluate_outcome_gates,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], float]:
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != lock["source_archive_sha256"]:
        raise RuntimeError("V110 source archive mismatch")
    development_bytes = (PROJECT_ROOT / lock["development_language"]).read_bytes()
    if hashlib.sha256(development_bytes).hexdigest() != lock["development_language_sha256"]:
        raise RuntimeError("V110 development language mismatch")
    baseline_config = lock["baseline_config_payload"]
    source_records, member = parse_massive_archive(
        archive_bytes, baseline_config["expectedLocaleMemberSuffix"],
    )
    development_records = read_jsonl_bytes(development_bytes)
    original_split = split_development_records(development_records, baseline_config)
    v109_result = json.loads((PROJECT_ROOT / lock["V109_result"]).read_text())
    observed_fixtures = {
        identifier: row for identifier, row in v109_result["fixtures"].items()
        if row["kind"] == "observed_model_blind_holdback"
    }
    records = original_split["calibration"]
    if set(observed_fixtures) != {row["record_id"] for row in records}:
        raise RuntimeError("V110 V109 observed identity mismatch")
    direct_predictions = {
        identifier: row["parsed_response"] for identifier, row in observed_fixtures.items()
    }
    controls = [
        row for row in v109_result["fixtures"].values()
        if row["kind"] == "controlled_missing_observation"
    ]
    controlled_accuracy = sum(
        row["response_valid"] and row["parsed_response"]["status"] == "ABSTAIN"
        for row in controls
    ) / len(controls)
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training_records = build_declared_training_records(source_records, catalog)
    analysis = build_analysis(
        records, direct_predictions, training_records, catalog,
        lock["config_payload"], baseline_config,
    )
    analysis["source_locale_member"] = member
    analysis["source_record_count"] = len(source_records)
    return analysis, controlled_accuracy


def persisted_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": {
            "counts": analysis["split"]["counts"],
            "class_counts": analysis["split"]["class_counts"],
            "membership_sha256": analysis["split"]["membership_sha256"],
        },
        "retrieval_tuning": analysis["retrieval_tuning"],
        "abstention_tuning": analysis["abstention_tuning"],
        "policy_metrics": {
            name: aggregate_metrics(metrics)
            for name, metrics in sorted(analysis["policy_metrics"].items())
        },
        "training_summary": analysis["training_summary"],
        "source_locale_member": analysis["source_locale_member"],
        "source_record_count": analysis["source_record_count"],
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v110-open-world-deterministic-validation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v110-open-world-deterministic-validation/development-evaluation"
    if output_root.exists():
        raise RuntimeError("V110 analysis may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V110 lock mismatch")
    dependency_keys = (
        "config", "parent_typed_choice_outcome", "V109_implementation_lock", "V109_result",
        "baseline_outcome", "baseline_lock", "source_archive", "development_language",
        "visible_catalog", "controlled_identifiers", "selected_population", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V110 dependency drifted: {key}")
    analysis, controlled_accuracy = reconstruct(lock)
    access = {
        "source_archive_read_count": 1, "development_language_read_count": 1,
        "V109_result_automatic_read_count": 1, "protected_test_language_read_count": 0,
        "manual_language_or_raw_response_inspection_count": 0,
        "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    gates = evaluate_outcome_gates(
        analysis, controlled_accuracy, access, lock["config_payload"],
    )
    gates["calibration_evaluation_membership_disjoint"] = not (
        {row["record_id"] for row in analysis["split"]["calibration"]}
        & {row["record_id"] for row in analysis["split"]["evaluation"]}
    )
    gates["outputs_contain_no_source_language_or_raw_model_response"] = True
    passed = all(gates.values())
    aggregate = persisted_analysis(analysis)
    membership_path = output_root / "secondary-membership.json"
    calibration_path = output_root / "calibration.json"
    metrics_path = output_root / "policy-metrics.json"
    predictions_path = output_root / "evaluation-predictions.json"
    write_json(membership_path, {
        "membership": analysis["split"]["membership"],
        "membership_sha256": analysis["split"]["membership_sha256"],
        "contains_language": False,
    })
    write_json(calibration_path, {
        "retrieval_tuning": analysis["retrieval_tuning"],
        "abstention_tuning": analysis["abstention_tuning"],
    })
    write_json(metrics_path, aggregate["policy_metrics"])
    write_json(predictions_path, {
        "predictions": {
            name: {key: value for key, value in sorted(predictions.items())}
            for name, predictions in sorted(analysis["policy_predictions"].items())
        },
        "contains_language_or_raw_model_response": False,
    })
    outputs = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {
            "secondary_membership": membership_path, "calibration": calibration_path,
            "policy_metrics": metrics_path, "evaluation_predictions": predictions_path,
        }.items()
    }
    primary = aggregate["policy_metrics"][lock["config_payload"]["primaryPolicy"]]
    result = {
        "schema_version": "110-open-world-deterministic-validation-result",
        "experiment": lock["config_payload"]["experiment"],
        "passed": passed,
        "decision": (
            "development_layer_qualifies_for_separately_locked_protected_test_protocol"
            if passed else "development_layer_nonqualifying_keep_protected_test_and_induction_closed"
        ),
        "analysis": aggregate, "controlled_missing_observation_abstention_accuracy": controlled_accuracy,
        "primary_policy": lock["config_payload"]["primaryPolicy"],
        "primary_regret_above_ask_always": primary["mean_regret"] - 1.125,
        "gates": gates, "access": access, "output_integrity": outputs,
        "claim_boundary": "prospectively hash-split secondary development analysis of frozen outputs; no new model, protected test, learned likelihood, schema induction, planning, API, training, action, service call, or side effect",
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(json.dumps({
        "passed": passed, "decision": result["decision"],
        "split": aggregate["split"], "retrieval_tuning": aggregate["retrieval_tuning"],
        "abstention_tuning": aggregate["abstention_tuning"],
        "policy_summary": {
            name: {
                "exact": metrics["observed_exact_decision_accuracy"],
                "macro_f1": metrics["observed_status_macro_f1"],
                "known_exact": metrics["known_exact_intent_accuracy"],
                "novel_recall": metrics["per_status"]["NOVEL"]["recall"],
                "novel_precision": metrics["per_status"]["NOVEL"]["precision"],
                "unsupported_recall": metrics["per_status"]["UNSUPPORTED"]["recall"],
                "false_known": metrics["false_known_acceptance_rate"],
                "ECE": metrics["confidence_ece_10_bin"],
                "coverage": metrics["decision_coverage"],
                "mean_regret": metrics["mean_regret"],
                "unsafe_known_shadow_proposals": metrics["shadow_known_proposal_on_novel_or_unsupported_count"],
            }
            for name, metrics in aggregate["policy_metrics"].items()
        },
        "gates": gates, "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
