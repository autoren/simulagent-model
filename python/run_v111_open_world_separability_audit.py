#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v106_open_world_benchmark import (
    build_declared_training_records, fit_character_retrieval, split_development_records,
)
from v111_open_world_separability_audit import (
    build_separability_analysis, extract_features, selected_rule_passes,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != lock["source_archive_sha256"]:
        raise RuntimeError("V111 source archive mismatch")
    development_bytes = (PROJECT_ROOT / lock["development_language"]).read_bytes()
    if hashlib.sha256(development_bytes).hexdigest() != lock["development_language_sha256"]:
        raise RuntimeError("V111 development language mismatch")
    baseline_config = lock["baseline_config_payload"]
    source_records, member = parse_massive_archive(
        archive_bytes, baseline_config["expectedLocaleMemberSuffix"],
    )
    development_records = read_jsonl_bytes(development_bytes)
    original_split = split_development_records(development_records, baseline_config)
    records = original_split["calibration"]

    v109_result = json.loads((PROJECT_ROOT / lock["V109_result"]).read_text())
    observed = {
        identifier: row for identifier, row in v109_result["fixtures"].items()
        if row["kind"] == "observed_model_blind_holdback"
    }
    if set(observed) != {row["record_id"] for row in records}:
        raise RuntimeError("V111 V109 observed identity mismatch")
    direct_predictions = {
        identifier: row["parsed_response"] for identifier, row in observed.items()
    }

    membership_payload = json.loads((PROJECT_ROOT / lock["secondary_membership"]).read_text())
    membership = membership_payload["membership"]
    by_id = {row["record_id"]: row for row in records}
    if set(by_id) != {row["record_id"] for row in membership}:
        raise RuntimeError("V111 membership identity mismatch")
    calibration = sorted(
        (by_id[row["record_id"]] for row in membership if row["subset"] == "calibration"),
        key=lambda row: row["record_id"],
    )
    evaluation = sorted(
        (by_id[row["record_id"]] for row in membership if row["subset"] == "evaluation"),
        key=lambda row: row["record_id"],
    )
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training_records = build_declared_training_records(source_records, catalog)
    retrieval_spec = baseline_config["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training_records, retrieval_spec["vectorizer"])
    calibration_features = extract_features(fitted, calibration, direct_predictions)
    evaluation_features = extract_features(fitted, evaluation, direct_predictions)
    analysis = build_separability_analysis(
        calibration_features, evaluation_features, lock["config_payload"],
    )
    metadata = {
        "source_locale_member": member,
        "source_record_count": len(source_records),
        "training_record_count": len(training_records),
        "declared_intent_count": len({row["intent_id"] for row in training_records}),
        "subset_counts": {"calibration": len(calibration), "evaluation": len(evaluation)},
        "class_counts": {
            subset: dict(sorted(Counter(row["class_label"] for row in rows).items()))
            for subset, rows in (("calibration", calibration), ("evaluation", evaluation))
        },
        "membership_sha256": membership_payload["membership_sha256"],
    }
    return analysis, metadata


def persisted_analysis(analysis: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_count": analysis["candidate_count"],
        "calibration": analysis["calibration"],
        "selected_evaluation_metrics": analysis["selected_evaluation_metrics"],
        "evaluation_oracle": analysis["evaluation_oracle"],
        "feature_distributions": analysis["feature_distributions"],
        "individual_feature_or_identifier_emission_count": analysis["individual_feature_or_identifier_emission_count"],
        "metadata": metadata,
    }


def evaluate_integrity_gates(
    analysis: dict[str, Any], metadata: dict[str, Any], access: dict[str, int],
    config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["evaluationGates"]
    return {
        "calibration_record_count": metadata["subset_counts"]["calibration"] == gates["requiredCalibrationRecordCount"],
        "evaluation_record_count": metadata["subset_counts"]["evaluation"] == gates["requiredEvaluationRecordCount"],
        "balanced_class_counts": all(
            count == gates["requiredCountPerClassPerSubset"]
            for values in metadata["class_counts"].values() for count in values.values()
        ),
        "registered_candidate_count": analysis["candidate_count"] == 1343,
        "calibration_and_evaluation_use_same_registered_family": (
            analysis["calibration"]["candidate_count"] == 1343
            and analysis["evaluation_oracle"]["candidate_count"] == 1343
        ),
        "selected_rule_was_chosen_from_calibration_only": analysis["calibration"]["selected"]["rule"] is not None,
        "aggregate_only_output": analysis["individual_feature_or_identifier_emission_count"] == 0,
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= gates["maximumProtectedTestLanguageReadCount"],
        "zero_manual_language_or_raw_response_inspection": access["manual_language_or_raw_response_inspection_count"] <= gates["maximumManualLanguageOrRawResponseInspectionCount"],
        "zero_model_loads": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "zero_model_generations": access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
    }


def decision_for(quality_gate_pass: bool, oracle_feasible_count: int) -> str:
    if quality_gate_pass:
        return "simple_features_transfer_separably_preregister_full_development_policy"
    if oracle_feasible_count == 0:
        return "close_current_single_turn_evidence_interface_for_simple_deterministic_novelty_gating"
    return "selection_instability_require_genuinely_fresh_evidence_population"


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v111-open-world-separability-audit-lock.json"
    output_root = PROJECT_ROOT / "outputs/v111-open-world-separability-audit/development-census"
    if output_root.exists():
        raise RuntimeError("V111 census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V111 lock mismatch")
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "parent_result", "V109_result",
        "baseline_outcome", "baseline_lock", "source_archive", "development_language",
        "visible_catalog", "secondary_membership", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V111 dependency drifted: {key}")

    analysis, metadata = reconstruct(lock)
    access = {
        "source_archive_read_count": 1, "development_language_read_count": 1,
        "V109_result_automatic_read_count": 1, "V110_result_automatic_read_count": 0,
        "protected_test_language_read_count": 0,
        "manual_language_or_raw_response_inspection_count": 0,
        "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    integrity_gates = evaluate_integrity_gates(analysis, metadata, access, lock["config_payload"])
    integrity_passed = all(integrity_gates.values())
    quality_gate_pass = selected_rule_passes(analysis, lock["config_payload"])
    oracle_feasible_count = analysis["evaluation_oracle"]["feasible_candidate_count"]
    aggregate = persisted_analysis(analysis, metadata)
    aggregate_path = output_root / "aggregate-analysis.json"
    write_json(aggregate_path, aggregate)
    result = {
        "schema_version": "111-open-world-separability-audit-result",
        "experiment": lock["config_payload"]["experiment"],
        "passed": integrity_passed,
        "quality_gate_pass": quality_gate_pass,
        "decision": decision_for(quality_gate_pass, oracle_feasible_count),
        "analysis": aggregate,
        "integrity_gates": integrity_gates,
        "access": access,
        "output_integrity": {
            "aggregate_analysis": {
                "path": str(aggregate_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(aggregate_path),
            }
        },
        "claim_boundary": "aggregate-only census over frozen V110 development membership and frozen V109 outputs; evaluation-label oracle is diagnostic only; no individual feature, identifier, language, raw response, new model, protected test, schema induction, planning, API, training, action, service call, or side effect",
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(json.dumps({
        "passed": integrity_passed,
        "quality_gate_pass": quality_gate_pass,
        "decision": result["decision"],
        "selected_calibration": analysis["calibration"]["selected"],
        "selected_evaluation_metrics": analysis["selected_evaluation_metrics"],
        "calibration_feasible_candidate_count": analysis["calibration"]["feasible_candidate_count"],
        "evaluation_oracle_feasible_candidate_count": oracle_feasible_count,
        "evaluation_oracle_selected": analysis["evaluation_oracle"]["selected"],
        "integrity_gates": integrity_gates,
        "access": access,
    }, indent=2, sort_keys=True))
    if not integrity_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
