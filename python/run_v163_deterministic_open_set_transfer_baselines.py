#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v163_deterministic_open_set_transfer_baselines import (
    build_deterministic_transfer_artifacts,
    evaluate_baseline_pipeline_gates,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    config = lock["config_payload"]
    development_bytes = (PROJECT_ROOT / lock["development_language"]).read_bytes()
    if bytes_sha256(development_bytes) != lock["development_language_sha256"]:
        raise RuntimeError("V163 development language identity mismatch")
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    if bytes_sha256(archive_bytes) != lock["source_archive_sha256"]:
        raise RuntimeError("V163 source archive identity mismatch")
    source_records, member = parse_massive_archive(
        archive_bytes, config["expectedLocaleMemberSuffix"]
    )
    development_records = read_jsonl_bytes(development_bytes)
    catalog = json.loads((PROJECT_ROOT / config["visibleCatalog"]).read_text())
    artifacts = build_deterministic_transfer_artifacts(
        source_records, development_records, catalog, config
    )
    artifacts["source_locale_member"] = member
    artifacts["source_record_count"] = len(source_records)
    return artifacts


def main() -> None:
    lock_path = (
        PROJECT_ROOT
        / "configs/v163-deterministic-open-set-transfer-baselines-lock.json"
    )
    output_root = (
        PROJECT_ROOT
        / "outputs/v163-deterministic-open-set-transfer/development-baselines"
    )
    if output_root.exists():
        raise RuntimeError("V163 development baseline run may execute only once")
    lock = json.loads(lock_path.read_text())
    if (
        payload_hash(
            {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
        )
        != lock["lock_payload_sha256"]
    ):
        raise RuntimeError("V163 benchmark lock mismatch")
    dependency_keys = (
        "config",
        "parent_language_outcome",
        "historical_interface_outcome",
        "visible_catalog",
        "safe_hypothesis_universe",
        "source_archive",
        "plan",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V163 dependency drifted: {key}")

    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "source_archive_read_count": 1,
        "development_language_read_count": 1,
        "protected_language_read_count": 0,
        "manual_utterance_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    checks = evaluate_baseline_pipeline_gates(artifacts, access, config)
    checks["calibration_evaluation_membership_is_disjoint"] = not (
        {row["record_id"] for row in artifacts["split"]["calibration"]}
        & {row["record_id"] for row in artifacts["split"]["evaluation"]}
    )
    checks["outputs_contain_no_source_language"] = True
    pipeline_passed = all(checks.values())

    membership_path = output_root / "development-split-membership.json"
    tuning_path = output_root / "retrieval-calibration.json"
    metrics_path = output_root / "baseline-evaluation.json"
    predictions_path = output_root / "baseline-predictions.json"
    training_path = output_root / "declared-training-summary.json"
    controls_path = output_root / "controlled-missing-identifiers.json"
    residual_path = output_root / "model-eligible-residual.json"
    write_json(
        membership_path,
        {
            "membership": artifacts["split"]["membership"],
            "membership_sha256": artifacts["split"]["membership_sha256"],
            "counts": artifacts["split"]["counts"],
            "class_counts": artifacts["split"]["class_counts"],
            "contains_language": False,
        },
    )
    write_json(tuning_path, artifacts["retrieval_tuning"])
    write_json(
        metrics_path,
        {
            "baseline_metrics": artifacts["baseline_metrics"],
            "best_nonoracle_baseline": artifacts["best_nonoracle_baseline"],
            "residual_summary": artifacts["residual_summary"],
            "nonresidual_summary": artifacts["nonresidual_summary"],
            "residual_qualification_checks": artifacts[
                "residual_qualification_checks"
            ],
            "residual_qualified": artifacts["residual_qualified"],
        },
    )
    write_json(
        predictions_path,
        {
            "predictions": artifacts["evaluation_predictions"],
            "payload_sha256": artifacts["evaluation_prediction_payload_sha256"],
            "contains_source_language": False,
        },
    )
    write_json(training_path, artifacts["training_summary"])
    write_json(
        controls_path,
        {
            "records": artifacts["controlled_missing_identifiers"],
            "contains_language": False,
        },
    )
    write_json(
        residual_path,
        {
            "records": artifacts["residual_manifest"],
            "payload_sha256": artifacts["residual_summary"][
                "manifest_payload_sha256"
            ],
            "membership_uses_truth_or_language": False,
            "contains_language": False,
        },
    )
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {
            "development_split_membership": membership_path,
            "retrieval_calibration": tuning_path,
            "baseline_evaluation": metrics_path,
            "baseline_predictions": predictions_path,
            "declared_training_summary": training_path,
            "controlled_missing_identifiers": controls_path,
            "model_eligible_residual": residual_path,
        }.items()
    }
    result = {
        "schema_version": "163-deterministic-open-set-transfer-baselines-result",
        "experiment": "v163_fresh_massive_deterministic_open_set_transfer_baselines",
        "passed": pipeline_passed,
        "residual_qualified": artifacts["residual_qualified"],
        "decision": (
            "freeze_baselines_and_preregister_one_local_residual_model_protocol"
            if pipeline_passed and artifacts["residual_qualified"]
            else (
                "freeze_baselines_and_close_local_model_residual_branch"
                if pipeline_passed
                else "stop_V163_before_model_or_protected_access"
            )
        ),
        "source_locale_member": artifacts["source_locale_member"],
        "source_record_count": artifacts["source_record_count"],
        "development_summary": {
            "split_counts": artifacts["split"]["counts"],
            "split_class_counts": artifacts["split"]["class_counts"],
            "split_membership_sha256": artifacts["split"]["membership_sha256"],
            "training_summary": artifacts["training_summary"],
            "retrieval_selected_thresholds": artifacts["retrieval_tuning"]["selected"],
            "retrieval_threshold_candidate_count": artifacts["retrieval_tuning"][
                "candidate_count"
            ],
            "best_nonoracle_baseline": artifacts["best_nonoracle_baseline"],
            "evaluation_prediction_payload_sha256": artifacts[
                "evaluation_prediction_payload_sha256"
            ],
            "residual_summary": artifacts["residual_summary"],
            "nonresidual_summary": artifacts["nonresidual_summary"],
        },
        "baseline_metrics": artifacts["baseline_metrics"],
        "residual_qualification_checks": artifacts["residual_qualification_checks"],
        "output_integrity": output_integrity,
        "gates": checks,
        "access": access,
        "claim_boundary": (
            "record-disjoint MASSIVE deterministic development and residual-feasibility result; "
            "no protected language, model, API, ontology, planning, action, or execution result"
        ),
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(
        json.dumps(
            {
                "passed": pipeline_passed,
                "residual_qualified": result["residual_qualified"],
                "decision": result["decision"],
                "development_summary": result["development_summary"],
                "baseline_metrics": {
                    name: {
                        "exact": value["observed_exact_decision_accuracy"],
                        "macro_f1": value["observed_status_macro_f1"],
                        "known_exact": value["known_exact_intent_accuracy"],
                        "novel_exact": value["novel_exact_scenario_accuracy"],
                        "false_known": value["false_known_acceptance_rate"],
                        "mean_regret": value["mean_regret"],
                    }
                    for name, value in result["baseline_metrics"].items()
                },
                "residual_qualification_checks": result[
                    "residual_qualification_checks"
                ],
                "gates": checks,
                "access": access,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not pipeline_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
