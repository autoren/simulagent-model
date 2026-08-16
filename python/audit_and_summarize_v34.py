#!/usr/bin/env python3
"""Reproduce V34 metrics, audit access, and write the durable result summary."""

from __future__ import annotations

import argparse
import json

import numpy as np

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v34_operation import qualification, score_operations, select_prompt_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v34-operation-interface-lock.json")
    parser.add_argument("--result", default="outputs/v34-operation-interface/result.json")
    parser.add_argument("--audit", default="outputs/v34-operation-interface/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v34-results.md")
    args = parser.parse_args()
    lock_path, result_path, audit_path, markdown_path = map(
        lambda value: (PROJECT_ROOT / value).resolve(),
        (args.lock, args.result, args.audit, args.markdown),
    )
    lock, result = json.loads(lock_path.read_text()), json.loads(result_path.read_text())
    config, errors = lock["config_payload"], []
    if result["protocol_lock_sha256"] != file_sha256(lock_path):
        errors.append("V34 result does not bind the protocol lock")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"V34 implementation changed: {path}")
    metadata_path = PROJECT_ROOT / result["feature_metadata"]
    prediction_path = PROJECT_ROOT / result["predictions"]
    if file_sha256(metadata_path) != result["feature_metadata_sha256"]:
        errors.append("V34 feature metadata changed")
    if file_sha256(prediction_path) != result["predictions_sha256"]:
        errors.append("V34 predictions changed")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    fit_rows = [row for row in rows if row["split"] == "factor_fit"]
    calibration_rows = [row for row in rows if row["split"] == "factor_calibration"]
    saved = np.load(prediction_path)
    reproduced = {}
    for name in config["readouts"]["methods"]:
        reproduced[name] = {
            "fit": score_operations(fit_rows, saved[f"{name}_fit"], {**config, "v32_config": lock["v32_config_payload"]}),
            "calibration": score_operations(calibration_rows, saved[f"{name}_calibration"], {**config, "v32_config": lock["v32_config_payload"]}),
        }
    metric_match = all(
        reproduced[name][split] == result["methods"][name][split]
        for name in reproduced for split in ("fit", "calibration")
    )
    selected = select_prompt_method(result["methods"])
    selected_qualification = qualification(
        reproduced[selected]["calibration"], reproduced["legacyHiddenRidge"]["calibration"], config
    )
    comparisons = {
        "metrics_reproduce": metric_match,
        "fit_selected_prompt_method": selected == result["fit_selected_prompt_method"],
        "qualification_reproduces": selected_qualification == result["qualification"],
    }
    if not all(comparisons.values()):
        errors.append("V34 result does not reproduce")
    access = result["data_access"]
    limits = config["limits"]
    access_checks = {
        "backbone_forward_passes": access["backbone_forward_passes"] == limits["backboneForwardPasses"],
        "ridge_training_fits": access["ridge_training_fits"] == limits["ridgeTrainingFits"],
        "no_calibration_selection": access["calibration_selections"] == 0,
        "no_v32_evaluation_records": access["v32_evaluation_records_read"] == 0,
        "no_v32_evaluation_features": access["v32_evaluation_features_read"] == 0,
        "no_v32_evaluation_predictions": access["v32_evaluation_predictions_read"] == 0,
        "no_adapter": access["adapter_training_runs"] == 0,
        "no_v28": access["v28_integration_replays"] == 0,
        "no_fresh_suite": access["fresh_suite_constructions"] == 0,
    }
    if not all(access_checks.values()):
        errors.append("V34 data-access ledger violates the lock")
    audit = {
        "schema_version": 34, "experiment": "v34_post_result_audit",
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "passed": not errors, "decision": "accept_v34_development_result" if not errors else "reject_v34_development_result",
        "errors": errors, "reproduction_checks": comparisons, "access_checks": access_checks,
        "fit_selected_prompt_method": selected, "qualification": selected_qualification,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    names = {
        "legacyHiddenRidge": "Legacy hidden ridge", "semanticHiddenRidge": "Focused hidden ridge",
        "nativeLogitRidge": "Native-logit ridge", "nativeArgmax": "Native argmax",
    }
    lines = [
        "# V34 results: operation-focused frozen interface", "",
        f"Decision: `{result['decision']}`.", "",
        "V34 is a fit/calibration-only representation diagnostic. It does not reuse V32 evaluation, open a fresh suite, or authorize V28.", "",
        "## Operation classification", "",
        "| Method | Fit | Calibration | Worst calibration operation | Oracle-sign compiled truth |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in config["readouts"]["methods"]:
        fit, cal = reproduced[name]["fit"], reproduced[name]["calibration"]
        lines.append(f"| {names[name]} | {fit['operation_accuracy']:.3f} | {cal['operation_accuracy']:.3f} | {cal['worst_operation_accuracy']:.3f} | {cal['oracle_sign_compiled_truth_accuracy']:.3f} |")
    lines.extend([
        "", "## Interpretation", "",
        f"Fit-only cross-validation selected `{selected}`. Its calibration operation gain over the legacy hidden-state ridge baseline is {selected_qualification['calibration_operation_gain_over_legacy']:+.3f}.", "",
        f"All registered qualification gates passed: `{str(selected_qualification['passed']).lower()}`.", "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
